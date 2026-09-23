# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""``SnodeCostProfile`` / ``SnodeCostTable``, on synthetic snode lists.

The profile pass decides what gets priced, in what order the estimator is
driven (warm-up, then the rank-lockstep sync, then the re-read), and what the
passes after it see.  Inductor's wait / collective / weight-gather predicates
are decided by ``isinstance`` against IR classes no synthetic node can be, so
those are stubbed by kind; the weight-load predicate is the real one.
"""

import copy
import gc
import pickle

import pytest

from magi_compiler.passes import snode_cost, snode_utils
from magi_compiler.passes.snode_cost import SnodeCostProfile, SnodeCostTable
from magi_compiler.profiling import ProfilingRuntimeEstimator
from magi_compiler.profiling import runtime_estimator as re_mod
from magi_compiler.profiling.runtime_estimator import ProfileEntry


class _IR:
    def __init__(self, op_overload):
        self.op_overload = op_overload


class _Snode:
    snodes = None

    def __init__(self, name, kind):
        from magi_compiler.passes.weight_offload.h2d_op import H2D_LOAD

        self.name = name
        self.kind = kind
        self.node = _IR(H2D_LOAD if kind == "load" else "fake.op")

    def get_name(self):
        return self.name

    def __repr__(self):
        return f"<{self.kind} {self.name}>"


@pytest.fixture(autouse=True)
def stub_predicates(monkeypatch):
    is_wait = lambda s: getattr(s, "kind", None) == "wait"  # noqa: E731
    is_gather = lambda s: getattr(s, "kind", None) == "gather"  # noqa: E731
    monkeypatch.setattr(snode_utils, "contains_wait", is_wait)
    monkeypatch.setattr(snode_utils, "issues_transfer", is_gather)
    monkeypatch.setattr(snode_cost, "issues_transfer", is_gather)
    monkeypatch.setattr(snode_cost, "is_weight_gather", is_gather)


class _Estimator:
    """The surface of ``ProfilingRuntimeEstimator`` the profile pass drives."""

    def __init__(self, sync=True, fail_sync=False):
        self._sync_across_ranks = sync
        self.fail_sync = fail_sync
        self.warmed: list[str] = []
        self.synced = False

    def __call__(self, s):
        self.warmed.append(s.name)
        return 100.0

    def warm_and_sync(self):
        if self.fail_sync:
            raise RuntimeError("gloo went away")
        self.synced = True
        return 1

    def requery(self, s):
        return 200.0 if self.synced else 100.0


def _graph(*kinds):
    return [_Snode(f"{k}{i}", k) for i, k in enumerate(kinds)]


def test_prices_compute_and_gathers_never_a_load_or_a_wait():
    order = _graph("compute", "load", "wait", "gather", "compute")
    c0, load, wait, gather, c4 = order
    est, table = _Estimator(), SnodeCostTable()
    out = SnodeCostProfile(table, est)(order)

    assert out is order, "the profile pass must never reorder"
    assert est.warmed == ["compute0", "gather3", "compute4"]
    assert est.synced
    # The passes after it read the synced value, not the warm-up seed.
    assert table(c0) == table(gather) == table(c4) == 200.0
    assert table.ok
    assert load not in table and wait not in table
    assert table(load) == 0.0, "a load is priced from bytes / bandwidth, never by the table"


def test_a_load_alone_is_reason_to_price_the_compute_around_it():
    order = _graph("compute", "load", "wait", "compute")
    est = _Estimator()
    SnodeCostProfile(SnodeCostTable(), est)(order)
    assert est.warmed == ["compute0", "compute3"]


def test_graph_without_gather_or_load_is_not_priced():
    est, table = _Estimator(), SnodeCostTable()
    SnodeCostProfile(table, est)(_graph("compute", "compute"))
    assert est.warmed == [] and not est.synced
    assert len(table) == 0


def test_analytical_mode_records_the_same_snodes():
    order = _graph("compute", "load", "wait", "gather")
    table = SnodeCostTable()
    SnodeCostProfile(table, None)(order)
    assert [s.name for s in order if s in table] == ["compute0", "gather3"]


def test_sync_failure_marks_the_table_but_still_records():
    order = _graph("compute", "gather")
    table = SnodeCostTable()
    SnodeCostProfile(table, _Estimator(fail_sync=True))(order)
    assert not table.ok
    assert table(order[0]) == 100.0


def test_each_compile_starts_from_an_empty_table():
    table = SnodeCostTable()
    profile = SnodeCostProfile(table, _Estimator())
    first = _graph("compute", "gather")
    profile(first)
    table.ok = False
    second = _graph("gather", "compute")
    profile(second)
    assert first[0] not in table and second[1] in table
    assert table.ok


def test_a_miss_on_a_priced_kind_warns_once_per_compile(monkeypatch):
    """Compute and collectives are always recorded by the profile pass, so a
    miss on one means the chain is broken; waits and loads are never recorded,
    so missing them is silent."""
    warned: list = []
    monkeypatch.setattr(snode_cost.magi_logger, "warning", lambda msg, *a, **k: warned.append(a[0]))
    table = SnodeCostTable()
    SnodeCostProfile(table, _Estimator())(_graph("gather", "compute"))

    for s in _graph("wait", "load"):
        table(s)
    assert warned == []

    stray_compute, stray_gather = _Snode("new_c", "compute"), _Snode("new_g", "gather")
    table(stray_compute)
    table(stray_gather)
    table(stray_compute)
    assert warned == ["new_c"], "one WARNING per compile; later misses go to DEBUG"

    table.reset()
    table(_Snode("next_compile", "compute"))
    assert warned == ["new_c", "next_compile"]


def test_table_does_not_keep_a_finished_graph_alive():
    table = SnodeCostTable()
    s = _Snode("c", "compute")
    table.record(s, 1.0)
    assert len(table) == 1
    del s
    gc.collect()
    assert len(table) == 0


def test_pass_list_copies_share_one_empty_table_and_pickle():
    """Inductor deepcopies the pass list into the fx-graph cache key and pickles
    it: the copies must drop the snode-keyed entries, yet still point at ONE
    table between them, exactly like the originals do."""
    from magi_compiler.passes.fsdp_overlap import FsdpOverlapReorder
    from magi_compiler.passes.weight_offload import H2dLoadReorder

    table = SnodeCostTable()
    s = _Snode("c", "compute")
    table.record(s, 5.0)
    passes = [
        SnodeCostProfile(table, ProfilingRuntimeEstimator(sync_across_ranks=True)),
        FsdpOverlapReorder(cost_fn=table),
        H2dLoadReorder(bandwidth_bytes_per_ns=1.0, cost_fn=table),
    ]
    clone = copy.deepcopy(passes)
    profile, fsdp, h2d = clone
    assert profile.table is fsdp._cost_fn is h2d._cost_fn
    assert profile.table is not table and len(profile.table) == 0
    assert profile.estimator._sync_across_ranks is True
    pickle.dumps(clone)
    assert len(pickle.loads(pickle.dumps(table))) == 0


# ---------------------------------------------------------------------------
# ProfilingRuntimeEstimator changes the profile pass relies on
# ---------------------------------------------------------------------------
def test_estimator_never_prices_a_load():
    est = ProfilingRuntimeEstimator(sync_across_ranks=True)
    assert est(_Snode("ld", "load")) == 0.0
    assert est.table == {} and est._key_snode == {}


def test_single_rank_sync_measures_the_deferred_entries(monkeypatch):
    """Without peers there is nobody to desync: a deferred entry is measured in
    place rather than keeping its analytical seed for good."""
    est = ProfilingRuntimeEstimator(sync_across_ranks=True)
    est._table[("k",)] = ProfileEntry(ns=1.0, kind="extern", label="x", measured=False)
    est._key_snode[("k",)] = object()
    monkeypatch.setattr(est, "_measure_one", lambda snode: (42.0, True))
    assert est.warm_and_sync() == 1
    entry = est.table[("k",)]
    assert entry.ns == 42.0 and entry.measured
    assert est._key_snode == {}


def test_requery_is_not_counted_as_a_reuse(monkeypatch):
    est = ProfilingRuntimeEstimator()
    est._table[("k",)] = ProfileEntry(ns=7.0, kind="compute", label="x", measured=True)
    monkeypatch.setattr(re_mod, "_structural_key", lambda snode: ("k",))
    s = _Snode("c", "compute")
    assert est(s) == 7.0
    assert est.requery(s) == 7.0
    assert est.table[("k",)].reuse_count == 1 and est.n_cache_hits == 1
