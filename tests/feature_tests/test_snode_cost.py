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
from torch._inductor import ir

from magi_compiler.passes import snode_cost, snode_utils
from magi_compiler.passes.snode_cost import SnodeCostProfile, SnodeCostTable
from magi_compiler.profiling import ProfilingRuntimeEstimator, runtime_estimator
from magi_compiler.profiling.runtime_estimator import ProfileEntry


class FakeSnode:
    snodes = None

    def __init__(self, name, kind):
        from magi_compiler.passes.weight_offload.runtime.h2d_op import H2D_LOAD

        self.name = name
        self.kind = kind
        self.node = object.__new__(ir.FallbackKernel)
        self.node.op_overload = H2D_LOAD if kind == "load" else "fake.op"

    def get_name(self):
        return self.name

    def __repr__(self):
        return f"<{self.kind} {self.name}>"


@pytest.fixture(autouse=True)
def stub_predicates(monkeypatch):
    is_wait = lambda s: getattr(s, "kind", None) == "wait"  # noqa: E731
    is_gather = lambda s: getattr(s, "kind", None) == "gather"  # noqa: E731
    monkeypatch.setattr(snode_utils, "contains_wait", is_wait)
    monkeypatch.setattr(snode_utils, "issues_comm", is_gather)
    monkeypatch.setattr(snode_cost, "issues_comm", is_gather)
    monkeypatch.setattr(snode_cost, "is_all_gather_launch", is_gather)


class FakeRuntimeEstimator:
    """The surface of ``ProfilingRuntimeEstimator`` the profile pass drives."""

    def __init__(self, sync=True, fail_sync=False):
        self.sync_across_ranks = sync
        self.fail_sync = fail_sync
        self.warmed: list[str] = []
        self.synced = False
        self.table: dict = {}
        self.n_measured = 0
        self.n_cache_hits = 0

    def summary(self):
        return "stub"

    def __call__(self, s):
        self.warmed.append(s.name)
        return 100.0

    def warm_and_sync(self):
        if self.fail_sync:
            raise RuntimeError("gloo went away")
        self.synced = True
        return 1

    def estimate_without_reuse_count(self, s):
        return 200.0 if self.synced else 100.0


def _make_snode_order(*kinds):
    return [FakeSnode(f"{k}{i}", k) for i, k in enumerate(kinds)]


def test_prices_compute_and_gathers_never_a_load_or_a_wait():
    order = _make_snode_order("compute", "load", "wait", "gather", "compute")
    c0, load, wait, gather, c4 = order
    estimator, table = FakeRuntimeEstimator(), SnodeCostTable()
    out = SnodeCostProfile(table, estimator)(order)

    assert out is order, "the profile pass must never reorder"
    assert estimator.warmed == ["compute0", "gather3", "compute4"]
    assert estimator.synced
    # The passes after it read the synced value, not the warm-up seed.
    assert table(c0) == table(gather) == table(c4) == 200.0
    assert table.is_profiling_valid
    assert load not in table and wait not in table
    assert table(load) == 0.0, "a load is priced from bytes / bandwidth, never by the table"


def test_a_load_alone_is_reason_to_price_the_compute_around_it():
    order = _make_snode_order("compute", "load", "wait", "compute")
    estimator = FakeRuntimeEstimator()
    SnodeCostProfile(SnodeCostTable(), estimator)(order)
    assert estimator.warmed == ["compute0", "compute3"]


def test_graph_without_gather_or_load_is_not_priced():
    estimator, table = FakeRuntimeEstimator(), SnodeCostTable()
    SnodeCostProfile(table, estimator)(_make_snode_order("compute", "compute"))
    assert estimator.warmed == [] and not estimator.synced
    assert len(table) == 0


def test_analytical_mode_records_the_same_snodes():
    order = _make_snode_order("compute", "load", "wait", "gather")
    table = SnodeCostTable()
    SnodeCostProfile(table, None)(order)
    assert [s.name for s in order if s in table] == ["compute0", "gather3"]


def test_sync_failure_marks_the_table_but_still_records():
    order = _make_snode_order("compute", "gather")
    table = SnodeCostTable()
    SnodeCostProfile(table, FakeRuntimeEstimator(fail_sync=True))(order)
    assert not table.is_profiling_valid
    assert table(order[0]) == 100.0


def test_each_compile_starts_from_an_empty_table():
    table = SnodeCostTable()
    profile = SnodeCostProfile(table, FakeRuntimeEstimator())
    first = _make_snode_order("compute", "gather")
    profile(first)
    table.is_profiling_valid = False
    second = _make_snode_order("gather", "compute")
    profile(second)
    assert first[0] not in table and second[1] in table
    assert table.is_profiling_valid


def test_a_miss_on_a_priced_kind_warns_once_per_compile(monkeypatch):
    """Compute and collectives are always recorded by the profile pass, so a
    miss on one means the chain is broken; waits and loads are never recorded,
    so missing them is silent."""
    warned_node_names: list = []
    monkeypatch.setattr(snode_cost.magi_logger, "warning", lambda msg, *a, **k: warned_node_names.append(a[0]))
    table = SnodeCostTable()
    SnodeCostProfile(table, FakeRuntimeEstimator())(_make_snode_order("gather", "compute"))

    for s in _make_snode_order("wait", "load"):
        table(s)
    assert warned_node_names == []

    stray_compute, stray_gather = FakeSnode("new_c", "compute"), FakeSnode("new_g", "gather")
    table(stray_compute)
    table(stray_gather)
    table(stray_compute)
    assert warned_node_names == ["new_c"], "one WARNING per compile; later misses go to DEBUG"

    table.reset()
    table(FakeSnode("next_compile", "compute"))
    assert warned_node_names == ["new_c", "next_compile"]


def test_table_does_not_keep_a_finished_graph_alive():
    table = SnodeCostTable()
    s = FakeSnode("c", "compute")
    table.record_runtime_ns(s, 1.0)
    assert len(table) == 1
    del s
    gc.collect()
    assert len(table) == 0


def test_pass_list_copies_share_one_empty_table_and_pickle():
    """Inductor deepcopies the pass list into the fx-graph cache key and pickles
    it: the copies must drop the snode-keyed entries, yet still point at ONE
    table between them, exactly like the originals do."""
    from magi_compiler.passes.fsdp_overlap import FsdpAutoBucket, FsdpOverlapReorder
    from magi_compiler.passes.weight_offload import H2dLoadReorder

    table = SnodeCostTable()
    s = FakeSnode("c", "compute")
    table.record_runtime_ns(s, 5.0)
    passes = [
        SnodeCostProfile(table, ProfilingRuntimeEstimator(sync_across_ranks=True)),
        FsdpAutoBucket(runtime_ns_fn=table),
        FsdpOverlapReorder(runtime_ns_fn=table),
        H2dLoadReorder(bandwidth_bytes_per_ns=1.0, runtime_ns_fn=table),
    ]
    clone = copy.deepcopy(passes)
    profile, bucket, fsdp, h2d = clone
    assert profile.table is bucket._runtime_ns_fn is fsdp._runtime_ns_fn is h2d._runtime_ns_fn
    assert profile.table is not table and len(profile.table) == 0
    assert profile.estimator.sync_across_ranks is True
    pickle.dumps(clone)
    assert len(pickle.loads(pickle.dumps(table))) == 0


# ---------------------------------------------------------------------------
# ProfilingRuntimeEstimator changes the profile pass relies on
# ---------------------------------------------------------------------------
def test_estimator_never_prices_a_load():
    estimator = ProfilingRuntimeEstimator(sync_across_ranks=True)
    assert estimator(FakeSnode("ld", "load")) == 0.0
    assert estimator.table == {} and estimator._key_snode == {}


def test_single_rank_sync_measures_the_deferred_entries(monkeypatch):
    """Without peers there is nobody to desync: a deferred entry is measured in
    place rather than keeping its analytical seed for good."""
    estimator = ProfilingRuntimeEstimator(sync_across_ranks=True)
    estimator._table[("k",)] = ProfileEntry(ns=1.0, kind="extern", label="x", measured=False)
    estimator._key_snode[("k",)] = object()
    monkeypatch.setattr(estimator, "_measure_one", lambda snode: (42.0, True))
    assert estimator.warm_and_sync() == 1
    entry = estimator.table[("k",)]
    assert entry.ns == 42.0 and entry.measured
    assert estimator._key_snode == {}


def test_requery_is_not_counted_as_a_reuse(monkeypatch):
    estimator = ProfilingRuntimeEstimator()
    estimator._table[("k",)] = ProfileEntry(ns=7.0, kind="compute", label="x", measured=True)
    monkeypatch.setattr(runtime_estimator, "_structural_key", lambda snode: ("k",))
    s = FakeSnode("c", "compute")
    assert estimator(s) == 7.0
    assert estimator.estimate_without_reuse_count(s) == 7.0
    assert estimator.table[("k",)].reuse_count == 1 and estimator.n_cache_hits == 1
