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

"""Reorder pass #0: price the graph once, for every reorder pass after it.

The FSDP all-gather reorder and the weight-load reorder both size their overlap
windows from snode runtimes, and on several ranks those runtimes are only
trustworthy after a rank-lockstep re-measurement.  That measurement is not a
side effect of either pass: ``SnodeCostProfile`` runs first in
``reorder_for_compute_comm_overlap_passes``, prices the graph, and records one
number per snode in a ``SnodeCostTable`` that the later passes take as their
``cost_fn``.  Inductor chains the passes over the same snode objects
(``order = p(order)``), so a table keyed by snode stays valid down the chain.

Priced: compute and collectives.  Not priced: a weight load, which
``H2dLoadReorder`` prices from its bytes and the bus bandwidth, and a wait, which
takes no time of its own.
"""

from __future__ import annotations

import copy
import logging
import weakref

import torch
import torch.distributed as dist
from torch._inductor.scheduler import BaseSchedulerNode

from magi_compiler.utils import magi_logger

from .snode_utils import is_compute, is_weight_gather, issues_transfer
from .weight_offload.schedule.h2d_snode import is_h2d_load


def _analytical(snode: BaseSchedulerNode) -> float:
    from torch._inductor.comms import estimate_op_runtime

    try:
        return max(0.0, float(estimate_op_runtime(snode)))
    except Exception:  # noqa: BLE001
        return 0.0


def _multi_rank() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _is_priced(snode: BaseSchedulerNode) -> bool:
    """The snodes ``SnodeCostProfile`` records; the table warns on a miss for these alone."""
    return is_compute(snode) or issues_transfer(snode)


class SnodeCostTable:
    """snode -> ns for the compile in flight.  Callable, so it is a ``cost_fn``.

    Weakly keyed: the table lives in the pass list across compiles and must not
    keep a finished compile's scheduler graph alive.

    A miss falls back to Inductor's analytical estimate, except for a weight
    load, which this table never prices.  Waits are never recorded either, so
    missing them is expected; missing a compute or collective snode is not --
    either the profile pass is not ahead of the reader, or a pass between them
    built new snodes -- and is warned about, since the reader then sizes its
    windows from the roofline without saying so.
    """

    def __init__(self) -> None:
        self._ns: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        # False: pricing failed on this rank.  The FSDP reorder reduces it across
        # ranks before trusting any placement built on these numbers.
        self.ok = True
        self._misses = 0

    def reset(self) -> None:
        self._ns = weakref.WeakKeyDictionary()
        self.ok = True
        self._misses = 0

    def record(self, snode: BaseSchedulerNode, ns: float) -> None:
        self._ns[snode] = max(0.0, float(ns))

    def __contains__(self, snode: BaseSchedulerNode) -> bool:
        return snode in self._ns

    def __len__(self) -> int:
        return len(self._ns)

    def __call__(self, snode: BaseSchedulerNode) -> float:
        ns = self._ns.get(snode)
        if ns is not None:
            return ns
        if is_h2d_load(snode):
            return 0.0
        ns = _analytical(snode)
        if _is_priced(snode):
            self._report_miss(snode, ns)
        self._ns[snode] = ns
        return ns

    def _report_miss(self, snode: BaseSchedulerNode, ns: float) -> None:
        self._misses += 1
        # Once per compile at WARNING: a pass that bypassed the profile misses on
        # every snode it reads.
        log = magi_logger.warning if self._misses == 1 else magi_logger.debug
        log(
            "snode cost table: %s was not priced by SnodeCostProfile, using Inductor's analytical "
            "%.1fus instead. Install SnodeCostProfile ahead of every pass that reads this table, and "
            "make sure no pass between them creates snodes (later misses in this compile log at DEBUG)",
            snode.get_name(),
            ns / 1e3,
            rank="all",
        )

    # Inductor deepcopies the pass list into the fx-graph cache key and pickles
    # the copy; snodes hold FakeTensors, so neither may carry the entries.
    def __deepcopy__(self, memo):
        new = SnodeCostTable()
        memo[id(self)] = new
        return new

    def __getstate__(self):
        return {}

    def __setstate__(self, state) -> None:
        self.__init__()


class SnodeCostProfile:
    """Reorder pass that measures and never reorders.  Install it first.

    ``estimator`` is ``None`` for Inductor's analytical estimate, or a
    ``ProfilingRuntimeEstimator``; one built with ``sync_across_ranks=True`` is
    re-measured in rank lockstep by ``warm_and_sync`` before anything is recorded.
    """

    def __init__(self, table: SnodeCostTable, estimator=None) -> None:
        self.table = table
        self.estimator = estimator

    def __deepcopy__(self, memo):
        new = SnodeCostProfile.__new__(SnodeCostProfile)
        memo[id(self)] = new
        new.table = copy.deepcopy(self.table, memo)
        new.estimator = copy.deepcopy(self.estimator, memo)
        return new

    @property
    def _syncs(self) -> bool:
        return bool(getattr(self.estimator, "_sync_across_ranks", False))

    def __call__(self, snodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
        self.table.reset()
        if not self._needed(snodes):
            return snodes
        priced = [s for s in snodes if _is_priced(s)]
        if self.estimator is None:
            for s in priced:
                self.table.record(s, _analytical(s))
            magi_logger.info("snode cost profile: priced %d snode(s) analytically", len(priced))
            return snodes

        ok = self._warm(priced)
        n_changed = 0
        if self._syncs:
            # Every rank that got past _needed reaches this call, even one whose
            # warm-up failed: warm_and_sync is a lockstep collective.
            try:
                n_changed = self.estimator.warm_and_sync()
            except Exception as exc:  # noqa: BLE001
                magi_logger.warning("snode cost profile: rank-synchronized profiling failed (%s)", exc, rank="all")
                ok = False
        requery = getattr(self.estimator, "requery", self.estimator)
        for s in priced:
            try:
                self.table.record(s, requery(s))
            except Exception:  # noqa: BLE001
                self.table.record(s, _analytical(s))
        self.table.ok = ok
        self._report(len(priced), n_changed)
        return snodes

    def _needed(self, snodes) -> bool:
        """Whether a pass downstream will read the table for this graph.

        Only graphs with a weight gather or a weight load have one.  When the
        estimator syncs, the answer is reduced across ranks: ``warm_and_sync`` is
        a collective, so a rank whose own graph needs no pricing still has to
        join it if any peer's does.  A load counts here because the load reorder
        needs the COMPUTE around it priced -- the load itself never is.
        """
        local = any(is_weight_gather(s) or is_h2d_load(s) for s in snodes)
        if not (self._syncs and _multi_rank()):
            return local
        from magi_compiler.profiling.runtime_estimator import _get_cost_sync_group

        t = torch.tensor([int(local)], dtype=torch.int32)
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=_get_cost_sync_group())
        return bool(t.item())

    def _warm(self, priced) -> bool:
        try:
            for s in priced:
                self.estimator(s)
            return True
        except Exception as exc:  # noqa: BLE001
            magi_logger.warning("snode cost profile: warm-up measurement failed (%s)", exc, rank="all")
            return False

    def _report(self, n_priced: int, n_changed: int) -> None:
        est = self.estimator
        magi_logger.info(
            "snode cost profile: priced %d snode(s) (cost table: %d distinct ops, measured=%s reused=%s; "
            "%d entr%s changed by the rank-synchronized re-measurement)",
            n_priced,
            len(getattr(est, "table", {}) or {}),
            getattr(est, "n_measured", None),
            getattr(est, "n_cache_hits", None),
            n_changed,
            "y" if n_changed == 1 else "ies",
        )
        if not self.table.ok:
            magi_logger.warning(
                "snode cost profile: pricing failed on this rank; the FSDP reorder will leave the graph "
                "unchanged on every rank, and the load reorder places against the partial costs",
                rank="all",
            )
        # summary() builds the whole table eagerly; only pay for it when read.
        if hasattr(est, "summary") and logging.getLogger("magi_compiler").isEnabledFor(logging.DEBUG):
            magi_logger.debug("snode cost %s", est.summary())
