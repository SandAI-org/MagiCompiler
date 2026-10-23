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

"""Price compute and collectives once before the overlap reorder passes.

``SnodeCostProfile`` fills a shared ``SnodeCostTable`` keyed by scheduler nodes.
With synchronized profiling, rank-lockstep measurement precedes table writes.
Downstream passes that create nodes must price them separately.
Waits take no time here; ``H2dLoadReorder`` prices loads from bytes and bandwidth.
"""

from __future__ import annotations

import logging
import weakref

from torch._inductor.scheduler import BaseSchedulerNode

from magi_compiler.profiling.runtime_estimator import analytical_ns
from magi_compiler.utils import magi_logger

from .overlap import rank_sync
from .snode_utils import is_all_gather_launch, is_compute, issues_comm
from .weight_offload.schedule.h2d_snode import is_h2d_load


def _is_priced(snode: BaseSchedulerNode) -> bool:
    """The snodes ``SnodeCostProfile`` records; the table warns on a miss for these alone."""
    return is_compute(snode) or issues_comm(snode)


class SnodeCostTable:
    """Callable snode-to-nanoseconds table, weakly keyed to release finished graphs.

    Missing compute or collective entries warn and use Inductor's analytical estimate.
    Loads are unpriced; missing waits are expected and do not warn."""

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
        ns = max(0.0, analytical_ns(snode))
        if _is_priced(snode):
            self._report_miss(snode, ns)
        self._ns[snode] = ns
        return ns

    def _report_miss(self, snode: BaseSchedulerNode, ns: float) -> None:
        self._misses += 1
        # Once per compile at WARNING: a pass that bypassed the profile misses on
        # every snode it reads.
        log = magi_logger.warning if self._misses == 1 else magi_logger.debug
        hint = " Check profile ordering and pricing of new nodes." if self._misses == 1 else ""
        log("snode cost table: unpriced node=%s; analytical fallback=%.1fus%s", snode.get_name(), ns / 1e3, hint, rank="all")

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

    @property
    def _syncs(self) -> bool:
        return self.estimator is not None and self.estimator.sync_across_ranks

    def __call__(self, snodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
        self.table.reset()
        if not self._needed(snodes):
            return snodes
        priced = [s for s in snodes if _is_priced(s)]
        if self.estimator is None:
            for s in priced:
                self.table.record(s, analytical_ns(s))
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
        for s in priced:
            self.table.record(s, self.estimator.requery(s))
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
        local = any(is_all_gather_launch(s) or is_h2d_load(s) for s in snodes)
        return rank_sync.any_rank(local) if self._syncs else local

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
            len(est.table),
            est.n_measured,
            est.n_cache_hits,
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
        if logging.getLogger("magi_compiler").isEnabledFor(logging.DEBUG):
            magi_logger.debug("snode cost %s", est.summary())
