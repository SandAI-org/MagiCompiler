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

from magi_compiler.profiling.runtime_estimator import analytical_runtime_ns
from magi_compiler.utils import magi_logger

from .snode_utils import is_all_gather_launch, is_compute, is_h2d_load, issues_comm
from .transfer_overlap import rank_sync


def _should_record_cost(snode: BaseSchedulerNode) -> bool:
    """The snodes ``SnodeCostProfile`` records; the table warns on a miss for these alone."""
    return is_compute(snode) or issues_comm(snode)


class SnodeCostTable:
    """Callable snode-to-nanoseconds table, weakly keyed to release finished graphs.

    Missing compute or collective entries warn and use Inductor's analytical estimate.
    Loads are unpriced; missing waits are expected and do not warn."""

    def __init__(self) -> None:
        self._runtime_ns_by_node: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        # False: pricing failed on this rank.  The FSDP reorder reduces it across
        # ranks before trusting any placement built on these numbers.
        self.is_profiling_valid = True
        self._cache_miss_count = 0

    def reset(self) -> None:
        self._runtime_ns_by_node = weakref.WeakKeyDictionary()
        self.is_profiling_valid = True
        self._cache_miss_count = 0

    def record_runtime_ns(self, snode: BaseSchedulerNode, runtime_ns: float) -> None:
        self._runtime_ns_by_node[snode] = max(0.0, float(runtime_ns))

    def __contains__(self, snode: BaseSchedulerNode) -> bool:
        return snode in self._runtime_ns_by_node

    def __len__(self) -> int:
        return len(self._runtime_ns_by_node)

    def __call__(self, snode: BaseSchedulerNode) -> float:
        runtime_ns = self._runtime_ns_by_node.get(snode)
        if runtime_ns is not None:
            return runtime_ns
        if is_h2d_load(snode):
            return 0.0
        runtime_ns = max(0.0, analytical_runtime_ns(snode))
        if _should_record_cost(snode):
            self._report_miss(snode, runtime_ns)
        self._runtime_ns_by_node[snode] = runtime_ns
        return runtime_ns

    def _report_miss(self, snode: BaseSchedulerNode, runtime_ns: float) -> None:
        self._cache_miss_count += 1
        # Once per compile at WARNING: a pass that bypassed the profile misses on
        # every snode it reads.
        log = magi_logger.warning if self._cache_miss_count == 1 else magi_logger.debug
        hint = " Check profile ordering and pricing of new nodes." if self._cache_miss_count == 1 else ""
        log(
            "snode cost table: unpriced node=%s; analytical fallback=%.1fus%s",
            snode.get_name(),
            runtime_ns / 1e3,
            hint,
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

    @property
    def _should_sync_ranks(self) -> bool:
        return self.estimator is not None and self.estimator.sync_across_ranks

    def __call__(self, snodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
        self.table.reset()
        if not self._needs_cost_profile(snodes):
            return snodes
        priced_nodes = [s for s in snodes if _should_record_cost(s)]
        if self.estimator is None:
            for s in priced_nodes:
                self.table.record_runtime_ns(snode=s, runtime_ns=analytical_runtime_ns(s))
            magi_logger.info("snode cost profile: priced %d snode(s) analytically", len(priced_nodes))
            return snodes

        is_profiling_valid = self._populate_estimator(priced_nodes)
        changed_entry_count = 0
        if self._should_sync_ranks:
            # Every rank that got past _needs_cost_profile reaches this call, even one whose
            # warm-up failed: warm_and_sync is a lockstep collective.
            try:
                changed_entry_count = self.estimator.warm_and_sync()
            except Exception as exc:  # noqa: BLE001
                magi_logger.warning("snode cost profile: rank-synchronized profiling failed (%s)", exc, rank="all")
                is_profiling_valid = False
        for s in priced_nodes:
            self.table.record_runtime_ns(snode=s, runtime_ns=self.estimator.estimate_without_reuse_count(s))
        self.table.is_profiling_valid = is_profiling_valid
        self._log_profile_summary(priced_node_count=len(priced_nodes), changed_entry_count=changed_entry_count)
        return snodes

    def _needs_cost_profile(self, snodes) -> bool:
        """Whether a pass downstream will read the table for this graph.

        Only graphs with a weight gather or a weight load have one.  When the
        estimator syncs, the answer is reduced across ranks: ``warm_and_sync`` is
        a collective, so a rank whose own graph needs no pricing still has to
        join it if any peer's does.  A load counts here because the load reorder
        needs the COMPUTE around it priced -- the load itself never is.
        """
        needs_local_profile = any(is_all_gather_launch(s) or is_h2d_load(s) for s in snodes)
        return rank_sync.any_rank_has_flag(needs_local_profile) if self._should_sync_ranks else needs_local_profile

    def _populate_estimator(self, priced_nodes) -> bool:
        try:
            for s in priced_nodes:
                self.estimator(s)
            return True
        except Exception as exc:  # noqa: BLE001
            magi_logger.warning("snode cost profile: warm-up measurement failed (%s)", exc, rank="all")
            return False

    def _log_profile_summary(self, priced_node_count: int, changed_entry_count: int) -> None:
        estimator = self.estimator
        magi_logger.info(
            "snode cost profile: priced %d snode(s) (cost table: %d distinct ops, measured=%s reused=%s; "
            "%d entr%s changed by the rank-synchronized re-measurement)",
            priced_node_count,
            len(estimator.table),
            estimator.n_measured,
            estimator.n_cache_hits,
            changed_entry_count,
            "y" if changed_entry_count == 1 else "ies",
        )
        if not self.table.is_profiling_valid:
            magi_logger.warning(
                "snode cost profile: pricing failed on this rank; the FSDP reorder will leave the graph "
                "unchanged on every rank, and the load reorder places against the partial costs",
                rank="all",
            )
        # summary() builds the whole table eagerly; only pay for it when read.
        if logging.getLogger("magi_compiler").isEnabledFor(logging.DEBUG):
            magi_logger.debug("snode cost %s", estimator.summary())
