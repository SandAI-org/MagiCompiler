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

"""The shape every transfer/compute overlap pass shares.

Installed in Inductor's ``reorder_for_compute_comm_overlap_passes``: Inductor
calls it with the scheduler-node order and takes back whatever it returns, and
does NOT repair that order -- it must be topological.  A subclass says which
transfers exist and how to place them; this class owns the rest:

    has_transfer_candidates -> build_transfer_tasks -> assign_launch_indices
    -> rebuild_node_order + is_topological_order -> can_commit_order -> on_commit -> log_reorder_summary

``build_transfer_tasks`` returning ``None`` leaves the graph untouched.
``assign_launch_indices`` returns the placement, anchor -> launch index; the
tasks themselves stay inputs.  What one hook hands a later one lives in a fresh
``call_state_type`` instance per call, never on the pass object, which Inductor
deep-copies into its cache key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from torch._inductor.scheduler import BaseSchedulerNode

from magi_compiler.utils import magi_logger

from .cost import CostView, default_runtime_ns_fn
from .graph import SnodeGraph
from .task import TransferTask

DEFAULT_WINDOW_MARGIN_NS = 5_000.0
"""Headroom added to every transfer's window, absorbing estimator error and
launch latency so the wait rarely stalls."""


@dataclass
class OverlapCallState:
    """State one call's hooks share.  A pass subclasses it with the fields its
    hooks set and read, and names the subclass in ``call_state_type``."""


class OverlapPass:
    """Callable reorder pass; subclass and implement ``has_transfer_candidates``,
    ``build_transfer_tasks`` and ``assign_launch_indices``."""

    log_prefix = "overlap reorder"
    call_state_type: type[OverlapCallState] = OverlapCallState

    def __init__(self, runtime_ns_fn=None) -> None:
        # snode -> ns.  Normally the SnodeCostTable filled by the SnodeCostProfile
        # pass ahead of this one; Inductor's estimate_op_runtime when run alone.
        self._runtime_ns_fn = runtime_ns_fn if runtime_ns_fn is not None else default_runtime_ns_fn()

    # -- the template -------------------------------------------------------
    def __call__(self, snodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
        order = list(snodes)
        if not self.has_transfer_candidates(order):
            return order
        graph = SnodeGraph(order)
        cost = CostView(self._runtime_ns_fn)
        call_state = self.call_state_type()
        tasks = self.build_transfer_tasks(graph, cost, call_state)
        if tasks is None:
            return order
        launch_index_by_anchor = self.assign_launch_indices(graph, cost, tasks, call_state)

        new_order = graph.rebuild_node_order(tasks, launch_index_by_anchor)
        is_order_valid = graph.is_topological_order(new_order)
        if not is_order_valid:
            magi_logger.warning("%s: rebuilt order failed validation; leaving graph unchanged", self.log_prefix)
        committed = self.can_commit_order(is_order_valid, call_state)
        if committed:
            self.on_commit(graph, tasks, call_state)
            order[:] = new_order
        self.log_reorder_summary(graph, cost, tasks, launch_index_by_anchor, committed, call_state)
        return order

    # -- hooks ----------------------------------------------------------------
    def has_transfer_candidates(self, order: list[BaseSchedulerNode]) -> bool:
        """Cheap check that the graph holds anything this pass moves."""
        raise NotImplementedError

    def build_transfer_tasks(
        self, graph: SnodeGraph, cost: CostView, call_state: OverlapCallState
    ) -> list[TransferTask] | None:
        """One task per transfer, in program order; ``None`` to leave the graph alone."""
        raise NotImplementedError

    def assign_launch_indices(
        self, graph: SnodeGraph, cost: CostView, tasks: list[TransferTask], call_state: OverlapCallState
    ) -> Mapping[BaseSchedulerNode, int]:
        """Each moving task's launch index, keyed by anchor; a task without one stays put."""
        raise NotImplementedError

    def can_commit_order(self, is_order_valid: bool, call_state: OverlapCallState) -> bool:
        """Whether to commit the rebuilt order; by default, exactly when this rank found it valid."""
        return is_order_valid

    def on_commit(self, graph: SnodeGraph, tasks: list[TransferTask], call_state: OverlapCallState) -> None:
        """Side effects that may only happen once the new order is committed."""

    def log_reorder_summary(
        self,
        graph: SnodeGraph,
        cost: CostView,
        tasks: list[TransferTask],
        launch_index_by_anchor: Mapping[BaseSchedulerNode, int],
        committed: bool,
        call_state: OverlapCallState,
    ) -> None:
        """Log what was done."""
