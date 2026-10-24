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

"""Hoist H2D loads after FSDP overlap placement, leaving waits and gathers in place.

Compute costs come from ``SnodeCostTable``; transfer costs use bytes and configured
bandwidth. Residency selection searches the lowest feasible bus density across
compute prefixes ending at load deadlines, within the residency budget.

A unified device-weight budget starts with minimal pipelining room and assigns
the remainder to residency. More in-flight room is tried only when that budget
limits overlap. Separate resident/in-flight budgets remain supported.

``schedule_transfers_alap`` places transfers by deadline. Serialized bus time and buffer
lifetime through the last consumer are separate constraints. Resident loads keep
their existing position and are charged to residency, not in-flight storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from torch._inductor.scheduler import BaseSchedulerNode

from magi_compiler.utils import magi_logger

from ...transfer_overlap import (
    DEFAULT_WINDOW_MARGIN_NS,
    CostView,
    OverlapCallState,
    OverlapPass,
    ScheduleResult,
    SnodeGraph,
    TransferTask,
    inflight_peak_bytes,
    schedule_transfers_alap,
    unhoisted_index,
    weight_storage_peak_bytes,
)
from .h2d_snode import host_slot_of_load, is_h2d_load, load_bytes

_MAX_BUDGET_SPLIT_ROUNDS = 5
"""Times ``assign_launch_indices`` may grow the in-flight budget out of the residency one."""

_DENSITY_SEARCH_STEPS = 8
"""Bisection steps for the target bus density; 8 lands within 0.4% of it."""


@dataclass(frozen=True, eq=False)
class H2dLoadTask(TransferTask):
    """One load's placement problem: how much transfer to hide, and where it may go.

    ``anchor`` is the load, ``deadline_index`` its earliest wait (the hard upper bound).
    """

    host_slot: int = field(kw_only=True)  # host-pool slot this load pulls


@dataclass(frozen=True)
class LoadSchedule:
    """One residency / in-flight budget split and the placement it produced."""

    # Loads bought out of the offload plan: they copy device-to-device in place, off the bus.
    resident_anchors: frozenset
    inflight_budget_bytes: int
    schedule_result: ScheduleResult


@dataclass
class H2dLoadCallState(OverlapCallState):
    """The chosen schedule and what committing it cost, for the later hooks."""

    load_schedule: LoadSchedule | None = None  # set by assign_launch_indices
    promoted_bytes: int = 0  # set by on_commit: device bytes the resident weights took back


class H2dLoadReorder(OverlapPass):
    """Callable reorder pass.  Run me after ``FsdpOverlapReorder``."""

    log_prefix = "h2d load reorder"
    call_state_type = H2dLoadCallState

    def __init__(
        self,
        bandwidth_bytes_per_ns: float,
        window_margin_ns: float = DEFAULT_WINDOW_MARGIN_NS,
        window_scale: float = 1.0,
        max_resident_bytes: int = 0,
        max_inflight_bytes: int = 0,
        weight_budget_bytes: int = 0,
        bus_utilization: float = 1.0,
        runtime_ns_fn=None,
    ) -> None:
        super().__init__(runtime_ns_fn)
        self.bandwidth_bytes_per_ns = max(1e-6, bandwidth_bytes_per_ns)
        self.window_margin_ns = window_margin_ns
        self.window_scale = window_scale
        self.max_resident_bytes = max_resident_bytes
        self.max_inflight_bytes = max_inflight_bytes
        self.weight_budget_bytes = weight_budget_bytes
        self.bus_utilization = min(1.0, max(1e-3, bus_utilization))

    # -- OverlapPass hooks ------------------------------------------------
    def has_transfer_candidates(self, order: list[BaseSchedulerNode]) -> bool:
        if any(is_h2d_load(s) for s in order):
            return True
        magi_logger.debug("h2d load reorder: no weight load among %d snodes", len(order))
        return False

    def build_transfer_tasks(
        self, graph: SnodeGraph, cost: CostView, call_state: H2dLoadCallState
    ) -> list[H2dLoadTask] | None:
        original_index_by_node = graph.original_index_by_node
        tasks: list[H2dLoadTask] = []
        for load in (s for s in graph.order if is_h2d_load(s)):
            moving_nodes, waits = self._load_nodes_and_waits(load=load, graph=graph)
            if not waits:
                magi_logger.debug("h2d load reorder: %s has no wait; leaving it in place", load.get_name())
                continue
            transfer_bytes = load_bytes(moving_nodes)
            required_overlap_ns = transfer_bytes / self.bandwidth_bytes_per_ns * self.window_scale + self.window_margin_ns
            tasks.append(
                H2dLoadTask(
                    anchor=load,
                    moving_nodes=tuple(moving_nodes),
                    host_slot=host_slot_of_load(load),
                    deadline_index=min(original_index_by_node[w] for w in waits),
                    last_user_index=graph.last_user_index(moving_nodes, waits),
                    required_overlap_ns=required_overlap_ns,
                    earliest_launch_index=graph.earliest_legal_index(moving_nodes),
                    transfer_bytes=transfer_bytes,
                )
            )
        return tasks or None

    def assign_launch_indices(
        self, graph: SnodeGraph, cost: CostView, tasks: list[H2dLoadTask], call_state: H2dLoadCallState
    ) -> dict:
        original_index_by_node = graph.original_index_by_node
        compute_prefix_ns = cost.compute_prefix_ns(graph.order)
        best_schedule: LoadSchedule | None = None
        for resident_budget_bytes, inflight_budget_bytes in self._iter_budget_splits(tasks):
            resident_anchors = self._search_resident_weights(
                tasks=tasks, compute_prefix_ns=compute_prefix_ns, resident_budget_bytes=resident_budget_bytes
            )
            schedule_result = schedule_transfers_alap(
                tasks, compute_prefix_ns, original_index_by_node, inflight_budget_bytes, off_bus_anchors=resident_anchors
            )
            total_unhidden_ns = schedule_result.total_unhidden_ns
            if best_schedule is None or total_unhidden_ns < best_schedule.schedule_result.total_unhidden_ns - 1e-9:
                best_schedule = LoadSchedule(
                    resident_anchors=resident_anchors,
                    inflight_budget_bytes=inflight_budget_bytes,
                    schedule_result=schedule_result,
                )
            # Only an in-flight shortage is worth buying more in-flight room for;
            # if the bus itself is full, taking bytes off residency makes it worse.
            if total_unhidden_ns <= 0 or not schedule_result.inflight_budget_limited_anchors:
                break
        assert best_schedule is not None  # _iter_budget_splits always yields at least once
        call_state.load_schedule = best_schedule
        return best_schedule.schedule_result.launch_index_by_anchor

    def on_commit(self, graph: SnodeGraph, tasks: list[H2dLoadTask], call_state: H2dLoadCallState) -> None:
        resident_anchors = call_state.load_schedule.resident_anchors
        call_state.promoted_bytes = self._make_selected_weights_resident(tasks=tasks, resident_anchors=resident_anchors)

    def log_reorder_summary(
        self,
        graph: SnodeGraph,
        cost: CostView,
        tasks: list[H2dLoadTask],
        launch_index_by_anchor: dict,
        committed: bool,
        call_state: H2dLoadCallState,
    ) -> None:
        if not committed:
            return
        load_schedule = call_state.load_schedule
        original_index_by_node = graph.original_index_by_node
        schedule_result = load_schedule.schedule_result
        launch_index_by_anchor = schedule_result.launch_index_by_anchor
        inflight_budget_bytes = load_schedule.inflight_budget_bytes
        resident_anchors = load_schedule.resident_anchors

        peak_inflight_bytes = inflight_peak_bytes(tasks, launch_index_by_anchor)
        hoisted_load_count = sum(
            1 for load, launch_index in launch_index_by_anchor.items() if launch_index != original_index_by_node[load]
        )
        bus_transfer_bytes = sum(p.transfer_bytes for p in tasks if p.anchor not in resident_anchors)
        # graph.order is already reordered; deadlines still index the original order.
        original_order = list(original_index_by_node)
        overlap_capacity_bytes = self._overlap_capacity_bytes(
            tasks=tasks, compute_prefix_ns=cost.compute_prefix_ns(original_order)
        )
        uncovered_bytes = bus_transfer_bytes - overlap_capacity_bytes
        if uncovered_bytes > 0 and schedule_result.total_unhidden_ns > 0:
            magi_logger.warning(
                "h2d load reorder: estimated transfer coverage deficit=%.1f MiB (%.1fms); consider more residency",
                uncovered_bytes / 2**20,
                uncovered_bytes / self.bandwidth_bytes_per_ns / 1e6,
            )
        peak_weight_storage_bytes = weight_storage_peak_bytes(
            tasks, launch_index_by_anchor, original_index_by_node, resident_anchors
        )
        magi_logger.info(
            "h2d load reorder: promoted %.1f MiB; resident loads=%d; bus=%.1f MiB at %.1f GB/s; "
            "Hoisted %d/%d load(s); in-flight peak %.1f MiB of the %.0f MiB budget; "
            "device-weight peak=%.1f MiB; exposed=%.1fms (budget-bound=%.1fms)",
            call_state.promoted_bytes / 2**20,
            len(resident_anchors),
            bus_transfer_bytes / 2**20,
            self.bandwidth_bytes_per_ns,
            hoisted_load_count,
            len(tasks),
            peak_inflight_bytes / 2**20,
            inflight_budget_bytes / 2**20,
            peak_weight_storage_bytes / 2**20,
            schedule_result.total_unhidden_ns / 1e6,
            schedule_result.inflight_budget_limited_unhidden_ns / 1e6,
        )
        assert peak_inflight_bytes <= inflight_budget_bytes, (
            f"h2d load reorder: in-flight peak {peak_inflight_bytes} B exceeds "
            f"the {inflight_budget_bytes} B budget it was placed under"
        )

    # -- planning ---------------------------------------------------------
    @staticmethod
    def _load_nodes_and_waits(load, graph: SnodeGraph) -> tuple[list, list]:
        """The load plus the ``MultiOutput`` unpack its result reaches readers through, and the waits guarding it."""
        return [load, *graph.unpacks_of(load)], graph.waits_of(load)

    # -- scheduling --------------------------------------------------------
    def _effective_inflight_budget_bytes(self, tasks) -> int:
        """Never go below the unhoisted live-buffer peak.
        Without an explicit limit, allow that peak plus one largest load."""
        minimum_inflight_bytes = inflight_peak_bytes(tasks, {p.anchor: unhoisted_index(p) for p in tasks})
        if self.max_inflight_bytes <= 0:
            return minimum_inflight_bytes + max((p.transfer_bytes for p in tasks), default=0)
        if self.max_inflight_bytes < minimum_inflight_bytes:
            magi_logger.warning(
                "h2d load reorder: in-flight budget raised from %.1f to %.1f MiB (unhoisted peak)",
                self.max_inflight_bytes / 2**20,
                minimum_inflight_bytes / 2**20,
            )
            return minimum_inflight_bytes
        return self.max_inflight_bytes

    def _overlap_capacity_bytes(self, tasks, compute_prefix_ns) -> float:
        """Compute after the last load deadline cannot hide any weight transfer."""
        last_wait_index = max(p.deadline_index for p in tasks)
        return (
            self.bus_utilization
            * compute_prefix_ns[min(last_wait_index, len(compute_prefix_ns) - 1)]
            * self.bandwidth_bytes_per_ns
        )

    def _iter_budget_splits(self, tasks):
        """Try residency-first splits, starting at the unhoisted floor or twice the largest load.
        Further splits trade one load's worth of residency for in-flight room."""
        minimum_inflight_bytes = inflight_peak_bytes(tasks, {p.anchor: unhoisted_index(p) for p in tasks})
        largest_load_bytes = max((p.transfer_bytes for p in tasks), default=0)
        if self.weight_budget_bytes <= 0:
            yield self.max_resident_bytes, self._effective_inflight_budget_bytes(tasks)
            return
        weight_budget_bytes = self.weight_budget_bytes
        if minimum_inflight_bytes >= weight_budget_bytes:
            magi_logger.warning(
                "h2d load reorder: device-weight budget %.0f MiB <= unhoisted peak %.1f MiB; residency disabled",
                weight_budget_bytes / 2**20,
                minimum_inflight_bytes / 2**20,
            )
            yield 0, minimum_inflight_bytes
            return
        inflight_budget_bytes = min(weight_budget_bytes, max(minimum_inflight_bytes, 2 * largest_load_bytes))
        for _ in range(_MAX_BUDGET_SPLIT_ROUNDS):
            yield weight_budget_bytes - inflight_budget_bytes, inflight_budget_bytes
            if inflight_budget_bytes >= weight_budget_bytes or largest_load_bytes <= 0:
                return
            inflight_budget_bytes = min(weight_budget_bytes, inflight_budget_bytes + largest_load_bytes)

    def _select_resident_for_density(
        self, tasks, compute_prefix_ns, resident_budget_bytes, trial_density
    ) -> tuple[frozenset, bool]:
        """Bound transferred bytes at each deadline by the target bus density and available compute time.
        Promote the smallest weight covering each deficit; the flag is false
        when the residency budget cannot pay for a deficit."""
        effective_bandwidth_bytes_per_ns = trial_density * self.bandwidth_bytes_per_ns
        resident_anchors: set = set()
        bus_tasks: list = []
        pending_transfer_bytes = resident_bytes = 0
        is_density_feasible = True
        for task in sorted(tasks, key=lambda t: (t.deadline_index, -t.transfer_bytes)):
            pending_transfer_bytes += task.transfer_bytes
            bus_tasks.append(task)
            allowed_transfer_bytes = (
                effective_bandwidth_bytes_per_ns * compute_prefix_ns[min(task.deadline_index, len(compute_prefix_ns) - 1)]
            )
            while pending_transfer_bytes > allowed_transfer_bytes:
                selected_weight = self._pick_weight_for_deficit(
                    candidates=bus_tasks,
                    deficit_bytes=pending_transfer_bytes - allowed_transfer_bytes,
                    remaining_resident_bytes=resident_budget_bytes - resident_bytes,
                )
                if selected_weight is None:
                    is_density_feasible = False
                    break
                bus_tasks.remove(selected_weight)
                resident_anchors.add(selected_weight.anchor)
                pending_transfer_bytes -= selected_weight.transfer_bytes
                resident_bytes += selected_weight.transfer_bytes
        return frozenset(resident_anchors), is_density_feasible

    @staticmethod
    def _pick_weight_for_deficit(candidates, deficit_bytes: float, remaining_resident_bytes: int):
        """Smallest affordable weight covering the deficit, or largest affordable partial relief."""
        affordable_weights = [p for p in candidates if p.transfer_bytes <= remaining_resident_bytes]
        if not affordable_weights:
            return None
        covering_weights = [p for p in affordable_weights if p.transfer_bytes >= deficit_bytes]
        return (
            min(covering_weights, key=lambda p: p.transfer_bytes)
            if covering_weights
            else max(affordable_weights, key=lambda p: p.transfer_bytes)
        )

    def _search_resident_weights(self, tasks, compute_prefix_ns, resident_budget_bytes) -> frozenset:
        """Search for the lowest feasible bus density within the residency budget.
        If none fits, use ``bus_utilization`` and spend the available budget there."""
        min_density, max_density = 0.0, self.bus_utilization
        best_resident_anchors: frozenset | None = None
        for _ in range(_DENSITY_SEARCH_STEPS):
            trial_density = (min_density + max_density) / 2
            resident_anchors, is_density_feasible = self._select_resident_for_density(
                tasks=tasks,
                compute_prefix_ns=compute_prefix_ns,
                resident_budget_bytes=resident_budget_bytes,
                trial_density=trial_density,
            )
            if is_density_feasible:
                best_resident_anchors = resident_anchors
                max_density = trial_density
            else:
                min_density = trial_density
        if best_resident_anchors is not None:
            return best_resident_anchors
        resident_anchors, _ = self._select_resident_for_density(
            tasks=tasks,
            compute_prefix_ns=compute_prefix_ns,
            resident_budget_bytes=resident_budget_bytes,
            trial_density=self.bus_utilization,
        )
        return resident_anchors

    # -- committing --------------------------------------------------------
    @staticmethod
    def _make_selected_weights_resident(tasks, resident_anchors) -> int:
        from ..runtime.host_weights import get_global_host_weight_store

        resident_slots = [p.host_slot for p in tasks if p.anchor in resident_anchors]
        return get_global_host_weight_store().make_weights_resident(resident_slots)
