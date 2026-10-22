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

``alap_schedule`` places transfers by deadline. Serialized bus time and buffer
lifetime through the last consumer are separate constraints. Resident loads keep
their existing position and are charged to residency, not in-flight storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from torch._inductor.scheduler import BaseSchedulerNode

from magi_compiler.utils import magi_logger

from ...overlap import (
    DEFAULT_WINDOW_MARGIN_NS,
    CostView,
    OverlapPass,
    SnodeGraph,
    TransferTask,
    alap_schedule,
    device_peak,
    inflight_peak,
    live_starts,
    unhoisted_index,
)
from .h2d_snode import is_h2d_load, load_bytes, slot_of

_MAX_SPLIT_ROUNDS = 5
"""Times ``_schedule`` may grow the in-flight budget out of the residency one."""

_DENSITY_SEARCH_ROUNDS = 8
"""Bisection steps for the target bus density; 8 lands within 0.4% of it."""


@dataclass(eq=False)
class LoadPlan(TransferTask):
    """One load's placement problem: how much transfer to hide, and where it may go.

    ``anchor`` is the load, ``deadline`` its earliest wait (the hard upper
    bound), and ``inactive`` means bought out of the offload plan: resident, so
    the transfer is now device-to-device and off the bus.
    """

    slot: int = field(kw_only=True)  # host-pool slot this load pulls


class H2dLoadReorder(OverlapPass):
    """Callable reorder pass.  Run me after ``FsdpOverlapReorder``."""

    name = "h2d load reorder"

    def __init__(
        self,
        bandwidth_bytes_per_ns: float,
        window_margin_ns: float = DEFAULT_WINDOW_MARGIN_NS,
        window_scale: float = 1.0,
        max_resident_bytes: int = 0,
        max_inflight_bytes: int = 0,
        max_device_weight_bytes: int = 0,
        bus_utilization: float = 1.0,
        cost_fn=None,
    ) -> None:
        super().__init__(cost_fn)
        self.bandwidth_bytes_per_ns = max(1e-6, bandwidth_bytes_per_ns)
        self.window_margin_ns = window_margin_ns
        self.window_scale = window_scale
        self.max_resident_bytes = max_resident_bytes
        self.max_inflight_bytes = max_inflight_bytes
        self.max_device_weight_bytes = max_device_weight_bytes
        self.bus_utilization = min(1.0, max(1e-3, bus_utilization))

    # -- OverlapPass hooks ------------------------------------------------
    def wants(self, order: list[BaseSchedulerNode]) -> bool:
        if any(is_h2d_load(s) for s in order):
            return True
        magi_logger.debug("h2d load reorder: no weight load among %d snodes", len(order))
        return False

    def plan(self, graph: SnodeGraph, cost: CostView, ctx: dict) -> list[LoadPlan] | None:
        return self._plan(graph) or None

    def place(self, graph: SnodeGraph, cost: CostView, plans: list[LoadPlan], ctx: dict) -> None:
        prefix = cost.prefix(graph.order)
        targets, budget = self._schedule(plans, graph.order, graph.index_of, prefix)
        ctx.update(targets=targets, budget=budget)

    def on_commit(self, graph: SnodeGraph, plans: list[LoadPlan], ctx: dict) -> None:
        ctx["given_back"] = self._promote(plans)

    def report(self, graph: SnodeGraph, cost: CostView, plans: list[LoadPlan], committed: bool, ctx: dict) -> None:
        if committed:
            self._report(plans, ctx["targets"], graph.index_of, ctx["given_back"], ctx["budget"])

    # -- planning ---------------------------------------------------------
    @staticmethod
    def _group_and_waits(load, graph: SnodeGraph) -> tuple[list, list]:
        """The load plus the ``MultiOutput`` unpack its result reaches readers through, and the waits guarding it."""
        return [load, *graph.unpacks_of(load)], graph.waits_of(load)

    def _plan(self, graph: SnodeGraph) -> list[LoadPlan]:
        index_of = graph.index_of
        plans: list[LoadPlan] = []
        for load in (s for s in graph.order if is_h2d_load(s)):
            group, waits = self._group_and_waits(load, graph)
            if not waits:
                magi_logger.debug("h2d load reorder: %s has no wait; leaving it in place", load.get_name())
                continue
            nbytes = load_bytes(group)
            need = nbytes / self.bandwidth_bytes_per_ns * self.window_scale + self.window_margin_ns
            plans.append(
                LoadPlan(
                    anchor=load,
                    group=group,
                    slot=slot_of(load),
                    deadline=min(index_of[w] for w in waits),
                    last_user=graph.last_user_index(group, waits),
                    need=need,
                    lower=graph.earliest_legal_index(group),
                    nbytes=nbytes,
                    exposed=need,
                )
            )
        return plans

    # -- scheduling --------------------------------------------------------
    def _inflight_budget(self, plans) -> int:
        """Never go below the unhoisted live-buffer peak.
        Without an explicit limit, allow that peak plus one largest load."""
        floor = inflight_peak(plans, {p.anchor: unhoisted_index(p) for p in plans})
        if self.max_inflight_bytes <= 0:
            return floor + max((p.nbytes for p in plans), default=0)
        if self.max_inflight_bytes < floor:
            magi_logger.warning(
                "h2d load reorder: in-flight budget raised from %.1f to %.1f MiB (unhoisted peak)",
                self.max_inflight_bytes / 2**20,
                floor / 2**20,
            )
            return floor
        return self.max_inflight_bytes

    def _hideable_bytes(self, plans, prefix) -> float:
        """Compute after the last load deadline cannot hide any weight transfer."""
        last_wait = max(p.deadline for p in plans)
        return self.bus_utilization * prefix[min(last_wait, len(prefix) - 1)] * self.bandwidth_bytes_per_ns

    def _budget_splits(self, plans):
        """Try residency-first splits, starting at the unhoisted floor or twice the largest load.
        Further splits trade one load's worth of residency for in-flight room."""
        floor = inflight_peak(plans, {p.anchor: unhoisted_index(p) for p in plans})
        biggest = max((p.nbytes for p in plans), default=0)
        if self.max_device_weight_bytes <= 0:
            yield self.max_resident_bytes, self._inflight_budget(plans)
            return
        device = self.max_device_weight_bytes
        if floor >= device:
            magi_logger.warning(
                "h2d load reorder: device-weight budget %.0f MiB <= unhoisted peak %.1f MiB; residency disabled",
                device / 2**20,
                floor / 2**20,
            )
            yield 0, floor
            return
        inflight = min(device, max(floor, 2 * biggest))
        for _ in range(_MAX_SPLIT_ROUNDS):
            yield device - inflight, inflight
            if inflight >= device or biggest <= 0:
                return
            inflight = min(device, inflight + biggest)

    def _schedule(self, plans, order, index_of, prefix) -> tuple[dict, int]:
        """Buy the residency the step cannot overlap, then place what is left."""
        total = sum(p.nbytes for p in plans)
        hideable = self._hideable_bytes(plans, prefix)
        best: tuple[float, dict, int, set] | None = None
        for resident_budget, inflight_budget in self._budget_splits(plans):
            promoted, spent = self._select_resident(plans, prefix, resident_budget)
            targets = self._sweep(plans, order, index_of, prefix, promoted, inflight_budget)
            exposed = sum(p.exposed for p in plans)
            if best is None or exposed < best[0] - 1e-9:
                best = (exposed, targets, inflight_budget, promoted)
            # Only an in-flight shortage is worth buying more in-flight room for;
            # if the bus itself is full, taking bytes off residency makes it worse.
            if exposed <= 0 or not any(p.budget_bound for p in plans):
                break
        assert best is not None  # _budget_splits always yields at least once
        _, targets, inflight_budget, promoted = best
        # Re-run the winner so the plans carry its placement, not the last try's.
        targets = self._sweep(plans, order, index_of, prefix, promoted, inflight_budget)
        self._log_shortfall(total, hideable, sum(p.nbytes for p in plans if p.inactive), sum(p.exposed for p in plans))
        return targets, inflight_budget

    def _select_resident_at(self, plans, prefix, resident_budget, density) -> tuple[set, int, bool]:
        """Bound transferred bytes at each deadline by ``density * bandwidth * prefix[wait]``.
        Promote the smallest weight covering each deficit; the final flag is false
        when the residency budget cannot pay for a deficit."""
        rate = density * self.bandwidth_bytes_per_ns
        promoted: set = set()
        on_bus: list = []
        kept = spent = 0
        feasible = True
        for plan in sorted(plans, key=lambda p: (p.deadline, -p.nbytes)):
            kept += plan.nbytes
            on_bus.append(plan)
            allowance = rate * prefix[min(plan.deadline, len(prefix) - 1)]
            while kept > allowance:
                pick = self._best_fit(on_bus, kept - allowance, resident_budget - spent)
                if pick is None:
                    feasible = False
                    break
                on_bus.remove(pick)
                promoted.add(pick.anchor)
                kept -= pick.nbytes
                spent += pick.nbytes
        return promoted, spent, feasible

    @staticmethod
    def _best_fit(candidates, deficit: float, room: int):
        """Smallest affordable weight covering the deficit, or largest affordable partial relief."""
        affordable = [p for p in candidates if p.nbytes <= room]
        if not affordable:
            return None
        covering = [p for p in affordable if p.nbytes >= deficit]
        return min(covering, key=lambda p: p.nbytes) if covering else max(affordable, key=lambda p: p.nbytes)

    def _select_resident(self, plans, prefix, resident_budget) -> tuple[set, int]:
        """Search for the lowest feasible prefix density within the residency budget.
        If none fits, use ``bus_utilization`` and spend the available budget there."""
        lo, hi = 0.0, self.bus_utilization
        best: tuple[set, int] | None = None
        for _ in range(_DENSITY_SEARCH_ROUNDS):
            mid = (lo + hi) / 2
            promoted, spent, feasible = self._select_resident_at(plans, prefix, resident_budget, mid)
            if feasible:
                best = (promoted, spent)
                hi = mid
            else:
                lo = mid
        if best is not None:
            return best
        promoted, spent, _ = self._select_resident_at(plans, prefix, resident_budget, self.bus_utilization)
        return promoted, spent

    def _sweep(self, plans, order, index_of, prefix, promoted, budget) -> dict:
        """Resident loads stay in place and consume neither bus time nor in-flight budget.
        Their bytes are already charged to residency; other loads use deadline order."""
        for plan in plans:
            plan.inactive = plan.anchor in promoted
        return alap_schedule(plans, prefix, index_of, budget)

    # -- committing --------------------------------------------------------
    @staticmethod
    def _promote(plans) -> int:
        from ..runtime import host_weights

        return host_weights.default_store().make_resident_many([p.slot for p in plans if p.inactive])

    # -- reporting --------------------------------------------------------
    def _report(self, plans, targets, index_of, given_back, budget) -> None:
        resident = [p for p in plans if p.inactive]
        exposed = sum(p.exposed for p in plans)
        by_budget = sum(p.exposed for p in plans if p.budget_bound)
        inflight = inflight_peak(plans, live_starts(plans, targets, index_of))
        moved = sum(1 for load, target in targets.items() if target != index_of[load])
        on_bus = sum(p.nbytes for p in plans if not p.inactive)
        magi_logger.info(
            "h2d load reorder: promoted %.1f MiB; resident loads=%d; bus=%.1f MiB at %.1f GB/s; "
            "Hoisted %d/%d load(s); in-flight peak %.1f MiB of the %.0f MiB budget; "
            "device-weight peak=%.1f MiB; exposed=%.1fms (budget-bound=%.1fms)",
            given_back / 2**20,
            len(resident),
            on_bus / 2**20,
            self.bandwidth_bytes_per_ns,
            moved,
            len(plans),
            inflight / 2**20,
            budget / 2**20,
            device_peak(plans, targets, index_of) / 2**20,
            exposed / 1e6,
            by_budget / 1e6,
        )
        assert (
            inflight <= budget
        ), f"h2d load reorder: in-flight peak {inflight} B exceeds the {budget} B budget it was placed under"

    def _log_shortfall(self, total, hideable, resident, exposed) -> None:
        """Warn when estimated compute coverage cannot hide the remaining transfers."""
        unavoidable = total - resident - hideable
        if unavoidable <= 0 or exposed <= 0:
            return
        magi_logger.warning(
            "h2d load reorder: estimated transfer coverage deficit=%.1f MiB (%.1fms); consider more residency",
            unavoidable / 2**20,
            unavoidable / self.bandwidth_bytes_per_ns / 1e6,
        )
