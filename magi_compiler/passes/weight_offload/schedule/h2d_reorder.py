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

"""Second-phase reorder: hoist each weight load off the gather it feeds.

Installed AFTER ``FsdpOverlapReorder`` in
``reorder_for_compute_comm_overlap_passes``; Inductor chains the passes
(``order = p(order)``), so this one sees the all-gather placement the first pass
settled on.  Compute is priced by the ``SnodeCostTable`` that ``SnodeCostProfile``
filled at the head of the chain; a load's own transfer is priced here, from its
bytes and the configured bandwidth.

Phase 1 moves ``h2d_load``, its wait and the all-gather as one block, which is
correct but leaves the load fully exposed: the wait sits directly in front of the
gather, so the compute stream stalls for the whole transfer before the gather is
even issued.  This pass pulls the load -- and only the load -- back out of that
block, opening a window of compute between it and its wait::

    ... compute ...  h2d_load  ... compute ...  h2d_wait  all_gather  ...
                     ^ moved here             ^ left where phase 1 put it

What a step can overlap at all
------------------------------

Before choosing where any load goes, there is a question of whether they can
fit: with ``C`` ns of compute in the step and a bus running at ``B`` bytes/ns,
no placement can hide more than ``C * B`` bytes of transfer.  Anything past that
is exposed wherever it is put, and the only way to remove it is to stop
transferring it -- residency.  So the pass computes

    schedule_size = bus_utilization * C * B

and requires ``total_bytes - schedule_size`` of residency before it places
anything.  Getting this backwards is not a small error: on a model with
``total_bytes`` at 1.4x ``C * B``, a sweep that only prices per-load exposure
reports every load hidden (each one individually is), buys no residency at all,
and runs a third slower than the same graph with a third of the weight resident.

Two budgets, and between them the whole policy
----------------------------------------------

* ``max_resident_bytes`` -- weight bytes handed back to the device permanently.
* ``max_inflight_bytes`` -- weight bytes that may be live in load buffers at once.

They add: the device holds at most ``max_resident_bytes + max_inflight_bytes`` of
weight.  Splitting what used to be a single peak cap is what makes the trade
expressible at all -- but the two are competing for one pool of device memory,
so ``max_device_weight_bytes`` lets the caller name the pool instead and have
the pass split it: the smallest in-flight budget that still hides everything,
and the remainder to residency, which is worth strictly more per byte.

Residency has to be spread, not stacked
---------------------------------------

Which weights become resident matters as much as how many.  Handing the budget
to the largest ones takes them all from the front of the graph and leaves the
back at its original bus-to-compute density, where the loads have nowhere to go;
the step then follows the saturated half.  ``_select_resident`` instead walks the
loads in wait order against a running allowance of ``utilization * B *
prefix[wait]`` and promotes whatever overruns it, which keeps every prefix of the
graph feasible and therefore spreads residency at the density the graph needs.
Short hoists follow from that on their own, and short hoists are what keeps the
in-flight peak small.

Bus occupancy is not live range
--------------------------------

One PCIe stream serializes transfers in *time*, not in snode-index windows.  A
load issued at index ``at`` occupies the bus for ``need`` ns starting when the
compute stream reaches ``prefix[at]`` (or later, if the bus is still busy).  That
interval is what must not overlap another transfer.  The buffer's live range
``[at, last_user]`` is a separate quantity, bounded only by the in-flight byte
budget.

So the sweep books the two separately: a running bus end on the time axis, and
``InflightMap`` on ``[at, last_user]`` in bytes.  Transfers are scheduled in
DEADLINE order and as late as their deadlines allow (see ``_sweep``), which is
what keeps the two from being confused for one another -- a load is hoisted
because the bus needs it earlier, never because the model could not find a
cheaper index.

Residency is spent to the budget, not to zero exposure
------------------------------------------------------

A resident weight does not merely get hidden, it stops crossing PCIe at all:
every step the bus carries ``total - resident`` bytes.  Modelled exposure can
reach zero while the bus is still the real limit -- the model prices one stream
at a nominal bandwidth, and eight ranks pulling from the same host in lockstep
do not get it.  So the buy-out runs in two phases: first the priced one, which
aims the budget at whatever the sweep still cannot hide, then a fill that hands
out the remainder largest-first.  Per byte the bus saving is identical whichever
weight is chosen, so the fill takes the big buckets: they dominate the in-flight
peak and the allocator churn that goes with it.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.utils import contains_wait

from magi_compiler.utils import magi_logger

from ...overlap import (
    DEFAULT_WINDOW_MARGIN_NS,
    CostView,
    OverlapPass,
    SnodeGraph,
    alap_schedule,
    device_peak,
    inflight_peak,
    live_starts,
    unhoisted_index,
)
from ...snode_utils import is_multi_output
from .h2d_snode import H2D_OPS, is_h2d_load, slots_of
from .load_plan import LoadPlan, load_bytes


def _magi_logger_enabled_for_debug() -> bool:
    """The per-load report builds a string per line; skip it when nobody reads it."""
    return logging.getLogger("magi_compiler").isEnabledFor(logging.DEBUG)


_MAX_SPLIT_ROUNDS = 5
"""Times ``_schedule`` may grow the in-flight budget out of the residency one."""

_DENSITY_SEARCH_ROUNDS = 8
"""Bisection steps for the target bus density; 8 lands within 0.4% of it."""


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
        bus_utilization: float = 0.9,
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

    def _transfer_ns(self, group: list[BaseSchedulerNode]) -> float:
        return load_bytes(group) / self.bandwidth_bytes_per_ns

    # -- OverlapPass hooks ------------------------------------------------
    def wants(self, order: list[BaseSchedulerNode]) -> bool:
        if any(is_h2d_load(s) for s in order):
            return True
        magi_logger.debug("h2d load reorder: no weight load among %d snodes (known ops: %s)", len(order), H2D_OPS)
        return False

    def plan(self, graph: SnodeGraph, cost: CostView, ctx: dict) -> list[LoadPlan] | None:
        return self._plan(graph) or None

    def place(self, graph: SnodeGraph, cost: CostView, plans: list[LoadPlan], ctx: dict) -> None:
        prefix = cost.prefix(graph.order)
        targets, budget = self._schedule(plans, graph.order, graph.index_of, prefix)
        ctx.update(prefix=prefix, targets=targets, budget=budget)

    def on_commit(self, graph: SnodeGraph, plans: list[LoadPlan], ctx: dict) -> None:
        ctx["given_back"] = self._promote(plans)

    def report(self, graph: SnodeGraph, cost: CostView, plans: list[LoadPlan], committed: bool, ctx: dict) -> None:
        if committed:
            self._report(plans, ctx["targets"], graph.index_of, ctx["given_back"], ctx["budget"], len(graph), ctx["prefix"])

    # -- planning ---------------------------------------------------------
    @staticmethod
    def _group_and_waits(load, graph: SnodeGraph) -> tuple[list, list]:
        """The load plus the unpacks that travel with it, and the waits guarding it."""

        def classify(u):
            if contains_wait(u):
                return "wait", False
            if is_multi_output(u):
                return "member", True
            return None, False

        found = graph.walk_users([load], classify)
        group = [load] + [u for label, u in found if label == "member"]
        waits = [u for label, u in found if label == "wait"]
        return group, waits

    def _plan(self, graph: SnodeGraph) -> list[LoadPlan]:
        index_of = graph.index_of
        plans: list[LoadPlan] = []
        for load in (s for s in graph.order if is_h2d_load(s)):
            group, waits = self._group_and_waits(load, graph)
            if not waits:
                magi_logger.debug("h2d load reorder: %s has no wait; leaving it in place", load.get_name())
                continue
            need = self._transfer_ns(group) * self.window_scale + self.window_margin_ns
            plans.append(
                LoadPlan(
                    anchor=load,
                    group=group,
                    slots=slots_of(load),
                    deadline=min(index_of[w] for w in waits),
                    last_user=graph.last_user_index(group, waits),
                    need=need,
                    lower=graph.earliest_legal_index(group),
                    nbytes=load_bytes(group),
                    exposed=need,
                )
            )
        return plans

    # -- scheduling --------------------------------------------------------
    def _inflight_budget(self, plans) -> int:
        """Bytes of load buffer this schedule may have live at once.

        Floored at what phase 1's own order already needs.  Unasked, that floor
        plus one load -- the least memory any overlap at all can cost.
        """
        floor = inflight_peak(plans, {p.load: unhoisted_index(p) for p in plans})
        if self.max_inflight_bytes <= 0:
            return floor + max((p.nbytes for p in plans), default=0)
        if self.max_inflight_bytes < floor:
            magi_logger.warning(
                "h2d load reorder: the %.1f MiB in-flight budget is under the %.1f MiB that phase 1's "
                "own unhoisted order already needs, so it is raised to that -- weights are still "
                "being read past where the next load has to start, and no placement changes it",
                self.max_inflight_bytes / 2**20,
                floor / 2**20,
            )
            return floor
        return self.max_inflight_bytes

    def _hideable_bytes(self, plans, prefix) -> float:
        """Weight bytes the step has compute to overlap, at the priced bandwidth.

        Bounded by the compute upstream of the *last* wait, not by the whole
        graph: compute after every load's deadline can hide nothing.
        """
        last_wait = max(p.wait_idx for p in plans)
        return self.bus_utilization * prefix[min(last_wait, len(prefix) - 1)] * self.bandwidth_bytes_per_ns

    def _budget_splits(self, plans):
        """``(resident, inflight)`` budgets to try, most residency first.

        With a single device-weight budget the two are one number split two ways,
        and the split is not symmetric: residency takes bytes off the bus for
        every step, while in-flight room only decides how far upstream a load may
        start.  So the first attempt keeps in-flight at the minimum that can
        pipeline at all -- the unhoisted floor, or two buckets, whichever is
        larger -- and spends everything else on residency; later attempts buy
        in-flight room back one bucket at a time, and only if the sweep says the
        budget and not the bus is what left transfer exposed.
        """
        floor = inflight_peak(plans, {p.load: unhoisted_index(p) for p in plans})
        biggest = max((p.nbytes for p in plans), default=0)
        if self.max_device_weight_bytes <= 0:
            yield self.max_resident_bytes, self._inflight_budget(plans)
            return
        device = self.max_device_weight_bytes
        if floor >= device:
            magi_logger.warning(
                "h2d load reorder: the %.0f MiB device-weight budget is under the %.1f MiB of load "
                "buffers phase 1's own unhoisted order already needs, so nothing is left for "
                "residency and the bus carries every byte",
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
        self._log_shortfall(total, hideable, sum(p.nbytes for p in plans if p.promoted), sum(p.exposed for p in plans))
        return targets, inflight_budget

    def _select_resident_at(self, plans, prefix, resident_budget, density) -> tuple[set, int, bool]:
        """Promote weights until every prefix of the graph holds ``density``.

        Walks the loads in deadline order against a running allowance of
        ``density * B * prefix[wait]`` -- the bytes the bus can have delivered by
        the time this load's gather needs them.  Overrunning it means the loads up
        to here cannot all fit in the compute up to here, no matter where they go,
        so one comes off the bus: the SMALLEST weight that closes the overrun.
        Best fit, not largest, on two counts -- it does not spend budget on relief
        the window did not ask for, and it leaves the window sitting at the target
        density instead of alternating between slack and saturation.  Loads then
        only have to travel as far as their own window, and short hoists are what
        keeps the in-flight peak small.

        Spreading falls out of walking prefixes rather than sizes.  Handing the
        budget to the largest weights globally would take them all from the front
        of the graph and leave the back saturated, which is where the step time
        would then come from.

        The third return value is whether the budget covered the target
        everywhere; ``False`` means the walk wanted another weight off the bus and
        could not pay for it.
        """
        rate = density * self.bandwidth_bytes_per_ns
        promoted: set = set()
        on_bus: list = []
        kept = spent = 0
        feasible = True
        for plan in sorted(plans, key=lambda p: (p.wait_idx, -p.nbytes)):
            kept += plan.nbytes
            if plan.slots:  # only a load with host slots can be given back
                on_bus.append(plan)
            allowance = rate * prefix[min(plan.wait_idx, len(prefix) - 1)]
            while kept > allowance:
                pick = self._best_fit(on_bus, kept - allowance, resident_budget - spent)
                if pick is None:
                    # No candidate at all is the graph's own doing (a load with no
                    # host slot cannot be given back); one that exists but does not
                    # fit is the budget refusing, and only that is infeasibility.
                    feasible = feasible and not on_bus
                    break
                on_bus.remove(pick)
                promoted.add(pick.load)
                kept -= pick.nbytes
                spent += pick.nbytes
        return promoted, spent, feasible

    @staticmethod
    def _best_fit(candidates, deficit: float, room: int):
        """Smallest candidate that closes ``deficit``, else the largest that fits.

        The fallback matters as much as the rule: when no single weight covers the
        overrun, taking the largest is what makes progress -- the loop then comes
        back for the remainder.
        """
        affordable = [p for p in candidates if p.nbytes <= room]
        if not affordable:
            return None
        covering = [p for p in affordable if p.nbytes >= deficit]
        return min(covering, key=lambda p: p.nbytes) if covering else max(affordable, key=lambda p: p.nbytes)

    def _select_resident(self, plans, prefix, resident_budget) -> tuple[set, int]:
        """The flattest bus-to-compute density the residency budget can hold.

        Feasibility is monotone in the target: a budget that holds density ``d``
        everywhere also holds anything looser, so the smallest holdable ``d`` is
        the one to take.  Asking for less than the bus needs is the point -- it is
        how a budget larger than feasibility requires gets spent without stacking
        every purchase at the front of the graph, and it buys margin against the
        cost table being optimistic.  When even the honest utilization does not
        fit, the walk runs there anyway and spends what it has; ``_log_shortfall``
        says how much transfer is then exposed no matter what.
        """
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
        """Schedule the bus by deadline, then issue each load just in time.

        ``overlap.alap_schedule`` does the placing.  Promoted loads are its
        inactive tasks: they keep phase 1's position and take no bus slot
        (device-to-device) and no in-flight room -- their bytes are charged to
        the residency budget, and charging them twice would let a filled
        residency squeeze the transfers that are still on the bus.

        Scheduling by size instead of by deadline is what the previous sweep
        did, and it cost both ways.  The big transfers claimed the bus first; a
        3ms bundle then found every mid-graph instant taken, and the only
        placement the model scored as fully hidden was the gap at the very top of
        the graph.  On gaga4 400B that put 28 such bundles at snode ~400 against
        deadlines 1000+ nodes later: 4% of the bytes holding 37% of the in-flight
        budget for the length of the graph, which under one device-weight budget
        comes straight out of residency and back onto the bus.
        """
        for plan in plans:
            plan.promoted = plan.load in promoted
        return alap_schedule(plans, prefix, index_of, budget)

    # -- committing --------------------------------------------------------
    @staticmethod
    def _promote(plans) -> int:
        from ..runtime import host_pool

        slots: list[int] = []
        for p in plans:
            if not p.promoted:
                continue
            if not p.slots:
                p.promoted = False
                continue
            slots.extend(p.slots)
        return host_pool.make_resident_many(slots)

    # -- reporting --------------------------------------------------------
    def _report(self, plans, targets, index_of, given_back, budget, n_snodes, prefix) -> None:
        resident = [p for p in plans if p.promoted]
        exposed = sum(p.exposed for p in plans)
        by_budget = sum(p.exposed for p in plans if p.budget_bound)
        inflight = inflight_peak(plans, live_starts(plans, targets, index_of))
        moved = sum(1 for load, target in targets.items() if target != index_of[load])
        on_bus = sum(p.nbytes for p in plans if not p.promoted)
        total = sum(p.nbytes for p in plans)
        hideable = self._hideable_bytes(plans, prefix)
        compute_ns = hideable / self.bus_utilization / self.bandwidth_bytes_per_ns
        magi_logger.info(
            "h2d load reorder: %.0fms of compute at %.1f GB/s can overlap %.1f MiB of the %.1f MiB "
            "offloaded (%.0f%% bus utilization assumed), so %.1f MiB had to become resident; it did "
            "over %d weight(s), leaving %.1f MiB on the bus, %.1fms of transfer. Hoisted %d/%d "
            "load(s); in-flight peak %.1f MiB of the %.0f MiB budget; %.1f MiB of weight on the "
            "device at peak; %.1fms still exposed%s",
            compute_ns / 1e6,
            self.bandwidth_bytes_per_ns,
            hideable / 2**20,
            total / 2**20,
            self.bus_utilization * 100,
            given_back / 2**20,
            len(resident),
            on_bus / 2**20,
            on_bus / self.bandwidth_bytes_per_ns / 1e6,
            moved,
            len(plans),
            inflight / 2**20,
            budget / 2**20,
            device_peak(plans, targets, index_of) / 2**20,
            exposed / 1e6,
            f", {by_budget / 1e6:.1f}ms of it because the in-flight budget and not the bus ran out" if by_budget > 0 else "",
        )
        self._log_density(plans, prefix)
        if inflight > budget:
            magi_logger.warning(
                "h2d load reorder: %.1f MiB of load buffers are live at once, over the %.0f MiB "
                "budget the sweep was supposed to hold -- the in-flight accounting and the emitted "
                "live ranges disagree, so treat the peak this pass reports as unreliable",
                inflight / 2**20,
                budget / 2**20,
            )
        self._log_placement(plans, targets, index_of, n_snodes)

    def _log_density(self, plans, prefix) -> None:
        """How full the bus is over every prefix of the graph, after residency.

        Measured on prefixes, not on the gap between neighbouring deadlines: a
        load can be hoisted anywhere upstream, so what has to hold is that the
        bytes due by each deadline fit the compute available by then.  Per-gap
        densities read as noise for exactly that reason -- two loads four snodes
        apart show one empty window and one impossible one, and nothing is wrong.

        A max above 1.0 is a schedule that does not exist: that prefix has more
        transfer than compute and the excess is exposed wherever it is placed.
        """
        if not _magi_logger_enabled_for_debug():
            return
        by_wait: dict[int, int] = defaultdict(int)
        for p in plans:
            if not p.promoted:
                by_wait[p.wait_idx] += p.nbytes
        ratios = []
        cumulative = 0
        for wait in sorted(by_wait):
            cumulative += by_wait[wait]
            compute = prefix[min(wait, len(prefix) - 1)]
            if compute > 0:
                ratios.append(cumulative / self.bandwidth_bytes_per_ns / compute)
        if not ratios:
            return
        magi_logger.debug(
            "h2d load reorder: bus occupancy over %d prefix(es): first %.2f, median %.2f, worst "
            "%.2f at prefix %d/%d (1.0 means the transfers due by then exactly fill the compute "
            "available by then, and above 1.0 cannot be hidden at any placement)",
            len(ratios),
            ratios[0],
            sorted(ratios)[len(ratios) // 2],
            max(ratios),
            ratios.index(max(ratios)) + 1,
            len(ratios),
        )

    @staticmethod
    def _weights_of(plan) -> str:
        from ..runtime import host_pool

        names = [host_pool.name_of(s) for s in plan.slots]
        names = [n for n in names if n] or ["?"]
        return ", ".join(names[:3]) + (f", +{len(names) - 3} more" if len(names) > 3 else "")

    def _log_placement(self, plans, targets, index_of, n_snodes) -> None:
        if not _magi_logger_enabled_for_debug():
            return
        magi_logger.debug(
            "h2d load placement (%d loads over %d snodes; 'at' is where the load ended up, "
            "'last_user' the last snode that still reads its bytes, 'floor' the dep floor and the "
            "in-flight floor):",
            len(plans),
            n_snodes,
        )
        for p in sorted(plans, key=lambda p: index_of[p.load]):
            if p.promoted:
                verdict = f"RESIDENT: {p.need / 1e6:.1f}ms off the bus"
            elif p.exposed <= 0:
                verdict = "hidden"
            elif p.budget_bound:
                verdict = f"EXPOSED {p.exposed / 1e6:.1f}ms (in-flight budget)"
            else:
                verdict = f"EXPOSED {p.exposed / 1e6:.1f}ms (bus/compute)"
            at = targets.get(p.load, index_of[p.load])
            magi_logger.debug(
                "  %-10s %2d slot(s) %7.1f MiB  at %5d (from %5d, floor %5d/%5d)  last_user %5d  " "need %6.1fms  %-38s  %s",
                p.load.get_name(),
                len(p.slots),
                p.nbytes / 2**20,
                at,
                index_of[p.load],
                p.lower,
                p.budget_floor,
                p.last_user,
                p.need / 1e6,
                verdict,
                self._weights_of(p),
            )

    def _log_shortfall(self, total, hideable, resident, exposed) -> None:
        """Say so when the step simply has no compute for the bytes left on the bus."""
        unavoidable = total - resident - hideable
        if unavoidable <= 0 or exposed <= 0:
            return
        magi_logger.warning(
            "h2d load reorder: %.1f MiB of weight is offloaded and %.1f MiB of it is resident, "
            "leaving %.1f MiB on the bus, but %.0f ms of compute at %.1f GB/s can only overlap "
            "%.1f MiB of it -- %.1f MiB (%.1fms) is exposed wherever the loads are placed, and only "
            "more residency removes it",
            total / 2**20,
            resident / 2**20,
            (total - resident) / 2**20,
            hideable / self.bus_utilization / self.bandwidth_bytes_per_ns / 1e6,
            self.bandwidth_bytes_per_ns,
            hideable / 2**20,
            unavoidable / 2**20,
            unavoidable / self.bandwidth_bytes_per_ns / 1e6,
        )
