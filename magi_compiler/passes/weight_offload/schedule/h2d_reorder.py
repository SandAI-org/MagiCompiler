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

import copy
from bisect import bisect_right
from collections import defaultdict

from torch._inductor.comms import _is_fake_dep
from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.utils import contains_wait

from magi_compiler.utils import magi_logger

from ...snode_utils import earliest_legal_index, is_compute, is_multi_output, validate_topological_order
from .h2d_reorder_report import H2dReorderReport
from .h2d_snode import H2D_OPS, is_h2d_load, slots_of
from .load_plan import InflightMap, LoadPlan, inflight_peak, load_bytes

_DEFAULT_WINDOW_MARGIN_NS = 5_000.0

_MAX_SPLIT_ROUNDS = 5
"""Times ``_schedule`` may grow the in-flight budget out of the residency one."""

_DENSITY_SEARCH_ROUNDS = 8
"""Bisection steps for the target bus density; 8 lands within 0.4% of it."""


class H2dLoadReorder(H2dReorderReport):
    """Callable reorder pass.  Run me after ``FsdpOverlapReorder``."""

    def __init__(
        self,
        bandwidth_bytes_per_ns: float,
        window_margin_ns: float = _DEFAULT_WINDOW_MARGIN_NS,
        window_scale: float = 1.0,
        max_resident_bytes: int = 0,
        max_inflight_bytes: int = 0,
        max_device_weight_bytes: int = 0,
        bus_utilization: float = 0.9,
        cost_fn=None,
    ) -> None:
        self.bandwidth_bytes_per_ns = max(1e-6, bandwidth_bytes_per_ns)
        self.window_margin_ns = window_margin_ns
        self.window_scale = window_scale
        self.max_resident_bytes = max_resident_bytes
        self.max_inflight_bytes = max_inflight_bytes
        self.max_device_weight_bytes = max_device_weight_bytes
        self.bus_utilization = min(1.0, max(1e-3, bus_utilization))
        if cost_fn is None:
            from torch._inductor.comms import estimate_op_runtime

            cost_fn = estimate_op_runtime
        self._cost_fn = cost_fn

    def __deepcopy__(self, memo):
        # Through memo, so this copy and the profile pass's copy share one table.
        new = H2dLoadReorder.__new__(H2dLoadReorder)
        memo[id(self)] = new
        new.bandwidth_bytes_per_ns = self.bandwidth_bytes_per_ns
        new.window_margin_ns = self.window_margin_ns
        new.window_scale = self.window_scale
        new.max_resident_bytes = self.max_resident_bytes
        new.max_inflight_bytes = self.max_inflight_bytes
        new.max_device_weight_bytes = self.max_device_weight_bytes
        new.bus_utilization = self.bus_utilization
        new._cost_fn = copy.deepcopy(self._cost_fn, memo)
        return new

    def _cost(self, snode: BaseSchedulerNode) -> float:
        try:
            return max(0.0, float(self._cost_fn(snode)))
        except Exception:  # noqa: BLE001
            return 0.0

    def _transfer_ns(self, group: list[BaseSchedulerNode]) -> float:
        return load_bytes(group) / self.bandwidth_bytes_per_ns

    def __call__(self, snodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
        order = list(snodes)
        loads = [s for s in order if is_h2d_load(s)]
        if not loads:
            magi_logger.debug("h2d load reorder: no weight load among %d snodes (known ops: %s)", len(order), H2D_OPS)
            return order

        buf_to_snode = {b: s for s in order for b in s.get_buffer_names()}
        users: dict[str, set] = defaultdict(set)
        for s in order:
            for d in s.unmet_dependencies:
                if not _is_fake_dep(d):
                    users[d.name].add(s)
        index_of = {s: i for i, s in enumerate(order)}

        plans = self._plan(loads, order, index_of, buf_to_snode, users)
        if not plans:
            return order

        prefix = self._compute_prefix(order)
        targets, budget = self._schedule(plans, order, index_of, prefix)

        new_order = self._rebuild(order, targets, index_of, {p.load: p.group for p in plans})
        if not validate_topological_order(new_order, buf_to_snode):
            magi_logger.warning("h2d load reorder: rebuilt order failed validation; leaving graph unchanged")
            return order

        given_back = self._promote(plans)
        order[:] = new_order
        self._report(plans, targets, index_of, given_back, budget, len(order), prefix)
        return order

    # -- planning ---------------------------------------------------------
    @staticmethod
    def _group_and_waits(load, users) -> tuple[list, list]:
        group = [load]
        waits: list = []
        stack = list(load.get_buffer_names())
        seen: set = set()
        while stack:
            for u in users.get(stack.pop(), ()):
                if u in seen:
                    continue
                seen.add(u)
                if contains_wait(u):
                    waits.append(u)
                elif is_multi_output(u):
                    group.append(u)
                    stack.extend(u.get_buffer_names())
        return group, waits

    @staticmethod
    def _last_user_index(group, waits, users, index_of) -> int:
        last = max(index_of[w] for w in waits)
        skip = set(group)
        for src in (*group, *waits):
            for name in src.get_buffer_names():
                for u in users.get(name, ()):
                    if u not in skip:
                        last = max(last, index_of[u])
        return last

    def _plan(self, loads, order, index_of, buf_to_snode, users) -> list[LoadPlan]:
        plans: list[LoadPlan] = []
        for load in sorted(loads, key=lambda s: index_of[s]):
            group, waits = self._group_and_waits(load, users)
            if not waits:
                magi_logger.debug("h2d load reorder: %s has no wait; leaving it in place", load.get_name())
                continue
            need = self._transfer_ns(group) * self.window_scale + self.window_margin_ns
            plans.append(
                LoadPlan(
                    load=load,
                    group=group,
                    slots=slots_of(load),
                    wait_idx=min(index_of[w] for w in waits),
                    last_user=self._last_user_index(group, waits, users, index_of),
                    need=need,
                    lower=earliest_legal_index(group, index_of, buf_to_snode),
                    nbytes=load_bytes(group),
                    exposed=need,
                )
            )
        return plans

    # -- scheduling --------------------------------------------------------
    @staticmethod
    def _unhoisted(plan) -> int:
        """Where a load sits when it is not hoisted at all: against its own wait."""
        return max(plan.lower, plan.wait_idx - 1)

    def _inflight_budget(self, plans) -> int:
        """Bytes of load buffer this schedule may have live at once.

        Floored at what phase 1's own order already needs.  Unasked, that floor
        plus one load -- the least memory any overlap at all can cost.
        """
        floor = inflight_peak(plans, {p.load: self._unhoisted(p) for p in plans})
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
        floor = inflight_peak(plans, {p.load: self._unhoisted(p) for p in plans})
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

        Every offloaded load is booked at its unhoisted position in the in-flight
        map before anything moves, and released only when placed -- so the budget
        is a promise.  Promoted loads keep phase 1's position, take no bus slot
        (device-to-device) and no in-flight room either: their bytes are charged
        to the residency budget, and charging them twice would let a filled
        residency squeeze the transfers that are still on the bus.

        The active loads are then list-scheduled in DEADLINE order, in two
        passes.  A backward pass over the waits gives each transfer the latest
        instant it may finish without pushing the ones after it past their own
        deadlines; a forward pass then runs the bus, giving each transfer the
        latest slot that respects both that limit and the bus still being busy.
        Where the bus is saturated the slots butt together and it never idles;
        where there is slack a transfer simply starts late, which costs nothing
        and keeps its buffer's live range short.  The snode a load is emitted at
        is the latest one whose compute prefix still reaches its slot -- just in
        time, because the DMA begins at that instant whatever index we choose and
        anything earlier only holds memory for longer.

        Scheduling by size instead is what the previous sweep did, and it cost
        both ways.  The big transfers claimed the bus first; a 3ms bundle then
        found every mid-graph instant taken, and the only placement the model
        scored as fully hidden was the gap at the very top of the graph.  On
        gaga4 400B that put 28 such bundles at snode ~400 against deadlines
        1000+ nodes later: 4% of the bytes holding 37% of the in-flight budget
        for the length of the graph, which under one device-weight budget comes
        straight out of residency and back onto the bus.
        """
        live = InflightMap(budget)
        for plan in plans:
            plan.promoted = plan.load in promoted
            if plan.promoted:
                plan.exposed = 0.0
                plan.budget_floor = index_of[plan.load]
                plan.budget_bound = False
            else:
                live.add(self._unhoisted(plan), plan.last_user, plan.nbytes)

        active = sorted((p for p in plans if not p.promoted), key=lambda p: (p.wait_idx, index_of[p.load]))
        latest_finish = self._latest_finish(active, prefix)

        targets: dict = {}
        bus_end = 0.0
        emitted = -1
        for plan in active:
            live.remove(self._unhoisted(plan), plan.last_user, plan.nbytes)
            floor = live.earliest_start(plan.last_user, plan.nbytes)
            plan.budget_floor = floor
            lo = max(plan.lower, floor)

            # As late as the deadlines allow, but never before the bus is free
            # or before the load is legal to issue.
            t_bus = max(prefix[lo], bus_end, latest_finish[plan.load] - plan.need)
            bus_end = t_bus + plan.need
            plan.exposed = max(0.0, bus_end - prefix[plan.wait_idx])
            # The in-flight floor is the limit only when relaxing it would have
            # started the DMA sooner; otherwise the bus was full regardless.
            plan.budget_bound = plan.exposed > 0 and prefix[lo] > prefix[plan.lower] and prefix[lo] > t_bus - plan.need

            at = self._issue_index(prefix, t_bus, lo, plan.wait_idx, emitted)
            targets[plan.load] = at
            emitted = at
            live.add(at, plan.last_user, plan.nbytes)
        return targets

    @staticmethod
    def _latest_finish(active, prefix) -> dict:
        """Per load, the last instant its transfer may end and still keep order.

        Walked back from the last deadline: a transfer may not finish later than
        its own wait, nor later than the start of the next one already pinned to
        its deadline.  Without this a load with slack would be scheduled the
        moment the bus is free, which is early, and then sit in memory until its
        wait -- the bus gains nothing and the in-flight budget pays for it.
        """
        limit = float("inf")
        out: dict = {}
        for plan in reversed(active):
            limit = min(prefix[plan.wait_idx], limit)
            out[plan.load] = limit
            limit -= plan.need
        return out

    @staticmethod
    def _issue_index(prefix, t_bus: float, lo: int, wait_idx: int, emitted: int) -> int:
        """Latest snode whose compute prefix still reaches ``t_bus``.

        Clamped into ``[lo, wait_idx)`` -- a load must follow its producers and
        precede its own wait -- and never before the load placed ahead of it,
        which holds an earlier bus slot, so the stream issues the transfers in
        the order the bus was scheduled to run them.
        """
        hi = max(lo, wait_idx - 1)
        at = bisect_right(prefix, t_bus, lo, max(lo + 1, wait_idx)) - 1
        return min(hi, max(lo, emitted, at))

    def _compute_prefix(self, order) -> list[float]:
        prefix = [0.0] * (len(order) + 1)
        for i, s in enumerate(order):
            prefix[i + 1] = prefix[i] + (self._cost(s) if is_compute(s) else 0.0)
        return prefix

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

    @staticmethod
    def _rebuild(order, targets, index_of, groups) -> list[BaseSchedulerNode]:
        member_target = {m: targets[load] for load, group in groups.items() if load in targets for m in group}

        def _key(s):
            target = member_target.get(s)
            return (target - 0.5, index_of[s]) if target is not None else (index_of[s], 0.0)

        return sorted(order, key=_key)
