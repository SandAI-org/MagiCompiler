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

from __future__ import annotations

"""Second-phase reorder: hoist each weight load off the gather it feeds.

Installed AFTER ``FsdpOverlapReorder`` in
``reorder_for_compute_comm_overlap_passes``; Inductor chains the passes
(``order = p(order)``), so this one sees the all-gather placement the first pass
settled on.

Phase 1 moves ``h2d_load``, its wait and the all-gather as one block, which is
correct but leaves the load fully exposed: the wait sits directly in front of the
gather, so the compute stream stalls for the whole transfer before the gather is
even issued.  This pass pulls the load -- and only the load -- back out of that
block, opening a window of compute between it and its wait::

    ... compute ...  h2d_load  ... compute ...  h2d_wait  all_gather  ...
                     ^ moved here             ^ left where phase 1 put it

The window that hides a load is therefore strictly upstream of the window that
hides its own gather, which is physics: a shard cannot be gathered before it has
arrived.  What is NOT serialized is one weight against another -- the load of
weight i+1 freely claims the same compute that hides the gather of weight i,
because PCIe and NVLink are different hardware.  That is why this is a second
sweep with its own compute pointer rather than more items in the first one.

Hoisting is bounded by one rule: **at most one load is live on the device at a
time** (one ``h2d_load``, which may be a whole FSDP bucket).  Those bytes occupy
memory from where the load runs until the last snode that still reads them --
the gather for a shard, the last matmul for an ungathered or unsharded weight --
not until the wait that only means the copy has landed.  An unconstrained sweep
would stack those intervals and rebuild, on the device, most of the residency
that offloading just paid PCIe to remove.  A load that cannot be placed under
the rule is handed back to the device rather than forced in, which removes its
transfer instead of leaving a stall.  The cost is that some compute runs with
the PCIe lane idle -- accepted deliberately: bounded memory is the point,
overlap is the bonus.

Three consequences worth stating, because they are what make the split cheap:

* No cross-rank negotiation.  This pass moves no collective, so the NCCL launch
  sequence is byte-identical before and after; ranks are free to place their
  loads differently, and a rank on slower PCIe *should*.
* No profiling.  A load's cost is ``bytes / bandwidth`` off one calibration
  probe.  The first pass never benchmarks an ``h2d_load`` either -- a transfer on
  a private stream neither hides an all-gather nor competes with one, so it has
  no business in a compute window.
* Failure is free.  Dropping this pass leaves phase 1's correct-but-slow order.
"""

from collections import defaultdict
from dataclasses import dataclass

from torch._inductor.comms import _is_fake_dep
from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.utils import contains_wait

from magi_compiler.utils import magi_logger


def magi_logger_enabled_for_debug() -> bool:
    """The per-load report builds a string per line; skip it when nobody reads it."""
    import logging

    return logging.getLogger("magi_compiler").isEnabledFor(logging.DEBUG)


from ..snode_utils import earliest_legal_index, is_multi_output, issues_transfer, validate_topological_order
from .ops import H2D_OPS, is_h2d_load, slots_of

_DEFAULT_WINDOW_MARGIN_NS = 5_000.0


def _snode_bytes(snode: BaseSchedulerNode) -> int:
    node = getattr(snode, "node", None)
    try:
        numel = 1
        for d in node.get_size():
            numel *= int(d)
        return numel * node.get_dtype().itemsize
    except Exception:  # noqa: BLE001 - an unsized node simply contributes nothing
        return 0


def _load_bytes(group: list[BaseSchedulerNode]) -> int:
    """Bytes this load pulls across PCIe.

    Measured on the unpacks rather than on the load itself: a coalesced load is a
    multi-output kernel whose own ``get_size`` describes no single tensor, and
    sizing a whole bucket's window from one member would under-count it by the
    bucket factor.
    """
    unpacks = [s for s in group if is_multi_output(s)]
    return sum(_snode_bytes(s) for s in (unpacks or group[:1]))


@dataclass
class _Plan:
    """One load's placement problem: how much transfer to hide, and where it may go."""

    load: BaseSchedulerNode
    group: list  # the load plus the alias snodes that must travel with it
    slots: list[int]  # host-pool slots this load pulls
    wait_idx: int  # earliest wait: the load's hard upper bound
    # Last snode that still reads this load's bytes (gather, matmul, ...).
    # The wait is only a floor: it means the copy has landed, not that the
    # buffer is free.  Sweep, peak and bubbles all use this.
    last_user: int
    need: float  # ns of compute that would fully hide the transfer
    lower: int  # earliest legal index (real-dep floor)
    nbytes: int
    exposed: float  # ns of transfer the sweep could not cover; set by _sweep
    promoted: bool = False  # shards put back on the device; transfer is now D2D
    relieved: bool = False  # re-offloaded into an idle window to meet the cap


class H2dLoadReorder:
    """Callable reorder pass.  Run me after ``FsdpOverlapReorder``."""

    def __init__(
        self,
        bandwidth_bytes_per_ns: float,
        window_margin_ns: float = _DEFAULT_WINDOW_MARGIN_NS,
        window_scale: float = 1.0,
        max_resident_bytes: int = 0,
        cost_fn=None,
    ) -> None:
        self.bandwidth_bytes_per_ns = max(1e-6, bandwidth_bytes_per_ns)
        self.window_margin_ns = window_margin_ns
        self.window_scale = window_scale
        # Ceiling on the weight bytes the sweep may leave resident.  0 = none:
        # take the fastest schedule and keep whatever it wants.  See _relieve.
        self.max_resident_bytes = max_resident_bytes
        if cost_fn is None:
            from torch._inductor.comms import estimate_op_runtime

            cost_fn = estimate_op_runtime
        self._cost_fn = cost_fn
        self._cost_cache: dict[BaseSchedulerNode, float] = {}

    def __deepcopy__(self, memo):
        # Fresh, cache-free instance: Inductor deepcopies passes into the
        # fx-graph cache key, and snode keys hold FakeTensors whose data_ptr
        # access raises.
        new = H2dLoadReorder.__new__(H2dLoadReorder)
        new.bandwidth_bytes_per_ns = self.bandwidth_bytes_per_ns
        new.window_margin_ns = self.window_margin_ns
        new.window_scale = self.window_scale
        new.max_resident_bytes = self.max_resident_bytes
        new._cost_fn = self._cost_fn
        new._cost_cache = {}
        memo[id(self)] = new
        return new

    def _cost(self, snode: BaseSchedulerNode) -> float:
        c = self._cost_cache.get(snode)
        if c is None:
            try:
                c = max(0.0, float(self._cost_fn(snode)))
            except Exception:  # noqa: BLE001
                c = 0.0
            self._cost_cache[snode] = c
        return c

    def _transfer_ns(self, group: list[BaseSchedulerNode]) -> float:
        return _load_bytes(group) / self.bandwidth_bytes_per_ns

    @staticmethod
    def _is_compute(snode: BaseSchedulerNode) -> bool:
        return not issues_transfer(snode) and not is_h2d_load(snode) and not contains_wait(snode)

    def __call__(self, snodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
        self._cost_cache = {}
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

        targets = self._sweep(plans, order, index_of)
        if self.max_resident_bytes > 0:
            self._relieve(plans, targets, index_of)

        new_order = self._rebuild(order, targets, index_of, {p.load: p.group for p in plans})
        if not validate_topological_order(new_order, buf_to_snode):
            magi_logger.warning("h2d load reorder: rebuilt order failed validation; leaving graph unchanged")
            return order

        # Only once the order is committed: promotion moves real bytes, and a
        # rejected order must not leave the pool half-rearranged.
        given_back = self._promote(plans)
        # In-place: the Inductor driver's peak-memory report reads the list it
        # handed us, not the one we return.
        order[:] = new_order
        self._report(plans, targets, index_of, len(loads), given_back, len(order))
        return order

    def _report(self, plans, targets, index_of, n_loads, given_back, n_snodes) -> None:
        moved = sum(1 for load, target in targets.items() if target != index_of[load])
        promoted = [p for p in plans if p.promoted]
        relieved = [p for p in plans if p.relieved]
        exposed = sum(p.exposed for p in plans if not p.promoted)
        magi_logger.info(
            "h2d load reorder: hoisted %d/%d weight load(s) at %.1f GB/s; %d weight(s) kept resident "
            "(%.1f MiB) because nothing could hide them%s; in-flight peak %.1f MiB; %.1fus still exposed",
            moved,
            n_loads,
            self.bandwidth_bytes_per_ns,
            len(promoted),
            given_back / 2**20,
            f", {len(relieved)} re-offloaded into idle windows to meet the " f"{self.max_resident_bytes / 2**20:.0f} MiB cap"
            if relieved
            else "",
            self._inflight_peak(plans, targets) / 2**20,
            exposed / 1e3,
        )
        self._log_placement(plans, targets, index_of, n_snodes)

    @staticmethod
    def _weights_of(plan) -> str:
        """The parameters behind one load, as the placement log wants them."""
        from . import host_pool

        names = [host_pool.name_of(s) for s in plan.slots]
        names = [n for n in names if n] or ["?"]
        return ", ".join(names[:3]) + (f", +{len(names) - 3} more" if len(names) > 3 else "")

    def _log_placement(self, plans, targets, index_of, n_snodes) -> None:
        """One line per load: which weights, how far it moved, and what it bought.

        The interesting question a profile raises is always "why is that load
        there", and the answer needs the weight names next to the indices -- a
        bucket of forty-layer MoE experts and a bucket of attention projections
        look identical as snode ids and behave nothing alike.
        """
        if not magi_logger_enabled_for_debug():
            return
        magi_logger.debug(
            "h2d load placement (%d loads over %d snodes; 'at' is where the load ended up, "
            "'last_user' the last snode that still reads its bytes):",
            len(plans),
            n_snodes,
        )
        for p in sorted(plans, key=lambda p: index_of[p.load]):
            if p.promoted:
                verdict = "RESIDENT: nothing upstream could hide it"
            elif p.relieved:
                verdict = "re-offloaded into an idle window (residency cap)"
            elif p.exposed <= 0:
                verdict = "hidden"
            else:
                verdict = f"EXPOSED {p.exposed / 1e3:.1f}us"
            at = targets.get(p.load, index_of[p.load])
            magi_logger.debug(
                "  %-10s %2d slot(s) %7.1f MiB  at %5d (from %5d, floor %5d)  last_user %5d  " "need %6.1fms  %-45s  %s",
                p.load.get_name(),
                len(p.slots),
                p.nbytes / 2**20,
                at,
                index_of[p.load],
                p.lower,
                p.last_user,
                p.need / 1e6,
                verdict,
                self._weights_of(p),
            )

    # -- planning ---------------------------------------------------------
    @staticmethod
    def _group_and_waits(load, users) -> tuple[list, list]:
        """The snodes that travel with the load, and the waits that guard it.

        A custom op's result reaches its consumers through a ``MultiOutput``
        unpack rather than directly, so a wait is two hops away and the unpacks
        have to move with the load -- they read the load's buffer and nothing
        else.  Searching only the load's direct readers finds the unpacks,
        decides they are not waits, and drops the load from the plan: no error,
        no hoist, no overlap.  ``FsdpOverlapReorder._wait_snodes`` carries the
        same scar.

        A coalesced load has one unpack and one wait per bucket member, so both
        are collected rather than stopping at the first.
        """
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
        """Latest snode that still reads this load's bytes.

        The wait only means the copy has landed.  Direct users of the load /
        unpack / wait buffers are the ones that still hold those bytes -- a
        gather for a shard, a matmul for an ungathered or plain weight.  Group
        members (the load and its unpacks) travel with the load, so they do not
        count: their final position is the placement, not a consumer.
        """
        last = max(index_of[w] for w in waits)
        skip = set(group)
        for src in (*group, *waits):
            for name in src.get_buffer_names():
                for u in users.get(name, ()):
                    if u not in skip:
                        last = max(last, index_of[u])
        return last

    def _plan(self, loads, order, index_of, buf_to_snode, users) -> list[_Plan]:
        """One entry per load, in program order, skipping the ones with no wait.

        A load whose wait this pass cannot find is left alone rather than guessed
        at: moving a transfer away from a synchronization we do not understand is
        how you get a race that only shows up under load.
        """
        plans: list[_Plan] = []
        for load in sorted(loads, key=lambda s: index_of[s]):
            group, waits = self._group_and_waits(load, users)
            if not waits:
                magi_logger.debug("h2d load reorder: %s has no wait; leaving it in place", load.get_name())
                continue
            need = self._transfer_ns(group) * self.window_scale + self.window_margin_ns
            plans.append(
                _Plan(
                    load=load,
                    group=group,
                    slots=slots_of(load),
                    # The earliest wait: the load has to precede every one of them.
                    wait_idx=min(index_of[w] for w in waits),
                    last_user=self._last_user_index(group, waits, users, index_of),
                    need=need,
                    lower=earliest_legal_index(group, index_of, buf_to_snode),
                    nbytes=_load_bytes(group),
                    exposed=need,
                )
            )
        return plans

    # -- placement --------------------------------------------------------
    def _sweep(self, plans, order, index_of) -> dict:
        """Latest-safe-launch, back to front, with one load live at a time.

        The two-pointer part is ``FsdpOverlapReorder``'s, for the same reason:
        one PCIe stream means the loads are serialized against each other, so
        each must claim a run of compute the next one cannot also spend.  The
        pointer and the carry are this pass's own, which is how the PCIe lane
        ends up free to reuse the compute the NVLink lane is already hiding
        behind.

        What is added here is the ``frontier``.  Having placed one load, the next
        one to move is not necessarily its immediate predecessor: if that load's
        last user would still be reading it where the placed one now starts, it
        is left alone and the sweep looks further back for one whose bytes are
        already unused -- which may be several loads back, or none.

        A load left alone is not left exposed: its weight goes back on the device
        and stays there.  Loading a weight that nothing can hide is pure cost
        every single forward, so this is both the faster answer and the simpler
        one, and it is why the sweep needs no separate notion of a buffer count.
        Keeping at most one load in flight falls out of the same rule.

        This produces the fastest schedule.  When the residency it asks for is
        more than the caller can afford, ``_relieve`` puts some of it back --
        that is the only other lever, and it runs after this.
        """
        targets: dict = {}
        compute_idx = len(order)
        carry = 0.0
        carry_idx = len(order)
        frontier = len(order)  # where the next-later load's bytes come alive

        for plan in reversed(plans):
            if plan.last_user > frontier:
                # Something still reads this load where the next one would start.
                plan.promoted = True
                continue

            cur = index_of[plan.load]
            compute_idx = min(compute_idx, cur)
            if cur < carry_idx:
                carry = 0.0
                carry_idx = compute_idx
            acc = carry
            t = compute_idx
            while acc < plan.need and t > plan.lower:
                s = order[t - 1]
                if self._is_compute(s):
                    acc += self._cost(s)
                    carry_idx = t - 1
                t -= 1
            target = max(plan.lower, t)
            targets[plan.load] = target
            plan.exposed = max(0.0, plan.need - acc)
            carry = max(0.0, acc - plan.need)
            compute_idx = target
            frontier = target
        return targets

    @staticmethod
    def _inflight_peak(plans, targets) -> int:
        """Most load bytes alive at once, over a sweep of the placed live ranges.

        One bucket under the sweep's rule alone; more once a residency cap has
        put loads back into idle windows, which is exactly the trade the cap
        makes and the reason it is worth reporting rather than assuming.
        """
        events: list[tuple[int, int]] = []
        for p in plans:
            at = targets.get(p.load)
            if at is not None:
                events.append((at, p.nbytes))
                events.append((p.last_user, -p.nbytes))
        events.sort()
        peak = live = 0
        for _idx, delta in events:
            live += delta
            peak = max(peak, live)
        return peak

    # -- residency relief --------------------------------------------------
    @staticmethod
    def _bubbles(targets, by_load) -> list[tuple[int, int]]:
        """Index ranges where the load stream has nothing to do.

        The sweep leaves these behind on purpose -- it never hoists a load into a
        window it does not need -- so they are the free space: a transfer put
        here competes with no other transfer for the bus.
        """
        placed = sorted((t, by_load[load].last_user) for load, t in targets.items())
        out: list[tuple[int, int]] = []
        prev_end = 0
        for start, end in placed:
            if start > prev_end:
                out.append((prev_end, start))
            prev_end = max(prev_end, end)
        return out

    def _relieve(self, plans, targets, index_of) -> None:
        """Give up speed for residency, one weight at a time, until the cap holds.

        The sweep hands back every weight it cannot schedule for free, which is
        the fastest answer but says nothing about how much device memory that
        costs.  When the bill is too high, weights come back into the offload
        plan -- and the ones to pick are those whose loads fit in a bubble the
        sweep already left idle, because a transfer there is the cheapest one
        available: the bus is free and no other load is waiting on it.

        Largest first, so the cap is met with the fewest weights re-offloaded and
        therefore the fewest new transfers on the critical path.
        """
        by_load = {p.load: p for p in plans}
        resident = sum(p.nbytes for p in plans if p.promoted)
        if resident <= self.max_resident_bytes:
            return

        while resident > self.max_resident_bytes:
            bubbles = self._bubbles(targets, by_load)
            best = None
            for plan in sorted((p for p in plans if p.promoted), key=lambda p: -p.nbytes):
                for lo, hi in bubbles:
                    at = max(plan.lower, lo)
                    if at < min(hi, plan.wait_idx):
                        best = (plan, at)
                        break
                if best is not None:
                    break
            if best is None:
                magi_logger.warning(
                    "h2d load reorder: %.1f MiB of weights stay resident, over the %.1f MiB cap -- "
                    "no idle window is left to schedule another load into",
                    resident / 2**20,
                    self.max_resident_bytes / 2**20,
                )
                return
            plan, at = best
            plan.promoted = False
            plan.relieved = True
            targets[plan.load] = at
            resident -= plan.nbytes

    @staticmethod
    def _promote(plans) -> int:
        """Put the shards of every skipped load back on the device.

        The graph is untouched: the load still runs, it just copies
        device-to-device now.  That is what lets this decision be made during
        scheduling without invalidating the artifact being scheduled.
        """
        from . import host_pool

        slots: list[int] = []
        for p in plans:
            if not p.promoted:
                continue
            if not p.slots:
                # No slot to hand back -- leave the load where it is.  Correct,
                # just a transfer this pass could not improve.
                p.promoted = False
                continue
            slots.extend(p.slots)
        return host_pool.make_resident_many(slots)

    @staticmethod
    def _rebuild(order, targets, index_of, groups) -> list[BaseSchedulerNode]:
        """Apply every move in one stable-sort rebuild: targets live in the
        original index space, so incremental moves would shift them.

        Group members sort to the same key as their load and keep their relative
        order, so a load and its unpack stay adjacent.
        """
        member_target = {m: targets[load] for load, group in groups.items() if load in targets for m in group}

        def _key(s):
            target = member_target.get(s)
            return (target - 0.5, index_of[s]) if target is not None else (index_of[s], 0.0)

        return sorted(order, key=_key)
