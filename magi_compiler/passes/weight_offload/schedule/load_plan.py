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

"""Per-load placement state and the device-memory accounting over it."""

from __future__ import annotations

from bisect import insort
from dataclasses import dataclass

from torch._inductor.scheduler import BaseSchedulerNode

from ...snode_utils import is_multi_output


def snode_bytes(snode: BaseSchedulerNode) -> int:
    node = getattr(snode, "node", None)
    try:
        numel = 1
        for d in node.get_size():
            numel *= int(d)
        return numel * node.get_dtype().itemsize
    except Exception:  # noqa: BLE001 - an unsized node simply contributes nothing
        return 0


def load_bytes(group: list[BaseSchedulerNode]) -> int:
    """Bytes this load pulls across PCIe.

    Measured on the unpacks rather than on the load itself: a coalesced load is a
    multi-output kernel whose own ``get_size`` describes no single tensor, and
    sizing a whole bucket's window from one member would under-count it by the
    bucket factor.
    """
    unpacks = [s for s in group if is_multi_output(s)]
    return sum(snode_bytes(s) for s in (unpacks or group[:1]))


@dataclass
class LoadPlan:
    """One load's placement problem: how much transfer to hide, and where it may go."""

    load: BaseSchedulerNode
    group: list  # the load plus the alias snodes that must travel with it
    slots: list[int]  # host-pool slots this load pulls
    wait_idx: int  # earliest wait: the load's hard upper bound
    # Inclusive right end of the closed live range [target, last_user].
    last_user: int
    need: float  # ns of compute / bus time that would fully hide the transfer
    lower: int  # earliest legal index (real-dep floor)
    nbytes: int
    exposed: float  # ns of transfer the placement could not cover; set by _sweep
    promoted: bool = False  # bought out of the offload plan; transfer is now D2D
    budget_floor: int = 0  # earliest index the in-flight budget left open
    budget_bound: bool = False  # the in-flight budget, not the bus, is what stopped it


class InflightMap:
    """Occupancy of the in-flight byte budget over the snode index space.

    Ranges are closed, ``[start, last_user]``, because ``_rebuild`` inserts a load
    *before* the node at its target: that node already sees the new buffer, while
    the earlier load's ``last_user`` still reads the old one.  Two ranges
    therefore overlap iff ``earlier.last_user >= later.start``.
    """

    def __init__(self, budget: int) -> None:
        self.budget = budget
        self._deltas: list[tuple[int, int]] = []

    def add(self, start: int, last_user: int, nbytes: int) -> None:
        insort(self._deltas, (start, nbytes))
        insort(self._deltas, (last_user + 1, -nbytes))

    def remove(self, start: int, last_user: int, nbytes: int) -> None:
        self._deltas.remove((start, nbytes))
        self._deltas.remove((last_user + 1, -nbytes))

    def earliest_start(self, last_user: int, nbytes: int) -> int:
        """Earliest index where ``[i, last_user]`` still fits ``nbytes`` in the budget."""
        room = self.budget - nbytes
        blocked = -1
        occupied = 0
        prev = 0
        for idx, delta in self._deltas:
            if prev > last_user:
                break
            if occupied > room and prev < idx:
                blocked = max(blocked, min(idx - 1, last_user))
            occupied += delta
            prev = idx
        return blocked + 1


def live_starts(plans, targets, index_of) -> dict:
    """Where each load buffer still on the bus comes alive; resident ones never do."""
    return {p.load: targets.get(p.load, index_of[p.load]) for p in plans if not p.promoted}


def peak_point(plans, starts) -> tuple[int, int]:
    events: list[tuple[int, int]] = []
    for p in plans:
        at = starts.get(p.load)
        if at is not None:
            events.append((at, p.nbytes))
            events.append((p.last_user, -p.nbytes))
    events.sort(key=lambda ev: (ev[0], ev[1] < 0))
    peak = live = where = 0
    for idx, delta in events:
        live += delta
        if live > peak:
            peak, where = live, idx
    return peak, where


def inflight_peak(plans, starts) -> int:
    return peak_point(plans, starts)[0]


def peak_with(plans, starts, promoted) -> int:
    permanent = sum(p.nbytes for p in plans if p.load in promoted)
    return permanent + inflight_peak(plans, starts)


def device_peak(plans, targets, index_of) -> int:
    """Weight bytes on the device at the worst point in the graph.

    Every load is counted where its buffer comes alive, promoted ones
    included: residency removes the PCIe crossing, not the copy, so
    ``h2d_load`` still allocates an output for a resident slot and its bytes
    are on the device twice while that buffer lives.  The in-flight *budget*
    deliberately does not charge for those (see ``H2dLoadReorder._sweep``);
    this is the memory report, where it would be a lie not to.
    """
    starts = {p.load: targets.get(p.load, index_of[p.load]) for p in plans}
    return peak_with(plans, starts, {p.load for p in plans if p.promoted})
