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

"""Device memory held by transfer buffers, over the snode index space.

A buffer is live over the closed range ``[start, last_user]``: ``rebuild``
inserts a group *before* the node at its target, so that node already sees the
new buffer while the earlier buffer's ``last_user`` still reads the old one.
Two ranges therefore overlap iff ``earlier.last_user >= later.start``.
"""

from __future__ import annotations

from bisect import insort


class InflightMap:
    """Occupancy of an in-flight byte budget."""

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


def live_starts(tasks, targets, index_of) -> dict:
    """Where each buffer still on the resource comes alive; inactive ones never do."""
    return {t.anchor: targets.get(t.anchor, index_of[t.anchor]) for t in tasks if not t.inactive}


def peak_point(tasks, starts) -> tuple[int, int]:
    events: list[tuple[int, int]] = []
    for t in tasks:
        at = starts.get(t.anchor)
        if at is not None:
            events.append((at, t.nbytes))
            events.append((t.last_user, -t.nbytes))
    events.sort(key=lambda ev: (ev[0], ev[1] < 0))
    peak = live = where = 0
    for idx, delta in events:
        live += delta
        if live > peak:
            peak, where = live, idx
    return peak, where


def inflight_peak(tasks, starts) -> int:
    return peak_point(tasks, starts)[0]


def peak_with(tasks, starts, permanent_anchors) -> int:
    permanent = sum(t.nbytes for t in tasks if t.anchor in permanent_anchors)
    return permanent + inflight_peak(tasks, starts)


def device_peak(tasks, targets, index_of) -> int:
    """Transfer bytes on the device at the worst point in the graph.

    Every buffer is counted where it comes alive, inactive ones included, and an
    inactive task's bytes are charged again for the whole graph: for a weight
    made resident, residency removes the PCIe crossing, not the copy, so
    ``h2d_load`` still allocates an output and its bytes are on the device twice
    while that buffer lives.  The in-flight *budget* deliberately does not
    charge for those (see ``alap_schedule``); this is the memory report, where
    it would be a lie not to.
    """
    starts = {t.anchor: targets.get(t.anchor, index_of[t.anchor]) for t in tasks}
    return peak_with(tasks, starts, {t.anchor for t in tasks if t.inactive})
