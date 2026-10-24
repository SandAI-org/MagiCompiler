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

A buffer is live over the closed range ``[start_index, last_user_index]``:
``rebuild_node_order`` inserts a task's moving nodes *before* the node at its
launch index, so that node already sees the new buffer while the earlier
buffer's last user still reads the old one.  Two ranges therefore overlap iff
``earlier.last_user_index >= later.start_index``.
"""

from __future__ import annotations

from bisect import insort


class InflightMemoryBudget:
    """Occupancy of an in-flight byte budget by closed live ranges."""

    def __init__(self, budget_bytes: int) -> None:
        self.budget_bytes = budget_bytes
        # (index, +bytes) where a range starts, (last_user_index + 1, -bytes) where it ends; kept sorted.
        self._byte_delta_events: list[tuple[int, int]] = []

    def reserve_live_range(self, start_index: int, last_user_index: int, transfer_bytes: int) -> None:
        insort(self._byte_delta_events, (start_index, transfer_bytes))
        insort(self._byte_delta_events, (last_user_index + 1, -transfer_bytes))

    def release_live_range(self, start_index: int, last_user_index: int, transfer_bytes: int) -> None:
        self._byte_delta_events.remove((start_index, transfer_bytes))
        self._byte_delta_events.remove((last_user_index + 1, -transfer_bytes))

    def earliest_start_index(self, last_user_index: int, transfer_bytes: int) -> int:
        """Earliest index ``i`` where ``[i, last_user_index]`` still fits ``transfer_bytes`` in the budget."""
        remaining_bytes = self.budget_bytes - transfer_bytes
        last_blocked_index = -1
        occupied_bytes = 0
        previous_index = 0
        for event_index, delta_bytes in self._byte_delta_events:
            if previous_index > last_user_index:
                break
            if occupied_bytes > remaining_bytes and previous_index < event_index:
                last_blocked_index = max(last_blocked_index, min(event_index - 1, last_user_index))
            occupied_bytes += delta_bytes
            previous_index = event_index
        return last_blocked_index + 1


def inflight_peak_bytes_and_index(tasks, start_index_by_anchor) -> tuple[int, int]:
    """Peak live bytes and the index where it is reached.

    ``start_index_by_anchor`` maps an anchor to where its buffer comes alive; a
    task without an entry is not counted.
    """
    byte_delta_events: list[tuple[int, int]] = []
    for task in tasks:
        start_index = start_index_by_anchor.get(task.anchor)
        if start_index is not None:
            byte_delta_events.append((start_index, task.transfer_bytes))
            byte_delta_events.append((task.last_user_index, -task.transfer_bytes))
    # At one index, starts count before ends: the ranges are closed.
    byte_delta_events.sort(key=lambda event: (event[0], event[1] < 0))
    peak_bytes = live_bytes = peak_index = 0
    for event_index, delta_bytes in byte_delta_events:
        live_bytes += delta_bytes
        if live_bytes > peak_bytes:
            peak_bytes, peak_index = live_bytes, event_index
    return peak_bytes, peak_index


def inflight_peak_bytes(tasks, start_index_by_anchor) -> int:
    peak_bytes, _ = inflight_peak_bytes_and_index(tasks, start_index_by_anchor)
    return peak_bytes


def peak_bytes_with_resident_weights(tasks, start_index_by_anchor, resident_anchors) -> int:
    """In-flight peak plus the resident weights, which hold their bytes for the whole graph."""
    resident_bytes = sum(t.transfer_bytes for t in tasks if t.anchor in resident_anchors)
    return resident_bytes + inflight_peak_bytes(tasks, start_index_by_anchor)


def weight_storage_peak_bytes(tasks, launch_index_by_anchor, original_index_by_node, resident_anchors) -> int:
    """Peak transfer storage, including resident weights and their separate D2D outputs.

    Unlike the H2D in-flight budget, this counts resident output lifetimes too;
    a task without a launch index runs at its original position.  Only weight
    storage, not the device's total memory.
    """
    start_index_by_anchor = {t.anchor: launch_index_by_anchor.get(t.anchor, original_index_by_node[t.anchor]) for t in tasks}
    return peak_bytes_with_resident_weights(tasks, start_index_by_anchor, resident_anchors)
