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

"""Placing transfers on one serial resource so compute hides them.

Every task has a duration (``required_overlap_ns``), a release
(``earliest_launch_index``) and a deadline, one stream runs them back to back,
and a task should start as LATE as that allows so its buffer lives briefly.
``schedule_transfers_alap`` solves that on the time axis (the compute prefix sum), where
a long kernel's leftover time is simply there for the next task.

A task's ``deadline_index`` is the first index its transfer has to be done by,
so ``deadline_index - 1`` is where it sits unhoisted: a load in front of its
wait, a gather at its own position.
"""

from __future__ import annotations

from bisect import bisect_right
from typing import Mapping

from .memory import InflightMemoryBudget
from .task import ScheduleResult, TransferTask


def unhoisted_index(task: TransferTask) -> int:
    """Where a task sits when it is not hoisted at all: against its own deadline."""
    return max(task.earliest_launch_index, task.deadline_index - 1)


def latest_finish_times_ns(active_tasks: list[TransferTask], compute_prefix_ns: list[float]) -> dict:
    """Per task, the last instant its transfer may end and still keep order.

    Walked back from the last deadline: a transfer may not finish later than
    its own deadline, nor later than the start of the next one already pinned to
    its deadline.  Without this a task with slack would be scheduled the moment
    the resource is free, which is early, and then sit in memory until its
    deadline -- the resource gains nothing and the in-flight budget pays for it.
    """
    finish_limit_ns = float("inf")
    finish_ns_by_anchor: dict = {}
    for task in reversed(active_tasks):
        finish_limit_ns = min(compute_prefix_ns[task.deadline_index], finish_limit_ns)
        finish_ns_by_anchor[task.anchor] = finish_limit_ns
        finish_limit_ns -= task.required_overlap_ns
    return finish_ns_by_anchor


def latest_legal_launch_index(
    compute_prefix_ns: list[float],
    transfer_start_ns: float,
    earliest_index: int,
    deadline_index: int,
    previous_launch_index: int,
) -> int:
    """Latest snode whose compute prefix still reaches ``transfer_start_ns``.

    Clamped into ``[earliest_index, deadline_index)`` -- a transfer must follow
    its producers and precede its own deadline -- and never before the task
    placed ahead of it, which holds an earlier slot, so the stream issues the
    transfers in the order the resource was scheduled to run them.
    """
    latest_index = max(earliest_index, deadline_index - 1)
    search_end = max(earliest_index + 1, deadline_index)
    launch_index = bisect_right(compute_prefix_ns, transfer_start_ns, earliest_index, search_end) - 1
    return min(latest_index, max(earliest_index, previous_launch_index, launch_index))


def schedule_transfers_alap(
    tasks: list[TransferTask],
    compute_prefix_ns: list[float],
    original_index_by_node: dict,
    inflight_budget_bytes: float,
    *,
    off_bus_anchors: frozenset = frozenset(),
) -> ScheduleResult:
    """Place transfers in deadline order on one serial resource, each as late as possible.

    Reserve each task's unhoisted live range before moving any task.
    A backward pass computes latest finishes;
    a forward pass places launches within dependency, bus and live-byte limits.
    Tasks whose anchor is in ``off_bus_anchors`` stay put and use no bus time or in-flight budget."""
    tasks_on_bus = [t for t in tasks if t.anchor not in off_bus_anchors]
    inflight_memory = InflightMemoryBudget(inflight_budget_bytes)
    for task in tasks_on_bus:
        inflight_memory.reserve_live_range(unhoisted_index(task), task.last_user_index, task.transfer_bytes)

    tasks_by_deadline = sorted(tasks_on_bus, key=lambda t: (t.deadline_index, original_index_by_node[t.anchor]))
    finish_ns_by_anchor = latest_finish_times_ns(tasks_by_deadline, compute_prefix_ns)

    launch_index_by_anchor: dict = {}
    unhidden_ns_by_anchor: dict = {}
    inflight_budget_limited_anchors: set = set()
    bus_available_ns = 0.0
    previous_launch_index = -1
    for task in tasks_by_deadline:
        inflight_memory.release_live_range(unhoisted_index(task), task.last_user_index, task.transfer_bytes)
        memory_lower_index = inflight_memory.earliest_start_index(task.last_user_index, task.transfer_bytes)
        earliest_launch_index = max(task.earliest_launch_index, memory_lower_index)

        # As late as the deadlines allow, but never before the task is legal to
        # issue; it starts once the resource is free as well.
        release_ns = compute_prefix_ns[earliest_launch_index]
        requested_start_ns = max(release_ns, finish_ns_by_anchor[task.anchor] - task.required_overlap_ns)
        transfer_start_ns = max(requested_start_ns, bus_available_ns)
        bus_available_ns = transfer_start_ns + task.required_overlap_ns
        unhidden_ns = max(0.0, bus_available_ns - compute_prefix_ns[task.deadline_index])
        unhidden_ns_by_anchor[task.anchor] = unhidden_ns
        # The in-flight floor is the limit only when relaxing it would have
        # started the transfer sooner; otherwise the resource was full regardless.
        inflight_floor_delayed_release = release_ns > compute_prefix_ns[task.earliest_launch_index]
        if unhidden_ns > 0 and inflight_floor_delayed_release and release_ns > transfer_start_ns - task.required_overlap_ns:
            inflight_budget_limited_anchors.add(task.anchor)

        launch_index = latest_legal_launch_index(
            compute_prefix_ns, transfer_start_ns, earliest_launch_index, task.deadline_index, previous_launch_index
        )
        launch_index_by_anchor[task.anchor] = launch_index
        previous_launch_index = launch_index
        inflight_memory.reserve_live_range(launch_index, task.last_user_index, task.transfer_bytes)
    return ScheduleResult(
        launch_index_by_anchor=launch_index_by_anchor,
        unhidden_ns_by_anchor=unhidden_ns_by_anchor,
        inflight_budget_limited_anchors=frozenset(inflight_budget_limited_anchors),
    )


def enforce_launch_order(tasks: list[TransferTask], launch_index_by_anchor: Mapping) -> dict:
    """Clamp launch indices non-decreasing in program order (``tasks`` order), so cost
    jitter or same-slot per-rank indices cannot swap two transfers on one stream."""
    ordered_launch_index_by_anchor: dict = {}
    previous_launch_index = -1
    for task in tasks:
        launch_index = launch_index_by_anchor.get(task.anchor)
        if launch_index is None:
            continue
        previous_launch_index = max(launch_index, previous_launch_index)
        ordered_launch_index_by_anchor[task.anchor] = previous_launch_index
    return ordered_launch_index_by_anchor
