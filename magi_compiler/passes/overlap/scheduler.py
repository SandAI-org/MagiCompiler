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
``alap_schedule`` solves that on the time axis (the compute prefix sum), where
a long kernel's leftover time is simply there for the next task.

A task's ``deadline_index`` is the first index its transfer has to be done by,
so ``deadline_index - 1`` is where it sits unhoisted: a load in front of its
wait, a gather at its own position.
"""

from __future__ import annotations

from bisect import bisect_right

from .memory import InflightMap
from .task import TransferTask


def unhoisted_index(task: TransferTask) -> int:
    """Where a task sits when it is not hoisted at all: against its own deadline."""
    return max(task.earliest_launch_index, task.deadline_index - 1)


def latest_finish(active: list[TransferTask], prefix: list[float]) -> dict:
    """Per task, the last instant its transfer may end and still keep order.

    Walked back from the last deadline: a transfer may not finish later than
    its own deadline, nor later than the start of the next one already pinned to
    its deadline.  Without this a task with slack would be scheduled the moment
    the resource is free, which is early, and then sit in memory until its
    deadline -- the resource gains nothing and the in-flight budget pays for it.
    """
    limit = float("inf")
    out: dict = {}
    for task in reversed(active):
        limit = min(prefix[task.deadline_index], limit)
        out[task.anchor] = limit
        limit -= task.required_overlap_ns
    return out


def issue_index(prefix: list[float], t_start: float, lo: int, deadline: int, emitted: int) -> int:
    """Latest snode whose compute prefix still reaches ``t_start``.

    Clamped into ``[lo, deadline)`` -- a transfer must follow its producers and
    precede its own deadline -- and never before the task placed ahead of it,
    which holds an earlier slot, so the stream issues the transfers in the order
    the resource was scheduled to run them.
    """
    hi = max(lo, deadline - 1)
    at = bisect_right(prefix, t_start, lo, max(lo + 1, deadline)) - 1
    return min(hi, max(lo, emitted, at))


def alap_schedule(tasks: list[TransferTask], prefix: list[float], index_of: dict, budget: float) -> dict:
    """Place transfers that use the bus in deadline order on one serial resource.

    Reserve each such task's unhoisted live range before moving any task.
    A backward pass computes latest finishes;
    a forward pass places launches within dependency, bus and live-byte limits.
    Tasks off the bus stay put and use no bus time or in-flight budget.
    Fill launch_index, unhidden_transfer_ns and budget_bound;
    return anchor-to-launch-index for the tasks on the bus."""
    live = InflightMap(budget)
    for task in tasks:
        if not task.uses_transfer_bus:
            task.unhidden_transfer_ns = 0.0
            task.launch_index = None
            task.budget_bound = False
        else:
            live.add(unhoisted_index(task), task.last_user_index, task.nbytes)

    active = sorted((t for t in tasks if t.uses_transfer_bus), key=lambda t: (t.deadline_index, index_of[t.anchor]))
    finish = latest_finish(active, prefix)

    targets: dict = {}
    bus_end = 0.0
    emitted = -1
    for task in active:
        live.remove(unhoisted_index(task), task.last_user_index, task.nbytes)
        floor = live.earliest_start(task.last_user_index, task.nbytes)
        lo = max(task.earliest_launch_index, floor)

        # As late as the deadlines allow, but never before the task is legal to
        # issue; it starts once the resource is free as well.
        t_issue = max(prefix[lo], finish[task.anchor] - task.required_overlap_ns)
        t_start = max(t_issue, bus_end)
        bus_end = t_start + task.required_overlap_ns
        task.unhidden_transfer_ns = max(0.0, bus_end - prefix[task.deadline_index])
        # The in-flight floor is the limit only when relaxing it would have
        # started the transfer sooner; otherwise the resource was full regardless.
        task.budget_bound = (
            task.unhidden_transfer_ns > 0
            and prefix[lo] > prefix[task.earliest_launch_index]
            and prefix[lo] > t_start - task.required_overlap_ns
        )

        at = issue_index(prefix, t_start, lo, task.deadline_index, emitted)
        task.launch_index = at
        targets[task.anchor] = at
        emitted = at
        live.add(at, task.last_user_index, task.nbytes)
    return targets


def enforce_monotone(tasks: list[TransferTask]) -> None:
    """Clamp launch indices non-decreasing in program order, so cost jitter or
    same-slot per-rank indices cannot swap two transfers on one stream."""
    running = -1
    for task in tasks:
        if task.launch_index is None:
            continue
        if task.launch_index < running:
            task.launch_index = running
        running = task.launch_index
