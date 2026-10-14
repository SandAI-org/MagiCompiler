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

Every task has a duration (``need``), a release (``lower``) and a deadline, one
stream runs them back to back, and a task should start as LATE as that allows so
its buffer lives briefly.  ``alap_schedule`` solves that on the time axis (the
compute prefix sum), where a long kernel's leftover time is simply there for
the next task.

A task's ``deadline`` is the first index its transfer has to be done by, so
``deadline - 1`` is where it sits unhoisted: a load in front of its wait, a
gather at its own position.
"""

from __future__ import annotations

from bisect import bisect_right

from .memory import InflightMap
from .task import TransferTask


def unhoisted_index(task: TransferTask) -> int:
    """Where a task sits when it is not hoisted at all: against its own deadline."""
    return max(task.lower, task.deadline - 1)


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
        limit = min(prefix[task.deadline], limit)
        out[task.anchor] = limit
        limit -= task.need
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
    """Schedule the resource by deadline, then issue each task just in time.

    Every active task is booked at its unhoisted position in the in-flight map
    before anything moves, and released only when placed -- so the budget is a
    promise.  Inactive tasks keep their position, take no resource slot and no
    in-flight room: whatever they cost is charged elsewhere, and charging them
    twice would let that squeeze the transfers still on the resource.

    The active tasks are then list-scheduled in DEADLINE order, in two passes.
    A backward pass (``latest_finish``) gives each the latest instant it may
    finish without pushing the ones after it past their own deadlines; a
    forward pass then runs the resource, giving each the latest slot that
    respects both that limit and the resource still being busy.  Where the
    resource is saturated the slots butt together and it never idles; where
    there is slack a task simply starts late, which costs nothing and keeps its
    buffer's live range short.  The snode a task is emitted at is the latest
    one whose compute prefix still reaches its slot -- just in time, because
    the transfer begins at that instant whatever index we choose and anything
    earlier only holds memory for longer.

    Fills ``target``, ``exposed`` and ``budget_bound`` on every task and
    returns ``{anchor: target}`` for the active ones.
    """
    live = InflightMap(budget)
    for task in tasks:
        if task.inactive:
            task.exposed = 0.0
            task.target = None
            task.budget_bound = False
        else:
            live.add(unhoisted_index(task), task.last_user, task.nbytes)

    active = sorted((t for t in tasks if not t.inactive), key=lambda t: (t.deadline, index_of[t.anchor]))
    finish = latest_finish(active, prefix)

    targets: dict = {}
    bus_end = 0.0
    emitted = -1
    for task in active:
        live.remove(unhoisted_index(task), task.last_user, task.nbytes)
        floor = live.earliest_start(task.last_user, task.nbytes)
        lo = max(task.lower, floor)

        # As late as the deadlines allow, but never before the task is legal to
        # issue; it starts once the resource is free as well.
        t_issue = max(prefix[lo], finish[task.anchor] - task.need)
        t_start = max(t_issue, bus_end)
        bus_end = t_start + task.need
        task.exposed = max(0.0, bus_end - prefix[task.deadline])
        # The in-flight floor is the limit only when relaxing it would have
        # started the transfer sooner; otherwise the resource was full regardless.
        task.budget_bound = task.exposed > 0 and prefix[lo] > prefix[task.lower] and prefix[lo] > t_start - task.need

        at = issue_index(prefix, t_start, lo, task.deadline, emitted)
        task.target = at
        targets[task.anchor] = at
        emitted = at
        live.add(at, task.last_user, task.nbytes)
    return targets


def enforce_monotone(tasks: list[TransferTask]) -> None:
    """Clamp targets non-decreasing in program order, so cost jitter or
    same-slot per-rank indices cannot swap two transfers on one stream."""
    running = -1
    for task in tasks:
        if task.target is None:
            continue
        if task.target < running:
            task.target = running
        running = task.target
