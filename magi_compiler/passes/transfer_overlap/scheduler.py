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

``blocked_windows_ns`` are sorted, disjoint spans of that axis taken by collectives
queued on the same stream; they split it into gaps.  A copy cannot pause, so one
longer than its gap runs into the next collective, which queues behind it: compute
stalls at that collective's wait for the overflow.  Only the gap a transfer starts
in hides it, however many gaps it runs across, so the overflow is its exposure and
the link is free again once that collective is done.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Mapping, Sequence

from .memory import InflightMemoryBudget
from .task import ScheduleResult, TransferTask


def unhoisted_index(task: TransferTask) -> int:
    """Where a task sits when it is not hoisted at all: against its own deadline."""
    return max(task.earliest_launch_index, task.deadline_index - 1)


def _idle_link_cost_per_ns(tasks: Sequence[TransferTask], compute_prefix_ns: Sequence[float]) -> float:
    """Exposure one ns of idle link is expected to cost later: the link's load factor over the axis, at most 1."""
    axis_ns = compute_prefix_ns[-1] if compute_prefix_ns else 0.0
    if axis_ns <= 0:
        return 1.0
    return min(1.0, sum(task.required_overlap_ns for task in tasks) / axis_ns)


@dataclass(frozen=True)
class _GapPlacement:
    """Where one transfer runs inside one gap between blocked windows."""

    start_ns: float
    exposed_ns: float
    link_free_ns: float  # when the next transfer may start: its own end, or once the collective it runs into is done
    runs_into_window: bool


def _gap_bounds_ns(gap_index: int, blocked_windows_ns: Sequence) -> tuple[float, float]:
    """Gap ``gap_index`` runs from the end of window ``gap_index - 1`` to the start of window ``gap_index``."""
    gap_start_ns = blocked_windows_ns[gap_index - 1][1] if gap_index > 0 else 0.0
    gap_end_ns = blocked_windows_ns[gap_index][0] if gap_index < len(blocked_windows_ns) else float("inf")
    return gap_start_ns, gap_end_ns


def _plan_start_between_windows(
    finish_limit_ns: float,
    deadline_ns: float,
    duration_ns: float,
    blocked_windows_ns: Sequence,
    window_starts_ns: Sequence[float],
    idle_link_cost_per_ns: float,
) -> float:
    """Latest planned start, scanning gaps backwards from ``finish_limit_ns``.

    A gap scores the transfer's exposure in it plus ``idle_link_cost_per_ns`` times the
    link it leaves idle up to ``finish_limit_ns``, which earlier transfers cannot use.
    A gap shorter than the transfer counts only when the transfer runs into a
    collective or its own wait there, never into the next transfer's planned start.
    The idle term only grows backwards, so the scan stops once it alone reaches the
    best score; with a gap scored at the limit, that bounds the hoist by
    ``duration / idle_link_cost_per_ns`` of compute.
    """
    best_score_ns = float("inf")
    best_start_ns = finish_limit_ns - duration_ns
    for gap_index in range(bisect_left(window_starts_ns, finish_limit_ns), -1, -1):
        gap_start_ns, gap_end_ns = _gap_bounds_ns(gap_index, blocked_windows_ns)
        occupied_until_ns = min(gap_end_ns, finish_limit_ns)
        idle_score_ns = idle_link_cost_per_ns * (finish_limit_ns - occupied_until_ns)
        if idle_score_ns >= best_score_ns:
            break
        room_ns = occupied_until_ns - gap_start_ns
        if room_ns >= duration_ns:
            score_ns, start_ns = idle_score_ns, occupied_until_ns - duration_ns
        elif room_ns > 0 and min(gap_end_ns, deadline_ns) <= finish_limit_ns:
            score_ns, start_ns = duration_ns - room_ns + idle_score_ns, gap_start_ns
        else:
            continue
        if score_ns < best_score_ns:
            best_score_ns, best_start_ns = score_ns, start_ns
    return best_start_ns


def _place_from(
    earliest_start_ns: float,
    deadline_ns: float,
    duration_ns: float,
    blocked_windows_ns: Sequence,
    window_starts_ns: Sequence[float],
) -> _GapPlacement:
    """The least exposed start at or after ``earliest_start_ns``: there, or at the start of a later gap.

    The transfer's hidden part ends at the first window after its start or at its
    deadline, whichever comes first; the rest of it is exposed.
    """
    window_index = bisect_right(window_starts_ns, earliest_start_ns) - 1
    if window_index >= 0 and earliest_start_ns < blocked_windows_ns[window_index][1]:
        earliest_start_ns = blocked_windows_ns[window_index][1]
    best_placement: _GapPlacement | None = None
    candidate_start_ns = earliest_start_ns
    while True:
        next_window_index = bisect_right(window_starts_ns, candidate_start_ns)
        _, gap_end_ns = _gap_bounds_ns(next_window_index, blocked_windows_ns)
        hidden_until_ns = min(gap_end_ns, deadline_ns)
        exposed_ns = max(0.0, duration_ns - (hidden_until_ns - candidate_start_ns))
        runs_into_window = exposed_ns > 0 and gap_end_ns <= deadline_ns
        if runs_into_window:
            link_free_ns = blocked_windows_ns[next_window_index][1]
        else:
            link_free_ns = candidate_start_ns + duration_ns
        if best_placement is None or exposed_ns < best_placement.exposed_ns:
            best_placement = _GapPlacement(candidate_start_ns, exposed_ns, link_free_ns, runs_into_window)
        if not runs_into_window:
            return best_placement
        candidate_start_ns = blocked_windows_ns[next_window_index][1]


def latest_finish_times_ns(
    active_tasks: list[TransferTask], compute_prefix_ns: list[float], blocked_windows_ns: Sequence = ()
) -> dict:
    """Per task, where its transfer is planned to end, chained backwards from the last deadline.

    Each transfer must be off the link by its deadline and the next transfer's planned
    start.  With blocked windows the start comes from ``_plan_start_between_windows``,
    and a transfer longer than its gap is planned to run past the window ending it.
    """
    window_starts_ns = [window_start_ns for window_start_ns, _ in blocked_windows_ns]
    idle_link_cost_per_ns = _idle_link_cost_per_ns(active_tasks, compute_prefix_ns)
    finish_limit_ns = float("inf")
    finish_ns_by_anchor: dict = {}
    for task in reversed(active_tasks):
        deadline_ns = compute_prefix_ns[task.deadline_index]
        finish_limit_ns = min(deadline_ns, finish_limit_ns)
        if blocked_windows_ns:
            start_ns = _plan_start_between_windows(
                finish_limit_ns,
                deadline_ns,
                task.required_overlap_ns,
                blocked_windows_ns,
                window_starts_ns,
                idle_link_cost_per_ns,
            )
        else:
            start_ns = finish_limit_ns - task.required_overlap_ns
        finish_ns_by_anchor[task.anchor] = start_ns + task.required_overlap_ns
        finish_limit_ns = start_ns
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
    blocked_windows_ns: Sequence = (),
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
    finish_ns_by_anchor = latest_finish_times_ns(tasks_by_deadline, compute_prefix_ns, blocked_windows_ns)
    window_starts_ns = [window_start_ns for window_start_ns, _ in blocked_windows_ns]

    launch_index_by_anchor: dict = {}
    unhidden_ns_by_anchor: dict = {}
    inflight_budget_limited_anchors: set = set()
    window_straddling_anchors: set = set()
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
        placement = _place_from(
            max(requested_start_ns, bus_available_ns),
            compute_prefix_ns[task.deadline_index],
            task.required_overlap_ns,
            blocked_windows_ns,
            window_starts_ns,
        )
        transfer_start_ns = placement.start_ns
        bus_available_ns = placement.link_free_ns
        unhidden_ns = placement.exposed_ns
        unhidden_ns_by_anchor[task.anchor] = unhidden_ns
        if placement.runs_into_window:
            window_straddling_anchors.add(task.anchor)
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
        window_straddling_anchors=frozenset(window_straddling_anchors),
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
