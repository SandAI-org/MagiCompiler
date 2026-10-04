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

"""Exposure and peak memory of one bucket plan, with the placement worked out exactly.

For a fixed partition of a run of weight gathers into buckets, on one serial
collective stream and a fixed compute order, placement has a best answer:

* forward, every bucket as early as it may start (no earlier than its shard
  is ready, nor before the previous bucket finishes): each finish is the
  earliest possible, so each bucket's lateness past its first reader is the
  least any placement can get;
* backward, every bucket as late as that lateness, its own deadline and the
  next bucket's start allow.

The result exposes no more than any other placement and, among those, launches
every bucket as late as possible -- which, since a gathered weight is allocated
at its launch and freed at a last read no placement moves, is also the least
memory.  ``FsdpOverlapReorder`` places by the same deadline chain, so this is
what the plan will get.

Memory is priced against ``base``, Inductor's per-step estimate of the order
the plan is made on, where each gather still sits at its own position: a member
merged into a bucket launched at ``L`` is allocated that much earlier, adding
its gathered bytes over ``[L, own position)``.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

from .auto_bucket import Member, PlanParams


@dataclass
class Placement:
    """Where one run's buckets launch, and how late each one finishes."""

    launches: list[int]  # launch index per bucket
    late_ns: list[float]  # per bucket: finish past its first reader

    @property
    def exposed_ns(self) -> float:
        return sum(self.late_ns)


def _need(params: PlanParams, out_bytes: int) -> float:
    return (params.alpha_ns + params.beta_ns_per_byte * out_bytes) * params.scale + params.margin_ns


def place(members: list[Member], segs: list[tuple[int, int]], prefix: list[float], params: PlanParams) -> Placement:
    """Place ``segs`` (contiguous ``[start, end)`` over ``members``) by the deadline chain above."""
    n = len(segs)
    need = [_need(params, sum(m.out_bytes for m in members[s:e])) for s, e in segs]
    release = [prefix[max(m.lo for m in members[s:e])] for s, e in segs]
    deadline = [prefix[min(m.deadline for m in members[s:e])] for s, e in segs]

    finish, late, t = [0.0] * n, [0.0] * n, 0.0
    for k in range(n):
        t = max(release[k], t) + need[k]
        finish[k] = t
        late[k] = max(0.0, t - deadline[k])

    starts, nxt = [0.0] * n, float("inf")
    for k in reversed(range(n)):
        starts[k] = min(max(deadline[k], finish[k]), nxt) - need[k]
        nxt = starts[k]

    launches, prev = [], 0
    for k, (s, e) in enumerate(segs):
        lo = max(m.lo for m in members[s:e])
        cap = min(m.wait for m in members[s:e])
        idx = bisect.bisect_right(prefix, starts[k] + 1e-6) - 1
        idx = min(max(idx, lo, prev), cap)
        launches.append(idx)
        prev = idx
    return Placement(launches=launches, late_ns=late)


def peak_with(base: list[int], runs: list[tuple[list[Member], list[tuple[int, int]], Placement]]) -> tuple[int, int, int]:
    """(peak bytes, its step, what the plans' earlier allocations add there) over ``base``."""
    delta = [0] * (len(base) + 1)
    for members, segs, placed in runs:
        for (s, e), launch in zip(segs, placed.launches):
            for m in members[s:e]:
                if launch < m.idx:
                    delta[launch] += m.out_bytes
                    delta[m.idx] -= m.out_bytes
    peak = peak_step = extra_at_peak = extra = 0
    for step, b in enumerate(base):
        extra += delta[step]
        if b + extra > peak:
            peak, peak_step, extra_at_peak = b + extra, step, extra
    return peak, peak_step, extra_at_peak
