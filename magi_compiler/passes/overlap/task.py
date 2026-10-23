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

"""One transfer's placement problem, and the placement the scheduler returns for a set of them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from torch._inductor.scheduler import BaseSchedulerNode


@dataclass(frozen=True, eq=False)
class TransferTask:
    """A transfer to hide behind compute, and where it may go.

    Indices are positions in the order the pass received.  A pure input: the
    scheduler returns its placement in a ``ScheduleResult`` instead of writing
    it back here.  Identity-hashed, so a task can key a dict.
    """

    anchor: BaseSchedulerNode  # the snode that issues the transfer
    moving_nodes: tuple  # snodes that move with it, anchor included
    earliest_launch_index: int  # 1 + the last real producer
    deadline_index: int  # first index the transfer has to be done by; unhoisted it sits at deadline_index - 1
    required_overlap_ns: float  # compute (or bus) time that fully hides it
    transfer_bytes: int = 0
    last_user_index: int = -1  # inclusive end of the buffer's live range


@dataclass(frozen=True)
class ScheduleResult:
    """Where the scheduler placed the tasks on the bus.

    Tasks kept off the bus have no entry in either map: they stay at their
    original index and hide nothing.
    """

    launch_index_by_anchor: Mapping[BaseSchedulerNode, int]
    unhidden_ns_by_anchor: Mapping[BaseSchedulerNode, float]
    # The in-flight budget, not the bus, is what left these transfers partly unhidden.
    inflight_budget_limited_anchors: frozenset

    @property
    def total_unhidden_ns(self) -> float:
        return sum(self.unhidden_ns_by_anchor.values())

    @property
    def inflight_budget_limited_unhidden_ns(self) -> float:
        return sum(self.unhidden_ns_by_anchor[anchor] for anchor in self.inflight_budget_limited_anchors)
