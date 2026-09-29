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

"""One transfer's placement problem, in the terms every overlap pass shares."""

from __future__ import annotations

from dataclasses import dataclass, field

from torch._inductor.scheduler import BaseSchedulerNode


@dataclass(eq=False)
class TransferTask:
    """A transfer to hide behind compute, and where it may go.

    Indices are positions in the order the pass received.  A pass declares the
    problem (``anchor`` .. ``nbytes``); the scheduler fills in the answer
    (``target`` onward).  Identity-hashed, so a task can key a dict.
    """

    anchor: BaseSchedulerNode  # the snode that issues the transfer
    group: list  # snodes that move with it, anchor included
    lower: int  # earliest legal index: 1 + the last real producer
    deadline: int  # first index the transfer has to be done by; unhoisted it sits at deadline - 1
    need: float  # ns of compute (or bus) time that fully hides it
    nbytes: int = 0
    last_user: int = -1  # inclusive end of the buffer's live range
    target: int | None = None  # where the scheduler placed the anchor
    exposed: float = 0.0  # ns the placement could not hide
    inactive: bool = False  # left in place and off the resource
    budget_floor: int = 0  # earliest index the in-flight budget left open
    budget_bound: bool = False  # the in-flight budget, not the resource, is what stopped it
    meta: dict = field(default_factory=dict)  # pass-specific diagnostics
