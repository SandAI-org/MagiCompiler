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

"""How an overlap pass reads snode runtimes.

``cost_fn`` is normally the ``SnodeCostTable`` that ``SnodeCostProfile`` filled
at the head of the reorder chain, and Inductor's ``estimate_op_runtime`` when a
pass runs on its own.  Either may raise or return garbage for an odd node; a
placement must never crash the compile over one, so every read is clamped here.
"""

from __future__ import annotations

from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.utils import contains_wait

from ..snode_utils import is_compute, is_multi_output


def default_cost_fn():
    from torch._inductor.comms import estimate_op_runtime

    return estimate_op_runtime


class CostView:
    """ns per snode, never negative, never raising."""

    def __init__(self, cost_fn) -> None:
        self.cost_fn = cost_fn

    @property
    def ok(self) -> bool:
        """False when the cost table says pricing failed on this rank."""
        return bool(getattr(self.cost_fn, "ok", True))

    def __call__(self, snode: BaseSchedulerNode) -> float:
        try:
            return max(0.0, float(self.cost_fn(snode)))
        except Exception:  # noqa: BLE001
            return 0.0

    def prefix(self, order: list[BaseSchedulerNode]) -> list[float]:
        """``prefix[i]``: ns of compute that runs before the snode at index ``i``.

        The time axis every transfer is placed on.  Only compute counts: a
        transfer or a wait on it hides nothing.
        """
        prefix = [0.0] * (len(order) + 1)
        for i, s in enumerate(order):
            prefix[i + 1] = prefix[i] + (self(s) if is_compute(s) else 0.0)
        return prefix

    def is_transparent(self, snode: BaseSchedulerNode) -> bool:
        """A forwarder that is not a real use of a transfer's result: waits,
        MultiOutput unpacks, and ~zero-cost view/reshape/getitem kernels."""
        if contains_wait(snode) or is_multi_output(snode):
            return True
        return self(snode) <= 1.0
