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

"""A weight load as an overlap task, and the bytes it moves."""

from __future__ import annotations

from dataclasses import dataclass, field

from torch._inductor.scheduler import BaseSchedulerNode

from ...overlap import TransferTask
from ...snode_utils import is_multi_output


def snode_bytes(snode: BaseSchedulerNode) -> int:
    node = getattr(snode, "node", None)
    try:
        numel = 1
        for d in node.get_size():
            numel *= int(d)
        return numel * node.get_dtype().itemsize
    except Exception:  # noqa: BLE001 - an unsized node simply contributes nothing
        return 0


def load_bytes(group: list[BaseSchedulerNode]) -> int:
    """Bytes this load pulls across PCIe.

    Measured on the ``MultiOutput`` unpack rather than on the load itself: a
    custom op lowers to a ``FallbackKernel`` whose own ``get_size`` describes no
    tensor, so the load alone would size as zero.
    """
    unpacks = [s for s in group if is_multi_output(s)]
    return sum(snode_bytes(s) for s in (unpacks or group[:1]))


@dataclass(eq=False)
class LoadPlan(TransferTask):
    """One load's placement problem: how much transfer to hide, and where it may go.

    ``anchor`` is the load, ``deadline`` its earliest wait (the hard upper
    bound), and ``inactive`` means bought out of the offload plan: resident, so
    the transfer is now device-to-device and off the bus.
    """

    slots: list[int] = field(default_factory=list)  # host-pool slots this load pulls

    @property
    def load(self) -> BaseSchedulerNode:
        return self.anchor

    @property
    def wait_idx(self) -> int:
        return self.deadline

    @property
    def promoted(self) -> bool:
        return self.inactive

    @promoted.setter
    def promoted(self, value: bool) -> None:
        self.inactive = value
