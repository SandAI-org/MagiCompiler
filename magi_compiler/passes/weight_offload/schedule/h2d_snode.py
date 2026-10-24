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

"""Read the host slot and transfer size of H2D load scheduler nodes."""

from __future__ import annotations

from torch._inductor.scheduler import BaseSchedulerNode

from ...snode_utils import ir_node_bytes, is_multi_output


def host_slot_of_load(load: BaseSchedulerNode) -> int:
    """The host-pool slot a load pulls.

    Inductor flattens a custom op's non-tensor arguments into ``constant_args``,
    and ``h2d_load(Tensor shard, int slot)`` has exactly one: the slot.
    """
    (slot,) = load.node.constant_args
    return int(slot)


def load_bytes(group: list[BaseSchedulerNode]) -> int:
    """Bytes a load pulls across PCIe.

    Measured on the ``MultiOutput`` unpack rather than on the load itself: a
    custom op lowers to a ``FallbackKernel`` whose own layout describes no
    tensor.
    """
    unpacks = [s for s in group if is_multi_output(s)]
    return sum(ir_node_bytes(s.node) for s in (unpacks or group[:1]))
