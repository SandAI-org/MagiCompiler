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

"""Recognize H2D loads, which provide no compute window for hiding transfers, and read what they move."""

from __future__ import annotations

from torch._inductor import ir
from torch._inductor.scheduler import BaseSchedulerNode

from ...snode_utils import ir_node_bytes, is_multi_output, iter_ir_nodes
from ..runtime.h2d_op import H2D_LOAD


def is_h2d_load(snode: BaseSchedulerNode) -> bool:
    """Recognize the fallback op; rank-private H2D loads stay outside the collective skeleton."""
    return any(isinstance(node, ir.ExternKernel) and node.op_overload is H2D_LOAD for node in iter_ir_nodes(snode))


def slot_of(load: BaseSchedulerNode) -> int:
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
