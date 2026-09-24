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

"""Recognizing a weight load among scheduler nodes.

Lives on its own because two passes need it and neither owns it: the load
reorder has to find the loads it places, and the all-gather reorder has to know
a load is NOT compute -- counting a PCIe transfer as compute that hides a
collective double-spends the same microseconds on two different transfers.
"""

from __future__ import annotations

from torch._inductor.scheduler import BaseSchedulerNode


def h2d_ops() -> tuple:
    """The host-to-device load ops, imported lazily so this module stays
    importable without a CUDA build."""
    try:
        from ..runtime.h2d_op import H2D_LOAD, H2D_LOAD_COALESCED

        return (H2D_LOAD, H2D_LOAD_COALESCED)
    except Exception:  # noqa: BLE001
        return ()


H2D_OPS = h2d_ops()


def is_h2d_load(snode: BaseSchedulerNode) -> bool:
    """``magi::h2d_load`` lowers to an ordinary FallbackKernel, so nothing in
    Inductor marks it as a transfer.

    It is deliberately NOT part of any collective skeleton -- a load issues no
    NCCL work, and putting it there would make two ranks that merely offload
    different weights look like two structurally different graphs.
    """
    node = getattr(snode, "node", None)
    if getattr(node, "op_overload", None) in H2D_OPS:
        return True
    for child in getattr(snode, "snodes", []) or []:
        if getattr(getattr(child, "node", None), "op_overload", None) in H2D_OPS:
            return True
    return False


def slots_of(snode: BaseSchedulerNode) -> list[int]:
    """The host-pool slots a load node pulls.

    Inductor flattens a custom op's non-tensor arguments into ``constant_args``,
    and both load ops take exactly one such argument -- the slot, or the list of
    them -- so this is the whole of it for the plain and the coalesced form alike.
    """
    node = getattr(snode, "node", None)
    try:
        return [int(a) for a in getattr(node, "constant_args", ())]
    except (TypeError, ValueError):  # not an int arg: not a shape this pass knows
        return []
