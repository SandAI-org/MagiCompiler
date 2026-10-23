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

"""Shared scheduler-node classification, collective discovery and IR size helpers."""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

import sympy
import torch
from torch._inductor import ir
from torch._inductor.scheduler import BaseSchedulerNode, FusedSchedulerNode, GroupedSchedulerNode
from torch._inductor.utils import contains_collective, contains_wait, is_collective
from torch._inductor.virtualized import V


@lru_cache(maxsize=1)
def ce_ag_ops() -> tuple:
    """``(single, coalesced)`` copy-engine gather ops, or ``(None, None)`` without
    symmetric memory.  Imported lazily so this stays importable without a CUDA build."""
    try:
        from magi_compiler.symm_mem.all_gather import CE_ALL_GATHER, CE_ALL_GATHER_COALESCED

        return CE_ALL_GATHER, CE_ALL_GATHER_COALESCED
    except Exception:  # noqa: BLE001
        return None, None


def iter_ir_nodes(snode: BaseSchedulerNode) -> Iterator[ir.Operation]:
    # Keep one-level traversal even for ForeachKernelSchedulerNode, whose get_nodes() flattens further.
    children = snode.snodes if isinstance(snode, (GroupedSchedulerNode, FusedSchedulerNode)) else (snode,)
    for child in children:
        if child.node is not None:
            yield child.node


def is_ce_ag_ir(node: ir.Operation | None) -> bool:
    """``magi::ce_all_gather`` lowers to an ordinary FallbackKernel, so Inductor's
    ``is_collective`` does not recognize it."""
    return isinstance(node, ir.ExternKernel) and node.op_overload is not None and node.op_overload in ce_ag_ops()


def is_ce_ag_coalesced_ir(node: ir.Operation | None) -> bool:
    return isinstance(node, ir.ExternKernel) and node.op_overload is not None and node.op_overload is ce_ag_ops()[1]


def is_comm_ir(node: ir.Operation | None) -> bool:
    """A c10d collective or a copy-engine gather: an IR node that talks to peer GPUs."""
    return node is not None and (is_collective(node) or is_ce_ag_ir(node))


def leaf_comm_node(snode: BaseSchedulerNode) -> ir.ExternKernel | None:
    """The underlying comm IR node for a (possibly grouped) snode, or None."""
    return next((node for node in iter_ir_nodes(snode) if is_comm_ir(node)), None)


def issues_comm(snode: BaseSchedulerNode) -> bool:
    """True if this snode exchanges bytes with peer GPUs; a host-to-device load is not comm."""
    return contains_collective(snode) or leaf_comm_node(snode) is not None


ALL_GATHER_OP = torch.ops._c10d_functional.all_gather_into_tensor.default
COALESCED_ALL_GATHER_OP = torch.ops._c10d_functional.all_gather_into_tensor_coalesced.default


def is_all_gather_launch(snode: BaseSchedulerNode) -> bool:
    """An all-gather launch of the shape FSDP gathers weights with -- NCCL or copy
    engine, single or coalesced -- whether or not lowering tagged it as a weight."""
    node = leaf_comm_node(snode)
    op = node.op_overload if node is not None else None
    return op is not None and (op in (ALL_GATHER_OP, COALESCED_ALL_GATHER_OP) or op in ce_ag_ops())


def _tagged_as_weight(snode: BaseSchedulerNode) -> bool:
    # Deferred: fsdp_overlap imports this module.
    from .fsdp_overlap.node_meta import is_weight_ag

    node = leaf_comm_node(snode)
    return any(is_weight_ag(o) for o in node.origins or ())


def weight_gathers(order: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    """The FSDP weight-gather launches, in program order.

    The ones lowering tagged as weight gathers, so activation gathers in the same
    graph are left alone; every all-gather launch when nothing carries the tag.
    """
    gathers = [s for s in order if is_all_gather_launch(s)]
    return [s for s in gathers if _tagged_as_weight(s)] or gathers


def ir_node_bytes(node: ir.IRNode) -> int:
    """Bytes of a single-tensor IR node; a symbolic dimension counts at its size hint."""
    numel = 1
    for d in node.get_size():
        numel *= int(d) if isinstance(d, (int, sympy.Integer)) else int(V.graph.sizevars.size_hint(d, fallback=0))
    return numel * node.get_dtype().itemsize


def is_compute(snode: BaseSchedulerNode) -> bool:
    """True if this snode's runtime is compute a transfer can hide behind.

    A weight load is excluded along with comm: counting a PCIe
    transfer as compute would spend the same microseconds hiding two transfers.
    """
    # Imported per call: weight_offload's package init imports this module.
    from .weight_offload.schedule.h2d_snode import is_h2d_load

    return not issues_comm(snode) and not is_h2d_load(snode) and not contains_wait(snode)


def is_multi_output(snode: BaseSchedulerNode) -> bool:
    """A custom op's result reaches its readers through one of these, not directly."""
    return isinstance(snode.node, ir.MultiOutput)
