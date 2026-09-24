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

"""Scheduler-node helpers shared by every reorder pass.

Kept apart from the passes themselves because two of them -- the FSDP all-gather
reorder and the weight-load reorder -- have to agree exactly on what a real
dependency is, and a second copy of that judgement is a miscompile waiting to
happen.
"""

from __future__ import annotations

import torch
from torch._inductor.comms import _is_fake_dep
from torch._inductor.ir import MultiOutput
from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.utils import contains_collective, contains_wait, is_collective

from magi_compiler.utils import magi_logger


def ce_ag_ops() -> tuple:
    """Copy-engine gather ops, imported lazily so this stays importable without a
    CUDA build."""
    try:
        from magi_compiler.symm_mem.all_gather import CE_ALL_GATHER, CE_ALL_GATHER_COALESCED

        return tuple(op for op in (CE_ALL_GATHER, CE_ALL_GATHER_COALESCED) if op is not None)
    except Exception:  # noqa: BLE001
        return ()


def is_ce_ag_ir(node) -> bool:
    """``magi::ce_all_gather`` lowers to an ordinary FallbackKernel, so Inductor's
    ``is_collective`` does not recognize it."""
    return getattr(node, "op_overload", None) in ce_ag_ops()


def is_gather_ir(node) -> bool:
    return node is not None and (is_collective(node) or is_ce_ag_ir(node))


def leaf_collective_node(snode: BaseSchedulerNode):
    """The underlying collective IR node for a (possibly grouped) snode, or None."""
    node = getattr(snode, "node", None)
    if is_gather_ir(node):
        return node
    # GroupedSchedulerNode: find the collective child.
    for child in getattr(snode, "snodes", []) or []:
        cn = getattr(child, "node", None)
        if is_gather_ir(cn):
            return cn
    return None


def issues_transfer(snode: BaseSchedulerNode) -> bool:
    """True if this snode moves bytes between devices rather than computing."""
    return contains_collective(snode) or leaf_collective_node(snode) is not None


_NCCL_WEIGHT_AG_OPS = (
    torch.ops._c10d_functional.all_gather_into_tensor.default,
    torch.ops._c10d_functional.all_gather_into_tensor_coalesced.default,
)


def is_weight_gather(snode: BaseSchedulerNode) -> bool:
    """An FSDP weight all-gather launch, over NCCL or the copy engine."""
    node = leaf_collective_node(snode)
    op = getattr(node, "op_overload", None) if node is not None else None
    return op is not None and (op in _NCCL_WEIGHT_AG_OPS or op in ce_ag_ops())


def is_compute(snode: BaseSchedulerNode) -> bool:
    """True if this snode's runtime is compute a transfer can hide behind.

    A weight load is excluded along with the collectives: counting a PCIe
    transfer as compute would spend the same microseconds hiding two transfers.
    """
    # Imported per call: weight_offload's package init imports this module.
    from .weight_offload.schedule.h2d_snode import is_h2d_load

    return not issues_transfer(snode) and not is_h2d_load(snode) and not contains_wait(snode)


def is_multi_output(snode: BaseSchedulerNode) -> bool:
    """A custom op's result reaches its readers through one of these, not directly.

    Matched by class name rather than ``isinstance``: Inductor has exactly one
    class by this name, and the name lets a scheduler-level test stand one in
    without building a real IR node.
    """
    return type(getattr(snode, "node", None)).__name__ == MultiOutput.__name__


def earliest_legal_index(group, index_of, buf_to_snode) -> int:
    """1 + max index of any REAL (non-fake buffer) producer the group needs.

    Deliberately NOT ``snode.ancestors``: that set is polluted by the fake
    ``WeakDep`` edges Inductor inserts between collectives for comm-stream
    serialization.  Weight gathers read independent param shards -- there is no
    real gather->gather dependency -- so counting the WeakDep would pin the
    launch right after the previous collective and forbid the very hoist the
    reorder passes exist for.  A gather's only real producer is its weight-shard
    placeholder (+ to_local/pad/cast chain), so real ``lower`` is ~0."""
    group_set = set(group)
    lo = 0
    for s in group:
        for d in s.unmet_dependencies:  # buffer names
            if _is_fake_dep(d):  # WeakDep / StarDep -- ordering hint, not data
                continue
            prod = buf_to_snode.get(d.name)
            if prod is None or prod in group_set:
                continue
            lo = max(lo, index_of.get(prod, 0) + 1)
    return lo


def validate_topological_order(new_order, buf_to_snode) -> bool:
    """Valid topological order w.r.t. REAL data deps: every node's non-fake
    buffer producers precede it (the driver does not repair the order, so a
    violation would silently miscompile).  Checking direct producers per node
    is a complete validation of the real-dep DAG.  ``snode.ancestors`` is NOT
    used -- it includes the fake WeakDep edges the passes intentionally cross
    (see ``earliest_legal_index``); an ancestors check would false-reject
    every legal hoist.  WeakDep is advisory, not a correctness constraint."""
    pos = {s: i for i, s in enumerate(new_order)}
    for s in new_order:
        sp = pos[s]
        for d in s.unmet_dependencies:  # buffer names
            if _is_fake_dep(d):  # WeakDep / StarDep -- advisory ordering, not data
                continue
            prod = buf_to_snode.get(d.name)
            if prod is s:  # fused snode may name its own internal buffers
                continue
            if prod is not None and pos.get(prod, -1) >= sp:
                magi_logger.debug(
                    "validate fail: %s@%d needs buffer-dep %s@%d (buf %s)",
                    s.get_name(),
                    sp,
                    prod.get_name(),
                    pos.get(prod, -1),
                    d.name,
                )
                return False
    return True
