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

"""Keeping a collective-moving placement identical on every rank.

A pass that moves NCCL-issuing snodes must emit the same collective sequence on
every rank, or the ranks deadlock.  Costs are rank-local measurements and
per-rank graphs can differ in compute, so agreement is negotiated, not assumed:

* ``negotiate_mode`` picks how much freedom the ranks can share
  (slot / pinned / abort);
* ``consensus_slot_targets`` puts each transfer in the same skeleton slot
  everywhere (the index inside the slot stays rank-local);
* ``agree`` ANDs a local commit verdict across ranks -- committing on some
  ranks and not on others is itself a divergent sequence.

A pass that moves only rank-private work (a host-to-device load) needs none of
this.
"""

from __future__ import annotations

import bisect

import torch
import torch.distributed as dist
from torch._inductor import ir
from torch._inductor.scheduler import BaseSchedulerNode

from magi_compiler.utils import magi_logger

from ..snode_utils import is_weight_gather, issues_transfer, leaf_collective_node


def _collective_kind_key(snode: BaseSchedulerNode) -> tuple:
    """Coarse, rank-comparable identity of one NCCL-issuing snode."""
    node = leaf_collective_node(snode) or snode.node
    op = (node.op_overload or node.python_kernel_name) if isinstance(node, ir.ExternKernel) else None
    op = op or type(node).__name__
    dims: tuple = ()
    try:
        dims = tuple("?" if getattr(d, "free_symbols", None) else int(d) for d in node.get_size())
    except Exception:  # noqa: BLE001
        pass
    return (is_weight_gather(snode), str(op), dims)


def collective_skeleton(order: list[BaseSchedulerNode]) -> tuple[list[int], list[tuple]]:
    """The graph's collective skeleton: indices (ascending) and rank-comparable
    kinds of every snode that issues a transfer -- functional NCCL collectives,
    custom ops with an internal collective, and copy-engine / symmetric-memory
    gathers.  This sequence is what must stay rank-identical; the compute
    between two consecutive entries is rank-private."""
    from magi_compiler.profiling.runtime_estimator import snode_issues_collective

    idx = [i for i, s in enumerate(order) if snode_issues_collective(s) or issues_transfer(s)]
    return idx, [_collective_kind_key(order[i]) for i in idx]


def negotiate_mode(anchors, skel_kinds, costs_ok: bool = True, who: str = "FSDP overlap") -> tuple[str, object, int]:
    """Return (mode, group, world) with a rank-identical placement policy.

    Matching collective skeletons use consensus slots; differing skeletons pin
    transfers between their existing neighbors. Candidate-count or pricing
    failures abort on every rank, so no peer skips a later collective alone.
    """
    from magi_compiler.profiling.runtime_estimator import _get_cost_sync_group

    group = _get_cost_sync_group()
    world = dist.get_world_size()
    mine = (len(anchors), tuple(skel_kinds), bool(costs_ok))
    peers: list = [None] * world
    dist.all_gather_object(peers, mine, group=group)
    failed = [rank for rank, p in enumerate(peers) if not p[2]]
    if failed:
        magi_logger.warning(
            who + " reorder: snode cost profiling failed on rank(s) %s; leaving the graph "
            "unchanged on every rank (overlap OFF).",
            failed,
        )
        return "abort", group, world

    desc = [(p[0], len(p[1])) for p in peers]
    if any(p[0] != peers[0][0] for p in peers[1:]):
        magi_logger.warning(
            who + " reorder: weight-AG counts diverge ((n_weight_gathers, n_collectives) per rank: %s); "
            "leaving the graph unchanged (overlap OFF).",
            desc,
        )
        return "abort", group, world
    if all(p[1] == peers[0][1] for p in peers[1:]):
        return "slot", group, world
    magi_logger.warning(
        who + " reorder: collective skeletons differ ((n_weight_gathers, n_collectives) per rank: %s). "
        "Continuing in PINNED mode: gathers keep their position relative to every NCCL-issuing snode "
        "(no hop over CP / EP kernels).",
        desc,
    )
    return "pinned", group, world


def agree(ok: bool, sync_group, world: int, who: str = "FSDP overlap") -> bool:
    """Reduce a local yes/no into a rank-identical one (AND over ranks)."""
    if world <= 1:
        return ok

    try:
        t = torch.tensor([1 if ok else 0], dtype=torch.int32)
        dist.all_reduce(t, op=dist.ReduceOp.MIN, group=sync_group)
        return bool(t.item())
    except Exception as exc:  # noqa: BLE001
        magi_logger.warning(who + " reorder: cross-rank agreement failed (%s); leaving graph unchanged", exc)
        return False


def consensus_slot_targets(
    anchors_in_order, task_of, skel_idx, index_of, sync_group, world, who: str = "FSDP overlap"
) -> None:
    """Put each transfer in the same skeleton slot on every rank (max of desired
    slot and dep floor, then non-decreasing).  Index inside the slot stays local.

    ``anchors_in_order`` lists every candidate in program order, planned or not,
    so the per-rank lists stay aligned; ``task_of`` maps the planned ones.
    """

    def slot_of(idx: int) -> int:
        return bisect.bisect_left(skel_idx, idx)

    mine = []
    for anchor in anchors_in_order:
        own = slot_of(index_of[anchor])
        task = task_of.get(anchor)
        mine.append((slot_of(task.target), slot_of(task.lower)) if task is not None else (own, own))
    peers: list = [None] * world
    dist.all_gather_object(peers, mine, group=sync_group)

    running = 0
    for j, anchor in enumerate(anchors_in_order):
        q = max(max(p[j][0] for p in peers), max(p[j][1] for p in peers))
        q = running = max(q, running)
        task = task_of.get(anchor)
        if task is None:
            continue
        target = task.target
        slot_lo = max(task.lower, skel_idx[q - 1] + 1 if q > 0 else 0)
        slot_hi = skel_idx[q] if q < len(skel_idx) else index_of[anchor]
        task.target = min(max(target, slot_lo), max(slot_hi, slot_lo))
        magi_logger.debug(
            who + " slot consensus: launch=%d slot=%d mine=%s target=%d->%d bounds=[%d,%d]",
            index_of[anchor],
            q,
            mine[j],
            target,
            task.target,
            slot_lo,
            slot_hi,
        )


def raise_lower_for_nccl_barriers(lower: int, anchor_idx: int, order: list) -> int:
    """Raise ``lower`` so a transfer cannot hop any NCCL-issuing snode that
    originally precedes it."""
    from magi_compiler.profiling.runtime_estimator import snode_issues_collective

    barrier = lower
    for i in range(lower, anchor_idx):
        if snode_issues_collective(order[i]):
            barrier = i + 1
    return barrier
