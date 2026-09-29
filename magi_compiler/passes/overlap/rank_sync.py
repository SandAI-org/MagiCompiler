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
  (identical / slot / pinned / abort);
* ``consensus_slot_targets`` puts each transfer in the same skeleton slot
  everywhere (the index inside the slot stays rank-local);
* ``agree`` ANDs a local commit verdict across ranks -- committing on some
  ranks and not on others is itself a divergent sequence.

A pass that moves only rank-private work (a host-to-device load) needs none of
this.
"""

from __future__ import annotations

import bisect
import hashlib

import torch
import torch.distributed as dist
from torch._inductor.scheduler import BaseSchedulerNode

from magi_compiler.utils import magi_logger

from ..snode_utils import is_weight_gather, issues_transfer, leaf_collective_node


def _size_hint_of(sym) -> int:
    """Rank-identical size hint for a sympy symbol (0 if unavailable, e.g. in
    unit tests without a live Inductor graph)."""
    try:
        from torch._inductor.virtualized import V

        return int(V.graph.sizevars.size_hint(sym, fallback=0))
    except Exception:  # noqa: BLE001
        return 0


def _collective_kind_key(snode: BaseSchedulerNode) -> tuple:
    """Coarse, rank-comparable identity of one NCCL-issuing snode."""
    node = leaf_collective_node(snode) or getattr(snode, "node", None)
    op = getattr(node, "op_overload", None) or getattr(node, "python_kernel_name", None) or type(node).__name__
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


def graph_fingerprint(order: list[BaseSchedulerNode]) -> str:
    """Rank-comparable digest of the snode sequence: type + op identity + output
    sizes + sorted origin fx TARGETS.  Origins are required -- a fused pointwise
    kernel is one ComputedBuffer whose class/size hide its contents (relu vs
    relu+sin look identical without them).  Targets only, not node names: names
    carry per-rank numbering noise.

    """
    import sympy

    h = hashlib.sha256()
    sym_canon: dict = {}  # sympy.Symbol -> canonical sympy.Symbol

    def _canon_size(size) -> str:
        dims = []
        for d in size:
            free = getattr(d, "free_symbols", None)
            if not free:
                dims.append(repr(d))
                continue
            fresh = [sym for sym in free if sym not in sym_canon]
            # Name-free assignment order; symbol name only as the last-resort
            # tie-break (see docstring: that case fails safe).
            fresh.sort(key=lambda sym: (_size_hint_of(sym), d.count(sym), sym.name))
            for sym in fresh:
                sym_canon[sym] = sympy.Symbol(f"c{len(sym_canon):04d}")
            dims.append(repr(d.xreplace(sym_canon)))
        return "[" + ", ".join(dims) + "]"

    for s in order:
        h.update(type(s).__name__.encode())
        for sub in getattr(s, "snodes", None) or (s,):
            n = getattr(sub, "node", None)
            if n is None:
                continue
            op = getattr(n, "op_overload", None) or getattr(n, "python_kernel_name", None) or type(n).__name__
            h.update(str(op).encode())
            try:
                h.update(_canon_size(n.get_size()).encode())
            except Exception:  # noqa: BLE001
                pass
            origins = getattr(n, "origins", None)
            if origins:
                h.update("|".join(sorted(str(getattr(o, "target", o)) for o in origins)).encode())
    return h.hexdigest()


def negotiate_mode(order, anchors, skel_kinds, costs_ok: bool = True, who: str = "FSDP overlap") -> tuple[str, object, int]:
    """Rank-identical placement mode: (mode, group, world).

    ``identical`` / ``slot``: skeletons match → consensus slots (in-slot index is per-rank).
    ``pinned``: skeletons differ → keep each transfer between its neighboring NCCL snodes.
    ``abort``: transfer counts differ, or some rank failed to price its graph
    → leave the graph unchanged.  The pricing verdict travels in the same
    exchange: a rank that bailed on its own would skip this collective and
    leave its peers blocked in it.
    """
    from magi_compiler.profiling.runtime_estimator import _get_cost_sync_group

    group = _get_cost_sync_group()
    world = dist.get_world_size()
    mine = ((graph_fingerprint(order), len(order), len(anchors)), tuple(skel_kinds), bool(costs_ok))
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
    if all(p == peers[0] for p in peers[1:]):
        return "identical", group, world

    desc = [(p[0][0][:12], p[0][1], p[0][2], len(p[1])) for p in peers]
    n_ag = [p[0][2] for p in peers]
    if any(g != n_ag[0] for g in n_ag[1:]):
        magi_logger.warning(
            who + " reorder: per-rank graphs differ AND weight-AG counts diverge "
            "((digest, n_snodes, n_weight_gathers, n_collectives) per rank: %s). No rank "
            "correspondence to reconcile; leaving the graph unchanged (overlap OFF).",
            desc,
        )
        return "abort", group, world
    if all(p[1] == peers[0][1] for p in peers[1:]):
        magi_logger.warning(
            who + " reorder: per-rank graphs are NOT structurally identical "
            "((digest, n_snodes, n_weight_gathers, n_collectives) per rank: %s), but the "
            "collective skeleton matches. Continuing in SLOT-consensus mode: gathers are "
            "placed in a rank-negotiated skeleton slot (they MAY hop CP / EP kernels, as "
            "long as every rank hops the same one).",
            desc,
        )
        return "slot", group, world
    magi_logger.warning(
        who + " reorder: per-rank graphs are NOT structurally identical AND their "
        "collective skeletons differ ((digest, n_snodes, n_weight_gathers, n_collectives) "
        "per rank: %s). Continuing in PINNED mode: gathers keep their position relative to "
        "every NCCL-issuing snode (no hop over CP / EP kernels).",
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
            who + " slot consensus: launch cur=%d slot=%d/%d (mine=%s) target %d -> %d [%d, %d]",
            index_of[anchor],
            q,
            slot_of(index_of[anchor]),
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
