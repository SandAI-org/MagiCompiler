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

* ``any_rank_has_flag`` ORs a local "anything to place" check, so a rank with
  nothing of its own still joins the collectives its peers enter;
* ``negotiate_placement_mode`` picks how much freedom the ranks can share
  (slot / pinned / abort);
* ``align_collective_slots`` puts each transfer in the same collective slot
  everywhere (the index inside the slot stays rank-local);
* ``all_ranks_agree`` ANDs a local commit verdict across ranks -- committing on
  some ranks and not on others is itself a divergent sequence.

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
from magi_compiler.utils.dist_utils import get_cpu_gloo_group

from ..snode_utils import issues_comm, leaf_comm_node, weight_gathers


def _collective_signature(snode: BaseSchedulerNode, is_weight: bool) -> tuple:
    """Coarse, rank-comparable identity of one NCCL-issuing snode: weight tag, op, shape, dtype."""
    node = leaf_comm_node(snode) or snode.node
    op = (node.op_overload or node.python_kernel_name) if isinstance(node, ir.ExternKernel) else None
    op = op or type(node).__name__
    dims: tuple = ()
    dtype = None
    try:
        dims = tuple("?" if getattr(d, "free_symbols", None) else int(d) for d in node.get_size())
        dtype = str(node.get_dtype())
    except (AttributeError, NotImplementedError):
        # A NoneLayout has a size but no dtype, a MultiOutputLayout neither.
        pass
    return (is_weight, str(op), dims, dtype)


def collective_skeleton(order: list[BaseSchedulerNode]) -> tuple[list[int], list[tuple]]:
    """The graph's collective skeleton: indices (ascending) and rank-comparable
    signatures of every snode that issues comm -- functional NCCL collectives,
    custom ops with an internal collective, and copy-engine / symmetric-memory
    gathers.  This sequence is what must stay rank-identical; the compute
    between two consecutive entries is rank-private."""
    from magi_compiler.profiling.runtime_estimator import snode_issues_collective

    collective_indices = [i for i, s in enumerate(order) if snode_issues_collective(s) or issues_comm(s)]
    weight_gather_nodes = set(weight_gathers(order))
    collective_signatures = [_collective_signature(order[i], order[i] in weight_gather_nodes) for i in collective_indices]
    return collective_indices, collective_signatures


def gather_rank_objects(local_object, sync_group, world_size: int) -> list:
    """Every rank's ``local_object``, in rank order: ``all_gather_object`` over ``sync_group``."""
    objects_by_rank: list = [None] * world_size
    dist.all_gather_object(objects_by_rank, local_object, group=sync_group)
    return objects_by_rank


def negotiate_placement_mode(
    candidate_count: int, collective_signatures, is_profiling_valid: bool = True, pass_name: str = "FSDP overlap"
) -> tuple[str, object, int]:
    """Return (placement_mode, sync_group, world_size) with a rank-identical placement policy.

    Matching collective skeletons use consensus slots; differing skeletons pin
    transfers between their existing neighbors. Candidate-count or pricing
    failures abort on every rank, so no peer skips a later collective alone.
    """
    sync_group = get_cpu_gloo_group()
    world_size = dist.get_world_size()
    rank_metadata = gather_rank_objects(
        (candidate_count, tuple(collective_signatures), bool(is_profiling_valid)), sync_group, world_size
    )
    candidate_counts = [count for count, _, _ in rank_metadata]
    signatures_by_rank = [signatures for _, signatures, _ in rank_metadata]
    failed_ranks = [rank for rank, (_, _, is_valid) in enumerate(rank_metadata) if not is_valid]
    if failed_ranks:
        magi_logger.warning(
            pass_name + " reorder: snode cost profiling failed on rank(s) %s; leaving the graph "
            "unchanged on every rank (overlap OFF).",
            failed_ranks,
        )
        return "abort", sync_group, world_size

    counts_by_rank = [(count, len(signatures)) for count, signatures in zip(candidate_counts, signatures_by_rank)]
    if any(count != candidate_counts[0] for count in candidate_counts[1:]):
        magi_logger.warning(
            pass_name + " reorder: weight-AG counts diverge ((n_weight_gathers, n_collectives) per rank: %s); "
            "leaving the graph unchanged (overlap OFF).",
            counts_by_rank,
        )
        return "abort", sync_group, world_size
    if all(signatures == signatures_by_rank[0] for signatures in signatures_by_rank[1:]):
        return "slot", sync_group, world_size
    magi_logger.warning(
        pass_name + " reorder: collective skeletons differ ((n_weight_gathers, n_collectives) per rank: %s). "
        "Continuing in PINNED mode: gathers keep their position relative to every NCCL-issuing snode "
        "(no hop over CP / EP kernels).",
        counts_by_rank,
    )
    return "pinned", sync_group, world_size


def all_ranks_agree(local_vote: bool, sync_group, world_size: int) -> bool:
    """Reduce a local yes/no into a rank-identical one (AND over ranks)."""
    if world_size <= 1:
        return local_vote
    vote_tensor = torch.tensor([1 if local_vote else 0], dtype=torch.int32)
    dist.all_reduce(vote_tensor, op=dist.ReduceOp.MIN, group=sync_group)
    return bool(vote_tensor.item())


def any_rank_has_flag(flag: bool) -> bool:
    """OR a local flag over the placement sync group, so every rank takes the same branch into a collective."""
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() <= 1:
        return flag
    vote_tensor = torch.tensor([int(flag)], dtype=torch.int32)
    dist.all_reduce(vote_tensor, op=dist.ReduceOp.MAX, group=get_cpu_gloo_group())
    return bool(vote_tensor.item())


def align_collective_slots(
    anchors_in_order, task_of, launch_index_by_anchor, collective_indices, original_index_by_node, sync_group, world_size
) -> dict:
    """Put each transfer in the same collective slot on every rank (max of desired
    slot and dep floor, then non-decreasing).  Index inside the slot stays local.

    A collective slot is a gap between two consecutive collectives of the
    skeleton, not a host-pool slot.  ``anchors_in_order`` lists every candidate
    in program order, planned or not, so the per-rank lists stay aligned;
    ``task_of`` maps the planned ones, and ``launch_index_by_anchor`` their local
    placement.  Returns the aligned placement.
    """

    def collective_slot_of_index(index: int) -> int:
        """Which gap between consecutive collectives in the skeleton ``index`` falls in."""
        return bisect.bisect_left(collective_indices, index)

    # Per candidate: (the slot this rank wants, the earliest slot its dependencies allow).
    local_slot_bounds = []
    for anchor in anchors_in_order:
        task = task_of.get(anchor)
        if task is None:
            # Unplanned: it stays where it is, so it wants its own slot and no earlier one.
            original_slot = collective_slot_of_index(original_index_by_node[anchor])
            local_slot_bounds.append((original_slot, original_slot))
        else:
            desired_slot = collective_slot_of_index(launch_index_by_anchor[anchor])
            dependency_floor_slot = collective_slot_of_index(task.earliest_launch_index)
            local_slot_bounds.append((desired_slot, dependency_floor_slot))
    slot_bounds_by_rank = gather_rank_objects(local_slot_bounds, sync_group, world_size)

    aligned_launch_index_by_anchor: dict = {}
    previous_slot = 0
    for candidate_index, anchor in enumerate(anchors_in_order):
        latest_desired_slot = max(bounds[candidate_index][0] for bounds in slot_bounds_by_rank)
        latest_floor_slot = max(bounds[candidate_index][1] for bounds in slot_bounds_by_rank)
        agreed_slot = max(latest_desired_slot, latest_floor_slot, previous_slot)
        previous_slot = agreed_slot
        task = task_of.get(anchor)
        if task is None:
            continue
        local_launch_index = launch_index_by_anchor[anchor]
        previous_collective_end = collective_indices[agreed_slot - 1] + 1 if agreed_slot > 0 else 0
        slot_start_index = max(task.earliest_launch_index, previous_collective_end)
        if agreed_slot < len(collective_indices):
            slot_end_index = collective_indices[agreed_slot]
        else:
            slot_end_index = original_index_by_node[anchor]
        aligned_launch_index = min(max(local_launch_index, slot_start_index), max(slot_end_index, slot_start_index))
        aligned_launch_index_by_anchor[anchor] = aligned_launch_index
        magi_logger.debug(
            "FSDP overlap slot consensus: launch=%d slot=%d mine=%s launch_index=%d->%d bounds=[%d,%d]",
            original_index_by_node[anchor],
            agreed_slot,
            local_slot_bounds[candidate_index],
            local_launch_index,
            aligned_launch_index,
            slot_start_index,
            slot_end_index,
        )
    return aligned_launch_index_by_anchor


def launch_floor_after_collectives(earliest_launch_index: int, anchor_index: int, order: list) -> int:
    """Raise ``earliest_launch_index`` so a transfer cannot hop any NCCL-issuing
    snode that originally precedes it."""
    from magi_compiler.profiling.runtime_estimator import snode_issues_collective

    floor_index = earliest_launch_index
    for index in range(earliest_launch_index, anchor_index):
        if snode_issues_collective(order[index]):
            floor_index = index + 1
    return floor_index
