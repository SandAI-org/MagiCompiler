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

"""Bind parked weights, then insert their graph-side loads.

``bind_host_slots_to_graph`` intersects candidates within each shard group;
it tags the graph without moving bytes. Unsharded parameters need no cross-rank vote.
``insert_h2d_loads`` inserts one load/wait pair per tagged holder before all readers.
The load preserves the stored dtype; existing casts and padding run on the device.
Different holders of a shared weight still need their own loads.
Weight discovery is delegated to ``WeightSource`` for plain and FSDP models.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
import torch.fx as fx

from magi_compiler.utils import magi_logger

from ..node_meta import host_slot, mark_host_offloaded, mark_host_slot
from .weight_source import OffloadCandidate, WeightSource, parameter_mesh_group

_WAIT_TENSOR_OP = torch.ops._c10d_functional.wait_tensor.default


def _candidate_key(candidate: OffloadCandidate) -> tuple:
    """Identity a peer rank can match without sharing tensor objects.

    Name + holder + layout, not the per-process slot integer: ranks vote on
    identity, and each process's pool minted its own slots.
    """
    return (
        candidate.parameter_name,
        candidate.weight_node.name,
        tuple(candidate.device_standin.shape),
        str(candidate.device_standin.dtype),
    )


def _collect_candidate_sync_groups(
    candidates: list[OffloadCandidate], placeholder_examples: Mapping[str, Any]
) -> dict[int, Any]:
    """Process groups this rank must enter, even if collect found nothing there.

    Scanning the graph inputs -- not just the plan -- is what keeps a rank
    that skipped every weight on a mesh from walking past the collective its
    peers are waiting on.
    """
    process_groups_by_id: dict[int, Any] = {}
    for obj in placeholder_examples.values():
        process_group = parameter_mesh_group(obj)
        if process_group is not None:
            process_groups_by_id[id(process_group)] = process_group
    for candidate in candidates:
        if candidate.process_group is not None:
            process_groups_by_id[id(candidate.process_group)] = candidate.process_group
    return process_groups_by_id


def _intersect_rank_candidates(
    candidates: list[OffloadCandidate], placeholder_examples: Mapping[str, Any], skip_counts: Counter
) -> list[OffloadCandidate]:
    """Keep the candidates every rank in each shard group also planned.

    Per group, not WORLD: expert-parallel ranks only vote with the mesh that
    holds those experts.  Per candidate, not all-or-nothing: a weight only
    some ranks want is dropped, the rest stay.  A weight with no mesh
    (unsharded ``nn.Parameter``) does not vote.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return candidates

    process_groups_by_id = _collect_candidate_sync_groups(candidates=candidates, placeholder_examples=placeholder_examples)
    if not process_groups_by_id:
        return candidates

    candidates_by_group: dict[int, list[OffloadCandidate]] = {group_id: [] for group_id in process_groups_by_id}
    for candidate in candidates:
        if candidate.process_group is not None and id(candidate.process_group) in candidates_by_group:
            candidates_by_group[id(candidate.process_group)].append(candidate)

    kept_candidate_ids: set[int] = set()
    for group_id, process_group in process_groups_by_id.items():
        group_candidates = candidates_by_group[group_id]
        candidate_keys = [_candidate_key(candidate) for candidate in group_candidates]
        world_size = dist.get_world_size(process_group)
        keys_by_rank: list[Any] = [None] * world_size
        dist.all_gather_object(keys_by_rank, candidate_keys, group=process_group)
        shared_key_counts = Counter(candidate_keys)
        for peer_candidate_keys in keys_by_rank:
            shared_key_counts &= Counter(peer_candidate_keys or [])
        remaining_key_counts = shared_key_counts
        dropped_count = 0
        for candidate, candidate_key in zip(group_candidates, candidate_keys):
            if remaining_key_counts[candidate_key] > 0:
                remaining_key_counts[candidate_key] -= 1
                kept_candidate_ids.add(id(candidate))
            else:
                dropped_count += 1
        if dropped_count:
            skip_counts["not planned by every rank in the shard group"] += dropped_count
            magi_logger.warning(
                "host offload: dropping %d weight(s) not planned by every rank in a %d-rank group; keeping %d",
                dropped_count,
                world_size,
                len(group_candidates) - dropped_count,
            )

    return [candidate for candidate in candidates if candidate.process_group is None or id(candidate) in kept_candidate_ids]


def _format_skip_reasons(skip_counts: Counter) -> str:
    if not skip_counts:
        return "no candidate was skipped"
    return "skipped: " + ", ".join(f"{skip_count}x {skip_reason}" for skip_reason, skip_count in skip_counts.most_common())


def bind_host_slots_to_graph(graph: fx.GraphModule, example_inputs: Sequence[Any] | None, source: WeightSource) -> int:
    """Tag already-parked weights on the graph so a later splice can load them.

    A weight reaches the pool only through host-first materialization.  This
    step does not copy or free anything. Candidate disagreement drops the
    affected candidates; mesh group lookup and collective errors propagate.

    Returns how many weights are now served from host memory.
    """
    placeholders = graph.graph.find_nodes(op="placeholder")
    placeholder_examples: Mapping[str, Any] = dict(zip((n.name for n in placeholders), example_inputs or ()))

    candidates, skip_counts = source.collect_candidates(graph=graph, placeholder_examples=placeholder_examples)

    # Before the empty check, so every rank still enters each of its groups.
    agreed_candidates = _intersect_rank_candidates(
        candidates=candidates, placeholder_examples=placeholder_examples, skip_counts=skip_counts
    )
    if not agreed_candidates:
        magi_logger.info("host offload: nothing to offload (%s)", _format_skip_reasons(skip_counts))
        return 0

    for candidate in agreed_candidates:
        mark_host_slot(candidate.weight_node, candidate.host_slot)
        if candidate.offload_tag_node is not None:
            mark_host_offloaded(candidate.offload_tag_node)

    weight_sizes_bytes = sorted(candidate.weight_bytes for candidate in agreed_candidates)
    magi_logger.info(
        "host offload: %d weight(s) bound to host slots, %.1f MiB; " "sizes %.1f / %.1f / %.1f MiB (min/median/max); %s",
        len(agreed_candidates),
        sum(candidate.weight_bytes for candidate in agreed_candidates) / 2**20,
        weight_sizes_bytes[0] / 2**20,
        weight_sizes_bytes[len(weight_sizes_bytes) // 2] / 2**20,
        weight_sizes_bytes[-1] / 2**20,
        _format_skip_reasons(skip_counts),
    )
    return len(agreed_candidates)


def insert_h2d_loads(graph: fx.GraphModule) -> int:
    """Splice a load in for every tagged weight.

    Returns how many load nodes were inserted.
    """
    node_indices = {n: i for i, n in enumerate(graph.graph.nodes)}
    weight_nodes = [n for n in graph.graph.nodes if host_slot(n) is not None]
    inserted_load_count = sum(
        _insert_weight_load(graph=graph, weight_node=h, host_slot=host_slot(h), node_indices=node_indices)
        for h in weight_nodes
    )

    if inserted_load_count:
        graph.graph.lint()
        graph.recompile()
    magi_logger.info("host offload: inserted %d h2d_load node(s) into the graph", inserted_load_count)
    return inserted_load_count


def _insert_weight_load(graph: fx.GraphModule, weight_node: fx.Node, host_slot: int, node_indices) -> int:
    """Insert a load for ``weight_node`` in front of its first reader and re-point its readers at it.

    The load has to sit above every reader, so whatever the graph does to the
    weight still runs on the device; the reorder pass is what places it properly.
    """
    from ..runtime.h2d_op import H2D_LOAD
    from ..runtime.host_weights import get_global_host_weight_store

    readers = list(weight_node.users)
    if not readers:
        return 0
    first_reader = min(readers, key=lambda n: node_indices.get(n, len(node_indices)))
    example_value = weight_node.meta.get("example_value")
    with graph.graph.inserting_before(first_reader):
        load = graph.graph.call_function(H2D_LOAD, (weight_node, host_slot))
        load.meta["example_value"] = example_value
        wait = graph.graph.call_function(_WAIT_TENSOR_OP, (load,))
        wait.meta["example_value"] = example_value
    # Everything that read the (now storage-free) weight reads the loaded copy
    # instead -- except the load itself, which still needs it.
    weight_node.replace_all_uses_with(wait, delete_user_cb=lambda user: user is not load)
    get_global_host_weight_store().mark_claimed(host_slot)
    return 1
