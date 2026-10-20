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

``bind_weights_to_host`` intersects candidates within each shard group;
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
from .weight_source import OffloadCandidate, WeightSource, mesh_group

_WAIT = torch.ops._c10d_functional.wait_tensor.default


def _candidate_key(c: OffloadCandidate) -> tuple:
    """Identity a peer rank can match without sharing tensor objects.

    Name + holder + layout, not the per-process slot integer: ranks vote on
    identity, and each process's pool minted its own slots.
    """
    return (c.name, c.holder.name, tuple(c.local.shape), str(c.local.dtype))


def _groups_to_sync(plan: list[OffloadCandidate], placeholder_examples: Mapping[str, Any]) -> dict[int, Any]:
    """Process groups this rank must enter, even if collect found nothing there.

    Scanning the graph inputs -- not just the plan -- is what keeps a rank
    that skipped every weight on a mesh from walking past the collective its
    peers are waiting on.
    """
    groups: dict[int, Any] = {}
    for obj in placeholder_examples.values():
        pg = mesh_group(obj)
        if pg is not None:
            groups[id(pg)] = pg
    for c in plan:
        if c.group is not None:
            groups[id(c.group)] = c.group
    return groups


def _align_across_ranks(
    plan: list[OffloadCandidate], placeholder_examples: Mapping[str, Any], skipped: Counter
) -> list[OffloadCandidate]:
    """Keep the candidates every rank in each shard group also planned.

    Per group, not WORLD: expert-parallel ranks only vote with the mesh that
    holds those experts.  Per candidate, not all-or-nothing: a weight only
    some ranks want is dropped, the rest stay.  A weight with no mesh
    (unsharded ``nn.Parameter``) does not vote.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return plan

    groups = _groups_to_sync(plan, placeholder_examples)
    if not groups:
        return plan

    by_group: dict[int, list[OffloadCandidate]] = {gid: [] for gid in groups}
    for c in plan:
        if c.group is not None and id(c.group) in by_group:
            by_group[id(c.group)].append(c)

    keep: set[int] = set()
    for gid, pg in groups.items():
        cands = by_group[gid]
        keys = [_candidate_key(c) for c in cands]
        world = dist.get_world_size(pg)
        gathered: list[Any] = [None] * world
        dist.all_gather_object(gathered, keys, group=pg)
        common = Counter(keys)
        for peer in gathered:
            common &= Counter(peer or [])
        remaining = common
        dropped = 0
        for c, key in zip(cands, keys):
            if remaining[key] > 0:
                remaining[key] -= 1
                keep.add(id(c))
            else:
                dropped += 1
        if dropped:
            skipped["not planned by every rank in the shard group"] += dropped
            magi_logger.warning(
                "host offload: dropping %d weight(s) not planned by every rank in a %d-rank group; keeping %d",
                dropped,
                world,
                len(cands) - dropped,
            )

    return [c for c in plan if c.group is None or id(c) in keep]


def _describe(skipped: Counter) -> str:
    if not skipped:
        return "no candidate was skipped"
    return "skipped: " + ", ".join(f"{n}x {why}" for why, n in skipped.most_common())


def bind_weights_to_host(graph: fx.GraphModule, example_inputs: Sequence[Any] | None, source: WeightSource) -> int:
    """Tag already-parked weights on the graph so a later splice can load them.

    A weight reaches the pool only through host-first materialization.  This
    step does not copy or free anything. Candidate disagreement drops the
    affected candidates; mesh group lookup and collective errors propagate.

    Returns how many weights are now served from host memory.
    """
    placeholders = graph.graph.find_nodes(op="placeholder")
    placeholder_examples: Mapping[str, Any] = dict(zip((n.name for n in placeholders), example_inputs or ()))

    plan, skipped = source.collect(graph, placeholder_examples)

    # Before the empty check, so every rank still enters each of its groups.
    plan = _align_across_ranks(plan, placeholder_examples, skipped)
    if not plan:
        magi_logger.info("host offload: nothing to offload (%s)", _describe(skipped))
        return 0

    for c in plan:
        mark_host_slot(c.holder, c.slot)
        if c.tag is not None:
            mark_host_offloaded(c.tag)

    sizes = sorted(c.nbytes for c in plan)
    magi_logger.info(
        "host offload: %d weight(s) bound to host slots, %.1f MiB; " "sizes %.1f / %.1f / %.1f MiB (min/median/max); %s",
        len(plan),
        sum(c.nbytes for c in plan) / 2**20,
        sizes[0] / 2**20,
        sizes[len(sizes) // 2] / 2**20,
        sizes[-1] / 2**20,
        _describe(skipped),
    )
    return len(plan)


def insert_h2d_loads(graph: fx.GraphModule) -> int:
    """Splice a load in for every tagged weight.

    Returns how many load nodes were inserted.
    """
    order = {n: i for i, n in enumerate(graph.graph.nodes)}
    holders = [n for n in graph.graph.nodes if host_slot(n) is not None]
    inserted = sum(_splice_load(graph, h, host_slot(h), order) for h in holders)

    if inserted:
        graph.graph.lint()
        graph.recompile()
    magi_logger.info("host offload: inserted %d h2d_load node(s) into the graph", inserted)
    return inserted


def _splice_load(graph: fx.GraphModule, holder: fx.Node, slot: int, order) -> int:
    """Insert a load for ``holder`` in front of its first reader and re-point its readers at it.

    The load has to sit above every reader, so whatever the graph does to the
    weight still runs on the device; the reorder pass is what places it properly.
    """
    from ..runtime import host_weights
    from ..runtime.h2d_op import H2D_LOAD

    readers = list(holder.users)
    if not readers:
        return 0
    anchor = min(readers, key=lambda n: order.get(n, len(order)))
    example = holder.meta.get("example_value")
    with graph.graph.inserting_before(anchor):
        load = graph.graph.call_function(H2D_LOAD, (holder, slot))
        load.meta["example_value"] = example
        wait = graph.graph.call_function(_WAIT, (load,))
        wait.meta["example_value"] = example
    # Everything that read the (now storage-free) weight reads the loaded copy
    # instead -- except the load itself, which still needs it.
    holder.replace_all_uses_with(wait, delete_user_cb=lambda user: user is not load)
    host_weights.mark_claimed(slot)
    return 1
