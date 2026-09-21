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

"""Tag already-parked weights, and load them back inside the graph.

Weights reach the host pool only through host-first materialization
(``reserve`` + ``adopt``).  Two steps, deliberately separate:

``bind_weights_to_host``
    asks the source which parked weights the graph should load, intersects
    the pick with peers in each shard group (per-candidate, not WORLD-wide
    all-or-nothing), and tags the graph.  Nothing about the graph's shape
    changes, and no bytes move -- a failure here is a no-op.

``insert_h2d_loads``
    splices ``magi::h2d_load`` + ``wait_tensor`` in behind each tagged weight.
    The load goes ABOVE every reader of that weight, so anything the graph did
    to it -- a dtype cast, an uneven-shard pad -- still runs on the device:
    casting on the host would both burn CPU and, for a fp32-master/bf16-forward
    weight, double the bytes crossing PCIe.

What a weight *is* belongs to the source (see ``sources.py``); everything here
is written once and works for a sharded model and a plain one alike.
"""

from __future__ import annotations

import operator
from collections import Counter
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
import torch.fx as fx

from magi_compiler.utils import magi_logger

from .node_meta import host_slot, mark_host_offloaded, mark_host_slot
from .sources import OffloadCandidate, WeightSource, mesh_group

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
        try:
            world = dist.get_world_size(pg)
            gathered: list[Any] = [None] * world
            dist.all_gather_object(gathered, keys, group=pg)
        except Exception as exc:  # noqa: BLE001
            magi_logger.warning(
                "host offload: agreement on a %d-candidate process group failed (%s); dropping that group", len(cands), exc
            )
            skipped["process-group agreement failed"] += len(cands)
            continue
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


def bind_weights_to_host(
    graph: fx.GraphModule, example_inputs: Sequence[Any] | None, source: WeightSource, *, min_bytes: int = 0
) -> int:
    """Tag already-parked weights on the graph so a later splice can load them.

    A weight reaches the pool only through host-first materialization.  This
    step does not copy or free anything.  Failures are logged and dropped,
    never raised: the un-offloaded graph is always a valid fallback.

    Returns how many weights are now served from host memory.
    """
    placeholders = graph.graph.find_nodes(op="placeholder")
    placeholder_examples: Mapping[str, Any] = dict(zip((n.name for n in placeholders), example_inputs or ()))

    plan, skipped = source.collect(graph, placeholder_examples, min_bytes)

    # Before the empty check, so every rank still enters each of its groups.
    plan = _align_across_ranks(plan, placeholder_examples, skipped)
    parked: list[OffloadCandidate] = []
    for c in plan:
        if c.slot is None:
            skipped["weight was not materialized in host memory"] += 1
            continue
        parked.append(c)
    if not parked:
        magi_logger.info("host offload: nothing to offload (%s)", _describe(skipped))
        return 0

    for c in parked:
        mark_host_slot(c.holder, c.slot)
        if c.tag is not None:
            mark_host_offloaded(c.tag)

    sizes = sorted(c.nbytes for c in parked)
    magi_logger.info(
        "host offload: %d weight(s) served from host memory (%.1f MiB never allocated on device); "
        "sizes %.1f / %.1f / %.1f MiB (min/median/max, floor %.1f); %s",
        len(parked),
        sum(c.nbytes for c in parked) / 2**20,
        sizes[0] / 2**20,
        sizes[len(sizes) // 2] / 2**20,
        sizes[-1] / 2**20,
        min_bytes / 2**20,
        _describe(skipped),
    )
    return len(parked)


def apply_weight_offload(
    graph: fx.GraphModule, example_inputs: Sequence[Any] | None, source: WeightSource, *, min_bytes: int = 0
) -> int:
    """Bind then splice loads, for a graph that is NOT going through FSDP bucketing.

    FSDP cannot use this: it must bind *before* bucketing and insert *after*, so
    a bucket's members share one load.  The backend's weight pipeline keeps the
    two calls apart for that reason and does not go through here.
    """
    bound = bind_weights_to_host(graph, example_inputs, source, min_bytes=min_bytes)
    if not bound:
        return 0
    return insert_h2d_loads(graph, source)


def insert_h2d_loads(graph: fx.GraphModule, source: WeightSource) -> int:
    """Splice a load in for every tagged weight, grouped as the source asks.

    Returns how many load nodes were inserted.
    """
    holders = {n for n in graph.graph.nodes if host_slot(n) is not None}
    if not holders:
        magi_logger.info("host offload: inserted 0 h2d_load node(s) into the graph")
        return 0

    order = {n: i for i, n in enumerate(graph.graph.nodes)}
    inserted = 0
    for group in source.group(graph, holders):
        pairs = [(h, host_slot(h)) for h in group if host_slot(h) is not None]
        if pairs:
            inserted += _splice_loads(graph, pairs, order)

    if inserted:
        graph.graph.lint()
        graph.recompile()
    magi_logger.info("host offload: inserted %d h2d_load node(s) into the graph", inserted)
    return inserted


def _claim(slots) -> None:
    """Record that these shards now have a load that will put their bytes back.

    The pool cannot tell a shard that is waiting for its load from one that will
    never get one, and the second kind is a kernel reading freed storage.
    Claiming here -- at the splice, not at the bind -- is what lets
    ``restore_unclaimed`` tell them apart once the graph is final.
    """
    from . import host_pool

    for slot in slots:
        host_pool.mark_claimed(slot)


def _splice_loads(graph: fx.GraphModule, pairs, order) -> int:
    """Insert one load for ``pairs`` and re-point the weights' readers at it.

    The load has to sit above every reader of every weight in the group, so
    whatever the graph does to the weight still runs on the device.  The weights
    themselves are hoisted to meet it; they read nothing but a placeholder, so
    moving them up is always legal.
    """
    from .h2d_op import H2D_LOAD, H2D_LOAD_COALESCED

    holders = [h for h, _ in pairs]
    slots = [s for _, s in pairs]
    examples = [h.meta.get("example_value") for h in holders]

    readers = [u for h in holders for u in h.users]
    if not readers:
        return 0
    anchor = min(readers, key=lambda n: order.get(n, len(order)))
    for holder in sorted(holders, key=lambda n: order.get(n, 0)):
        if holder.op in ("placeholder", "get_attr"):
            continue
        # Only a holder that reads nothing but a placeholder.  The anchor is the
        # group's EARLIEST reader, so for any other holder the move can be
        # upwards -- past a producer of its own.  An unsharded weight's holder
        # reads a redistribute and is exactly that case; leaving it where it is
        # costs nothing, since the load is spliced in front of the anchor either
        # way and the reorder pass is what places it properly.
        if all(inp.op in ("placeholder", "get_attr") for inp in holder.all_input_nodes):
            anchor.prepend(holder)

    with graph.graph.inserting_before(anchor):
        if len(pairs) == 1:
            load = graph.graph.call_function(H2D_LOAD, (holders[0], slots[0]))
            load.meta["example_value"] = examples[0]
            outs = [load]
        else:
            load = graph.graph.call_function(H2D_LOAD_COALESCED, (list(holders), list(slots)))
            load.meta["example_value"] = list(examples)
            outs = []
            for i, example in enumerate(examples):
                out = graph.graph.call_function(operator.getitem, (load, i))
                out.meta["example_value"] = example
                outs.append(out)

        for holder, out, example in zip(holders, outs, examples):
            wait = graph.graph.call_function(_WAIT, (out,))
            wait.meta["example_value"] = example
            # Everything that read the (now storage-free) weight reads the loaded
            # copy instead -- except the load itself, which still needs it.
            holder.replace_all_uses_with(wait, delete_user_cb=lambda user: user is not load)

    _claim(slots)
    return 1
