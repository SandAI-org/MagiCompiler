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

"""Graph-side host offload: park weight shards in host memory, load them back in the graph.

Runs right after redistribute lowering, in the slot the copy engine uses for its
own binding -- and mutually exclusive with it, because a copy-engine gather reads
its peers' *device*-resident shards and offload is exactly the act of freeing
those.

Two steps, deliberately separate:

``bind_weights_to_host``
    picks the shards to offload, agrees on the pick across ranks, moves the bytes
    into the host pool and tags the graph.  Nothing about the graph's shape
    changes yet, so a failure here is a no-op.

``insert_h2d_loads``
    splices ``magi::h2d_load`` + ``wait_tensor`` in behind every tagged
    ``to_local``.  The load goes *above* the dtype cast and the uneven-shard pad
    so those still run on the device: casting on the host would both burn CPU and
    double the bytes crossing PCIe for a fp32-master/bf16-forward weight.
"""

from __future__ import annotations

import operator
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
import torch.fx as fx

from magi_compiler.utils import magi_logger

from .node_meta import HOST_SLOT, is_weight_ag, mark_host_offloaded, mark_host_slot

_ALL_GATHER = torch.ops._c10d_functional.all_gather_into_tensor.default
_ALL_GATHER_COALESCED = torch.ops._c10d_functional.all_gather_into_tensor_coalesced.default
_WAIT = torch.ops._c10d_functional.wait_tensor.default

# Producers we walk through to get from a gather back to its ``to_local``.
_PREP_METHODS = {"to_local", "contiguous", "to", "view", "reshape"}
_PREP_FUNCTIONS = ("constant_pad_nd", "_to_copy", "convert_element_type", "view", "reshape", "clone")


@dataclass
class OffloadCandidate:
    """One weight shard that could be parked in host memory."""

    holder: fx.Node  # the ``to_local`` producing the shard
    gather: fx.Node  # the all_gather it feeds
    param: Any  # the live DTensor behind the placeholder
    nbytes: int
    slot: int | None = None  # already bound by an earlier compile of the same model


def _is_prep(node: fx.Node) -> bool:
    if node.op == "call_method":
        return str(node.target) in _PREP_METHODS
    if node.op == "call_function":
        name = getattr(node.target, "__name__", "") or str(node.target)
        return any(t in name for t in _PREP_FUNCTIONS)
    return False


def _shard_holder(gather: fx.Node) -> fx.Node | None:
    """The ``to_local(placeholder|get_attr)`` whose output this gather consumes.

    Walks the prep chain rather than looking only at ``args[0]``: a weight with a
    ``forward_dtype`` or an uneven shard has a cast and/or a pad in between, and
    matching only the bare shape would silently skip exactly the biggest weights.
    """
    q: deque[fx.Node] = deque(gather.all_input_nodes)
    seen: set[fx.Node] = set()
    while q:
        node = q.popleft()
        if node in seen:
            continue
        seen.add(node)
        if node.op == "call_method" and node.target == "to_local":
            owner = node.args[0] if node.args else None
            if isinstance(owner, fx.Node) and owner.op in ("placeholder", "get_attr"):
                return node
            return None
        if _is_prep(node):
            q.extend(node.all_input_nodes)
    return None


def _param_name(holder: fx.Node) -> str:
    """The parameter a ``to_local`` reads, as something a human can place.

    Dynamo names a lifted parameter after its module path, so
    ``L_self_modules_layers_3_modules_mlp_parameters_w1_`` becomes
    ``layers.3.mlp.w1`` -- which is what the placement logs need to be readable
    at forty layers.
    """
    owner = holder.args[0] if holder.args else None
    raw = str(getattr(owner, "target", "") or getattr(owner, "name", "") or "?")
    parts = [p for p in raw.strip("_").split("_") if p and p not in ("L", "self", "modules", "parameters", "parameter")]
    return ".".join(parts) or raw


def _resolve(graph: fx.GraphModule, holder: fx.Node, placeholder_examples: Mapping[str, Any]) -> Any:
    """The live object behind a ``to_local``'s owner, or None."""
    owner = holder.args[0]
    if owner.op == "placeholder":
        return placeholder_examples.get(owner.name)
    if owner.op == "get_attr":
        obj: Any = graph
        for part in str(owner.target).split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj
    return None


def _unoffloadable(param: Any, min_shard_bytes: int) -> str | None:
    """Why ``param``'s shard cannot be parked in host memory, or None if it can."""
    from torch._subclasses.fake_tensor import FakeTensor
    from torch.distributed.tensor import DTensor, Shard

    if not isinstance(param, DTensor):
        return "graph input is not a DTensor"

    local = param._local_tensor
    if isinstance(local, FakeTensor) or local.is_meta:
        return "graph input is a fake/meta tensor"

    placements = param.placements
    if len(placements) != 1 or not isinstance(placements[0], Shard) or placements[0].dim != 0:
        return f"placement {tuple(placements)} is not a single Shard(0)"
    if local.device.type != "cuda":
        return f"shard already lives on {local.device.type}"
    if not local.is_contiguous():
        return "shard is not contiguous"
    if local.untyped_storage().nbytes() == 0:
        return "shard storage was already freed"
    if local.numel() * local.element_size() < min_shard_bytes:
        return "shard is below the size floor"
    return None


def _collect(
    graph: fx.GraphModule, placeholder_examples: Mapping[str, Any], min_shard_bytes: int
) -> tuple[list[OffloadCandidate], Counter]:
    from magi_compiler.offload import host_pool

    candidates: list[OffloadCandidate] = []
    skipped: Counter = Counter()
    seen: set[int] = set()

    for node in graph.graph.nodes:
        if node.op != "call_function" or node.target is not _ALL_GATHER or not is_weight_ag(node):
            continue
        holder = _shard_holder(node)
        if holder is None:
            skipped["gather does not reach a to_local(parameter)"] += 1
            continue
        param = _resolve(graph, holder, placeholder_examples)
        if param is None:
            skipped["graph input has no live parameter behind it"] += 1
            continue

        local = getattr(param, "_local_tensor", None)
        # A shard an earlier compile of this same model already bound.  It is a
        # candidate again, not a skip: every graph over these parameters needs
        # its own load, and the bytes are gone from the device either way.  This
        # only shows up on a model compiled for more than one shape, where
        # skipping it leaves the second graph gathering freed storage -- an
        # illegal access, far from here and with nothing pointing back.
        slot = host_pool.slot_of(local) if local is not None else None
        if slot is None:
            why = _unoffloadable(param, min_shard_bytes)
            if why is not None:
                skipped[why] += 1
                continue

        if id(local) in seen:
            skipped["shard is tied to an earlier weight"] += 1
            continue
        seen.add(id(local))
        candidates.append(
            OffloadCandidate(holder=holder, gather=node, param=param, nbytes=local.numel() * local.element_size(), slot=slot)
        )
    return candidates, skipped


def _select(candidates: list[OffloadCandidate]) -> list[OffloadCandidate]:
    """Offload every eligible shard.

    Which weights should actually stay on the device is not decided here, and
    deliberately so: the answer depends on how much compute sits upstream of each
    gather, which nothing at FX time can see.  ``H2dLoadReorder`` makes that call
    during scheduling, where it has per-kernel costs and a placement, and gives
    back the shards whose transfer it could not hide -- bounded by
    ``gpu_resident_weight_ratio``.  The size floor still applies (see
    ``_unoffloadable``), because a shard too small to amortize a DMA is a bad
    trade at any schedule.
    """
    return list(candidates)


def _agree_across_ranks(plan: list[OffloadCandidate]) -> bool:
    """True if every rank arrived at the same plan, in the same order.

    A shard offloaded on only some ranks is not a hang the way a half-retargeted
    gather is -- the all-gather still runs everywhere -- but the graphs would
    diverge structurally, which costs ``FsdpOverlapReorder`` its identical-graph
    fast path and turns every placement into a negotiation.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return True

    mine = [
        (c.holder.name, tuple(c.param._local_tensor.shape), str(c.param._local_tensor.dtype), c.slot is None) for c in plan
    ]
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, mine)
    for rank, theirs in enumerate(gathered):
        if theirs == mine:
            continue
        i = next((i for i, (a, b) in enumerate(zip(mine, theirs)) if a != b), min(len(mine), len(theirs)))
        magi_logger.warning(
            "host offload: this rank plans %d shard(s) and rank %d plans %d; they first differ at index "
            "%d, where this rank has %s and rank %d has %s",
            len(mine),
            rank,
            len(theirs),
            i,
            mine[i] if i < len(mine) else "<end of plan>",
            rank,
            theirs[i] if i < len(theirs) else "<end of plan>",
        )
        return False
    return True


def _describe(skipped: Counter) -> str:
    if not skipped:
        return "no candidate was skipped"
    return "skipped: " + ", ".join(f"{n}x {why}" for why, n in skipped.most_common())


def bind_weights_to_host(graph: fx.GraphModule, example_inputs: Sequence[Any] | None, *, min_shard_bytes: int = 0) -> int:
    """Move the selected weight shards into the host pool and tag the graph.

    Returns how many shards are now host-resident.  Failures are logged and
    dropped, never raised: the un-offloaded graph is always a valid fallback.
    """
    from magi_compiler.offload import host_pool

    placeholders = graph.graph.find_nodes(op="placeholder")
    placeholder_examples = dict(zip((n.name for n in placeholders), example_inputs or ()))

    candidates, skipped = _collect(graph, placeholder_examples, min_shard_bytes)
    plan = _select(candidates)

    # Before the empty check, so every rank reaches the collective.
    if not _agree_across_ranks(plan):
        magi_logger.warning("host offload: ranks disagree on which shards to offload; offloading none")
        return 0

    if not plan:
        magi_logger.info("host offload: nothing to offload (%s)", _describe(skipped))
        return 0

    fresh = [c for c in plan if c.slot is None]
    bound = host_pool.bind_many([c.param._local_tensor for c in fresh], names=[_param_name(c.holder) for c in fresh])
    for c, slot in zip(fresh, bound):
        c.slot = slot
    for c in plan:
        mark_host_slot(c.holder, c.slot)
        mark_host_offloaded(c.gather)

    magi_logger.info(
        "host offload: %d shard(s) served from host memory, %d newly parked (%.1f MiB freed on device); %s",
        len(plan),
        len(fresh),
        sum(c.nbytes for c in fresh) / 2**20,
        _describe(skipped),
    )
    return len(plan)


def insert_h2d_loads(graph: fx.GraphModule) -> int:
    """Splice ``h2d_load`` + ``wait_tensor`` in behind every tagged ``to_local``.

    Runs AFTER bucketing, so a coalesced gather's members are known and their
    loads can be merged to match: one submission, one event, one wait per bucket
    instead of one of each per weight.

    Returns how many load nodes were inserted.
    """
    from .node_meta import host_slot

    order = {n: i for i, n in enumerate(graph.graph.nodes)}
    inserted = 0
    done: set[fx.Node] = set()

    for gather in list(graph.graph.nodes):
        if gather.op != "call_function" or gather.target not in (_ALL_GATHER, _ALL_GATHER_COALESCED):
            continue
        shard_args = gather.args[0] if gather.target is _ALL_GATHER_COALESCED else [gather.args[0]]

        pairs = []
        for shard in shard_args:
            holder = _holder_with_slot(shard)
            if holder is not None and holder not in done:
                pairs.append((holder, host_slot(holder)))
        if not pairs:
            continue
        if len(pairs) != len(shard_args):
            # split_by=is_host_offloaded keeps buckets single-transport, so a
            # partial bucket means an assumption broke somewhere upstream.
            magi_logger.warning(
                "host offload: %s mixes %d offloaded and %d resident shard(s); "
                "loading them separately rather than as one bucket",
                gather.name,
                len(pairs),
                len(shard_args) - len(pairs),
            )

        done.update(h for h, _ in pairs)
        inserted += _splice_loads(graph, pairs, order)

    if inserted:
        graph.graph.lint()
        graph.recompile()
    magi_logger.info("host offload: inserted %d h2d_load node(s) into the graph", inserted)
    return inserted


def _holder_with_slot(node) -> fx.Node | None:
    """Walk back from a gather's shard argument to the ``to_local`` carrying a slot."""
    if not isinstance(node, fx.Node):
        return None
    q: deque[fx.Node] = deque([node])
    seen: set[fx.Node] = set()
    while q:
        n = q.popleft()
        if n in seen:
            continue
        seen.add(n)
        if n.meta.get(HOST_SLOT) is not None:
            return n
        if _is_prep(n):
            q.extend(n.all_input_nodes)
    return None


def _splice_loads(graph: fx.GraphModule, pairs, order) -> int:
    """Insert one load for ``pairs`` and re-point the shards' readers at it.

    The load has to sit above every reader of every shard in the group, so the
    dtype cast and the uneven-shard pad still run on the device.  The shards
    themselves are hoisted to meet it; they read nothing but a placeholder, so
    moving them up is always legal.
    """
    from magi_compiler.offload.h2d_op import H2D_LOAD, H2D_LOAD_COALESCED

    holders = [h for h, _ in pairs]
    slots = [s for _, s in pairs]
    examples = [h.meta.get("example_value") for h in holders]

    readers = [u for h in holders for u in h.users]
    anchor = min(readers, key=lambda n: order.get(n, len(order)))
    for holder in sorted(holders, key=lambda n: order.get(n, 0)):
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
            # Everything that read the (now storage-free) shard reads the loaded
            # copy instead -- except the load itself, which still needs the shard.
            holder.replace_all_uses_with(wait, delete_user_cb=lambda user: user is not load)

    return 1
