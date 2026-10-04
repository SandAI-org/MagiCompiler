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

"""What counts as an offloadable weight, and where its load goes.

``PlainParamSource`` and ``FsdpShardSource`` are the two answers.  Everything
downstream -- binding, splicing the load in, placing it during scheduling -- is
written once against ``OffloadCandidate``.

Two differences are worth naming because they change the arithmetic downstream,
not just the lookup:

* **What dies, and when.**  Under FSDP the loaded bytes are a shard that an
  all-gather consumes into a fresh buffer, so the load's output dies almost
  immediately.  Without FSDP the loaded bytes *are* the weight the matmul reads,
  so they live until the last consumer -- a much longer range for the same
  placement.
* **How much moves.**  A shard is 1/world_size of a weight.  Offloading a plain
  parameter moves the whole thing, so the same model costs world_size times the
  PCIe traffic.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import torch
import torch.fx as fx

from .fx_walk import is_prep, param_name, resolve, walk_back_to_holder


def mesh_group(param: Any):
    """The process group this weight is sharded or replicated over, or None.

    Binder agreement is per this group, not WORLD: an expert-parallel rank
    only has to match the ranks that share its mesh.  A plain Parameter has
    no mesh and does not enter a collective.
    """
    mesh = getattr(getattr(param, "_spec", None), "mesh", None)
    if mesh is None:
        return None
    try:
        return mesh.get_group()
    except RuntimeError:
        # Multi-dim mesh needs an explicit dim; we only offload single Shard(0)
        # / Replicate, which live on dim 0.
        try:
            return mesh.get_group(0)
        except Exception:  # noqa: BLE001
            return None
    except Exception:  # noqa: BLE001
        return None


@dataclass
class OffloadCandidate:
    """One weight whose bytes could live in host memory."""

    holder: fx.Node  # the node whose output is the weight; the load goes right after it
    local: Any  # the live stand-in the slot is keyed on
    name: str  # the parameter it came from, for the placement logs
    nbytes: int
    tag: fx.Node | None = None  # node to mark offloaded, so later passes can tell
    slot: int | None = None  # host-pool slot from adopt; None = never parked
    group: Any = None  # ProcessGroup of the DTensor mesh; None = no cross-rank vote


class WeightSource(Protocol):
    """How to find offloadable weights in one flavour of graph."""

    def collect(
        self, graph: fx.GraphModule, placeholder_examples: Mapping[str, Any], min_bytes: int
    ) -> tuple[list[OffloadCandidate], Counter]:
        """Every candidate, plus a tally of why the rest were passed over."""


def parked_slot(local: Any, min_bytes: int) -> tuple[int | None, str | None]:
    """The host-pool slot for ``local``, or ``(None, why)`` if it cannot be loaded.

    A weight reaches the pool only through host-first materialization.  Collect
    never copies a resident shard off the device.
    """
    from ..runtime import host_pool

    if not isinstance(local, torch.Tensor):
        return None, "graph input is not a tensor"
    slot = host_pool.slot_of(local)
    if slot is None:
        return None, "weight was not materialized in host memory"
    if host_pool.slot_bytes(slot) < min_bytes:
        return None, "weight is below the size floor"
    return slot, None


@dataclass
class PlainParamSource(WeightSource):
    """Weights of a model that is NOT sharded: lifted parameter placeholders.

    Without FSDP there is no redistribute to lower and no all-gather to key off,
    so a weight is just a graph input that happens to be a ``Parameter``.  The
    load goes directly in front of its first reader, and its bytes stay live
    until the last one -- there is no gather to hand them off to.
    """

    def collect(
        self, graph: fx.GraphModule, placeholder_examples: Mapping[str, Any], min_bytes: int
    ) -> tuple[list[OffloadCandidate], Counter]:
        from ..runtime import host_pool

        candidates: list[OffloadCandidate] = []
        skipped: Counter = Counter()

        for node in graph.graph.nodes:
            if node.op not in ("placeholder", "get_attr"):
                continue
            live = resolve(graph, node, placeholder_examples)
            if live is None:
                continue
            if not isinstance(live, torch.nn.Parameter):
                # Activations and buffers are graph inputs too; only weights are
                # worth moving, because only they are the same every forward.
                continue
            if not node.users:
                continue

            slot, why = parked_slot(live, min_bytes)
            if why is not None:
                skipped[why] += 1
                continue
            # A tied weight reaches the graph as two placeholders, and each one
            # needs its own load: the splice repoints the readers of the node it
            # was given, so a second node left untagged would read the empty
            # stand-in.  The nodes are distinct by construction here, so there
            # is nothing to dedupe -- the shard was adopted once.
            candidates.append(
                OffloadCandidate(
                    holder=node, local=live, name=param_name(node), nbytes=host_pool.slot_bytes(slot), tag=None, slot=slot
                )
            )
        return candidates, skipped


_ALL_GATHER = torch.ops._c10d_functional.all_gather_into_tensor.default
_ALL_GATHER_COALESCED = torch.ops._c10d_functional.all_gather_into_tensor_coalesced.default


def _is_to_local(node: fx.Node) -> bool:
    if not (node.op == "call_method" and node.target == "to_local"):
        return False
    owner = node.args[0] if node.args else None
    return isinstance(owner, fx.Node) and owner.op in ("placeholder", "get_attr")


def shard_holder(gather: fx.Node) -> fx.Node | None:
    """The ``to_local(placeholder|get_attr)`` whose output this gather consumes.

    Walks the prep chain rather than looking only at ``args[0]``: a weight with a
    ``forward_dtype`` or an uneven shard has a cast and/or a pad in between, and
    matching only the bare shape would silently skip exactly the biggest weights.
    """
    for inp in gather.all_input_nodes:
        found = walk_back_to_holder(inp, _is_to_local)
        if found is not None:
            return found
    return None


def _is_local_extraction(node: fx.Node) -> bool:
    """A ``to_local``, in either shape SimpleFSDP's parametrization leaves behind.

    The lowering rewrites the ``Shard(0)`` ones into ``to_local`` + all-gather;
    what still carries the ``prim_to_local`` form after it has run is a weight it
    declined to lower.
    """
    return (node.op == "call_function" and getattr(node.target, "__name__", None) == "prim_to_local") or (
        node.op == "call_method" and node.target == "to_local"
    )


def _feeds_a_weight_gather(node: fx.Node) -> bool:
    """True if an all-gather consumes this node, through the usual prep chain.

    The forward mirror of ``shard_holder``, and the thing that makes "has no
    gather" a property of the graph rather than of what an earlier sweep
    happened to record.  ``h2d_load`` and ``wait_tensor`` are walked through for
    the same reason the backward walk does it: once a load has been spliced in,
    a walk that stops at it stops one node short of the gather it is looking for.
    """
    q: deque[fx.Node] = deque(node.users)
    seen: set[fx.Node] = set()
    while q:
        user = q.popleft()
        if user in seen:
            continue
        seen.add(user)
        if user.op == "call_function" and user.target in (_ALL_GATHER, _ALL_GATHER_COALESCED):
            return True
        name = getattr(user.target, "__name__", "") or str(user.target)
        if is_prep(user) or "h2d_load" in name or "wait_tensor" in name:
            q.extend(user.users)
    return False


def _weight_behind(to_local: fx.Node) -> fx.Node | None:
    """The parameter placeholder a ``to_local`` reads, through its redistribute."""
    src = to_local.args[0] if to_local.args else None
    if not isinstance(src, fx.Node):
        return None
    if src.op not in ("placeholder", "get_attr"):
        name = getattr(src.target, "__name__", None)
        if not (name == "prim_redistribute" or (src.op == "call_method" and src.target == "redistribute")):
            return None
        src = src.args[0] if src.args else None
    if not isinstance(src, fx.Node) or src.op not in ("placeholder", "get_attr"):
        return None
    # Only weights.  An activation that happens to be a DTensor changes every
    # forward, so parking it would serve stale bytes -- and unlike a wrong
    # placement that is not something any later check would notice.
    text = f"{src.name} {src.target}".lower()
    return src if any(t in text for t in ("parameter", "parameters", "weight", "bias")) else None


@dataclass
class FsdpShardSource(WeightSource):
    """Weights of a SimpleFSDP model: the shards its weight all-gathers read.

    A shard's loaded bytes die at the all-gather that consumes them, which is why
    the loads here are cheaper to schedule than a plain parameter's: the live
    range is a handful of snodes rather than the rest of the layer.

    ``collect`` runs BEFORE bucketing, so it only ever meets the
    one-gather-per-weight form the lowering produces: bucketing needs to know
    which shards are offloaded to keep a bucket single-kind, and that is exactly
    what binding decides.  Every weight still gets a load of its own after
    bucketing; a bucket's launch waits for each member's load separately.
    """

    def collect(
        self, graph: fx.GraphModule, placeholder_examples: Mapping[str, Any], min_bytes: int
    ) -> tuple[list[OffloadCandidate], Counter]:
        from magi_compiler.passes.fsdp_overlap.node_meta import is_weight_ag

        from ..runtime import host_pool

        candidates: list[OffloadCandidate] = []
        skipped: Counter = Counter()
        seen: set[fx.Node] = set()

        for node in graph.graph.nodes:
            if node.op != "call_function" or not is_weight_ag(node):
                continue
            if node.target is _ALL_GATHER_COALESCED:
                # Binding has to precede bucketing -- bucketing splits offloaded
                # from resident gathers on the tag binding sets -- so a bucket
                # here means the two ran in the wrong order.  Counted rather than
                # ignored: every weight would drop out, the caller would log
                # "nothing to offload", and the first sign of it would be the
                # OOM that offload was turned on to prevent.
                skipped["weight gather was already bucketed; binding must run before bucketing"] += 1
                continue
            if node.target is not _ALL_GATHER:
                continue
            holder = shard_holder(node)
            if holder is None:
                skipped["gather does not reach a to_local(parameter)"] += 1
                continue
            param = resolve(graph, holder.args[0], placeholder_examples)
            if param is None:
                skipped["graph input has no live parameter behind it"] += 1
                continue

            local = getattr(param, "_local_tensor", None)
            # A shard an earlier compile of this same model already adopted.  It
            # is a candidate again, not a skip: every graph over these
            # parameters needs its own load.  Skipping it leaves the second
            # graph gathering an empty stand-in -- an illegal access, far from
            # here and with nothing pointing back.
            slot, why = parked_slot(local, min_bytes)
            if why is not None:
                skipped[why] += 1
                continue

            # By holder, not by shard.  A weight two all-gathers read has two
            # holders, and each one needs its own load: the splice repoints the
            # readers of the holder it was given, so a second holder left
            # untagged would gather the empty stand-in.  The shard was adopted
            # once.
            if holder in seen:
                skipped["gather shares a holder with an earlier weight"] += 1
                continue
            seen.add(holder)
            candidates.append(
                OffloadCandidate(
                    holder=holder,
                    local=local,
                    name=param_name(holder.args[0]),
                    nbytes=host_pool.slot_bytes(slot),
                    tag=node,
                    slot=slot,
                    group=mesh_group(param),
                )
            )

        self._collect_ungathered(graph, placeholder_examples, min_bytes, candidates, skipped, seen)
        return candidates, skipped

    @staticmethod
    def _collect_ungathered(graph, placeholder_examples, min_bytes, candidates, skipped, seen) -> None:
        """Weights SimpleFSDP never shards, and which therefore have no gather.

        A ``Shard(0)`` whose dim0 does not divide its mesh pads only the trailing
        ranks, which makes the graph differ per rank and deadlocks NCCL, so
        athena replicates those weights instead.  The lowering then leaves them
        on the prim path, and keying off all-gathers misses them entirely.

        They are the ones most worth taking.  A replicated weight is a FULL copy
        on every rank rather than 1/N, so per GPU it costs world_size times what
        the same tensor costs sharded -- and it was excluded from the one feature
        that exists to get weights off the device.

        What changes downstream is when the bytes die: a shard is consumed by its
        gather and freed, while these ARE the weight the matmul reads and live to
        its last consumer.
        """
        from ..runtime import host_pool

        for node in graph.graph.nodes:
            if node in seen or not _is_local_extraction(node) or not node.users:
                continue
            src = _weight_behind(node)
            if src is None or _feeds_a_weight_gather(node):
                continue
            param = resolve(graph, src, placeholder_examples)
            local = getattr(param, "_local_tensor", None)
            if local is None:
                continue
            slot, why = parked_slot(local, min_bytes)
            if why is not None:
                skipped[why] += 1
                continue
            seen.add(node)
            candidates.append(
                OffloadCandidate(
                    holder=node,
                    local=local,
                    name=param_name(src),
                    nbytes=host_pool.slot_bytes(slot),
                    # No gather to tag: bucketing splits offloaded gathers from
                    # resident ones, and this weight has neither.
                    tag=None,
                    slot=slot,
                    group=mesh_group(param),
                )
            )
