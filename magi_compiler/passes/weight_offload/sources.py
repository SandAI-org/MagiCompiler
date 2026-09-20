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
from typing import Any, Callable, Mapping, Protocol

import torch
import torch.fx as fx

# Producers we walk through when tracing a value back to the parameter behind it.
_PREP_METHODS = {"to_local", "contiguous", "to", "view", "reshape"}
_PREP_FUNCTIONS = ("constant_pad_nd", "_to_copy", "convert_element_type", "view", "reshape", "clone")


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

    def group(self, graph: fx.GraphModule, holders: set[fx.Node]) -> list[list[fx.Node]]:
        """Which tagged weights share one load.

        A group is one submission: one stream sync, one event, one wait.  Takes
        graph nodes rather than the candidates ``collect`` returned, because
        bucketing runs in between and rebuilds nodes -- only what is still in the
        graph, carrying its slot in ``node.meta``, is safe to read here.
        """


def is_prep(node: fx.Node) -> bool:
    """A cheap reshaping producer that a weight's value may pass through."""
    if node.op == "call_method":
        return str(node.target) in _PREP_METHODS
    if node.op == "call_function":
        name = getattr(node.target, "__name__", "") or str(node.target)
        return any(t in name for t in _PREP_FUNCTIONS)
    return False


def param_name(node: fx.Node) -> str:
    """A parameter's module path, as something a human can place.

    Dynamo names a lifted parameter after that path, so
    ``L_self_modules_layers_3_modules_mlp_parameters_w1_`` becomes
    ``layers.3.mlp.w1`` -- which is what the placement logs need to be readable
    at forty layers.
    """
    raw = str(getattr(node, "target", "") or getattr(node, "name", "") or "?")
    parts = [p for p in raw.strip("_").split("_") if p and p not in ("L", "self", "modules", "parameters", "parameter")]
    return ".".join(parts) or raw


def resolve(graph: fx.GraphModule, node: fx.Node, placeholder_examples: Mapping[str, Any]) -> Any:
    """The live object a placeholder or get_attr stands for, or None."""
    if node.op == "placeholder":
        return placeholder_examples.get(node.name)
    if node.op == "get_attr":
        obj: Any = graph
        for part in str(node.target).split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj
    return None


def parked_slot(local: Any, min_bytes: int) -> tuple[int | None, str | None]:
    """The host-pool slot for ``local``, or ``(None, why)`` if it cannot be loaded.

    A weight reaches the pool only through host-first materialization.  Collect
    never copies a resident shard off the device.
    """
    from . import host_pool

    if not isinstance(local, torch.Tensor):
        return None, "graph input is not a tensor"
    slot = host_pool.slot_of(local)
    if slot is None:
        return None, "weight was not materialized in host memory"
    if host_pool.slot_bytes(slot) < min_bytes:
        return None, "weight is below the size floor"
    return slot, None


@dataclass
class PlainParamSource:
    """Weights of a model that is NOT sharded: lifted parameter placeholders.

    Without FSDP there is no redistribute to lower and no all-gather to key off,
    so a weight is just a graph input that happens to be a ``Parameter``.  The
    load goes directly in front of its first reader, and its bytes stay live
    until the last one -- there is no gather to hand them off to.

    Weights are grouped into one load while they are consumed close together and
    the group stays under ``group_bytes``, for the same reason the FSDP path
    buckets: the per-load stream sync, event and Work registration is ~10us of
    CPU that would otherwise be paid once per weight.
    """

    group_bytes: int = 0  # 0 = one load per weight

    def collect(
        self, graph: fx.GraphModule, placeholder_examples: Mapping[str, Any], min_bytes: int
    ) -> tuple[list[OffloadCandidate], Counter]:
        from . import host_pool

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

    def group(self, graph: fx.GraphModule, holders: set[fx.Node]) -> list[list[fx.Node]]:
        from . import host_pool
        from .node_meta import host_slot

        order = {n: i for i, n in enumerate(graph.graph.nodes)}

        # By first reader, not by declaration: Dynamo lifts every parameter to the
        # top of the graph, so their placeholder order says nothing about when
        # they are used, and grouping by it would put layer 0 and layer 39 in one
        # submission -- which is the thing that makes a bucket span a model.
        def first_use(node: fx.Node) -> int:
            return min((order.get(u, len(order)) for u in node.users), default=len(order))

        groups: list[list[fx.Node]] = []
        run_bytes = 0
        for node in sorted(holders, key=first_use):
            nbytes = host_pool.slot_bytes(host_slot(node))
            if groups and self.group_bytes > 0 and run_bytes + nbytes <= self.group_bytes:
                groups[-1].append(node)
                run_bytes += nbytes
            else:
                groups.append([node])
                run_bytes = nbytes
        return groups


def walk_back_to_holder(node: fx.Node, stop: Callable[[fx.Node], bool]) -> fx.Node | None:
    """Walk a prep chain backwards until ``stop`` accepts a node."""
    q: deque[fx.Node] = deque([node])
    seen: set[fx.Node] = set()
    while q:
        n = q.popleft()
        if n in seen:
            continue
        seen.add(n)
        if stop(n):
            return n
        if is_prep(n):
            q.extend(n.all_input_nodes)
    return None


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
class FsdpShardSource:
    """Weights of a SimpleFSDP model: the shards its weight all-gathers read.

    A shard's loaded bytes die at the all-gather that consumes them, which is why
    the loads here are cheaper to schedule than a plain parameter's: the live
    range is a handful of snodes rather than the rest of the layer.

    The two halves see different graphs, and that is forced rather than
    incidental.  ``collect`` runs BEFORE bucketing, so it only ever meets the
    one-gather-per-weight form the lowering produces: bucketing needs to know
    which shards are offloaded to keep a bucket single-kind, and that is exactly
    what binding decides.  ``group`` runs AFTER, so a bucket's members are known
    and their loads can be merged to match, and it has to understand the
    coalesced form as well.  Swapping the order looks harmless and is not --
    see the note in ``collect``.
    """

    def collect(
        self, graph: fx.GraphModule, placeholder_examples: Mapping[str, Any], min_bytes: int
    ) -> tuple[list[OffloadCandidate], Counter]:
        from magi_compiler.passes.fsdp_overlap.node_meta import is_weight_ag

        from . import host_pool

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
        its last consumer.  ``group`` gives each its own load for that reason.
        """
        from . import host_pool

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

    def group(self, graph: fx.GraphModule, holders: set[fx.Node]) -> list[list[fx.Node]]:
        """One load per all-gather bucket, plus one apiece for the ungathered weights.

        Runs after bucketing, so a bucket's members are known and their loads can
        be merged to match: one submission and one wait per bucket, not per
        weight.  Mirroring the buckets is not an optimization but the point --
        every member has to have landed before the single launch that reads them
        all.

        A weight with no gather has no bucket to mirror, so it gets a load to
        itself.  Merging it into one would be wrong twice over: its bytes live
        until its last reader rather than dying at a gather, so it would hold a
        full-size buffer open for as long as the bucket's shortest-lived member
        needs; and the splice hoists a group's holders to their earliest common
        reader, which is only safe while a holder reads nothing but a
        placeholder -- these read a redistribute.
        """
        from magi_compiler.passes.fsdp_overlap.node_meta import is_weight_ag

        groups: list[list[fx.Node]] = []
        done: set[fx.Node] = set()
        for gather in graph.graph.nodes:
            if gather.op != "call_function" or gather.target not in (_ALL_GATHER, _ALL_GATHER_COALESCED):
                continue
            # The same question ``collect`` asks. A model has other collectives --
            # gaga4 alone gathers activations for context parallelism -- and
            # walking back from their arguments only fails to find a tagged
            # holder, which is the right answer arrived at the slow way.
            if not is_weight_ag(gather):
                continue
            shard_args = gather.args[0] if gather.target is _ALL_GATHER_COALESCED else [gather.args[0]]
            members = []
            for shard in shard_args:
                holder = shard if shard in holders else self._holder_of(shard, holders)
                if holder is not None and holder not in done:
                    members.append(holder)
            if not members:
                continue
            if len(members) != len(shard_args):
                # split_by keeps buckets single-kind, so a partial bucket means an
                # assumption broke somewhere upstream.
                from magi_compiler.utils import magi_logger

                magi_logger.warning(
                    "host offload: %s mixes %d offloaded and %d resident shard(s); loading them "
                    "separately rather than as one bucket",
                    gather.name,
                    len(members),
                    len(shard_args) - len(members),
                )
            done.update(members)
            groups.append(members)

        order = {n: i for i, n in enumerate(graph.graph.nodes)}
        groups.extend([h] for h in sorted(holders - done, key=lambda n: order.get(n, len(order))))
        return groups

    @staticmethod
    def _holder_of(node, holders: set[fx.Node]) -> fx.Node | None:
        """Walk back from a gather's shard argument to the tagged ``to_local``."""
        from .node_meta import host_slot

        if not isinstance(node, fx.Node):
            return None
        return walk_back_to_holder(node, lambda n: n in holders and host_slot(n) is not None)
