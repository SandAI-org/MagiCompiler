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

"""Discover plain and FSDP weights as OffloadCandidates for binding and scheduling.

Gathered weights load local shards whose outputs live through their consumers,
typically the gather. Plain and replicated weights load full local copies that
remain live through computation. Payload size and lifetime drive H2D placement.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import torch
import torch.fx as fx
from torch.distributed.tensor import DTensor

# Producers we walk through when tracing a value back to the parameter behind it.
_PREP_METHODS = {"to_local", "contiguous", "to", "view", "reshape"}
_PREP_FUNCTIONS = ("constant_pad_nd", "_to_copy", "convert_element_type", "view", "reshape", "clone")


def _is_prep(node: fx.Node) -> bool:
    """A cheap reshaping producer that a weight's value may pass through."""
    if node.op == "call_method":
        return str(node.target) in _PREP_METHODS
    if node.op == "call_function":
        name = getattr(node.target, "__name__", "") or str(node.target)
        return any(t in name for t in _PREP_FUNCTIONS)
    return False


def _param_name(node: fx.Node) -> str:
    raw = str(node.target or node.name or "?")
    parts = [p for p in raw.strip("_").split("_") if p and p not in ("L", "self", "modules", "parameters", "parameter")]
    return ".".join(parts) or raw


def _resolve(graph: fx.GraphModule, node: fx.Node, placeholder_examples: Mapping[str, Any]) -> Any:
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


def mesh_group(param: object):
    """Binder agreement uses the DTensor's group, selecting dim 0 for a multi-dim mesh.
    Non-DTensor inputs do not vote; group lookup errors propagate.
    """
    if not isinstance(param, DTensor):
        return None
    mesh = param.device_mesh
    return mesh.get_group(mesh_dim=0 if mesh.ndim > 1 else None)


@dataclass
class OffloadCandidate:
    """One weight whose bytes could live in host memory."""

    holder: fx.Node  # the node whose output is the weight; the load goes right after it
    local: Any  # the live stand-in the slot is keyed on
    name: str  # the parameter it came from, for the placement logs
    nbytes: int
    slot: int  # host-pool slot from adopt
    tag: fx.Node | None = None  # node to mark offloaded, so later passes can tell
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
    from ..runtime import host_weights

    if not isinstance(local, torch.Tensor):
        return None, "graph input is not a tensor"
    slot = host_weights.slot_of(local)
    if slot is None:
        return None, "weight was not materialized in host memory"
    if host_weights.slot_bytes(slot) < min_bytes:
        return None, "weight is below the size floor"
    return slot, None


@dataclass
class PlainParamSource(WeightSource):
    """Discover registered Parameters behind placeholders and get_attr nodes."""

    def collect(
        self, graph: fx.GraphModule, placeholder_examples: Mapping[str, Any], min_bytes: int
    ) -> tuple[list[OffloadCandidate], Counter]:
        from ..runtime import host_weights

        candidates: list[OffloadCandidate] = []
        skipped: Counter = Counter()

        for node in graph.graph.nodes:
            if node.op not in ("placeholder", "get_attr"):
                continue
            live = _resolve(graph=graph, node=node, placeholder_examples=placeholder_examples)
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
            # Each placeholder needs a load, even when tied weights share one host slot.
            candidates.append(
                OffloadCandidate(
                    holder=node, local=live, name=_param_name(node), nbytes=host_weights.slot_bytes(slot), tag=None, slot=slot
                )
            )
        return candidates, skipped


_ALL_GATHER = torch.ops._c10d_functional.all_gather_into_tensor.default


def _is_to_local(node: fx.Node) -> bool:
    if not (node.op == "call_method" and node.target == "to_local"):
        return False
    owner = node.args[0] if node.args else None
    return isinstance(owner, fx.Node) and owner.op in ("placeholder", "get_attr")


def _find_shard_holder(node: fx.Node) -> fx.Node | None:
    q: deque[fx.Node] = deque([node])
    seen: set[fx.Node] = set()
    while q:
        n = q.popleft()
        if n in seen:
            continue
        seen.add(n)
        if _is_to_local(n):
            return n
        if _is_prep(n):
            q.extend(n.all_input_nodes)
    return None


def shard_holder(gather: fx.Node) -> fx.Node | None:
    """Find the parameter's to_local through preparation, including dtype casts and shard padding."""
    for inp in gather.all_input_nodes:
        found = _find_shard_holder(inp)
        if found is not None:
            return found
    return None


def _is_local_extraction(node: fx.Node) -> bool:
    """Recognize both lowered to_local methods and unlowered prim_to_local calls."""
    return (node.op == "call_function" and getattr(node.target, "__name__", None) == "prim_to_local") or (
        node.op == "call_method" and node.target == "to_local"
    )


def _feeds_a_weight_gather(node: fx.Node) -> bool:
    """Find a downstream gather through preparation, H2D loads, and waits."""
    q: deque[fx.Node] = deque(node.users)
    seen: set[fx.Node] = set()
    while q:
        user = q.popleft()
        if user in seen:
            continue
        seen.add(user)
        if user.op == "call_function" and user.target is _ALL_GATHER:
            return True
        name = getattr(user.target, "__name__", "") or str(user.target)
        if _is_prep(user) or "h2d_load" in name or "wait_tensor" in name:
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


def _shard_candidate(*, holder, source, param, min_bytes, skipped, tag=None) -> OffloadCandidate | None:
    from ..runtime import host_weights

    local = param._local_tensor if isinstance(param, DTensor) else None
    # Reuse the adopted slot; each compiled graph still needs its own load.
    slot, why = parked_slot(local=local, min_bytes=min_bytes)
    if why is not None:
        skipped[why] += 1
        return None
    return OffloadCandidate(
        holder=holder,
        local=local,
        name=_param_name(source),
        nbytes=host_weights.slot_bytes(slot),
        tag=tag,
        slot=slot,
        group=mesh_group(param),
    )


@dataclass
class FsdpShardSource(WeightSource):
    """Find SimpleFSDP weight holders, including weights without an all-gather.

    Gathered shards can die at the gather; ungathered weights live through their readers.
    Each distinct holder needs a load, even when several holders share a host slot.
    Local bucket planning separates offloaded and resident gathers;
    rank 0's adopted plan may mix them on another rank."""

    def collect(
        self, graph: fx.GraphModule, placeholder_examples: Mapping[str, Any], min_bytes: int
    ) -> tuple[list[OffloadCandidate], Counter]:
        from magi_compiler.passes.fsdp_overlap.node_meta import is_weight_ag

        candidates: list[OffloadCandidate] = []
        skipped: Counter = Counter()
        seen: set[fx.Node] = set()

        for node in graph.graph.nodes:
            if node.op != "call_function" or not is_weight_ag(node):
                continue
            if node.target is not _ALL_GATHER:
                continue
            holder = shard_holder(node)
            if holder is None:
                skipped["gather does not reach a to_local(parameter)"] += 1
                continue
            param = _resolve(graph=graph, node=holder.args[0], placeholder_examples=placeholder_examples)
            if param is None:
                skipped["graph input has no live parameter behind it"] += 1
                continue

            # Deduplicate by holder: distinct holders need separate loads of a shared slot.
            if holder in seen:
                skipped["gather shares a holder with an earlier weight"] += 1
                continue
            candidate = _shard_candidate(
                holder=holder, source=holder.args[0], param=param, min_bytes=min_bytes, skipped=skipped, tag=node
            )
            if candidate is not None:
                seen.add(holder)
                candidates.append(candidate)

        self._collect_ungathered(graph, placeholder_examples, min_bytes, candidates, skipped, seen)
        return candidates, skipped

    @staticmethod
    def _collect_ungathered(graph, placeholder_examples, min_bytes, candidates, skipped, seen) -> None:
        """Collect weight holders without downstream gathers, including replicated weights."""
        for node in graph.graph.nodes:
            if node in seen or not _is_local_extraction(node) or not node.users:
                continue
            src = _weight_behind(node)
            if src is None or _feeds_a_weight_gather(node):
                continue
            param = _resolve(graph=graph, node=src, placeholder_examples=placeholder_examples)
            if not isinstance(param, DTensor):
                continue
            # This holder feeds no weight gather to tag for bucket planning.
            candidate = _shard_candidate(holder=node, source=src, param=param, min_bytes=min_bytes, skipped=skipped)
            if candidate is not None:
                seen.add(node)
                candidates.append(candidate)
