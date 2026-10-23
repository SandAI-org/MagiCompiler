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

from ...snode_utils import ALL_GATHER_OP
from ..runtime.h2d_op import H2D_LOAD

aten = torch.ops.aten

# Producers we walk through when tracing a value back to the parameter behind it.
_PREP_METHODS = frozenset({"to_local", "contiguous", "to", "view", "reshape"})
# Overload packets, so every overload of a listed op matches and nothing else does.
_PREP_FUNCTIONS = frozenset(
    {
        aten.constant_pad_nd,
        aten._to_copy,
        aten.view,
        aten.reshape,
        aten.clone,
        torch.ops.prims.convert_element_type,
        torch.reshape,
        torch.clone,
    }
)
_LOAD_OR_WAIT = frozenset({H2D_LOAD.overloadpacket, torch.ops._c10d_functional.wait_tensor})


def _function_key(target: Any) -> Any:
    return target.overloadpacket if isinstance(target, torch._ops.OpOverload) else target


def _is_prep(node: fx.Node) -> bool:
    """A cheap reshaping producer that a weight's value may pass through."""
    if node.op == "call_method":
        return node.target in _PREP_METHODS
    if node.op == "call_function":
        return _function_key(node.target) in _PREP_FUNCTIONS
    return False


def _parameter_name_from_node(node: fx.Node) -> str:
    raw = str(node.target or node.name or "?")
    parts = [p for p in raw.strip("_").split("_") if p and p not in ("L", "self", "modules", "parameters", "parameter")]
    return ".".join(parts) or raw


def _resolve_node_value(graph: fx.GraphModule, node: fx.Node, placeholder_examples: Mapping[str, Any]) -> Any:
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


def parameter_mesh_group(param: object):
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

    weight_node: fx.Node  # the node whose output is the weight; the load goes right after it
    device_standin: Any  # the live stand-in the slot is keyed on
    parameter_name: str  # the parameter it came from, for the placement logs
    weight_bytes: int
    host_slot: int  # host-pool slot from bind_host_buffer
    offload_tag_node: fx.Node | None = None  # node to mark offloaded, so later passes can tell
    process_group: Any = None  # ProcessGroup of the DTensor mesh; None = no cross-rank vote


class WeightSource(Protocol):
    """How to find offloadable weights in one flavour of graph."""

    def collect_candidates(
        self, graph: fx.GraphModule, placeholder_examples: Mapping[str, Any]
    ) -> tuple[list[OffloadCandidate], Counter]:
        """Every candidate, plus counts of why other weights were skipped."""


def lookup_host_slot(device_standin: Any) -> tuple[int | None, str | None]:
    """The host-pool slot for ``device_standin``, or ``(None, skip_reason)`` if it cannot be loaded.

    A weight reaches the pool only through host-first materialization, which
    already applied the size floor.  Collect never copies a resident shard off
    the device.
    """
    from ..runtime.host_weights import get_global_host_weight_store

    if not isinstance(device_standin, torch.Tensor):
        return None, "graph input is not a tensor"
    host_slot = get_global_host_weight_store().slot_for_tensor(device_standin)
    if host_slot is None:
        return None, "weight was not materialized in host memory"
    return host_slot, None


@dataclass
class PlainParamSource(WeightSource):
    """Discover registered Parameters behind placeholders and get_attr nodes."""

    def collect_candidates(
        self, graph: fx.GraphModule, placeholder_examples: Mapping[str, Any]
    ) -> tuple[list[OffloadCandidate], Counter]:
        from ..runtime.host_weights import get_global_host_weight_store

        candidates: list[OffloadCandidate] = []
        skip_counts: Counter = Counter()

        for node in graph.graph.nodes:
            if node.op not in ("placeholder", "get_attr"):
                continue
            parameter_value = _resolve_node_value(graph=graph, node=node, placeholder_examples=placeholder_examples)
            if parameter_value is None:
                continue
            if not isinstance(parameter_value, torch.nn.Parameter):
                # Activations and buffers are graph inputs too; only weights are
                # worth moving, because only they are the same every forward.
                continue
            if not node.users:
                continue

            host_slot, skip_reason = lookup_host_slot(parameter_value)
            if skip_reason is not None:
                skip_counts[skip_reason] += 1
                continue
            # Each placeholder needs a load, even when tied weights share one host slot.
            candidates.append(
                OffloadCandidate(
                    weight_node=node,
                    device_standin=parameter_value,
                    parameter_name=_parameter_name_from_node(node),
                    weight_bytes=get_global_host_weight_store().slot_bytes(host_slot),
                    offload_tag_node=None,
                    host_slot=host_slot,
                )
            )
        return candidates, skip_counts


def _is_to_local(node: fx.Node) -> bool:
    if not (node.op == "call_method" and node.target == "to_local"):
        return False
    owner = node.args[0] if node.args else None
    return isinstance(owner, fx.Node) and owner.op in ("placeholder", "get_attr")


def _find_input_to_local(node: fx.Node) -> fx.Node | None:
    pending_nodes: deque[fx.Node] = deque([node])
    visited_nodes: set[fx.Node] = set()
    while pending_nodes:
        node_to_visit = pending_nodes.popleft()
        if node_to_visit in visited_nodes:
            continue
        visited_nodes.add(node_to_visit)
        if _is_to_local(node_to_visit):
            return node_to_visit
        if _is_prep(node_to_visit):
            pending_nodes.extend(node_to_visit.all_input_nodes)
    return None


def find_weight_node(gather: fx.Node) -> fx.Node | None:
    """Find the parameter's to_local through preparation, including dtype casts and shard padding."""
    for inp in gather.all_input_nodes:
        found = _find_input_to_local(inp)
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
    pending_users: deque[fx.Node] = deque(node.users)
    visited_users: set[fx.Node] = set()
    while pending_users:
        user = pending_users.popleft()
        if user in visited_users:
            continue
        visited_users.add(user)
        if user.op == "call_function" and user.target is ALL_GATHER_OP:
            return True
        if _is_prep(user) or (user.op == "call_function" and _function_key(user.target) in _LOAD_OR_WAIT):
            pending_users.extend(user.users)
    return False


def _find_parameter_source(to_local: fx.Node) -> fx.Node | None:
    """The placeholder or get_attr a ``to_local`` reads, through its redistribute."""
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
    return src


def _build_shard_candidate(
    *, weight_node, parameter_node, parameter_value, skip_counts, offload_tag_node=None
) -> OffloadCandidate | None:
    from ..runtime.host_weights import get_global_host_weight_store

    # Only weights.  An activation that happens to be a DTensor changes every
    # forward, so parking it would serve stale bytes that no later check notices.
    if not isinstance(parameter_value, torch.nn.Parameter):
        skip_counts["graph input is not a parameter"] += 1
        return None
    local_standin = parameter_value._local_tensor if isinstance(parameter_value, DTensor) else None
    # Reuse the adopted slot; each compiled graph still needs its own load.
    host_slot, skip_reason = lookup_host_slot(device_standin=local_standin)
    if skip_reason is not None:
        skip_counts[skip_reason] += 1
        return None
    return OffloadCandidate(
        weight_node=weight_node,
        device_standin=local_standin,
        parameter_name=_parameter_name_from_node(parameter_node),
        weight_bytes=get_global_host_weight_store().slot_bytes(host_slot),
        offload_tag_node=offload_tag_node,
        host_slot=host_slot,
        process_group=parameter_mesh_group(parameter_value),
    )


@dataclass
class FsdpShardSource(WeightSource):
    """Find SimpleFSDP weight holders, including weights without an all-gather.

    Gathered shards can die at the gather; ungathered weights live through their readers.
    Each distinct holder needs a load, even when several holders share a host slot.
    Local bucket planning separates offloaded and resident gathers;
    rank 0's adopted plan may mix them on another rank."""

    def collect_candidates(
        self, graph: fx.GraphModule, placeholder_examples: Mapping[str, Any]
    ) -> tuple[list[OffloadCandidate], Counter]:
        from magi_compiler.passes.fsdp_overlap.node_meta import is_weight_ag

        candidates: list[OffloadCandidate] = []
        skip_counts: Counter = Counter()
        seen_weight_nodes: set[fx.Node] = set()

        for node in graph.graph.nodes:
            if node.op != "call_function" or not is_weight_ag(node):
                continue
            if node.target is not ALL_GATHER_OP:
                continue
            weight_node = find_weight_node(node)
            if weight_node is None:
                skip_counts["gather does not reach a to_local(parameter)"] += 1
                continue
            parameter_value = _resolve_node_value(
                graph=graph, node=weight_node.args[0], placeholder_examples=placeholder_examples
            )
            if parameter_value is None:
                skip_counts["graph input has no live parameter behind it"] += 1
                continue

            # Deduplicate by holder: distinct holders need separate loads of a shared slot.
            if weight_node in seen_weight_nodes:
                skip_counts["gather shares a holder with an earlier weight"] += 1
                continue
            candidate = _build_shard_candidate(
                weight_node=weight_node,
                parameter_node=weight_node.args[0],
                parameter_value=parameter_value,
                skip_counts=skip_counts,
                offload_tag_node=node,
            )
            if candidate is not None:
                seen_weight_nodes.add(weight_node)
                candidates.append(candidate)

        self._append_ungathered_candidates(
            graph=graph,
            placeholder_examples=placeholder_examples,
            candidates=candidates,
            skip_counts=skip_counts,
            seen_weight_nodes=seen_weight_nodes,
        )
        return candidates, skip_counts

    @staticmethod
    def _append_ungathered_candidates(graph, placeholder_examples, candidates, skip_counts, seen_weight_nodes) -> None:
        """Collect weight holders without downstream gathers, including replicated weights."""
        for node in graph.graph.nodes:
            if node in seen_weight_nodes or not _is_local_extraction(node) or not node.users:
                continue
            parameter_node = _find_parameter_source(node)
            if parameter_node is None or _feeds_a_weight_gather(node):
                continue
            parameter_value = _resolve_node_value(graph=graph, node=parameter_node, placeholder_examples=placeholder_examples)
            if not isinstance(parameter_value, DTensor):
                continue
            # This holder feeds no weight gather to tag for bucket planning.
            candidate = _build_shard_candidate(
                weight_node=node, parameter_node=parameter_node, parameter_value=parameter_value, skip_counts=skip_counts
            )
            if candidate is not None:
                seen_weight_nodes.add(node)
                candidates.append(candidate)
