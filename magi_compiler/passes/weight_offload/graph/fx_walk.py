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

"""Tracing a graph value back to the parameter behind it."""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Mapping

import torch.fx as fx

# Producers we walk through when tracing a value back to the parameter behind it.
_PREP_METHODS = {"to_local", "contiguous", "to", "view", "reshape"}
_PREP_FUNCTIONS = ("constant_pad_nd", "_to_copy", "convert_element_type", "view", "reshape", "clone")


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
