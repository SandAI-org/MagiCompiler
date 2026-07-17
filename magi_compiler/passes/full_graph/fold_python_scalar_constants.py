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

"""Fold pure Python scalar expressions before piecewise graph splitting."""

import operator
from typing import Any

import torch
from torch.utils._pytree import tree_map

from ...magi_depyf.timeline import emit_pass_lifecycle
from ..pass_base import MagiInductorPass


class FoldPythonScalarConstantsPass(MagiInductorPass):
    """Replace allowlisted constant-only scalar FX nodes with Python literals.

    Dynamo may recapture a tensor-derived ``SymFloat`` as a Python constant but
    retain scalar arithmetic such as ``operator.neg(448.0)`` as an FX node.  If
    that value is reused after a piecewise boundary, ``split_module`` promotes
    the node to a subgraph output.  Inductor does not accept a bare Python float
    as a compiled graph output.

    Only explicitly allowlisted pure operators are evaluated, and every input
    must already be a plain Python scalar.  Tensor and symbolic values are never
    folded.
    """

    _SCALAR_TYPES = (bool, int, float, complex)
    _PURE_SCALAR_OPS = {
        operator.abs,
        operator.add,
        operator.floordiv,
        operator.mod,
        operator.mul,
        operator.neg,
        operator.pos,
        operator.pow,
        operator.sub,
        operator.truediv,
    }

    def is_applicable(self, graph: torch.fx.Graph, shape: int | None = None) -> bool:
        return any(node.op == "call_function" and node.target in self._PURE_SCALAR_OPS for node in graph.nodes)

    @classmethod
    def _is_plain_scalar_tree(cls, value: Any) -> bool:
        if isinstance(value, cls._SCALAR_TYPES):
            return True
        if isinstance(value, (tuple, list)):
            return all(cls._is_plain_scalar_tree(item) for item in value)
        if isinstance(value, dict):
            return all(isinstance(key, str) and cls._is_plain_scalar_tree(item) for key, item in value.items())
        return False

    @emit_pass_lifecycle
    def __call__(self, graph: torch.fx.Graph) -> None:
        folded_values: dict[torch.fx.Node, bool | int | float | complex] = {}

        def replace_folded(value: Any) -> Any:
            if isinstance(value, torch.fx.Node):
                return folded_values.get(value, value)
            return value

        for node in list(graph.nodes):
            if node.op != "call_function" or node.target not in self._PURE_SCALAR_OPS:
                continue

            args = tree_map(replace_folded, node.args)
            kwargs = tree_map(replace_folded, node.kwargs)
            if not self._is_plain_scalar_tree(args) or not self._is_plain_scalar_tree(kwargs):
                continue

            try:
                result = node.target(*args, **kwargs)
            except (ArithmeticError, TypeError, ValueError):
                continue
            if not isinstance(result, self._SCALAR_TYPES):
                continue

            folded_values[node] = result

        if not folded_values:
            return

        for node in graph.nodes:
            node.args = tree_map(replace_folded, node.args)
            node.kwargs = tree_map(replace_folded, node.kwargs)

        graph.eliminate_dead_code()
        graph.lint()
