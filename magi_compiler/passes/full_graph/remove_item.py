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

import operator

import torch

from ...magi_depyf.timeline import emit_pass_lifecycle
from ..pass_base import MagiInductorPass


class RemoveItemPass(MagiInductorPass):
    """
    Remove item() calls in the pattern: Tensor(placeholder) -> item() -> binary_op.

    When a graph input (placeholder) tensor is consumed only via item() and then
    fed into a supported binary op, this pass removes the item() indirection and
    lets the binary op take the tensor directly. This avoids the
    aten._local_scalar_dense lowering issue and keeps computation in tensor land.

    Reference: torch/fx/passes/_tensorify_python_scalars.py
    """

    SUPPORTED_OPS = {
        # ATen-level ops (post-AOTAutograd graphs)
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.div.Tensor,
        torch.ops.aten.gt.Scalar,
        torch.ops.aten.lt.Scalar,
        torch.ops.aten.ge.Scalar,
        torch.ops.aten.le.Scalar,
        torch.ops.aten.eq.Scalar,
        torch.ops.aten.ne.Scalar,
        # Symbolic scalar conversion (e.g. float(scalar_tensor))
        torch.sym_float,
        torch.sym_int,
        # Python-level operators (Dynamo-traced graphs)
        operator.add,
        operator.mul,
        operator.sub,
        operator.truediv,
        operator.gt,
        operator.lt,
        operator.ge,
        operator.le,
        operator.eq,
        operator.ne,
    }

    # Scalar variant -> Tensor variant (needed when item() removal
    # turns a scalar arg into a tensor arg)
    SCALAR_TO_TENSOR_OPS = {
        torch.ops.aten.gt.Scalar: torch.ops.aten.gt.Tensor,
        torch.ops.aten.lt.Scalar: torch.ops.aten.lt.Tensor,
        torch.ops.aten.ge.Scalar: torch.ops.aten.ge.Tensor,
        torch.ops.aten.le.Scalar: torch.ops.aten.le.Tensor,
        torch.ops.aten.eq.Scalar: torch.ops.aten.eq.Tensor,
        torch.ops.aten.ne.Scalar: torch.ops.aten.ne.Tensor,
    }

    @staticmethod
    def _is_item_node(node: torch.fx.Node) -> bool:
        if node.op == "call_method" and node.target == "item":
            return True
        if node.op == "call_function" and node.target is torch.ops.aten._local_scalar_dense.default:
            return True
        return False

    def is_applicable(self, graph: torch.fx.Graph, shape: int | None = None) -> bool:
        for node in graph.nodes:
            if self._is_item_node(node):
                input_node = node.args[0]
                if isinstance(input_node, torch.fx.Node) and input_node.op == "placeholder":
                    return True
        return False

    @emit_pass_lifecycle
    def __call__(self, graph: torch.fx.Graph):
        nodes_to_remove = []

        for node in list(graph.nodes):
            if not self._is_item_node(node):
                continue

            input_node = node.args[0]
            if not isinstance(input_node, torch.fx.Node) or input_node.op != "placeholder":
                continue

            # Allow removal if all users are call_function with supported ops,
            # OR if all users are call_function (any target) — custom ops like
            # athena.gaga4_fa_with_sink_cp accept scalar tensors transparently.
            can_remove = all(
                user.op == "call_function" and (
                    user.target in self.SUPPORTED_OPS
                    or hasattr(user.target, '__module__')  # custom op / torch op
                )
                for user in node.users
            )
            if not can_remove:
                continue

            original_users = list(node.users.keys())
            node.replace_all_uses_with(input_node)

            for user in original_users:
                if user.target in self.SCALAR_TO_TENSOR_OPS:
                    user.target = self.SCALAR_TO_TENSOR_OPS[user.target]

            nodes_to_remove.append(node)

        for node in nodes_to_remove:
            graph.erase_node(node)
