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

import torch
import torch._inductor.fx_passes.pre_grad

from ...magi_depyf.timeline import emit_pass_lifecycle
from ..pass_base import MagiInductorPass


class EliminateIdentityViewCastPass(MagiInductorPass):
    """
    Remove useless convert, view, reshape operations.
    When their input already has the target type and shape, these operations are redundant.
    """

    TARGET_METHODS = {
        "view",
        "reshape",
        "to",
        "type",
        "contiguous",
        "flatten",
        "permute",
        "transpose",
        "t",
        "unsqueeze",
        "squeeze",
        "expand",
        "repeat",
        "bfloat16",
        "float",
        "half",
        "int",
        "long",
        "short",
        "double",
        "bool",
        "byte",
    }

    @staticmethod
    def _get_tensor_info(node: torch.fx.Node):
        # Get tensor info from example_value
        if "example_value" in node.meta:
            val = node.meta["example_value"]
            if isinstance(val, torch.Tensor):
                return val.shape, val.dtype, val.stride()
            elif isinstance(val, (list, tuple)) and len(val) > 0 and isinstance(val[0], torch.Tensor):
                return val[0].shape, val[0].dtype, val[0].stride()

        return None, None, None

    def is_applicable(self, graph: torch.fx.Graph, shape: int | None = None) -> bool:
        for node in graph.nodes:
            if node.op == "call_method" and node.target in self.TARGET_METHODS:
                return True
        return False

    @emit_pass_lifecycle
    def __call__(self, graph: torch.fx.Graph):
        nodes_to_remove = []

        for node in graph.nodes:
            is_target_method = node.op == "call_method" and node.target in self.TARGET_METHODS
            if not is_target_method:
                continue

            # Need at least one argument (the input tensor)
            if not node.args or not isinstance(node.args[0], torch.fx.Node):
                continue

            input_node = node.args[0]

            node_shape, node_dtype, node_stride = self._get_tensor_info(node)
            input_shape, input_dtype, input_stride = self._get_tensor_info(input_node)
            if node_shape is None or input_shape is None:
                continue
            if node_dtype is None or input_dtype is None:
                continue
            # Some ops or metadata might not have stride properly captured,
            # but if they do, we should require them to match to be totally safe against contiguous-forcing ops.
            if node_stride is not None and input_stride is not None and node_stride != input_stride:
                continue

            # Check if shape and dtype match exactly
            if node_shape == input_shape and node_dtype == input_dtype:
                # For _to_copy, ensure we are not changing memory format or device or other properties implicitly,
                # but typically in full graph if shape and dtype match, and it's on the same device, it's safe.
                # Let's also check device just in case if it's available.
                def get_device(n):
                    if "example_value" in n.meta and isinstance(n.meta["example_value"], torch.Tensor):
                        return n.meta["example_value"].device

                node_device = get_device(node)
                input_device = get_device(input_node)
                if node_device is not None and input_device is not None and node_device != input_device:
                    continue

                # Replace uses
                node.replace_all_uses_with(input_node)
                nodes_to_remove.append(node)

        for node in nodes_to_remove:
            graph.erase_node(node)
