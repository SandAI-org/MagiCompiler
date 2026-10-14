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

"""compute_graph_hash must stay stable across TensorifyScalarRestartAnalysis retrace."""

from __future__ import annotations

import torch
import torch.nn as nn

from magi_compiler.utils import compute_graph_hash


class _Add(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


def test_compute_graph_hash_stable_across_tensorify_scalar() -> None:
    """Restart-analysis retrace can replace a Tensor with a Python scalar.

    Including those types in the cache key makes the second pass miss first-pass artifacts.
    """
    graph = torch.fx.symbolic_trace(_Add())
    tensor_inputs = compute_graph_hash(graph, [torch.zeros(2, dtype=torch.float32)])
    scalar_inputs = compute_graph_hash(graph, [1])
    assert tensor_inputs == scalar_inputs
    assert tensor_inputs == compute_graph_hash(graph, [torch.zeros(2, dtype=torch.bfloat16)])
