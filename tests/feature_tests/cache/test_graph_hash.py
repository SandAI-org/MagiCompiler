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

"""compute_graph_hash must change when input dtypes change; placements are covered by test_placement_cache_key."""

from __future__ import annotations

import torch
import torch.nn as nn

from magi_compiler.utils import compute_graph_hash


class _Add(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


def test_compute_graph_hash_changes_with_input_dtype() -> None:
    graph = torch.fx.symbolic_trace(_Add())
    fp32 = compute_graph_hash(graph, [torch.zeros(2, dtype=torch.float32)])
    bf16 = compute_graph_hash(graph, [torch.zeros(2, dtype=torch.bfloat16)])
    assert fp32 != bf16
    assert fp32 == compute_graph_hash(graph, [torch.ones(8, dtype=torch.float32)])
