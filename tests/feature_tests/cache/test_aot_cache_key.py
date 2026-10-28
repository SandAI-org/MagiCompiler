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

"""The AOT cache key tells weight layouts apart under one model_tag: an AOT load drops every guard."""

from __future__ import annotations

from pathlib import Path

import torch

from magi_compiler.config import CompileConfig
from magi_compiler.magi_backend.magi_compiler_base import MagiCompileState


class _Stack(torch.nn.Module):
    def __init__(self, num_layers: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.layers = torch.nn.ModuleList(torch.nn.Linear(8, 8, dtype=dtype) for _ in range(num_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _aot_path(module: torch.nn.Module, cache_root: Path) -> str:
    config = CompileConfig(cache_root_dir=str(cache_root))
    return MagiCompileState(
        module, config, model_tag="stack", dynamic_arg_dims={"x": 0}, target_method_name="forward"
    ).aot_compilation_path


def test_the_same_layout_shares_the_aot_artifact(tmp_path: Path) -> None:
    assert _aot_path(_Stack(2), tmp_path) == _aot_path(_Stack(2), tmp_path)


def test_another_layer_count_gets_its_own_aot_artifact(tmp_path: Path) -> None:
    assert _aot_path(_Stack(2), tmp_path) != _aot_path(_Stack(3), tmp_path)


def test_another_weight_dtype_gets_its_own_aot_artifact(tmp_path: Path) -> None:
    assert _aot_path(_Stack(2), tmp_path) != _aot_path(_Stack(2, dtype=torch.bfloat16), tmp_path)
