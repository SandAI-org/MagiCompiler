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

"""compute_graph_hash tells weight layouts apart, and ignores what a restart retrace or an AOT load changes."""

from __future__ import annotations

import torch
import torch.fx as fx

from magi_compiler.utils import compute_graph_hash

WEIGHT = "l_self_modules_layers_modules_{}_parameters_weight_"


def _hash(**inputs) -> str:
    graph = fx.Graph()
    for name in inputs:
        graph.placeholder(name)
    graph.output(None)
    return compute_graph_hash(fx.GraphModule(torch.nn.Module(), graph), list(inputs.values()))


def _layers(n: int, dtype: torch.dtype = torch.bfloat16, shape: tuple[int, ...] = (8, 8)) -> dict[str, torch.Tensor]:
    return {WEIGHT.format(i): torch.zeros(shape, dtype=dtype) for i in range(n)}


def test_the_restart_retrace_and_the_aot_load_hash_the_same() -> None:
    """The retrace drops the 0-d tensor that carried a Python float (a layer norm's eps, say); the AOT compile sees a
    concrete size where the load sees a symbolic one."""
    eps = torch.tensor(1e-5, dtype=torch.float64)
    first = _hash(l_x_=torch.zeros(4, 8), **_layers(2), l_self_eps=eps, s0=3)
    assert first == _hash(l_x_=torch.zeros(16, 8), **_layers(2), s0=3)


def test_another_layer_count_hashes_differently() -> None:
    assert _hash(**_layers(2)) != _hash(**_layers(3))


def test_another_weight_layout_hashes_differently() -> None:
    assert _hash(**_layers(2)) != _hash(**_layers(2, dtype=torch.float8_e4m3fn))
    assert _hash(**_layers(2)) != _hash(**_layers(2, shape=(8, 4)))
