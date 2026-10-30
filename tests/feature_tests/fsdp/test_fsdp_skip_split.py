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

"""enable_fsdp is process-wide; split-graph models (VAE) must not enter fullgraph overlap."""

from __future__ import annotations

import torch
import torch.nn as nn

from magi_compiler.config import CompileConfig, FSDPConfig
from magi_compiler.magi_backend.magi_backend import MagiBackend
from magi_compiler.utils.ordered_set import OrderedSet


class _Add(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


def test_split_graph_skips_fsdp_overlap_when_graph_split_enabled(monkeypatch) -> None:
    called: list[bool] = []
    cfg = CompileConfig(fsdp_config=FSDPConfig(enable_fsdp=True))
    assert cfg.disable_graph_split is False
    backend = MagiBackend(cfg, "vae", OrderedSet(), {})
    monkeypatch.setattr(backend, "_apply_fsdp_fullgraph_overlap", lambda *a, **k: called.append(True))
    graph = torch.fx.symbolic_trace(_Add())
    backend._split_graph(graph, (torch.zeros(2),))
    assert called == []


def test_fullgraph_still_applies_fsdp_overlap(monkeypatch) -> None:
    called: list[bool] = []
    cfg = CompileConfig(disable_graph_split=True, fsdp_config=FSDPConfig(enable_fsdp=True))
    backend = MagiBackend(cfg, "base", OrderedSet(), {})
    monkeypatch.setattr(backend, "_apply_fsdp_fullgraph_overlap", lambda *a, **k: called.append(True))
    graph = torch.fx.symbolic_trace(_Add())
    backend._split_graph(graph, (torch.zeros(2),))
    assert called == [True]
