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
from magi_compiler.magi_backend import weight_transfer_pipeline
from magi_compiler.magi_backend.magi_backend import MagiBackend
from magi_compiler.passes.fsdp_overlap import FsdpOverlapReorder
from magi_compiler.utils.ordered_set import OrderedSet

_PASSES_KEY = "reorder_for_compute_comm_overlap_passes"


class _Add(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


def _apply_weight_transfer_pipeline(cfg: CompileConfig, monkeypatch) -> tuple[list, dict]:
    """Run the backend's weight pipeline on a tiny graph; return the lowered graphs and the Inductor config."""
    lowered: list = []
    monkeypatch.setattr(weight_transfer_pipeline, "lower_prim_redistribute_to_collectives", lowered.append)
    backend = MagiBackend(cfg, 0, "model", OrderedSet(), {})
    backend._apply_weight_transfer_pipeline(torch.fx.symbolic_trace(_Add()), (torch.zeros(2),))
    return lowered, backend.inductor_compile_config


def test_split_graph_skips_fsdp_overlap_when_graph_split_enabled(monkeypatch) -> None:
    cfg = CompileConfig(fsdp_config=FSDPConfig(enable_fsdp=True))
    assert cfg.disable_graph_split is False

    lowered, inductor_config = _apply_weight_transfer_pipeline(cfg, monkeypatch)

    assert lowered == []
    assert not any(isinstance(p, FsdpOverlapReorder) for p in inductor_config.get(_PASSES_KEY) or ())


def test_fullgraph_still_applies_fsdp_overlap(monkeypatch) -> None:
    cfg = CompileConfig(disable_graph_split=True, fsdp_config=FSDPConfig(enable_fsdp=True))

    lowered, inductor_config = _apply_weight_transfer_pipeline(cfg, monkeypatch)

    assert len(lowered) == 1
    assert any(isinstance(p, FsdpOverlapReorder) for p in inductor_config[_PASSES_KEY])
