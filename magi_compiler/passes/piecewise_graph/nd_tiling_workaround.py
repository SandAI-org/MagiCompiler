# Copyright (c) 2025 SandAI. All Rights Reserved.
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

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.torch_version import TorchVersion

from ...magi_depyf.timeline import emit_pass_lifecycle
from ..pass_base import MagiInductorPass


class ND_TilingWorkaroundPass(MagiInductorPass):
    inductor_config_keys_potentially_mutated_by_this_pass = (
        "triton.prefer_nd_tiling",
        "triton.max_tiles",
        "triton.tile_reductions",
    )

    def __init__(self):
        super().__init__()
        self.is_target_torch_version = TorchVersion(torch.__version__) < (2, 11, 0)

    @emit_pass_lifecycle
    def __call__(self, graph: torch.fx.Graph):
        if not self.is_target_torch_version or not self.is_dynamic(graph) or not self.is_conv_heavy(graph):
            return False

        # On PyTorch < 2.11.0, Inductor's coalesce tiling analysis bails out on
        # symbolic numels, so dynamic-shape transpose/permute/channels-last kernels
        # degrade to untiled Grid1D. Forcing prefer_nd_tiling restores ND tiling
        # (WAN 2.2 VAE 540p decode: ~1.45x).
        torch._inductor.config.triton.prefer_nd_tiling = True
        torch._inductor.config.triton.max_tiles = 3
        torch._inductor.config.triton.tile_reductions = True
