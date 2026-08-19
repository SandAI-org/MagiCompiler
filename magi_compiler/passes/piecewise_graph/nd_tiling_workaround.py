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

from ...magi_depyf.timeline import emit_pass_lifecycle
from ...utils.envs import IS_PT_212
from ..pass_base import MagiInductorPass


class ND_TilingWorkaroundPass(MagiInductorPass):
    inductor_config_keys_potentially_mutated_by_this_pass = (
        "triton.prefer_nd_tiling",
        "triton.max_tiles",
        "triton.tile_reductions",
    )

    @emit_pass_lifecycle
    def __call__(self, graph: torch.fx.Graph):
        if not self.is_dynamic(graph) or not self.is_conv_heavy(graph):
            return False

        # Inductor's coalesce tiling analysis bails out on symbolic numels, so
        # dynamic-shape transpose/permute/channels-last kernels degrade to untiled Grid1D.
        # Forcing prefer_nd_tiling restores ND tiling.
        torch._inductor.config.triton.prefer_nd_tiling = True
        torch._inductor.config.triton.tile_reductions = True

        # max_tiles=3 causes two known issues:
        # - PT 2.12+: invalid 3D-grid reduction kernels (program_id(2) mapped
        #   to a non-existent grid dim).
        # - All versions: conv-heavy dynamic-shape graphs (e.g. turbo VAE at
        #   1080p) can overflow CUDA's z-grid limit (65535) when Inductor's
        #   coalesce_tiling_analysis collapses high-dim tensors into 3D.
        # max_tiles=3 is documented as "experimental and may have bugs".
        torch._inductor.config.triton.max_tiles = 2
