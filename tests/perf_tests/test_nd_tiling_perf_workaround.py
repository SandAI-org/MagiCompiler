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

"""Performance test: Triton ND-tiling workaround under dynamic shapes.

Background
----------
On PyTorch < 2.11.0, Inductor's coalesce tiling analysis bails out on symbolic
numels (``tiling_utils.extract_normalized_read_writes`` returns ``None``), so
transpose/permute/channels-last pointwise kernels in a dynamic-shape graph
degrade to untiled Grid1D. ``ND_TilingWorkaroundPass`` works around this by
enabling ``triton.prefer_nd_tiling`` (+ ``max_tiles=3`` + ``tile_reductions``)
when the post-grad graph is dynamic AND conv-heavy (the regime where the
degraded kernels dominate); see
``magi_compiler.passes.piecewise_graph.nd_tiling_workaround.ND_TilingWorkaroundPass``.

This test exercises a WAN-2.2-VAE-decode-like workload (stacked 3D conv resblocks
+ spatial upsampling) compiled with **dynamic H/W** — a dynamic, conv-heavy graph
that triggers the pass — and checks that magi_compile beats vanilla
``torch.compile`` on that path.

Real WAN 2.2 VAE decode (540p, dynamic H/W) numbers that motivate this:
  - with conv channels-last layout: 1.252s -> 542ms / decode (~2.3x)
  - without conv channels-last:       770ms -> 535ms / decode (~1.44x)
This synthetic decoder (no weights, no conv channels-last pass) reproduces the
"~1.4x" regime. The absolute ratio is GPU-dependent, so ND_TILING_SPEEDUP_THRESHOLD
is set to a conservative lower bound that still proves a clear, non-noise win.
"""

import pytest
import torch

from magi_compiler import magi_compile
from tests.model_definition import VAEDecoderLike
from tests.perf_tests import cuda_benchmark, print_perf_comparison
from tests.perf_tests.utils import assert_magi_vs_torch

# WAN 2.2 VAE 540p latent: [C, T, H, W]; dynamic dims are H and W.
LATENT_C, LATENT_T, LATENT_H, LATENT_W = 48, 7, 34, 60

# magi_compile (workaround on) vs vanilla torch.compile, both on the dynamic path.
# Observed ~1.36x (torch=2.20ms -> magi=1.63ms) on H100; assert a conservative
# lower bound that still proves a clear, non-noise win.
ND_TILING_SPEEDUP_THRESHOLD = 1.20


@pytest.fixture(scope="function")
def decoder_input(device):
    return torch.randn(1, LATENT_C, LATENT_T, LATENT_H, LATENT_W, device=device, dtype=torch.bfloat16)


def _compile_decoder(device: torch.device):
    def _patch(cfg):
        # This decoder is dynamic + conv-heavy, so the pass's heuristics fire and
        # the workaround is applied. The pass mutates triton configs only inside the
        # compilation's config.patch scope, so nothing leaks to the torch baseline.
        cfg.pass_config.enable_nd_tiling_workaround = True
        return cfg

    model = VAEDecoderLike().to(device).to(torch.bfloat16).eval()
    # Dynamic H, W (latent dims 3, 4) — the regime where coalesce tiling bails out.
    return magi_compile(model, dynamic_arg_dims={"z": [3, 4]}, config_patch=_patch)


def _compile_torch(device: torch.device):
    model = VAEDecoderLike().to(device).to(torch.bfloat16).eval()
    return torch.compile(model, backend="inductor")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support")
def test_nd_tiling_workaround_speedup(device, decoder_input):
    """ND-tiling ON should beat vanilla torch.compile on the dynamic path."""
    # Build isolated inputs to prevent dynamic shape marking leakage
    eager_input = decoder_input.clone()
    magi_input = decoder_input.clone()
    torch_input = decoder_input.clone()

    # Explicitly mark dynamic dimensions for the vanilla torch.compile environment
    torch._dynamo.mark_dynamic(torch_input, [3, 4])

    eager_model = VAEDecoderLike().to(device).to(torch.bfloat16).eval()
    magi_compiled = _compile_decoder(device)
    torch_compiled = _compile_torch(device)

    with torch.no_grad():
        eager_result = cuda_benchmark(lambda: eager_model(eager_input))
        torch_result = cuda_benchmark(lambda: torch_compiled(torch_input), compilation_warmup=3)
        magi_result = cuda_benchmark(lambda: magi_compiled(magi_input), compilation_warmup=3)

    speedup = torch_result.median / magi_result.median
    print_perf_comparison(
        "Dynamic ND-tiling: magi_compile vs torch.compile (dynamic H/W)",
        eager_result,
        magi_result,
        torch_result,
        extra_info=(f"latent=({LATENT_C}, {LATENT_T}, {LATENT_H}, {LATENT_W})  " f"speedup(torch/magi)={speedup:.2f}x"),
    )

    assert_magi_vs_torch(
        speedup, torch_result, magi_result, label="Dynamic ND-tiling workaround", threshold=ND_TILING_SPEEDUP_THRESHOLD
    )
