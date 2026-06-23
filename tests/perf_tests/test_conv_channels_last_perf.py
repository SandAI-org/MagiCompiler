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

"""Performance test: conv channels-last layout pass under static shapes.

Background
----------
cuDNN's channels-last (NHWC/NDHWC) conv kernels beat contiguous NC(D)HW on
Ampere+. ``ConvChannelsLastPass`` rewrites the post-grad ATen graph so every
``aten.convolution`` (conv2d/conv3d) consumes channels-last inputs/weights,
letting cuDNN pick those kernels; it sets ``layout_optimization=False`` so layout
is owned entirely by the pass. The pass applies only when the post-grad graph is
static AND conv-heavy (the regime where channels-last pays off and its transpose
doesn't tile badly); see
``magi_compiler.passes.piecewise_graph.conv_channels_last.ConvChannelsLastPass``.

This test exercises a WAN-2.2-VAE-decode-like workload (stacked 3D conv resblocks
+ spatial upsampling) compiled with **static shapes** — a static, conv-heavy graph
that triggers the pass — and checks that magi_compile beats vanilla
``torch.compile`` on that path.

Real WAN 2.2 VAE decode (540p, static shapes) numbers that motivate this:
  - with conv channels-last layout: 520ms -> 430ms / decode (**~1.2x speedup**)
This synthetic, weightless decoder reproduces this regime. The absolute ratio is
GPU-dependent, so CONV_CHANNELS_LAST_SPEEDUP_THRESHOLD is set to a conservative
lower bound of 1.20 that still proves a clear, non-noise win.
"""

import pytest
import torch

from magi_compiler import magi_compile
from tests.model_definition import VAEDecoderLike
from tests.perf_tests import cuda_benchmark, print_perf_comparison
from tests.perf_tests.utils import assert_magi_vs_torch

# WAN 2.2 VAE 540p latent: [C, T, H, W]; compiled with static shapes here.
LATENT_C, LATENT_T, LATENT_H, LATENT_W = 48, 7, 34, 60

# magi_compile (channels-last on) vs vanilla torch.compile, both on the static path.
# Real 540p decode lands ~1.2x; assert a conservative lower bound (calibrated GPUs only).
CONV_CHANNELS_LAST_SPEEDUP_THRESHOLD = 1.20


@pytest.fixture(scope="function")
def decoder_input(device):
    return torch.randn(1, LATENT_C, LATENT_T, LATENT_H, LATENT_W, device=device, dtype=torch.bfloat16)


def _compile_decoder(device: torch.device):
    def _patch(cfg):
        # This decoder is static + conv-dense, so the pass's heuristics fire and
        # channels-last is applied. The pass mutates layout_optimization only inside
        # the compilation's config.patch scope, so nothing leaks to the torch baseline.
        cfg.pass_config.enable_conv_channels_last = True
        return cfg

    model = VAEDecoderLike().to(device).to(torch.bfloat16).eval()
    # Empty dims => fully static; the pass forces channels-last without dynamic shapes.
    return magi_compile(model, dynamic_arg_dims={"z": []}, config_patch=_patch)


def _compile_torch(device: torch.device):
    model = VAEDecoderLike().to(device).to(torch.bfloat16).eval()
    return torch.compile(model, fullgraph=True, backend="inductor")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support")
def test_conv_channels_last_vs_torch_compile(device, decoder_input):
    """Channels-last pass ON should beat vanilla torch.compile on the static path."""
    # Build isolated inputs so the three paths don't share state.
    eager_input = decoder_input.clone()
    magi_input = decoder_input.clone()
    torch_input = decoder_input.clone()

    eager_model = VAEDecoderLike().to(device).to(torch.bfloat16).eval()
    magi_compiled = _compile_decoder(device)
    torch_compiled = _compile_torch(device)

    with torch.no_grad():
        eager_result = cuda_benchmark(lambda: eager_model(eager_input))
        torch_result = cuda_benchmark(lambda: torch_compiled(torch_input), compilation_warmup=3)
        magi_result = cuda_benchmark(lambda: magi_compiled(magi_input), compilation_warmup=3)

    speedup = torch_result.median / magi_result.median
    print_perf_comparison(
        "Conv channels-last: magi_compile vs torch.compile (static shapes)",
        eager_result,
        magi_result,
        torch_result,
        extra_info=(f"latent=({LATENT_C}, {LATENT_T}, {LATENT_H}, {LATENT_W})  " f"speedup(torch/magi)={speedup:.2f}x"),
    )

    assert_magi_vs_torch(
        speedup, torch_result, magi_result, label="Conv channels-last", threshold=CONV_CHANNELS_LAST_SPEEDUP_THRESHOLD
    )
