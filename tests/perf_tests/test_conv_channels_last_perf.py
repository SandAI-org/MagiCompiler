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

"""Performance test: conv channels-last layout pass.

cuDNN's channels-last (NHWC/NDHWC) conv kernels beat contiguous NC(D)HW on
Ampere+. ``ConvChannelsLastPass`` forces channels-last at every
``aten.convolution`` boundary so cuDNN picks those kernels.

This test runs a WAN-2.2-VAE-decode-like workload (stacked 3D conv resblocks +
spatial upsampling) with static shapes and compares ``magi_compile`` against
stock ``torch.compile``. The real win needs a weighted model with realistic
channel counts (real 540p decode: 520ms -> 430ms ~1.2x speedup); this synthetic, weightless
decoder doesn't fully reproduce that regime, so the assertion only checks
magi_compile stays at least on par with torch.compile (MAGI_VS_TORCH parity
bound, calibrated GPUs only).
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from magi_compiler import magi_compile
from tests.perf_tests import cuda_benchmark, print_perf_comparison
from tests.perf_tests.utils import assert_magi_vs_torch

# WAN 2.2 VAE 540p latent: [C, T, H, W].
LATENT_C, LATENT_T, LATENT_H, LATENT_W = 48, 7, 34, 60
BASE_CHANNELS = 128


class _ResBlock3D(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, cin)
        self.conv1 = nn.Conv3d(cin, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, padding=1)
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class VAEDecoderLike(nn.Module):
    """Stacked 3D conv resblocks + spatial upsampling, mimicking VAE decode."""

    def __init__(self, zc: int = LATENT_C, base: int = BASE_CHANNELS):
        super().__init__()
        self.conv_in = nn.Conv3d(zc, base, 3, padding=1)
        self.r1 = _ResBlock3D(base, base)
        self.up1 = nn.Conv3d(base, base, 3, padding=1)
        self.r2 = _ResBlock3D(base, base // 2)
        self.up2 = nn.Conv3d(base // 2, base // 2, 3, padding=1)
        self.r3 = _ResBlock3D(base // 2, base // 4)
        self.norm_out = nn.GroupNorm(32, base // 4)
        self.conv_out = nn.Conv3d(base // 4, 3, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(z)
        x = self.r1(x)
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode="nearest")
        x = self.up1(x)
        x = self.r2(x)
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode="nearest")
        x = self.up2(x)
        x = self.r3(x)
        return self.conv_out(F.silu(self.norm_out(x)))


@pytest.fixture(scope="module")
def decoder_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def decoder_input(decoder_device):
    return torch.randn(1, LATENT_C, LATENT_T, LATENT_H, LATENT_W, device=decoder_device, dtype=torch.bfloat16)


def _magi_decoder(device: torch.device):
    def _patch(cfg):
        cfg.pass_config.enable_conv_channels_last = True
        return cfg

    model = VAEDecoderLike().to(device).to(torch.bfloat16).eval()
    # Empty dims => fully static; the pass forces channels-last without dynamic shapes.
    return magi_compile(model, dynamic_arg_dims={"z": []}, config_patch=_patch)


def _eager_decoder(device: torch.device):
    return VAEDecoderLike().to(device).to(torch.bfloat16).eval()


def _torch_compiled_decoder(device: torch.device):
    model = VAEDecoderLike().to(device).to(torch.bfloat16).eval()
    return torch.compile(model, fullgraph=True, backend="inductor")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support")
def test_conv_channels_last_vs_torch_compile(decoder_device, decoder_input):
    """magi_compile (channels-last ON) vs stock torch.compile.

    MagiCompiler's pass forces NDHWC at the conv boundary, so magi_compile should
    be at least on par with torch.compile.
    """
    eager = _eager_decoder(decoder_device)
    magi = _magi_decoder(decoder_device)
    torch_compiled = _torch_compiled_decoder(decoder_device)

    with torch.no_grad():
        eager_result = cuda_benchmark(lambda: eager(decoder_input))
        torch_result = cuda_benchmark(lambda: torch_compiled(decoder_input), compilation_warmup=3)
        magi_result = cuda_benchmark(lambda: magi(decoder_input), compilation_warmup=3)

    magi_vs_torch = torch_result.median / magi_result.median
    print_perf_comparison(
        "Conv channels-last: magi_compile vs torch.compile",
        eager_result,
        magi_result,
        torch_compile=torch_result,
        extra_info=(f"latent=({LATENT_C}, {LATENT_T}, {LATENT_H}, {LATENT_W})  " f"speedup(torch/magi)={magi_vs_torch:.2f}x"),
    )

    assert_magi_vs_torch(magi_vs_torch, torch_result, magi_result, "conv_channels_last", threshold=1.2)
