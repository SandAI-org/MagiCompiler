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
degrade to untiled Grid1D. MagiCompiler works around this by auto-enabling
``triton.prefer_nd_tiling`` (+ ``max_tiles=3`` + ``tile_reductions``) for dynamic
compilation; see ``MagiBackend._should_enable_nd_tiling``.

This test exercises a WAN-2.2-VAE-decode-like workload (stacked 3D conv resblocks
+ spatial upsampling) compiled with **dynamic H/W**, and checks that the
workaround is a net win versus turning it off on the *same* magi_compile path.

Real WAN 2.2 VAE decode (540p, dynamic H/W) numbers that motivate this:
  - with conv channels-last layout: 1.252s -> 542ms / decode (~2.3x)
  - without conv channels-last:       770ms -> 535ms / decode (~1.44x)
This synthetic decoder (no weights, no conv channels-last pass) reproduces the
"~1.4x" regime; the absolute ratio is GPU-dependent so the strict assertion only
runs on calibrated GPUs.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from magi_compiler import magi_compile
from tests.perf_tests import cuda_benchmark, print_perf_comparison
from tests.perf_tests.utils import is_perf_calibrated_gpu

# WAN 2.2 VAE 540p latent: [C, T, H, W]; dynamic dims are H and W.
LATENT_C, LATENT_T, LATENT_H, LATENT_W = 48, 7, 34, 60
BASE_CHANNELS = 128

# nd_tiling(on) vs nd_tiling(off), both on the magi_compile dynamic path.
# Observed ~1.36x (off=2.209ms -> on=1.627ms) on H100; assert a conservative
# lower bound that still proves a clear, non-noise win.
ND_TILING_SPEEDUP_THRESHOLD = 1.20


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


def _compile_decoder(device: torch.device, enable_nd_tiling: bool):
    def _patch(cfg):
        cfg.enable_dynamic_nd_tiling = enable_nd_tiling
        return cfg

    model = VAEDecoderLike().to(device).to(torch.bfloat16).eval()
    # Dynamic H, W (latent dims 3, 4) — the regime where coalesce tiling bails out.
    return magi_compile(model, dynamic_arg_dims={"z": [3, 4]}, config_patch=_patch)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support")
def test_nd_tiling_workaround_speedup(decoder_device, decoder_input):
    """ND-tiling ON should beat ND-tiling OFF on the dynamic magi_compile path."""
    disabled = _compile_decoder(decoder_device, enable_nd_tiling=False)
    enabled = _compile_decoder(decoder_device, enable_nd_tiling=True)

    with torch.no_grad():
        disabled_result = cuda_benchmark(lambda: disabled(decoder_input), compilation_warmup=3)
        enabled_result = cuda_benchmark(lambda: enabled(decoder_input), compilation_warmup=3)

    speedup = disabled_result.median / enabled_result.median
    print_perf_comparison(
        "Dynamic ND-tiling: workaround ON vs OFF (magi_compile, dynamic H/W)",
        disabled_result,
        enabled_result,
        extra_info=(f"latent=({LATENT_C}, {LATENT_T}, {LATENT_H}, {LATENT_W})  " f"speedup(off/on)={speedup:.2f}x"),
    )

    if not is_perf_calibrated_gpu():
        return
    assert speedup >= ND_TILING_SPEEDUP_THRESHOLD, (
        f"ND-tiling workaround should be >= {ND_TILING_SPEEDUP_THRESHOLD:.2f}x faster than disabled "
        f"under dynamic shapes. Got {speedup:.2f}x "
        f"(disabled={disabled_result.median:.3f}ms, enabled={enabled_result.median:.3f}ms)"
    )
