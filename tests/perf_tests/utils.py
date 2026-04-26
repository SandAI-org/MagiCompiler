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

import functools

import torch

from tests.perf_tests import BenchmarkResult

MAGI_VS_TORCH_THRESHOLD = 0.97

# Absolute speedup-vs-eager thresholds are calibrated on H100.
# On other GPUs the operator mix (e.g. matmul vs memory-bound) may shift the
# ratio significantly, so we only enforce magi ≈ torch.compile (parity check).
_PERF_CALIBRATED_GPUS = ("H100",)


@functools.cache
def is_perf_calibrated_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    name = torch.cuda.get_device_name(0)
    return any(kw in name for kw in _PERF_CALIBRATED_GPUS)


def assert_speedup(
    magi_vs_eager: float, eager_result: BenchmarkResult, magi_result: BenchmarkResult, label: str, threshold: float
) -> None:
    if not is_perf_calibrated_gpu():
        return
    assert magi_vs_eager >= threshold, (
        f"[{label}] magi_compile must achieve >= {threshold:.2f}x over eager. "
        f"Got {magi_vs_eager:.2f}x "
        f"(eager={eager_result.median:.3f}ms, magi={magi_result.median:.3f}ms)"
    )


def assert_magi_vs_torch(
    magi_vs_torch: float,
    torch_result: BenchmarkResult,
    magi_result: BenchmarkResult,
    label: str,
    threshold: float = MAGI_VS_TORCH_THRESHOLD,
) -> None:
    assert magi_vs_torch >= threshold, (
        f"[{label}] magi_compile must be >= {threshold:.2f}x of torch.compile. "
        f"Got {magi_vs_torch:.2f}x "
        f"(torch={torch_result.median:.3f}ms, magi={magi_result.median:.3f}ms)"
    )
