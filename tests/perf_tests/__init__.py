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

import statistics
from collections.abc import Callable
from dataclasses import dataclass

import torch
from triton.testing import do_bench


@dataclass
class BenchmarkResult:
    times_ms: list[float]

    @property
    def median(self) -> float:
        return statistics.median(self.times_ms)

    @property
    def mean(self) -> float:
        return statistics.mean(self.times_ms)

    @property
    def min(self) -> float:
        return min(self.times_ms)

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.times_ms) if len(self.times_ms) > 1 else 0.0

    def summary(self, label: str = "") -> str:
        prefix = f"[{label}] " if label else ""
        return (
            f"{prefix}median={self.median:.3f}ms  mean={self.mean:.3f}ms  "
            f"min={self.min:.3f}ms  stdev={self.stdev:.3f}ms  (n={len(self.times_ms)})"
        )


def cuda_benchmark(
    fn: Callable[[], object],
    *,
    warmup: int = 25,
    rep: int = 100,
    grad_to_none: list[torch.Tensor] | None = None,
    compilation_warmup: int = 0,
) -> BenchmarkResult:
    if compilation_warmup > 0:
        for _ in range(compilation_warmup):
            fn()
        torch.cuda.synchronize()

    times = do_bench(fn, warmup=warmup, rep=rep, grad_to_none=grad_to_none, return_mode="all")
    return BenchmarkResult(times_ms=times)


def print_perf_comparison(
    title: str,
    eager: BenchmarkResult,
    magi: BenchmarkResult,
    torch_compile: BenchmarkResult | None = None,
    extra_info: str = "",
) -> tuple[float, float]:
    magi_vs_eager = eager.median / magi.median
    torch_vs_eager = eager.median / torch_compile.median if torch_compile else 0.0
    magi_vs_torch = torch_compile.median / magi.median if torch_compile else 0.0

    print(f"\n{'=' * 78}")
    print(title)
    if extra_info:
        print(f"  {extra_info}")
    print(f"{'=' * 78}")
    print(f"  {eager.summary('eager                       ')}")
    if torch_compile is not None:
        print(f"  {torch_compile.summary('torch.compile               ')}")
    print(f"  {magi.summary('magi_compile                ')}")
    print("  ---")
    if torch_compile is not None:
        print(f"  torch.compile     vs eager:         {torch_vs_eager:.2f}x")
    print(f"  magi_compile      vs eager:         {magi_vs_eager:.2f}x")
    if torch_compile is not None:
        print(f"  magi_compile      vs torch.compile: {magi_vs_torch:.2f}x")
    print(f"{'=' * 78}")
    return magi_vs_eager, magi_vs_torch
