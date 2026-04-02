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

import shutil
import statistics
from collections.abc import Callable
from dataclasses import dataclass

import torch
from triton.testing import do_bench

from magi_compiler.config import get_compile_config


class CleanupCacheContext:
    """Context manager for cleaning cache before and after execution"""

    def __enter__(self):
        shutil.rmtree(get_compile_config().cache_root_dir, ignore_errors=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        shutil.rmtree(get_compile_config().cache_root_dir, ignore_errors=True)


@dataclass
class BenchmarkResult:
    """Timing results from a CUDA benchmark run (all times in milliseconds).

    Follows the methodology of ``triton.testing.do_bench``: time-based warmup/rep,
    L2 cache flush between iterations, per-iteration CUDA event timing.
    """

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
    """Benchmark a GPU callable using ``triton.testing.do_bench`` methodology.

    Uses time-based warmup/rep (in ms), L2 cache flush between iterations,
    and per-iteration CUDA event timing -- the same approach torch inductor
    uses internally for autotuning fused triton kernels.

    Args:
        fn: Zero-argument callable to benchmark.
        warmup: Warmup duration in milliseconds (passed to ``do_bench``).
        rep: Benchmark repetition duration in milliseconds.
        grad_to_none: Tensors whose ``.grad`` should be cleared between iterations.
        compilation_warmup: Number of extra invocations **before** ``do_bench``
            to ensure lazy compilation (e.g. ``magi_compile``) is fully finished.
            These calls are *not* timed.

    Returns:
        BenchmarkResult with all per-iteration times in milliseconds.
    """
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
    torch_compile: BenchmarkResult,
    extra_info: str = "",
    magi_torch_compile: BenchmarkResult | None = None,
) -> tuple[float, float]:
    """Print a comparison table and return speedup ratios.

    When *magi_torch_compile* is provided the table includes the
    ``magi_compile(compile_mode=TORCH_COMPILE)`` variant as well.

    Returns:
        (magi_vs_eager_speedup, magi_vs_torch_compile_speedup) based on median.
    """
    magi_vs_eager = eager.median / magi.median
    torch_vs_eager = eager.median / torch_compile.median
    magi_vs_torch = torch_compile.median / magi.median

    print(f"\n{'=' * 78}")
    print(title)
    if extra_info:
        print(f"  {extra_info}")
    print(f"{'=' * 78}")
    print(f"  {eager.summary('eager                       ')}")
    print(f"  {torch_compile.summary('torch.compile               ')}")
    if magi_torch_compile is not None:
        print(f"  {magi_torch_compile.summary('magi (torch_compile mode)   ')}")
    print(f"  {magi.summary('magi_compile                ')}")
    print(f"  ---")
    print(f"  torch.compile     vs eager:         {torch_vs_eager:.2f}x")
    if magi_torch_compile is not None:
        mtc_vs_eager = eager.median / magi_torch_compile.median
        mtc_vs_torch = torch_compile.median / magi_torch_compile.median
        print(f"  magi(torch mode)  vs eager:         {mtc_vs_eager:.2f}x")
        print(f"  magi(torch mode)  vs torch.compile: {mtc_vs_torch:.2f}x")
    print(f"  magi_compile      vs eager:         {magi_vs_eager:.2f}x")
    print(f"  magi_compile      vs torch.compile: {magi_vs_torch:.2f}x")
    print(f"{'=' * 78}")
    return magi_vs_eager, magi_vs_torch


def enable_remote_debug():
    import os

    import debugpy

    ENABLE_MAGI_REMOTE_DEBUG = os.environ.get("ENABLE_MAGI_REMOTE_DEBUG", "false").lower()
    if ENABLE_MAGI_REMOTE_DEBUG == "false":
        return

    debug_ranks = []
    if ENABLE_MAGI_REMOTE_DEBUG == "true":
        debug_ranks = [0]
    elif ENABLE_MAGI_REMOTE_DEBUG == "all":
        debug_ranks = [i for i in range(1)]
    else:
        debug_ranks = [int(i) for i in ENABLE_MAGI_REMOTE_DEBUG.split(",")]

    rank = 0
    if rank in debug_ranks:
        debug_port = 5678 + int(rank)
        print(f"[rank {rank}] Starting remote debug on port {debug_port}")
        debugpy.listen(("0.0.0.0", debug_port))
        debugpy.wait_for_client()
        print(f"[rank {rank}] Remote debug attached")
