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

from tests.perf_tests import BenchmarkResult


def assert_speedup(
    magi_vs_eager: float, eager_result: BenchmarkResult, magi_result: BenchmarkResult, label: str, threshold: float
) -> None:
    assert magi_vs_eager >= threshold, (
        f"[{label}] magi_compile must achieve >= {threshold:.2f}x over eager. "
        f"Got {magi_vs_eager:.2f}x "
        f"(eager={eager_result.median:.3f}ms, magi={magi_result.median:.3f}ms)"
    )
