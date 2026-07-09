# Copyright (c) 2026 SandAI. All Rights Reserved.
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

"""Per-op runtime cost model (profiling) -- general infrastructure, not a pass.

Provides:
* :class:`.runtime_estimator.ProfilingRuntimeEstimator` -- a ``snode -> nanoseconds``
  cost model that MEASURES each scheduler node's real kernel time (Triton via
  ``benchmark_fused_nodes``, extern/custom via eager replay, collectives via a
  rank-lockstep replay), replacing Inductor's unreliable analytical roofline. Any
  pass that needs accurate per-op timing (e.g. the FSDP-overlap reorder) uses it.
* :mod:`.benchmark_inputs` -- a model-agnostic registry so the owning code (e.g.
  athena's gaga4) can supply valid replay inputs / lockstep flags for opaque custom
  ops that can't be replayed from generic size-hinted tensors alone.
"""

from .benchmark_inputs import get_benchmark_inputs_hook, op_has_internal_collective, register_benchmark_inputs
from .runtime_estimator import ProfilingRuntimeEstimator

__all__ = ["ProfilingRuntimeEstimator", "register_benchmark_inputs", "get_benchmark_inputs_hook", "op_has_internal_collective"]
