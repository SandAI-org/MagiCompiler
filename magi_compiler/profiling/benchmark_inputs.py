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

"""Per-op benchmark-input hooks for the profiling runtime estimator.

Some custom ops cannot be replayed from generic size-hinted tensors: they carry
VALUE-DEPENDENT metadata that must be self-consistent or they raise (e.g. a CP
attention op whose split-sizes must sum to the sequence length) -- and would fall
back to a 0 cost.  The model package that defines such an op registers a hook
here that builds valid replay inputs; MagiCompiler stays free of model-specific
op knowledge.

Hook: ``fn(fx_node, realize) -> (args, kwargs) | None`` -- ``fx_node`` is the op's
``torch.fx.Node`` (shapes in ``meta['val']``), ``realize`` is the generic arg
realizer to reuse for plain tensor args; return None to fall back to the generic
path.  Register at import time of the op-defining module, e.g.::

    register_benchmark_inputs("mylib::attn_cp", _attn_cp_inputs, has_internal_collective=True)

Hooks MUST produce rank-identical inputs (derive everything from shapes, no
per-rank state) so the rank-lockstep ``warm_and_sync`` measurement issues any
internal collective in lockstep.
"""

from __future__ import annotations

from typing import Callable

# op name (OpOverload string, e.g. "mylib::attn_cp") -> hook; plus the set of ops
# that issue an internal collective (need fixed-iter lockstep replay).
_BENCHMARK_INPUT_HOOKS: dict[str, Callable] = {}
_INTERNAL_COLLECTIVE_OPS: set[str] = set()


def register_benchmark_inputs(op_name: str, fn: Callable, *, has_internal_collective: bool = False) -> None:
    """Register a replay-input builder for ``op_name`` (see module docstring).
    ``has_internal_collective``: replay with a fixed iteration count under barriers
    (an adaptive count would desync the internal NCCL op across ranks)."""
    _BENCHMARK_INPUT_HOOKS[op_name] = fn
    if has_internal_collective:
        _INTERNAL_COLLECTIVE_OPS.add(op_name)


def get_benchmark_inputs_hook(op_name: str) -> Callable | None:
    return _BENCHMARK_INPUT_HOOKS.get(op_name)


def op_has_internal_collective(op_name: str) -> bool:
    return op_name in _INTERNAL_COLLECTIVE_OPS
