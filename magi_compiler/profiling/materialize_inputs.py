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

"""Per-op materialize-input hooks for the profiling runtime estimator.

Some custom ops cannot be replayed from generic size-hinted tensors: they carry
VALUE-DEPENDENT metadata that must be self-consistent or they raise (e.g. a CP
attention op whose split-sizes must sum to the sequence length) -- and would fall
back to a 0 cost.  The model package that defines such an op registers a hook
that rebuilds valid replay inputs from the same arguments the custom op sees;
MagiCompiler stays free of model-specific op knowledge.

Hook signature matches the custom op.  MagiCompiler generic-realizes every
argument first, then calls::

    hook(*realized_args, **realized_kwargs) -> tuple | None

Return a positional-arg tuple to replay (same slots as the op), or ``None`` to
keep the generic realize.  Prefer attaching the hook on the op decorator::

    def _attn_cp_inputs(q, k, v, cp_split_sizes):
        seq = int(q.shape[1] if q.dim() == 4 else q.shape[0])
        return q, k, v, [seq] * len(cp_split_sizes)

    @magi_register_custom_op("mylib::attn_cp", materialize_inputs=_attn_cp_inputs, has_internal_collective=True)
    def attn_cp(q, k, v, cp_split_sizes):
        ...

Hooks MUST produce rank-identical inputs (derive everything from shapes, no
per-rank state) so the rank-lockstep ``warm_and_sync`` measurement issues any
internal collective in lockstep.
"""

from __future__ import annotations

from typing import Callable

# op name (OpOverload string, e.g. "mylib::attn_cp") -> hook; plus the set of ops
# that issue an internal collective (need fixed-iter lockstep replay).
_MATERIALIZE_INPUT_HOOKS: dict[str, Callable] = {}
_INTERNAL_COLLECTIVE_OPS: set[str] = set()


def register_materialize_inputs(op_name: str, fn: Callable | None = None, *, has_internal_collective: bool = False) -> None:
    """Register a same-signature replay-input builder for ``op_name``.

    ``fn`` is optional when only ``has_internal_collective`` is needed (generic
    realize is already valid).  ``has_internal_collective``: replay with a fixed
    iteration count under barriers (an adaptive count would desync the internal
    NCCL op across ranks).
    """
    if fn is not None:
        _MATERIALIZE_INPUT_HOOKS[op_name] = fn
    if has_internal_collective:
        _INTERNAL_COLLECTIVE_OPS.add(op_name)


def get_materialize_inputs_hook(op_name: str) -> Callable | None:
    return _MATERIALIZE_INPUT_HOOKS.get(op_name)


def op_has_internal_collective(op_name: str) -> bool:
    return op_name in _INTERNAL_COLLECTIVE_OPS


def apply_materialize_inputs(hook: Callable | None, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
    """Run ``hook`` on already-realized custom-op inputs.

    ``hook`` has the same signature as the op.  ``None`` keeps ``(args, kwargs)``;
    a tuple/list replaces the positional args used for replay.
    """
    if hook is None:
        return args, kwargs
    out = hook(*args, **kwargs)
    if out is None:
        return args, kwargs
    if isinstance(out, (tuple, list)):
        return tuple(out), {}
    raise TypeError(f"materialize_inputs hook must return None or a tuple of op args, got {type(out).__name__}")
