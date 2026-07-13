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

Some opaque boundary custom ops cannot be replayed for timing from generic
size-hinted tensors alone, because they carry VALUE-DEPENDENT metadata that must
be self-consistent (not just right-shaped) or they raise -- e.g. a context-parallel
attention op taking a split-sizes tensor whose values must sum to the sequence
length, else its internal all_to_all asserts and the op falls back to a 0 cost
estimate.

The owning code (the model package that defines the custom op) registers a hook
here that builds a VALID, RANK-DETERMINISTIC set of replay inputs.  MagiCompiler
stays free of model-specific knowledge: ``_measure_extern`` just looks up the op
by name and, if a hook exists, uses it instead of the generic realizer.

A hook is ``fn(fx_node, realize) -> (args, kwargs) | None``:
  * ``fx_node``   -- the ``torch.fx.Node`` for the op (read its ``args``/``kwargs``
                     and their ``meta['val']`` for shapes);
  * ``realize``   -- MagiCompiler's default arg realizer (fx.Node/SymInt/container
                     -> concrete tensor/int), so a hook can reuse it for the plain
                     tensor args and only special-case the metadata ones;
  * returns ``(args, kwargs)`` of real objects to call the op with, or ``None`` to
    fall back to the generic path.

Hooks MUST produce rank-identical inputs (derive everything from shapes / static
sizes, no per-rank timing or state) so the rank-lockstep ``warm_and_sync`` measure
issues any internal collective in lockstep across ranks.

Example -- a custom attention op ``mylib::attn_cp(q, k, v, split_sizes, scale)``
whose internal all_to_all requires ``split_sizes`` values to sum to the sequence
length (a zero-filled tensor from the generic realizer would make it raise)::

    from magi_compiler.profiling import register_benchmark_inputs

    def _attn_cp_benchmark_inputs(fx_node, realize):
        q_node, k_node, v_node, _split_node, scale = fx_node.args
        q = realize(q_node)          # plain tensor args: reuse the generic realizer
        k = realize(k_node)
        v = realize(v_node)
        # Rebuild the VALUE-dependent arg from SHAPE hints only (rank-identical):
        # a uniform split summing to the per-rank seqlen.
        seq, group_size = q.shape[0], 4
        split_sizes = torch.full((group_size,), seq // group_size, dtype=torch.int32, device=q.device)
        split_sizes[-1] += seq - int(split_sizes.sum())
        return (q, k, v, split_sizes, float(scale)), {}

    register_benchmark_inputs(
        "mylib::attn_cp", _attn_cp_benchmark_inputs, has_internal_collective=True
    )

Register at import time of the module that defines the op (so the hook exists
before any compile).  Returning ``None`` from the hook falls back to the generic
realizer for that call.
"""

from __future__ import annotations

from typing import Callable

# op name (``torch.ops.ns.op`` overload string, e.g. "mylib::attn_cp")
# -> hook.  Also records which ops issue an internal collective (need fixed-iter
# lockstep replay) -- so MagiCompiler no longer hardcodes model-specific op names.
_BENCHMARK_INPUT_HOOKS: dict[str, Callable] = {}
_INTERNAL_COLLECTIVE_OPS: set[str] = set()


def register_benchmark_inputs(op_name: str, fn: Callable, *, has_internal_collective: bool = False) -> None:
    """Register a replay-input builder for ``op_name`` (see module docstring).

    ``has_internal_collective``: mark this op as issuing an internal collective, so
    the estimator replays it with a FIXED iteration count under barriers (keeps every
    rank in lockstep -- a duration-adaptive count would desync the internal NCCL op).
    """
    _BENCHMARK_INPUT_HOOKS[op_name] = fn
    if has_internal_collective:
        _INTERNAL_COLLECTIVE_OPS.add(op_name)


def get_benchmark_inputs_hook(op_name: str) -> Callable | None:
    return _BENCHMARK_INPUT_HOOKS.get(op_name)


def op_has_internal_collective(op_name: str) -> bool:
    return op_name in _INTERNAL_COLLECTIVE_OPS
