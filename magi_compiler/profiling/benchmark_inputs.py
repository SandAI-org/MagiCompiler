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
be self-consistent (not just right-shaped) or they raise -- e.g.
``gaga4_fa3_with_sink_cp`` takes a ``cp_split_sizes`` tensor whose values must sum
to the sequence length and equal the per-rank seqlen, else its internal CP
all_to_all asserts and the op falls back to a 0 cost estimate.

The owning code (which knows the op's semantics -- e.g. athena's gaga4 model)
registers a hook here that builds a VALID, RANK-DETERMINISTIC set of replay inputs.
MagiCompiler stays free of model-specific knowledge: ``_measure_extern`` just looks
up the op by name and, if a hook exists, uses it instead of the generic realizer.

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
"""

from __future__ import annotations

from typing import Callable

# op name (``torch.ops.ns.op`` overload string, e.g. "athena::gaga4_fa3_with_sink_cp")
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
