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

"""Graph-side copy-engine transport: bind eligible weights, then retarget tagged gathers.

``bind_weights_for_copy_engine`` runs between lowering and bucketing so a bucket
stays all-bound or all-unbound. ``rewrite_weight_ag_to_copy_engine`` retargets
whatever carries ``node_meta.CE_BOUND``. Operators live in ``symm_mem/``.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.fx as fx

from magi_compiler.utils import magi_logger

from .node_meta import is_ce_bound, is_uneven_shard, is_weight_ag, mark_ce_bound

_ALL_GATHER = torch.ops._c10d_functional.all_gather_into_tensor.default
_ALL_GATHER_COALESCED = torch.ops._c10d_functional.all_gather_into_tensor_coalesced.default


def _weight_holder(src) -> fx.Node | None:
    """The placeholder/get_attr behind ``to_local(...)``, or None for anything else."""
    if not isinstance(src, fx.Node) or src.op != "call_method" or src.target != "to_local":
        return None
    owner = src.args[0] if src.args else None
    return owner if isinstance(owner, fx.Node) and owner.op in ("placeholder", "get_attr") else None


def copy_engine_weight_candidates(graph: fx.GraphModule) -> list[tuple[fx.Node, fx.Node]]:
    """Weight all-gathers the copy engine could serve, as ``(gather, holder)`` pairs.

    Holder must be exactly ``to_local(placeholder|get_attr)``: anything in between
    is a temporary with no peer views. Binding attempts only these.
    """
    candidates: list[tuple[fx.Node, fx.Node]] = []
    for node in graph.graph.nodes:
        if node.op != "call_function" or node.target is not _ALL_GATHER:
            continue
        if not is_weight_ag(node) or is_uneven_shard(node):
            continue
        holder = _weight_holder(node.args[0] if node.args else None)
        if holder is not None:
            candidates.append((node, holder))
    return candidates


def bind_weights_for_copy_engine(graph: fx.GraphModule, example_inputs: Sequence[Any] | None, min_shard_bytes: int = 0) -> int:
    """Move eligible weights into symmetric memory and tag the gathers that got served.

    Must run before bucketing. ``example_inputs`` are the live tensors Dynamo
    captured; without them nothing binds. Returns how many gathers are now
    copy-engine backed.
    """
    from magi_compiler.symm_mem.bind import bind_graph_weights

    placeholders = graph.graph.find_nodes(op="placeholder")
    served = bind_graph_weights(
        graph=graph,
        candidates=copy_engine_weight_candidates(graph),
        # Built here so the NCCL path never pays for it.
        placeholder_examples=dict(zip((n.name for n in placeholders), example_inputs or ())),
        min_shard_bytes=min_shard_bytes,
    )
    for node in served:
        mark_ce_bound(node)
    return len(served)


def rewrite_weight_ag_to_copy_engine(graph: fx.GraphModule) -> int:
    """Retarget CE_BOUND weight gathers onto copy-engine ops. Returns how many."""
    from magi_compiler.symm_mem.all_gather import CE_ALL_GATHER, CE_ALL_GATHER_COALESCED

    rewritten = 0
    skipped = 0
    for node in graph.graph.nodes:
        if node.op != "call_function" or node.target not in (_ALL_GATHER, _ALL_GATHER_COALESCED):
            continue
        if not is_weight_ag(node):
            continue
        if not is_ce_bound(node):
            skipped += 1
            continue
        node.target = CE_ALL_GATHER if node.target is _ALL_GATHER else CE_ALL_GATHER_COALESCED
        rewritten += 1

    if rewritten:
        graph.graph.lint()
        graph.recompile()
    magi_logger.info(
        "FSDP copy-engine rewrite: %d weight all-gather(s) retargeted, %d left on NCCL (weight not bound)", rewritten, skipped
    )
    return rewritten
