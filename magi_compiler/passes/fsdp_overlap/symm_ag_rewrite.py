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

from __future__ import annotations

import torch
import torch.fx as fx

from magi_compiler.utils import magi_logger

_ALL_GATHER = torch.ops._c10d_functional.all_gather_into_tensor.default
_ALL_GATHER_COALESCED = torch.ops._c10d_functional.all_gather_into_tensor_coalesced.default


def _input_is_arena_shard(node: fx.Node) -> bool:
    """True when the gather input is ``to_local(placeholder/get_attr)`` with no intervening op."""
    src = node.args[0] if node.args else None
    if not isinstance(src, fx.Node) or src.op != "call_method" or src.target != "to_local":
        return False
    owner = src.args[0] if src.args else None
    return isinstance(owner, fx.Node) and owner.op in ("placeholder", "get_attr")


def _coalesced_inputs_are_arena_shards(node: fx.Node) -> bool:
    locs = node.args[0] if node.args else None
    if not isinstance(locs, (list, tuple)) or not locs:
        return False
    return all(isinstance(loc, fx.Node) and _input_is_arena_shard_from_local(loc) for loc in locs)


def _input_is_arena_shard_from_local(src: fx.Node) -> bool:
    if src.op != "call_method" or src.target != "to_local":
        return False
    owner = src.args[0] if src.args else None
    return isinstance(owner, fx.Node) and owner.op in ("placeholder", "get_attr")


def rewrite_weight_ag_to_copy_engine(graph: fx.GraphModule) -> int:
    """Retarget marked weight gathers to ``magi::symm_all_gather``. Returns count rewritten."""
    from magi_compiler.symm_mem.all_gather import SYMM_ALL_GATHER, SYMM_ALL_GATHER_COALESCED

    rewritten = 0
    skipped = 0
    for node in graph.graph.nodes:
        if node.op != "call_function":
            continue
        if not node.meta.get("magi_fsdp_weight_ag"):
            continue
        if node.target is _ALL_GATHER:
            if not _input_is_arena_shard(node):
                skipped += 1
                continue
            node.target = SYMM_ALL_GATHER
            rewritten += 1
        elif node.target is _ALL_GATHER_COALESCED:
            if not _coalesced_inputs_are_arena_shards(node):
                skipped += 1
                continue
            node.target = SYMM_ALL_GATHER_COALESCED
            rewritten += 1

    if rewritten:
        graph.graph.lint()
        graph.recompile()
    magi_logger.info(
        "FSDP copy-engine rewrite: %d weight all-gather(s) retargeted, "
        "%d left on NCCL (input is a cast/pad of the shard, not the shard itself)",
        rewritten,
        skipped,
    )
    return rewritten
