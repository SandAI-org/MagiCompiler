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

from typing import Any, Sequence

import torch.fx as fx

from magi_compiler.utils import magi_logger

from .bucket_all_gather import bucket_weight_all_gather_coalesced
from .copy_engine import bind_weights_for_copy_engine, rewrite_weight_ag_to_copy_engine
from .node_meta import is_ce_bound
from .redistribute_lowering import lower_prim_redistribute_to_collectives


def lower_and_bucket_full_graph(
    graph: fx.GraphModule,
    bucket_mode: str,
    bucket_size_bytes: int = 0,
    transport: str = "nccl",
    example_inputs: Sequence[Any] | None = None,
    min_shard_bytes: int = 0,
) -> int:
    """Lower SimpleFSDP weight redistribute -> explicit collectives, then
    optionally bucket them across the WHOLE graph (no subgraph partitioning).

    ``bucket_mode``:
      * ``"none"``      -- lowering only (N individual all_gather + N waits).
      * ``"coalesced"`` -- one all_gather_into_tensor_coalesced per bucket
                           (ONE launch, N getitems, N waits).

    ``bucket_size_bytes`` (coalesced mode only): when > 0, split the gathers into
    buckets of at most this many local-shard bytes, breaking at dtype changes and
    the byte cap in program order (see ``bucket_weight_all_gather_coalesced``).
    0 = no cap (one bucket per (group, dtype) run).

    ``transport="copy_engine"`` wraps the bucketing in the two steps ``copy_engine``
    owns: binding right after lowering, the retarget at the very end.  Bucketing
    then keys off what binding served, so bound and unbound gathers are split into
    separate buckets rather than the unbound ones being dropped from bucketing --
    losing the copy engine must not also lose coalescing.  ``example_inputs`` and
    ``min_shard_bytes`` are read only on this path.

    Returns the number of buckets created.
    """
    lowered = lower_prim_redistribute_to_collectives(graph)
    magi_logger.info("Whole-graph FSDP lowering: %d weight redistribute -> collectives", lowered)

    if transport == "copy_engine":
        bind_weights_for_copy_engine(graph, example_inputs, min_shard_bytes)

    bucket_mode = (bucket_mode or "none").lower()
    n = 0
    if bucket_mode == "coalesced":
        split_by = is_ce_bound if transport == "copy_engine" else None
        n = bucket_weight_all_gather_coalesced(graph, bucket_size_bytes=bucket_size_bytes, split_by=split_by)
        magi_logger.info("Whole-graph FSDP bucketing (%s): created %d buckets", bucket_mode, n)
    elif bucket_mode not in ("none", ""):
        raise ValueError(f"Unknown bucket_mode={bucket_mode!r}; expected 'none' or 'coalesced'")

    if transport == "copy_engine":
        rewrite_weight_ag_to_copy_engine(graph)

    return n
