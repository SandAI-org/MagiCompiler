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

import torch.fx as fx

from magi_compiler.utils import magi_logger

from .bucket_all_gather import bucket_weight_all_gather_coalesced
from .redistribute_lowering import lower_prim_redistribute_to_collectives
from .symm_ag_rewrite import rewrite_weight_ag_to_copy_engine


def lower_and_bucket_full_graph(
    graph: fx.GraphModule, bucket_mode: str, bucket_size_bytes: int = 0, transport: str = "nccl"
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

    ``transport="copy_engine"`` buckets *first* (only arena-shard gathers, so
    cast/pad stays out of the bucket), then retargets both the leftover singles
    and the coalesced launches at the copy-engine ops.  The wrapper still runs
    one gather per member; reorder just sees one comm node per bucket.

    Returns the number of buckets created.
    """
    lowered = lower_prim_redistribute_to_collectives(graph)
    magi_logger.info("Whole-graph FSDP lowering: %d weight redistribute -> collectives", lowered)

    bucket_mode = (bucket_mode or "none").lower()
    n = 0
    if bucket_mode == "coalesced":
        from .symm_ag_rewrite import _input_is_arena_shard

        eligible = _input_is_arena_shard if transport == "copy_engine" else None
        n = bucket_weight_all_gather_coalesced(graph, bucket_size_bytes=bucket_size_bytes, eligible=eligible)
        magi_logger.info("Whole-graph FSDP bucketing (%s): created %d buckets", bucket_mode, n)
    elif bucket_mode not in ("none", ""):
        raise ValueError(f"Unknown bucket_mode={bucket_mode!r}; expected 'none' or 'coalesced'")

    if transport == "copy_engine":
        rewrite_weight_ag_to_copy_engine(graph)

    return n
