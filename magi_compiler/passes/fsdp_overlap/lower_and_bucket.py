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

"""Whole-graph FSDP weight all-gather lowering + bucketing.

The piecewise pipeline (see ``magi_backend._split_graph``) lowers SimpleFSDP
weight ``prim_redistribute`` into explicit collectives and then buckets them
*per submod*, keyed by ``node_to_subgraph_id``.  Under whole-graph compilation
(``disable_graph_split=True``) there is no split and therefore no
``node_to_subgraph_id`` -- the entire model is a single Inductor graph.

This module reuses the exact same lowering/bucketing implementations, but drives
them with a **region-numbered** sid map computed the same way ``_split_graph``
numbers subgraphs (increment at each boundary op) -- WITHOUT actually splitting.
That keeps bucketing granularity per compute-region: the weight gathers used by
one layer's compute (between two boundary ops, e.g. attention / MoE) coalesce
into one collective, but different regions do NOT collapse into a single
model-wide gather (which would force the whole unsharded model resident and OOM,
defeating FSDP).

The scheduler-level :mod:`magi_compiler.passes.fsdp_overlap.reorder` pass then
places each (possibly coalesced) launch at its latest-safe position.  Bucketing
here is intentionally graph-level (not scheduler-level) so the collective node
structure is fixed before Inductor lowering.
"""

import torch
import torch.fx as fx

from magi_compiler.utils import magi_logger

from .bucket_all_gather import bucket_weight_all_gather_coalesced_per_region
from .redistribute_lowering import lower_prim_redistribute_to_collectives


def _build_region_sid_map(graph: fx.GraphModule, boundary_ops: list["torch._ops.OpOverload"]) -> dict[fx.Node, int]:
    """Number graph nodes into regions delimited by ``boundary_ops``.

    Mirrors ``_split_graph`` Step 2's ``node_to_subgraph_id`` numbering (compute
    regions get even ids, boundary ops get their own odd id), but is used only to
    group weight all-gathers for bucketing -- the graph is never split.  Weights
    within the same region share a sid, so bucketing coalesces them; weights in
    different regions stay in separate buckets.
    """
    resolved = set(boundary_ops)
    sid = 0
    mapping: dict[fx.Node, int] = {}
    for node in graph.graph.nodes:
        if node.op in ("output", "placeholder"):
            continue
        is_boundary = node.op == "call_function" and (
            node.target in resolved or (hasattr(node.target, "default") and node.target.default in resolved)
        )
        if is_boundary:
            sid += 1
            mapping[node] = sid
            sid += 1
        else:
            mapping[node] = sid
    return mapping


def lower_and_bucket_full_graph(
    graph: fx.GraphModule,
    bucket_mode: str,
    boundary_ops: list["torch._ops.OpOverload"] | None = None,
    bucket_size_bytes: int = 0,
) -> int:
    """Lower SimpleFSDP weight redistribute -> explicit collectives, then
    optionally bucket them per compute-region across the WHOLE graph.

    ``bucket_mode``:
      * ``"none"``      -- lowering only (N individual all_gather + N waits).
      * ``"coalesced"`` -- per region: one all_gather_into_tensor_coalesced
                           (ONE launch, N getitems, N waits).

    ``boundary_ops`` are the subgraph-boundary op overloads (the model's
    ``splitting_ops``); they delimit bucketing regions.  When None/empty, all
    same-(group, dtype) gathers fall in one region -- only safe for tiny graphs.

    ``bucket_size_bytes`` (coalesced mode only): when > 0, further split each region
    into buckets of at most this many local-shard bytes, breaking at dtype changes
    and the byte cap in program order (see
    ``bucket_weight_all_gather_coalesced_per_region``).  0 = no cap (one bucket per
    region/group/dtype).

    Returns the number of buckets created.
    """
    lowered = lower_prim_redistribute_to_collectives(graph)
    magi_logger.info("Whole-graph FSDP lowering: %d weight redistribute -> collectives", lowered)

    bucket_mode = (bucket_mode or "none").lower()
    if bucket_mode == "none":
        return 0

    sid_map = _build_region_sid_map(graph, boundary_ops or [])
    if bucket_mode == "coalesced":
        n = bucket_weight_all_gather_coalesced_per_region(graph, sid_map, bucket_size_bytes=bucket_size_bytes)
    else:
        raise ValueError(f"Unknown bucket_mode={bucket_mode!r}; expected 'none' or 'coalesced'")

    magi_logger.info("Whole-graph FSDP bucketing (%s): created %d buckets", bucket_mode, n)
    return n
