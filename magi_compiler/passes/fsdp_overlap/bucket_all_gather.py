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

import operator
from collections import defaultdict, deque

import torch
import torch.fx as fx

from magi_compiler.utils import magi_logger

_ALL_GATHER = torch.ops._c10d_functional.all_gather_into_tensor.default
_ALL_GATHER_COALESCED = torch.ops._c10d_functional.all_gather_into_tensor_coalesced.default
_WAIT = torch.ops._c10d_functional.wait_tensor.default


def _is_param_like(node: fx.Node) -> bool:
    """True for a placeholder/get_attr that names a SimpleFSDP weight shard."""
    if node.op not in ("placeholder", "get_attr"):
        return False
    name = f"{node.name} {node.target}".lower()
    return any(t in name for t in ("parameter", "parameters", "weight", "bias", "shard"))


def _gathers_a_weight(node: fx.Node) -> bool:
    """Walk back from an all_gather through cheap producers (to_local / cast /
    pad / view / reshape); return True if the gathered source is a weight shard.

    Used so the bucketing pass also fires on graphs whose weight gathers are NOT
    tagged ``magi_fsdp_weight_ag`` (e.g. the demo, which emits explicit
    ``all_gather_into_tensor`` directly off a ``*_shard`` parameter)."""
    q: deque[fx.Node] = deque(node.all_input_nodes)
    seen: set[fx.Node] = set()
    while q:
        dep = q.popleft()
        if dep in seen:
            continue
        seen.add(dep)
        if _is_param_like(dep):
            return True
        if dep.op == "call_method" and str(dep.target) in {"to_local", "contiguous", "to", "view", "reshape"}:
            q.extend(dep.all_input_nodes)
        elif dep.op == "call_function":
            nm = getattr(dep.target, "__name__", "") or str(dep.target)
            if any(t in nm for t in ("constant_pad_nd", "_to_copy", "convert_element_type", "view", "reshape", "clone")):
                q.extend(dep.all_input_nodes)
    return False


def _is_weight_all_gather(node: fx.Node) -> bool:
    """A SimpleFSDP weight ``all_gather_into_tensor`` launch.

    Recognized either by the ``magi_fsdp_weight_ag`` tag (set by the redistribute
    lowering pass on the real gaga4 graph) OR structurally, when the gathered
    source traces back to a weight/param placeholder (covers untagged graphs such
    as the demo)."""
    if node.op != "call_function" or node.target is not _ALL_GATHER:
        return False
    if node.meta.get("magi_fsdp_weight_ag"):
        return True
    return _gathers_a_weight(node)


def _local_shard_bytes(ag: fx.Node) -> int:
    """Bytes of the local (pre-gather) shard feeding a weight all_gather -- i.e. how
    much this rank contributes to the collective.  Used to cap coalesced bucket size."""
    loc = ag.args[0]
    m = loc.meta.get("example_value")
    if m is None:
        return 0
    return int(m.numel()) * int(m.element_size())


def _split_region_by_dtype_and_size(
    ag_nodes: list[fx.Node], node_index: dict[fx.Node, int], bucket_size_bytes: int
) -> list[list[fx.Node]]:
    """Split a region's weight all-gathers into buckets by PROGRAM-ADJACENCY (plan B).

    Walk ``ag_nodes`` in program order and start a NEW bucket whenever:
      * the dtype changes (a different-dtype weight physically breaks the run -- so
        e.g. bf16, bf16, fp32, bf16 -> {bf16,bf16}, {fp32}, {bf16}); or
      * adding the next weight's local-shard bytes would exceed ``bucket_size_bytes``
        (only when the cap is > 0).

    A single weight larger than the cap forms its own bucket (we never split one
    weight's all_gather).  With ``bucket_size_bytes <= 0`` only the dtype-change rule
    applies, so program-adjacent same-dtype weights still coalesce without a size cap.
    Caller filters buckets of size < 2 (nothing to coalesce)."""
    ordered = sorted(ag_nodes, key=lambda n: node_index[n])
    buckets: list[list[fx.Node]] = []
    cur: list[fx.Node] = []
    cur_dtype = None
    cur_bytes = 0
    for ag in ordered:
        dt = ag.meta["example_value"].dtype
        b = _local_shard_bytes(ag)
        breaks_dtype = cur and dt != cur_dtype
        breaks_size = cur and bucket_size_bytes > 0 and (cur_bytes + b) > bucket_size_bytes
        if breaks_dtype or breaks_size:
            buckets.append(cur)
            cur, cur_bytes = [], 0
        cur.append(ag)
        cur_dtype = dt
        cur_bytes += b
    if cur:
        buckets.append(cur)
    return buckets


def _producer_chain(node: fx.Node) -> list[fx.Node]:
    """The local-shard prep nodes feeding ``node`` (to_local / _to_copy / pad).
    These depend only on the weight placeholder, so they may be hoisted.  Returns
    the movable producers (excludes placeholders/get_attr)."""
    chain: list[fx.Node] = []
    seen: set[fx.Node] = set()
    stack = [node]
    while stack:
        n = stack.pop()
        for dep in n.all_input_nodes:
            if dep in seen:
                continue
            if dep.op in ("call_function", "call_method"):
                seen.add(dep)
                chain.append(dep)
                stack.append(dep)
    return chain


def _coalesce_one_bucket(
    graph: fx.GraphModule,
    node_to_subgraph_id: dict[fx.Node, int],
    node_index: dict[fx.Node, int],
    sid: int,
    ag_nodes: list[fx.Node],
) -> None:
    """Merge ONE bucket of same-dtype weight all_gathers into a single
    ``all_gather_into_tensor_coalesced`` (launch + per-member getitem + per-member
    wait).  ``ag_nodes`` must be same (group, dtype); caller guarantees len >= 2."""
    world = int(ag_nodes[0].args[1])
    ag_nodes = sorted(ag_nodes, key=lambda n: node_index[n])
    _, _world, group_name = ag_nodes[0].args

    locals_ = [ag.args[0] for ag in ag_nodes]
    ag_metas = [ag.meta["example_value"] for ag in ag_nodes]  # (W*chunk_i, *rest_i)

    first_ag = ag_nodes[0]
    waits = [next(iter(ag.users)) for ag in ag_nodes]  # sole user of each ag is its wait

    # Hoist every member's local-shard prep above the FIRST member's all_gather.
    for loc in locals_:
        chain = [loc, *_producer_chain(loc)] if loc.op in ("call_function", "call_method") else _producer_chain(loc)
        for prod in sorted(chain, key=lambda n: node_index[n]):
            first_ag.prepend(prod)
            node_to_subgraph_id[prod] = sid

    with graph.graph.inserting_before(first_ag):
        coalesced = graph.graph.call_function(_ALL_GATHER_COALESCED, (list(locals_), world, group_name))
        coalesced.meta["example_value"] = list(ag_metas)
        coalesced.meta["magi_fsdp_weight_ag"] = True
        coalesced.meta["magi_fsdp_weight_ag_coalesced"] = True
        node_to_subgraph_id[coalesced] = sid

        outs = []
        for i, am in enumerate(ag_metas):
            gi = graph.graph.call_function(operator.getitem, (coalesced, i))
            gi.meta["example_value"] = am
            node_to_subgraph_id[gi] = sid
            outs.append(gi)

    for out_i, old_wait, am in zip(outs, waits, ag_metas):
        with graph.graph.inserting_before(old_wait):
            wait_i = graph.graph.call_function(_WAIT, (out_i,))
            wait_i.meta["example_value"] = am
            node_to_subgraph_id[wait_i] = node_to_subgraph_id.get(old_wait, sid)
        old_wait.replace_all_uses_with(wait_i)

    for ag_old, wait_old in zip(ag_nodes, waits):
        node_to_subgraph_id.pop(wait_old, None)
        node_to_subgraph_id.pop(ag_old, None)
        graph.graph.erase_node(wait_old)
        graph.graph.erase_node(ag_old)


def bucket_weight_all_gather_coalesced_per_region(
    graph: fx.GraphModule, node_to_subgraph_id: dict[fx.Node, int], bucket_size_bytes: int = 0
) -> int:
    """Coalesce, per compute region, the SimpleFSDP weight all-gathers into ONE
    ``all_gather_into_tensor_coalesced`` per ``(region_id, group_name, dtype)``
    (or into several buckets when ``bucket_size_bytes > 0``).

    When ``bucket_size_bytes > 0``, a region is further split into multiple buckets:
    the weight all_gathers are walked in PROGRAM ORDER and a new bucket starts
    whenever the dtype changes or the accumulated local-shard bytes would exceed the
    cap (see :func:`_split_region_by_dtype_and_size`).  So only same-dtype,
    program-adjacent weights within the byte budget coalesce; each bucket is one
    ``all_gather_into_tensor_coalesced``.  ``bucket_size_bytes == 0`` (default) keeps
    the original behavior (one bucket per (region, group, dtype), no size cap).

    This does NOT cat the shards into one buffer.  ``all_gather_into_tensor_coalesced``
    fuses the N launches into a single NCCL group while returning one
    *weight-major* output buffer per input, so each member is recovered with a
    zero-copy ``operator.getitem`` -- no ``cat`` on the compute stream, no
    ``split_with_sizes`` clone, and no transient ~2x memory spike.

    For a group of N weight gathers (each a single ``all_gather_into_tensor`` whose
    input is the padded local shard ``(chunk_i, *rest_i)``) this builds::

        # launch side (the coalesced launch + its getitems stay together):
        coalesced = all_gather_into_tensor_coalesced([local_0, ..., local_{N-1}], W, group)
        out_i     = getitem(coalesced, i)          # (W*chunk_i, *rest_i), weight-major
        # use side (one wait per member, left at its consumer):
        wait_i    = wait_tensor(out_i)

    ``out_i`` has exactly the shape of the member's old ``all_gather`` output, so
    each member's existing downstream (slice / real use) is simply re-pointed from
    its old ``wait_tensor`` to ``wait_i``.  The ``coalesced`` launch and its
    ``getitem`` nodes stay together; the scheduler-level ``FsdpOverlapReorder`` pass
    later moves the launch as one unit for compute/comm overlap.

    Runs AFTER redistribute lowering (called from ``lower_and_bucket_full_graph``).
    Returns the number of coalesced buckets created.
    """
    node_index = {n: i for i, n in enumerate(graph.graph.nodes)}

    # Group by (region, group_name) ONLY -- dtype is NOT part of the key.  Program
    # order + the dtype-change rule in _split_region_by_dtype_and_size decide the
    # actual buckets, so a different-dtype weight physically breaks a same-dtype run
    # (plan B / strict program-adjacency).  With bucket_size_bytes==0 this reduces to
    # one bucket per (region, group, dtype) run.
    groups: dict[tuple[int, str], list[fx.Node]] = defaultdict(list)
    for node in graph.graph.nodes:
        if not _is_weight_all_gather(node):
            continue
        sid = node_to_subgraph_id.get(node)
        if sid is None:
            continue
        _, _world, group_name = node.args
        groups[(sid, group_name)].append(node)

    buckets = 0
    for (sid, group_name), ag_nodes in groups.items():
        for sub in _split_region_by_dtype_and_size(ag_nodes, node_index, bucket_size_bytes):
            if len(sub) < 2:
                continue  # single weight -> keep its own all_gather (nothing to coalesce)
            _coalesce_one_bucket(graph, node_to_subgraph_id, node_index, sid, sub)
            buckets += 1

    if buckets:
        graph.graph.lint()
        graph.recompile()
    magi_logger.info(
        "FSDP weight all-gather bucketing (coalesced): created %d coalesced buckets across submods " "(bucket_size_bytes=%d)",
        buckets,
        bucket_size_bytes,
    )
    return buckets
