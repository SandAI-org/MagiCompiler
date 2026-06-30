from __future__ import annotations

import math
import operator
from collections import defaultdict, deque

import torch
import torch.fx as fx

from magi_compiler.utils import magi_logger

_ALL_GATHER = torch.ops._c10d_functional.all_gather_into_tensor.default
_ALL_GATHER_COALESCED = torch.ops._c10d_functional.all_gather_into_tensor_coalesced.default
_WAIT = torch.ops._c10d_functional.wait_tensor.default
_CAT = torch.ops.aten.cat.default
_RESHAPE = torch.ops.aten.reshape.default
_SPLIT = torch.ops.aten.split_with_sizes.default


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


def bucket_weight_all_gather_per_submod(
    graph: fx.GraphModule,
    node_to_subgraph_id: dict[fx.Node, int],
) -> int:
    """Coalesce, per submod, the SimpleFSDP weight all-gathers into ONE
    ``all_gather_into_tensor`` per ``(subgraph_id, group_name, dtype)`` using the
    torch.compile-style flatten/cat/gather/split merge (single collective, single
    wait), instead of N individual gathers.

    For a group of N weight gathers (each a single ``all_gather_into_tensor`` from
    the redistribute-lowering pass, tagged ``magi_fsdp_weight_ag``, input = padded
    local shard of shape ``(chunk_i, *rest_i)``), this builds::

        # launch side (movable by the prefetch pass):
        flat_i   = reshape(local_i, [-1])              for each member
        cat_in   = cat([flat_0, ..., flat_{N-1}])      # (sum_numel,)
        ag       = all_gather_into_tensor(cat_in, W, group)   # (W*sum_numel,)
        # use side (wait stays before the consuming compute):
        waited   = wait_tensor(ag)                     # ONE wait
        resh     = reshape(waited, [W, sum_numel])
        splits   = split_with_sizes(resh, [numel_0, ...], dim=1)
        out_i    = reshape(splits[i], [W*chunk_i, *rest_i])

    ``out_i`` has exactly the shape of the member's old ``all_gather`` output, so
    each member's existing downstream (the optional ``slice`` back to the true
    full size, then the real use) is simply re-pointed from its old
    ``wait_tensor`` to ``out_i``.  The old per-member ``all_gather`` and
    ``wait_tensor`` are erased.

    Only the single ``ag`` tensor crosses submod boundaries (the ``split_with_sizes``
    list and its getitems live entirely on the use side), so this avoids the AOT
    output-spec mismatch that the list-returning ``all_gather_into_tensor_coalesced``
    op triggers under piecewise split.

    Runs AFTER redistribute lowering and BEFORE the collective prefetch pass.
    Returns the number of coalesced buckets created.
    """
    node_index = {n: i for i, n in enumerate(graph.graph.nodes)}

    groups: dict[tuple[int, str, torch.dtype], list[fx.Node]] = defaultdict(list)
    for node in graph.graph.nodes:
        if not _is_weight_all_gather(node):
            continue
        sid = node_to_subgraph_id.get(node)
        if sid is None:
            continue
        _, _world, group_name = node.args
        dtype = node.meta["example_value"].dtype
        groups[(sid, group_name, dtype)].append(node)

    buckets = 0
    for (sid, group_name, dtype), ag_nodes in groups.items():
        if len(ag_nodes) < 2:
            continue  # nothing to coalesce

        world = int(ag_nodes[0].args[1])
        ag_nodes.sort(key=lambda n: node_index[n])

        # Per-member: the padded local shard (ag input) and its full gathered meta.
        locals_ = [ag.args[0] for ag in ag_nodes]
        ag_metas = [ag.meta["example_value"] for ag in ag_nodes]  # (W*chunk_i, *rest_i)
        local_metas = [loc.meta["example_value"] for loc in locals_]  # (chunk_i, *rest_i)
        numels = [int(math.prod(m.shape)) for m in local_metas]
        sum_numel = sum(numels)
        dev = local_metas[0].device

        first_ag = ag_nodes[0]
        # Sole user of each member's all_gather is its wait_tensor.
        waits = [next(iter(ag.users)) for ag in ag_nodes]

        # Hoist every member's local-shard prep (to_local / _to_copy / pad chain)
        # above the FIRST member's all_gather, so all `locals_` are defined before
        # the coalesced launch we insert there.  These chains depend only on the
        # weight placeholder, so they can always move up.
        for loc in locals_:
            chain = [loc, *_producer_chain(loc)] if loc.op in ("call_function", "call_method") else _producer_chain(loc)
            for prod in sorted(chain, key=lambda n: node_index[n]):
                first_ag.prepend(prod)
                node_to_subgraph_id[prod] = sid

        # ---- launch side: flatten each local, concat, single all_gather ----
        # Insert just before the FIRST member's all_gather (all locals now precede it).
        with graph.graph.inserting_before(first_ag):
            flats = []
            for loc, lm, n in zip(locals_, local_metas, numels):
                fl = graph.graph.call_function(_RESHAPE, (loc, [-1]))
                fl.meta["example_value"] = lm.reshape(-1)
                node_to_subgraph_id[fl] = sid
                flats.append(fl)
            cat_in = graph.graph.call_function(_CAT, (flats, 0))
            cat_in.meta["example_value"] = local_metas[0].new_empty((sum_numel,))
            node_to_subgraph_id[cat_in] = sid

            ag = graph.graph.call_function(_ALL_GATHER, (cat_in, world, group_name))
            ag.meta["example_value"] = local_metas[0].new_empty((world * sum_numel,))
            ag.meta["magi_fsdp_weight_ag"] = True  # so the prefetch pass moves it
            node_to_subgraph_id[ag] = sid

        # ---- use side: one wait, reshape, split back to per-member tensors ----
        # Insert before the EARLIEST member's wait (its use site within this submod).
        first_wait = min(waits, key=lambda n: node_index[n])
        with graph.graph.inserting_before(first_wait):
            waited = graph.graph.call_function(_WAIT, (ag,))
            waited.meta["example_value"] = ag.meta["example_value"]
            node_to_subgraph_id[waited] = sid

            resh = graph.graph.call_function(_RESHAPE, (waited, [world, sum_numel]))
            resh.meta["example_value"] = local_metas[0].new_empty((world, sum_numel))
            node_to_subgraph_id[resh] = sid

            split = graph.graph.call_function(_SPLIT, (resh, numels, 1))
            split.meta["example_value"] = [local_metas[0].new_empty((world, n)) for n in numels]
            node_to_subgraph_id[split] = sid

            for i, (am, lm) in enumerate(zip(ag_metas, local_metas)):
                gi = graph.graph.call_function(operator.getitem, (split, i))
                gi.meta["example_value"] = local_metas[0].new_empty((world, numels[i]))
                node_to_subgraph_id[gi] = sid
                out_i = graph.graph.call_function(_RESHAPE, (gi, list(am.shape)))
                out_i.meta["example_value"] = am  # (W*chunk_i, *rest_i): same as old ag out
                node_to_subgraph_id[out_i] = sid

                # Re-point the member's downstream (slice / real use) from its old
                # wait to out_i, then drop the old wait + all_gather.
                waits[i].replace_all_uses_with(out_i)

        for ag_old, wait_old in zip(ag_nodes, waits):
            node_to_subgraph_id.pop(wait_old, None)
            node_to_subgraph_id.pop(ag_old, None)
            graph.graph.erase_node(wait_old)
            graph.graph.erase_node(ag_old)

        buckets += 1

    if buckets:
        graph.graph.lint()
        graph.recompile()
    magi_logger.info(
        "FSDP weight all-gather bucketing (concat): created %d coalesced buckets across submods", buckets
    )
    return buckets


def bucket_weight_all_gather_coalesced_per_submod(
    graph: fx.GraphModule,
    node_to_subgraph_id: dict[fx.Node, int],
) -> int:
    """Coalesce, per submod, the SimpleFSDP weight all-gathers into ONE
    ``all_gather_into_tensor_coalesced`` per ``(subgraph_id, group_name, dtype)``.

    Unlike :func:`bucket_weight_all_gather_per_submod` (the ``concat`` strategy),
    this does NOT cat the shards into one buffer.  ``all_gather_into_tensor_coalesced``
    fuses the N launches into a single NCCL group while returning one
    *weight-major* output buffer per input, so each member is recovered with a
    zero-copy ``operator.getitem`` -- no ``cat`` on the compute stream, no
    ``split_with_sizes`` clone, and no transient ~2x memory spike that the concat
    path incurs.

    For a group of N weight gathers (each a single ``all_gather_into_tensor`` whose
    input is the padded local shard ``(chunk_i, *rest_i)``) this builds::

        # launch side (kept together in ONE submod -- the list must not cross a
        # piecewise split boundary):
        coalesced = all_gather_into_tensor_coalesced([local_0, ..., local_{N-1}], W, group)
        out_i     = getitem(coalesced, i)          # (W*chunk_i, *rest_i), weight-major
        # use side (one wait per member, left at its consumer):
        wait_i    = wait_tensor(out_i)

    ``out_i`` has exactly the shape of the member's old ``all_gather`` output, so
    each member's existing downstream (slice / real use) is simply re-pointed from
    its old ``wait_tensor`` to ``wait_i``.  The ``coalesced`` launch and all its
    ``getitem`` nodes stay in the launch submod; only the per-member ``out_i``
    single tensors cross to the consumer submod.  The prefetch pass moves the
    launch together with its getitems (see ``apply_fsdp_collective_prefetch``).

    Runs AFTER redistribute lowering and BEFORE the collective prefetch pass.
    Returns the number of coalesced buckets created.
    """
    node_index = {n: i for i, n in enumerate(graph.graph.nodes)}

    groups: dict[tuple[int, str, torch.dtype], list[fx.Node]] = defaultdict(list)
    for node in graph.graph.nodes:
        if not _is_weight_all_gather(node):
            continue
        sid = node_to_subgraph_id.get(node)
        if sid is None:
            continue
        _, _world, group_name = node.args
        dtype = node.meta["example_value"].dtype
        groups[(sid, group_name, dtype)].append(node)

    buckets = 0
    for (sid, group_name, dtype), ag_nodes in groups.items():
        if len(ag_nodes) < 2:
            continue  # nothing to coalesce

        world = int(ag_nodes[0].args[1])
        ag_nodes.sort(key=lambda n: node_index[n])

        locals_ = [ag.args[0] for ag in ag_nodes]
        ag_metas = [ag.meta["example_value"] for ag in ag_nodes]  # (W*chunk_i, *rest_i)

        first_ag = ag_nodes[0]
        # Sole user of each member's all_gather is its wait_tensor.
        waits = [next(iter(ag.users)) for ag in ag_nodes]

        # Hoist every member's local-shard prep (to_local / _to_copy / pad chain)
        # above the FIRST member's all_gather so all `locals_` precede the
        # coalesced launch we insert there.  These depend only on the weight
        # placeholder, so they can always move up.
        for loc in locals_:
            chain = [loc, *_producer_chain(loc)] if loc.op in ("call_function", "call_method") else _producer_chain(loc)
            for prod in sorted(chain, key=lambda n: node_index[n]):
                first_ag.prepend(prod)
                node_to_subgraph_id[prod] = sid

        # ---- launch side: ONE coalesced all_gather + per-member getitem unpack ----
        # Insert before the FIRST member's all_gather (all locals now precede it).
        with graph.graph.inserting_before(first_ag):
            coalesced = graph.graph.call_function(_ALL_GATHER_COALESCED, (list(locals_), world, group_name))
            # The coalesced op returns a list[Tensor]; meta mirrors the per-member
            # gathered shapes (weight-major, same as each old all_gather output).
            coalesced.meta["example_value"] = list(ag_metas)
            coalesced.meta["magi_fsdp_weight_ag"] = True  # so prefetch moves it
            coalesced.meta["magi_fsdp_weight_ag_coalesced"] = True  # move getitems too
            node_to_subgraph_id[coalesced] = sid

            outs = []
            for i, am in enumerate(ag_metas):
                gi = graph.graph.call_function(operator.getitem, (coalesced, i))
                gi.meta["example_value"] = am  # (W*chunk_i, *rest_i)
                node_to_subgraph_id[gi] = sid
                outs.append(gi)

        # ---- use side: one wait per member, left before its consumer ----
        for i, (out_i, old_wait) in enumerate(zip(outs, waits)):
            with graph.graph.inserting_before(old_wait):
                wait_i = graph.graph.call_function(_WAIT, (out_i,))
                wait_i.meta["example_value"] = ag_metas[i]
                node_to_subgraph_id[wait_i] = node_to_subgraph_id.get(old_wait, sid)
            old_wait.replace_all_uses_with(wait_i)

        for ag_old, wait_old in zip(ag_nodes, waits):
            node_to_subgraph_id.pop(wait_old, None)
            node_to_subgraph_id.pop(ag_old, None)
            graph.graph.erase_node(wait_old)
            graph.graph.erase_node(ag_old)

        buckets += 1

    if buckets:
        graph.graph.lint()
        graph.recompile()
    magi_logger.info(
        "FSDP weight all-gather bucketing (coalesced): created %d coalesced buckets across submods", buckets
    )
    return buckets
