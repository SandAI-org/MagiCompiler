from __future__ import annotations

from collections import deque

import torch
import torch.fx as fx

from magi_compiler.utils import magi_logger

# The two functional-collective ops a SimpleFSDP weight gather lowers to once
# the DTensor ``redistribute``/``to_local`` prims have been turned into explicit
# collectives.  This pass operates on *these* nodes (not on the opaque DTensor
# prims), because only here are the launch (``all_gather_into_tensor``) and the
# wait (``wait_tensor``) separate, individually movable FX nodes.
_ALL_GATHER = torch.ops._c10d_functional.all_gather_into_tensor.default
_ALL_GATHER_COALESCED = torch.ops._c10d_functional.all_gather_into_tensor_coalesced.default
_WAIT_TENSOR = torch.ops._c10d_functional.wait_tensor.default


def _target_name(target: object) -> str:
    if hasattr(target, "name") and callable(getattr(target, "name")):
        try:
            return str(target.name())
        except Exception:
            pass
    name = getattr(target, "__name__", None)
    if name is not None:
        return str(name)
    return str(target)


def _is_all_gather(node: fx.Node) -> bool:
    # Both the single launch and the per-submod coalesced launch are movable.
    return node.op == "call_function" and node.target in (_ALL_GATHER, _ALL_GATHER_COALESCED)


def _is_wait_tensor(node: fx.Node) -> bool:
    return node.op == "call_function" and node.target is _WAIT_TENSOR


def _is_param_like(node: fx.Node) -> bool:
    """True for a placeholder/get_attr that names a SimpleFSDP weight shard."""
    if node.op not in ("placeholder", "get_attr"):
        return False
    name = f"{node.name} {node.target}".lower()
    return any(token in name for token in ("parameter", "parameters", "weight", "bias"))


def _is_transparent_producer(node: fx.Node) -> bool:
    """Cheap, structure-only ops that may sit *between* a weight placeholder and
    the ``all_gather`` (dtype cast / pad / view).  Used when walking backwards
    from the launch to (a) decide it gathers a weight and (b) collect the
    input-producing chain that must move together with the launch."""
    # The redistribute-lowering pass extracts the local shard with
    # ``call_method("to_local", (weight_placeholder,))`` before the gather, so
    # the chain from all_gather back to the weight passes through it.
    if node.op == "call_method" and str(node.target) in {"to_local", "contiguous", "to", "view", "reshape"}:
        return True
    if node.op != "call_function":
        return False
    name = _target_name(node.target)
    return any(
        token in name
        for token in (
            "constant_pad_nd",
            "_to_copy",
            "aten.to",
            "aten::to",
            "convert_element_type",
            # bucketing inserts cat(reshape(local),...) before the coalesced launch
            "aten.cat",
            "aten::cat",
            "aten.view",
            "aten::view",
            "aten.reshape",
            "aten::reshape",
            "_unsafe_view",
            "aten.clone",
            "aten::clone",
            "aten.detach",
            "aten::detach",
        )
    )


def _is_transparent_consumer(node: fx.Node) -> bool:
    """Nodes that forward the gathered weight to its real use without being the
    real use themselves: the ``wait_tensor`` plus cheap view/cast/slice ops."""
    if _is_wait_tensor(node):
        return True
    if node.op == "call_method" and str(node.target) in {
        "view",
        "reshape",
        "flatten",
        "squeeze",
        "unsqueeze",
        "permute",
        "transpose",
        "t",
        "contiguous",
        "to",
        "detach",
        "clone",
    }:
        return True
    if node.op != "call_function":
        return False
    name = _target_name(node.target)
    return any(
        token in name
        for token in (
            "aten.view",
            "aten::view",
            "aten.reshape",
            "aten::reshape",
            "_unsafe_view",
            "aten.squeeze",
            "aten::squeeze",
            "aten.unsqueeze",
            "aten::unsqueeze",
            "aten.permute",
            "aten::permute",
            "aten.transpose",
            "aten::transpose",
            "aten.t",
            "aten::t",
            "aten.slice",
            "aten::slice",
            # bucketing's use-side chain: wait -> reshape -> split_with_sizes -> getitem -> reshape
            "split_with_sizes",
            "aten.detach",
            "aten::detach",
            "aten.clone",
            "aten::clone",
            "aten.to",
            "aten::to",
            "_to_copy",
            "convert_element_type",
            "operator.getitem",
            "getitem",
        )
    )


def _gathers_a_weight(all_gather: fx.Node) -> bool:
    """Walk backwards from the launch through transparent producers; return True
    if the gathered source is a weight/bias placeholder."""
    q: deque[fx.Node] = deque(all_gather.all_input_nodes)
    seen: set[fx.Node] = set()
    while q:
        dep = q.popleft()
        if dep in seen:
            continue
        seen.add(dep)
        if _is_param_like(dep):
            return True
        if _is_transparent_producer(dep):
            q.extend(dep.all_input_nodes)
    return False


def _find_consumer_subgraph_id(
    all_gather: fx.Node, node_to_subgraph_id: dict[fx.Node, int]
) -> int | None:
    """Forward BFS through wait/transparent nodes to the first real consumer;
    return the minimum subgraph id among real consumers."""
    q: deque[fx.Node] = deque(all_gather.users)
    seen: set[fx.Node] = set()
    consumer_ids: list[int] = []
    while q:
        user = q.popleft()
        if user in seen:
            continue
        seen.add(user)
        if _is_transparent_consumer(user):
            q.extend(user.users)
            continue
        sid = node_to_subgraph_id.get(user)
        if sid is not None:
            consumer_ids.append(sid)
    if not consumer_ids:
        return None
    return min(consumer_ids)


def _collect_launch_input_chain(all_gather: fx.Node) -> list[fx.Node]:
    """Backwards from the launch, collect transparent producer nodes that feed
    *only* into this launch (their results are not used elsewhere).  These must
    move with the launch so it is self-contained in the previous submod.

    The weight placeholder/get_attr itself is NOT included (placeholders are not
    movable graph nodes); only intermediate compute that prepares the shard.
    """
    chain: list[fx.Node] = []
    q: deque[fx.Node] = deque(n for n in all_gather.all_input_nodes if _is_transparent_producer(n))
    seen: set[fx.Node] = set()
    frontier = {all_gather}
    while q:
        node = q.popleft()
        if node in seen:
            continue
        # Movable only if every consumer is within the chain we are moving.
        if not all(u in frontier or u in seen or u is all_gather for u in node.users):
            continue
        seen.add(node)
        chain.append(node)
        frontier.add(node)
        for dep in node.all_input_nodes:
            if _is_transparent_producer(dep):
                q.append(dep)
    return chain


def _build_first_compute_node_by_subgraph_id(
    graph: fx.GraphModule, node_to_subgraph_id: dict[fx.Node, int]
) -> dict[int, fx.Node]:
    """First *compute* node of each submod, used as the prefetch anchor.

    The anchor MUST be a node that genuinely STAYS in the target submod — i.e. a
    real compute node (matmul, custom op, norm, ...), NOT any weight-gather-chain
    node.  After redistribute lowering a gather is a chain
    ``to_local -> [_to_copy] -> [pad] -> all_gather -> wait -> [slice]``; every one
    of those is itself relocatable by this pass.  If we anchored on, say, the
    first ``to_local`` of the target submod, that node could be hoisted away to an
    even earlier submod, and prepending another launch before the (now-moved)
    anchor would drag it to the very front of the graph — corrupting submod
    ordering and making ``split_module`` raise KeyError on partition inputs.
    Skipping the whole gather-chain vocabulary keeps the anchor stable.
    """
    first: dict[int, fx.Node] = {}
    for node in graph.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue
        # Skip every node that is part of a (relocatable) weight-gather chain.
        if (
            _is_all_gather(node)
            or _is_wait_tensor(node)
            or _is_transparent_producer(node)
            or _is_transparent_consumer(node)
        ):
            continue
        sid = node_to_subgraph_id.get(node)
        if sid is not None and sid not in first:
            first[sid] = node
    return first


def _deps_available_before_anchor(
    nodes: list[fx.Node], anchor: fx.Node, node_index: dict[fx.Node, int]
) -> bool:
    """Every external dependency of the to-be-moved nodes must already be
    defined before the anchor (so the move does not read an undefined value)."""
    moved = set(nodes)
    anchor_index = node_index[anchor]
    for node in nodes:
        for dep in node.all_input_nodes:
            if dep in moved:
                continue
            if dep.op in ("placeholder", "get_attr"):
                continue
            if node_index.get(dep, anchor_index + 1) > anchor_index:
                return False
    return True


def apply_fsdp_collective_prefetch(
    graph: fx.GraphModule,
    node_to_subgraph_id: dict[fx.Node, int],
    *,
    distance: int = 1,
) -> int:
    """Prefetch SimpleFSDP weight all-gather across a submod boundary.

    For each ``all_gather_into_tensor`` whose gathered result feeds a weight use
    in submod ``N``, move ONLY the launch node (and the transparent producer
    chain that feeds exclusively into it) to the beginning of submod
    ``N - distance``.  The ``wait_tensor`` and everything after it are left
    untouched at the original use site in submod ``N``.

    Because ``split_module(keep_original_order=True)`` runs afterwards, the
    gathered tensor automatically becomes an output of submod ``N - distance``
    and an input of submod ``N`` -- so the collective is launched during the
    previous submod's compute and only waited on right before use, giving real
    compute/communication overlap.

    Unlike :func:`apply_fsdp_all_gather_prefetch`, this pass never moves the
    wait: at the explicit-collective level the launch and wait are distinct FX
    nodes, so leaving the wait behind actually keeps it at the consumer.

    IMPORTANT -- this FX move is necessary but NOT sufficient for overlap.  The
    hoisted ``all_gather`` has no consumer inside submod ``N - distance`` (its
    result is only a graph output), so Inductor's per-submod scheduler re-sinks
    it to just before ``return`` -- at runtime the launch is then issued AFTER
    that submod's compute and cannot overlap.  The caller must therefore also
    enable Inductor's comm/compute reorder with the ``raise_comms`` pass
    (``reorder_for_compute_comm_overlap=True``,
    ``reorder_for_compute_comm_overlap_passes=["raise_comms", "sink_waits"]``),
    which hoists the launch back to the top of the generated code.  MagiBackend
    wires this automatically when this pass moves any launch.
    """
    if distance <= 0:
        return 0

    first_node_by_subgraph_id = _build_first_compute_node_by_subgraph_id(graph, node_to_subgraph_id)
    node_index = {node: idx for idx, node in enumerate(graph.graph.nodes)}
    moved = 0
    changed = False

    for node in list(graph.graph.nodes):
        if not _is_all_gather(node):
            continue
        if not _gathers_a_weight(node):
            continue

        current_id = node_to_subgraph_id.get(node)
        if current_id is None:
            continue

        consumer_id = _find_consumer_subgraph_id(node, node_to_subgraph_id)
        if consumer_id is None:
            continue

        target_id = max(0, consumer_id - distance)
        if current_id <= target_id:
            # Already launched at or before the desired submod.
            continue

        anchor = first_node_by_subgraph_id.get(target_id)
        if anchor is None:
            continue

        # Move the launch plus the producer chain that feeds only into it.
        chain = _collect_launch_input_chain(node)
        # For a coalesced launch the op returns a list[Tensor] that must NOT cross
        # a submod boundary; its operator.getitem unpackers (whose only input is
        # the launch) move together with it so the list stays in the target submod
        # and only the per-member single tensors cross to the consumer.
        getitems = []
        if node.target is _ALL_GATHER_COALESCED or node.meta.get("magi_fsdp_weight_ag_coalesced"):
            getitems = [
                u for u in node.users
                if u.op == "call_function" and getattr(u.target, "__name__", "") == "getitem"
            ]
        # Order from earliest to latest so prepend keeps dependency order.
        move_group = sorted([node, *chain, *getitems], key=lambda n: node_index[n])

        if not _deps_available_before_anchor(move_group, anchor, node_index):
            magi_logger.debug(
                "Skip collective prefetch for %s: deps not available before %s", node.name, anchor.name
            )
            continue

        for moved_node in move_group:
            node_to_subgraph_id[moved_node] = target_id
            anchor.prepend(moved_node)

        node.meta["magi_fsdp_prefetch_from_subgraph"] = current_id
        node.meta["magi_fsdp_prefetch_to_subgraph"] = target_id
        node.meta["magi_fsdp_prefetch_for_consumer_subgraph"] = consumer_id

        moved += 1
        changed = True

    if changed:
        graph.graph.lint()
        graph.recompile()
    return moved
