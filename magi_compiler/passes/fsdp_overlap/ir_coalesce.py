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

"""Coalesce weight all-gathers after fusion while preserving buffer names.

Fused readers and waits refer to existing names;
each replacement unpack retains its member's names and SchedulerBuffer users:

    NCCL: ag_i = all_gather(in_i) -> packed = all_gather_coalesced(inputs)
                                   ag_i = packed[i]
    CE:   k_i = ce_all_gather(in_i); buf_i = k_i[0]
          -> packed = ce_all_gather_coalesced(inputs); buf_i = packed[i]

NCCL members become MultiOutputs; CE members reuse their existing unpack names.
Replacement IR nodes take the old slots in V.graph.buffers / operations.
CE launches remain in those lists, unscheduled, to prevent name reuse.

Inductor checks memory around each reorder pass using the list passed in.
The caller must update that list in place after its last merge;
``refresh_memory_planning`` then aligns buffer successor lists with the schedule.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch
from torch._inductor import ir
from torch._inductor.comms import _is_fake_dep
from torch._inductor.scheduler import BaseSchedulerNode, NodeUser
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

from ..overlap import SnodeGraph
from ..snode_utils import ALL_GATHER_OP, COALESCED_ALL_GATHER_OP, ce_ag_ops, ir_node_bytes, iter_ir_nodes


@dataclass(frozen=True)
class GatherInfo:
    """What bucketing needs to know about one single-output weight gather."""

    group_name: str
    world_size: int
    dtype: torch.dtype
    shard_bytes: int  # bytes this rank contributes
    transport: str = "nccl"  # "nccl" | "ce": which op, and which IR shape (see module docstring)


def parse_single_gather(snode: BaseSchedulerNode) -> GatherInfo | None:
    """``GatherInfo`` for a single-weight NCCL or copy-engine gather launch, else None.

    Coalesced gathers and anything grouped are not candidates: this module only
    knows how to merge the one-weight form.
    """
    node = snode.node
    if isinstance(node, ir._CollectiveKernel) and node.op_overload is ALL_GATHER_OP:
        if not isinstance(node.layout, ir.Layout):
            return None
        transport = "nccl"
    elif isinstance(node, ir.FallbackKernel) and node.op_overload is not None and node.op_overload is ce_ag_ops()[0]:
        transport = "ce"
    else:
        return None
    if len(node.inputs) != 1:
        return None
    try:
        world_size, group_name = node.constant_args
        world_size = int(world_size)
    except (TypeError, ValueError):
        return None
    input_ir = node.inputs[0]
    # Candidates are listed before the skeleton exchange, so an unsized input must not raise.
    if not isinstance(input_ir.maybe_get_layout(), ir.Layout):
        return None
    return GatherInfo(str(group_name), world_size, input_ir.get_dtype(), ir_node_bytes(input_ir), transport)


def _collect_merge_nodes(
    graph: SnodeGraph, members: list[BaseSchedulerNode]
) -> tuple[list[BaseSchedulerNode], set[BaseSchedulerNode]]:
    """``(prep_nodes, merge_nodes)``: the members' shard prep, and every snode that moves with their packed launch."""
    prep_nodes = graph.movable_prep_nodes(members)
    merge_nodes = set(members) | set(prep_nodes) | {u for m in members for u in graph.unpacks_of(m)}
    return prep_nodes, merge_nodes


@dataclass
class CoalescePlacement:
    """Where a bucket's launch goes, and which shard prep has to move up to it."""

    launch_index: int  # index in the ORIGINAL order the launch is inserted in front of
    hoisted_prep_nodes: list[BaseSchedulerNode]  # prep snodes at or after ``launch_index``, program order


def coalesce_placement(order: list[BaseSchedulerNode], members: list[BaseSchedulerNode]) -> CoalescePlacement | None:
    """Where the packed launch of ``members`` can go, or None if nowhere (see ``_find_coalesce_placement``)."""
    return _find_coalesce_placement(SnodeGraph(order), members)


def _find_coalesce_placement(graph: SnodeGraph, members: list[BaseSchedulerNode]) -> CoalescePlacement | None:
    """Where the packed launch of ``members`` can go, or None if nowhere.

    The launch has to follow every member's input and precede every member's
    wait.  In the order Inductor hands over, each gather sits just above its
    own consumer, so a later member's shard prep (cast, pad, host load) usually
    runs after the first member's wait.  That prep reads nothing but graph
    inputs, so it is hoisted with the launch -- the same travellers the reorder
    pass moves with a gather.  Prep that already runs above the launch stays
    put: moving a producer down could put it under one of its other readers.
    """
    prep_nodes, merge_nodes = _collect_merge_nodes(graph, members)
    original_index_by_node = graph.original_index_by_node
    launch_index = min(original_index_by_node[m] for m in members)
    launch_index = max(launch_index, graph.earliest_legal_index(merge_nodes))
    waits = [w for m in members for w in graph.waits_of(m)]
    if len(waits) < len(members) or launch_index > min(original_index_by_node[w] for w in waits):
        return None
    hoisted_prep_nodes = [p for p in prep_nodes if original_index_by_node[p] >= launch_index]
    return CoalescePlacement(launch_index=launch_index, hoisted_prep_nodes=hoisted_prep_nodes)


def _format_placement_constraints(graph: SnodeGraph, members: list[BaseSchedulerNode]) -> str:
    """Why ``_find_coalesce_placement`` found no index: each member's position, its waits, and the latest producer."""
    _, merge_nodes = _collect_merge_nodes(graph, members)
    original_index_by_node, producer_by_buffer = graph.original_index_by_node, graph.producer_by_buffer
    member_descriptions = []
    for m in members:
        unpack_indices = [original_index_by_node[u] for u in graph.unpacks_of(m)]
        wait_indices = [original_index_by_node[w] for w in graph.waits_of(m)]
        member_descriptions.append(f"{m.get_name()}@{original_index_by_node[m]} unpacks={unpack_indices} waits={wait_indices}")
    lower_index_by_node = {
        node: graph.earliest_legal_index(moving_nodes=[node], ignored_producers=merge_nodes) for node in merge_nodes
    }
    limiting_node = max(merge_nodes, key=lower_index_by_node.__getitem__)
    launch_floor = lower_index_by_node[limiting_node]
    blocking_producers = [
        f"{producer_by_buffer[d.name].get_name()}@{original_index_by_node[producer_by_buffer[d.name]]}"
        for d in limiting_node.unmet_dependencies
        if not _is_fake_dep(d) and d.name in producer_by_buffer and producer_by_buffer[d.name] not in merge_nodes
    ]
    return (
        f"members [{'; '.join(member_descriptions)}]; launch floor {launch_floor} "
        f"from {limiting_node.get_name()} reading {blocking_producers}"
    )


def _insert_coalesced_block(order, removed, placement: CoalescePlacement, block) -> list[BaseSchedulerNode]:
    """``order`` without ``removed``, with the hoisted prep and then ``block`` at the placement."""
    removed, hoisted = set(removed), set(placement.hoisted_prep_nodes)
    original_index_by_node = {s: i for i, s in enumerate(order)}
    remaining_nodes = [s for s in order if s not in removed and s not in hoisted]
    insert_position = sum(1 for s in remaining_nodes if original_index_by_node[s] < placement.launch_index)
    return remaining_nodes[:insert_position] + placement.hoisted_prep_nodes + list(block) + remaining_nodes[insert_position:]


def simulate_coalesced_order(
    order: list[BaseSchedulerNode], members: list[BaseSchedulerNode]
) -> list[BaseSchedulerNode] | None:
    """The order ``coalesce_all_gathers`` would produce, with the members standing
    in for the packed launch and its unpacks; None if the merge is not legal.

    Dependency-wise the stand-in is exact -- the launch reads what the members
    read, the unpacks produce what they produced -- so a planner can check a
    whole sequence of merges before touching any IR.
    """
    graph = SnodeGraph(order)
    members = sorted(members, key=graph.original_index_by_node.__getitem__)
    placement = _find_coalesce_placement(graph, members)
    if placement is None:
        return None
    unpacks = [u for m in members for u in graph.unpacks_of(m)]
    return _insert_coalesced_block(order, [*members, *unpacks], placement, [*members, *unpacks])


def ir_coalesce_supported() -> bool:
    """Whether this torch exposes the private Inductor pieces the surgery uses."""
    try:
        from torch._inductor.memory import assign_memory_planning_info_for_scheduler_buffers  # noqa: F401
        from torch._inductor.scheduler import Scheduler

        return (
            hasattr(ir._CollectiveKernel, "create_out_of_place")
            and hasattr(ir.FallbackKernel, "create")
            and hasattr(ir, "MultiOutput")
            and hasattr(Scheduler, "create_scheduler_node")
        )
    except ImportError:
        return False


def _as_kernel_input(input_ir: ir.IRNode) -> ir.IRNode:
    """Wrap an already-realized kernel input so ``realize_input`` takes it as is.

    A bare ``Buffer`` is not something lowering ever hands ``realize_input``,
    and it answers one with ``copy_input`` -- a fresh buffer that nothing
    schedules, which the wrapper then reads before it exists.
    """
    if isinstance(input_ir, (ir.StorageBox, ir.TensorBox)):
        return input_ir
    if isinstance(input_ir, ir.ReinterpretView):
        if isinstance(input_ir.data, (ir.StorageBox, ir.TensorBox)):
            input_buffer = input_ir.data
        else:
            input_buffer = ir.StorageBox(input_ir.data)
        return ir.ReinterpretView(data=input_buffer, layout=input_ir.layout)
    if isinstance(input_ir, ir.Buffer):
        return ir.StorageBox(input_ir)
    raise ValueError(f"unsupported all-gather input {type(input_ir).__name__}")


def _replace_ir_preserving_names(old_ir: ir.Buffer, replacement_ir: ir.Buffer) -> None:
    """Give ``replacement_ir`` the names and list slots of ``old_ir`` (see module docstring)."""
    graph = V.graph
    generated_buffer_name, old_buffer_name = replacement_ir.get_name(), old_ir.get_name()
    generated_operation_name, old_operation_name = replacement_ir.get_operation_name(), old_ir.get_operation_name()

    del graph.name_to_buffer[generated_buffer_name]
    replacement_ir.name = old_buffer_name
    del graph.name_to_op[generated_operation_name]
    replacement_ir.operation_name = old_operation_name

    graph.buffers.remove(replacement_ir)
    graph.buffers[graph.buffers.index(old_ir)] = replacement_ir
    graph.name_to_buffer[old_buffer_name] = replacement_ir
    graph.operations.remove(replacement_ir)
    graph.operations[graph.operations.index(old_ir)] = replacement_ir
    graph.name_to_op[old_operation_name] = replacement_ir

    if generated_buffer_name in graph.unaligned_buffers:
        graph.unaligned_buffers.discard(generated_buffer_name)
        graph.unaligned_buffers.add(old_buffer_name)


def _replace_input_references(snodes, old_ir: ir.Buffer, replacement_ir: ir.Buffer) -> None:
    """Swap ``old_ir`` for ``replacement_ir`` wherever an extern kernel holds it as an input.

    Codegen only needs the name, which is unchanged, but a stale object left in
    ``inputs`` answers ``isinstance`` questions -- ``_WaitKernel`` asks one to
    find the volatile read -- with the replaced node's class.
    """
    for s in snodes:
        for node in iter_ir_nodes(s):
            if not isinstance(node, ir.ExternKernel):
                continue
            for j, input_ir in enumerate(node.inputs):
                if input_ir is old_ir:
                    node.inputs[j] = replacement_ir
                elif isinstance(input_ir, ir.ReinterpretView) and input_ir.data is old_ir:
                    input_ir.data = replacement_ir
                elif isinstance(input_ir, ir.StorageBox) and input_ir.data is old_ir:
                    input_ir.data = replacement_ir


def _update_mutation_dependencies(scheduler, old_snodes, new_snode) -> None:
    """Carry mutation renames the replaced snodes' reads were given at init."""
    renamed_name_by_original_name = {}
    for old_snode in old_snodes:
        for dep in itertools.chain(old_snode.read_writes.reads, old_snode.unmet_dependencies):
            original_buffer_name = scheduler.mutation_real_name.get(dep.name)
            if original_buffer_name is not None:
                renamed_name_by_original_name[original_buffer_name] = dep.name
    if not renamed_name_by_original_name:
        return
    new_snode.unmet_dependencies = OrderedSet(d.rename(renamed_name_by_original_name) for d in new_snode.unmet_dependencies)
    new_snode.read_writes.reads = OrderedSet(d.rename(renamed_name_by_original_name) for d in new_snode.read_writes.reads)


def refresh_memory_planning(nodes: list[BaseSchedulerNode]) -> None:
    """Rebuild every buffer's size and successor info for ``nodes``."""
    from torch._inductor.memory import assign_memory_planning_info_for_scheduler_buffers

    if nodes:
        assign_memory_planning_info_for_scheduler_buffers(nodes, nodes[0].scheduler.name_to_buf)


def coalesce_all_gathers(
    order: list[BaseSchedulerNode], members: list[BaseSchedulerNode]
) -> tuple[list[BaseSchedulerNode], BaseSchedulerNode, list[BaseSchedulerNode]]:
    """Merge ``members`` (>= 2 single-weight gathers of one transport, group and dtype) into one.

    Returns ``(new_order, coalesced_snode, output_snodes)``: ``new_order`` is a
    fresh list -- ``order`` itself is not mutated -- with the members (and, for
    copy-engine gathers, their unpacks) replaced by the coalesced launch followed
    by one ``MultiOutput`` per member, in member order.  Raises ``ValueError``
    when the merge is not legal; the caller is expected to have checked
    ``coalesce_placement`` and ``parse_single_gather`` first, so on a multi-rank
    run a raise here is a divergence and must not be swallowed rank-locally.
    """
    if len(members) < 2:
        raise ValueError("a bucket needs at least two gathers")
    graph = SnodeGraph(order)
    members = sorted(members, key=graph.original_index_by_node.__getitem__)
    gather_infos = [parse_single_gather(m) for m in members]
    if any(info is None for info in gather_infos):
        raise ValueError("every member must be a single-weight all-gather")
    if len({(info.transport, info.group_name, info.world_size, info.dtype) for info in gather_infos}) != 1:
        raise ValueError("members span more than one (transport, group, world size, dtype)")
    placement = _find_coalesce_placement(graph, members)
    if placement is None:
        raise ValueError(
            "no index follows every member's input and precedes every member's wait: "
            + _format_placement_constraints(graph, members)
        )

    uses_copy_engine = gather_infos[0].transport == "ce"
    scheduler = members[0].scheduler
    launch_ir_nodes = [m.node for m in members]
    world_size, group_name = launch_ir_nodes[0].constant_args
    if uses_copy_engine:
        # The names to keep are the unpacks': waits and readers hang off them.
        output_snodes_to_preserve = []
        for m in members:
            unpacks = graph.unpacks_of(m)
            if len(unpacks) != 1:
                raise ValueError(f"copy-engine gather {m.get_name()} has {len(unpacks)} unpacks, expected one")
            output_snodes_to_preserve.append(unpacks[0])
        coalesced_op = ce_ag_ops()[1]
    else:
        output_snodes_to_preserve = list(members)
        coalesced_op = COALESCED_ALL_GATHER_OP
    old_output_ir_nodes = [s.node for s in output_snodes_to_preserve]

    # ``process_kernel`` consults the FX node being lowered for unbacked-symbol
    # bindings; lowering is long over, and a gather introduces no new symbols.
    lowering_fx_node = torch.fx.Graph().call_function(coalesced_op, ())
    buffer_count_before = len(V.graph.buffers)
    inputs = [_as_kernel_input(launch.inputs[0]) for launch in launch_ir_nodes]
    with V.set_current_node(lowering_fx_node):
        if uses_copy_engine:
            output_ir_nodes = list(ir.FallbackKernel.create(coalesced_op, inputs, world_size, group_name))
        else:
            output_ir_nodes = ir._CollectiveKernel.create_out_of_place(coalesced_op, inputs, world_size, group_name)
    assert isinstance(output_ir_nodes, list) and len(output_ir_nodes) == len(old_output_ir_nodes)
    coalesced_ir = output_ir_nodes[0].inputs[0]
    if coalesced_ir.get_name() in scheduler.name_to_buf or coalesced_ir.get_operation_name() in scheduler.name_to_node:
        raise ValueError(f"the coalesced gather was named {coalesced_ir.get_name()}, which is already in use")
    # Anything beyond the coalesced launch and its unpacks is a buffer lowering
    # made on the side (a copy of an input) that no scheduler node will run.
    if len(V.graph.buffers) != buffer_count_before + 1 + len(old_output_ir_nodes):
        raise ValueError("building the coalesced gather registered buffers it cannot schedule")
    if [i.get_name() for i in coalesced_ir.inputs] != [launch.inputs[0].get_name() for launch in launch_ir_nodes]:
        raise ValueError("the coalesced gather does not read the members' own inputs")
    coalesced_ir.origins = OrderedSet(o for launch in launch_ir_nodes for o in launch.origins)

    for old_output_ir, new_output_ir in zip(old_output_ir_nodes, output_ir_nodes):
        same_shape = tuple(old_output_ir.get_size()) == tuple(new_output_ir.get_size())
        if not same_shape or old_output_ir.get_dtype() != new_output_ir.get_dtype():
            raise ValueError(
                f"coalesced output {new_output_ir.get_size()} does not match "
                f"{old_output_ir.get_name()} {old_output_ir.get_size()}"
            )
        new_output_ir.origins = OrderedSet(old_output_ir.origins)
        _replace_ir_preserving_names(old_output_ir, new_output_ir)
    # A copy-engine member's own launch stays in ``V.graph.buffers`` / ``operations``,
    # unscheduled: Inductor names a new node after the lists' lengths, so shrinking
    # them would hand a later merge a name already in use.

    coalesced_snode = scheduler.create_scheduler_node(coalesced_ir)
    _update_mutation_dependencies(scheduler, members, coalesced_snode)
    coalesced_snode.ancestors = OrderedSet(a for m in members for a in m.ancestors)
    coalesced_snode.min_order = min(m.min_order for m in members)
    coalesced_snode.max_order = coalesced_snode.min_order
    scheduler.name_to_node[coalesced_snode.get_name()] = coalesced_snode
    scheduler.name_to_fused_node[coalesced_snode.get_name()] = coalesced_snode
    for buf in coalesced_snode.get_outputs():
        scheduler.name_to_buf[buf.get_name()] = buf
    if uses_copy_engine:
        for m in members:
            for out in m.get_outputs():
                scheduler.name_to_buf.pop(out.get_name(), None)

    output_snodes: list[BaseSchedulerNode] = []
    for preserved_output_snode, new_output_ir in zip(output_snodes_to_preserve, output_ir_nodes):
        output_snode = scheduler.create_scheduler_node(new_output_ir)
        output_snode.ancestors = OrderedSet(preserved_output_snode.ancestors) | OrderedSet([coalesced_snode.get_name()])
        output_snode.min_order = preserved_output_snode.min_order
        output_snode.max_order = preserved_output_snode.max_order
        scheduler.name_to_node[output_snode.get_name()] = output_snode
        scheduler.name_to_fused_node[output_snode.get_name()] = output_snode
        for new_out, old_out in zip(output_snode.get_outputs(), preserved_output_snode.get_outputs()):
            new_out.users = list(old_out.users)
            scheduler.name_to_buf[old_out.get_name()] = new_out
        output_snodes.append(output_snode)

    coalesced_buf = coalesced_snode.get_outputs()[0]
    coalesced_buf.users = [NodeUser(u, can_inplace=False) for u in output_snodes]

    member_set = set(members)
    for dep in coalesced_snode.unmet_dependencies:
        buf = scheduler.name_to_buf.get(dep.name)
        if buf is None:
            continue
        users = [u for u in buf.users if u.node not in member_set]
        users.append(NodeUser(coalesced_snode, can_inplace=False))
        buf.users = users

    removed = [*members, *output_snodes_to_preserve] if uses_copy_engine else members
    new_order = _insert_coalesced_block(order, removed, placement, [coalesced_snode, *output_snodes])
    remaining_snodes = [s for s in new_order if s is not coalesced_snode and s not in output_snodes]
    for old_output_ir, new_output_ir in zip(old_output_ir_nodes, output_ir_nodes):
        _replace_input_references(remaining_snodes, old_output_ir, new_output_ir)
    return new_order, coalesced_snode, output_snodes
