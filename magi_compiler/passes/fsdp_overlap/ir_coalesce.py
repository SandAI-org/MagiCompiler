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

"""Coalesce weight all-gathers after fusion, inside the Inductor scheduler.

By the time the reorder passes run, every weight gather is an out-of-place
``_CollectiveKernel`` whose operation name and buffer name are what the rest of
the graph refers to: its ``_WaitKernel`` mutates that buffer, and the kernels
that read the gathered weight -- many of them already fused into one Triton
body -- name it in their loop bodies.  Renaming any of that after fusion is not
practical, so a bucket keeps every name it can:

    before:  ag_i = all_gather(in_i)              (op_i / buf_i, one per member)
    after:   packed = all_gather_coalesced([in_0 .. in_k])      (new names)
             buf_i  = packed[i]                   (MultiOutput, takes op_i / buf_i)

Only the producer of ``buf_i`` changes; its waits and readers are untouched.

A copy-engine gather (``magi::ce_all_gather``) is a custom op, so it lowers to a
``FallbackKernel`` whose result reaches its readers through a ``MultiOutput``:
the gathered bytes, the waits and the readers all hang off the unpack.  There
the unpack's names are the ones kept, and the launch is simply no longer scheduled::

    before:  k_i = ce_all_gather(in_i); buf_i = k_i[0]     (one pair per member)
    after:   packed = ce_all_gather_coalesced([in_0 .. in_k])
             buf_i  = packed[i]                   (MultiOutput, takes buf_i's names)
The swap follows what ``Scheduler.finalize_multi_template_buffers`` does for a
template choice: the new IR node takes the old names and the old slots in
``V.graph.buffers`` / ``V.graph.operations``, and the new scheduler node takes
over the old one's ``SchedulerBuffer`` users.

Memory-planning info is the one piece of state that has to be rebuilt rather
than patched.  Inductor estimates peak memory around every reorder pass over the
list object it passed IN, and again in the wrapper over the final order; both
walk each buffer's ``mpi_buffer.succ_nodes`` and fail on a successor that is not
in the list.  So the pass that coalesces must write its result back into that
same list and then call ``refresh_memory_planning`` once, after its last merge.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch
from torch._inductor import ir
from torch._inductor.comms import _is_fake_dep
from torch._inductor.scheduler import BaseSchedulerNode, NodeUser
from torch._inductor.utils import contains_wait
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

_AG = torch.ops._c10d_functional.all_gather_into_tensor.default
_AG_COALESCED = torch.ops._c10d_functional.all_gather_into_tensor_coalesced.default


def _ce_ops() -> tuple:
    """(single, coalesced) copy-engine gather ops; (None, None) without symmetric memory."""
    try:
        from magi_compiler.symm_mem.all_gather import CE_ALL_GATHER, CE_ALL_GATHER_COALESCED

        return CE_ALL_GATHER, CE_ALL_GATHER_COALESCED
    except Exception:  # noqa: BLE001
        return None, None


@dataclass(frozen=True)
class GatherInfo:
    """What bucketing needs to know about one single-output weight gather."""

    group_name: str
    world: int
    dtype: torch.dtype
    shard_bytes: int  # bytes this rank contributes
    kind: str = "nccl"  # "nccl" | "ce": which op, and which IR shape (see module docstring)


def gather_info(snode: BaseSchedulerNode) -> GatherInfo | None:
    """``GatherInfo`` for a single-weight NCCL or copy-engine gather launch, else None.

    Coalesced gathers and anything grouped are not candidates: this module only
    knows how to merge the one-weight form.
    """
    node = getattr(snode, "node", None)
    op = getattr(node, "op_overload", None)
    if isinstance(node, ir._CollectiveKernel) and op is _AG:
        if not isinstance(node.layout, ir.Layout):
            return None
        kind = "nccl"
    elif isinstance(node, ir.FallbackKernel) and op is not None and op is _ce_ops()[0]:
        kind = "ce"
    else:
        return None
    if len(node.inputs) != 1:
        return None
    try:
        world, group_name = node.constant_args
        world = int(world)
    except (TypeError, ValueError):
        return None
    inp = node.inputs[0]
    try:
        numel = 1
        for d in inp.get_size():
            numel *= int(V.graph.sizevars.size_hint(d, fallback=0))
        dtype = inp.get_dtype()
    except Exception:  # noqa: BLE001
        return None
    return GatherInfo(str(group_name), world, dtype, numel * dtype.itemsize, kind)


def _unpacks_of(snode: BaseSchedulerNode, order: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    """The ``MultiOutput`` snodes a launch's result reaches its readers through (none for NCCL)."""
    names = set(snode.get_buffer_names())
    return [
        s
        for s in order
        if isinstance(getattr(s, "node", None), ir.MultiOutput)
        and any((not _is_fake_dep(d)) and d.name in names for d in s.unmet_dependencies)
    ]


def _waits_of(snode: BaseSchedulerNode, order: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    names = set(snode.get_buffer_names())
    for u in _unpacks_of(snode, order):
        names.update(u.get_buffer_names())
    return [
        s for s in order if contains_wait(s) and any((not _is_fake_dep(d)) and d.name in names for d in s.unmet_dependencies)
    ]


def _producer_index(snode, index_of, buf_to_snode, skip) -> int:
    """1 + index of the latest real producer ``snode`` reads, ignoring ``skip``."""
    lo = 0
    for d in snode.unmet_dependencies:
        if _is_fake_dep(d):
            continue
        prod = buf_to_snode.get(d.name)
        if prod is None or prod in skip:
            continue
        lo = max(lo, index_of[prod] + 1)
    return lo


@dataclass
class Placement:
    """Where a bucket's launch goes, and which shard prep has to move up to it."""

    at: int  # index in the ORIGINAL order the launch is inserted in front of
    hoisted: list[BaseSchedulerNode]  # prep snodes at or after ``at``, program order


def coalesce_placement(order: list[BaseSchedulerNode], members: list[BaseSchedulerNode]) -> Placement | None:
    """Where the packed launch of ``members`` can go, or None if nowhere.

    The launch has to follow every member's input and precede every member's
    wait.  In the order Inductor hands over, each gather sits just above its
    own consumer, so a later member's shard prep (cast, pad, host load) usually
    runs after the first member's wait.  That prep reads nothing but graph
    inputs, so it is hoisted with the launch -- the same travellers the reorder
    pass moves with a gather.  Prep that already runs above the launch stays
    put: moving a producer down could put it under one of its other readers.
    """
    from ..overlap import SnodeGraph
    from .reorder import FsdpOverlapReorder

    graph = SnodeGraph(order)
    index_of, buf_to_snode = graph.index_of, graph.buf_to_snode
    prep = FsdpOverlapReorder._prep_chain(list(members), graph)
    moving = set(members) | set(prep) | {u for m in members for u in _unpacks_of(m, order)}
    at = min(index_of[m] for m in members)
    at = max([at] + [_producer_index(s, index_of, buf_to_snode, moving) for s in moving])
    waits = [w for m in members for w in _waits_of(m, order)]
    if len(waits) < len(members) or at > min(index_of[w] for w in waits):
        return None
    return Placement(at=at, hoisted=[p for p in prep if index_of[p] >= at])


def _explain_placement(order: list[BaseSchedulerNode], members: list[BaseSchedulerNode]) -> str:
    """Why ``coalesce_placement`` found no index: each member's position, its waits, and the latest producer."""
    from ..overlap import SnodeGraph
    from .reorder import FsdpOverlapReorder

    graph = SnodeGraph(order)
    index_of, buf_to_snode = graph.index_of, graph.buf_to_snode
    prep = FsdpOverlapReorder._prep_chain(list(members), graph)
    moving = set(members) | set(prep) | {u for m in members for u in _unpacks_of(m, order)}
    parts = []
    for m in members:
        unpacks = _unpacks_of(m, order)
        waits = _waits_of(m, order)
        parts.append(
            f"{m.get_name()}@{index_of[m]} unpacks={[index_of[u] for u in unpacks]} waits={[index_of[w] for w in waits]}"
        )
    latest = max(moving, key=lambda s: _producer_index(s, index_of, buf_to_snode, moving))
    floor = _producer_index(latest, index_of, buf_to_snode, moving)
    culprit = [
        f"{buf_to_snode[d.name].get_name()}@{index_of[buf_to_snode[d.name]]}"
        for d in latest.unmet_dependencies
        if not _is_fake_dep(d) and d.name in buf_to_snode and buf_to_snode[d.name] not in moving
    ]
    return f"members [{'; '.join(parts)}]; launch floor {floor} from {latest.get_name()} reading {culprit}"


def _reordered(order, removed, placement: Placement, block) -> list[BaseSchedulerNode]:
    """``order`` without ``removed``, with the hoisted prep and then ``block`` at the placement."""
    removed, hoisted = set(removed), set(placement.hoisted)
    index_of = {s: i for i, s in enumerate(order)}
    rest = [s for s in order if s not in removed and s not in hoisted]
    cut = sum(1 for s in rest if index_of[s] < placement.at)
    return rest[:cut] + placement.hoisted + list(block) + rest[cut:]


def simulate_coalesce(order: list[BaseSchedulerNode], members: list[BaseSchedulerNode]) -> list[BaseSchedulerNode] | None:
    """The order ``coalesce_all_gathers`` would produce, with the members standing
    in for the packed launch and its unpacks; None if the merge is not legal.

    Dependency-wise the stand-in is exact -- the launch reads what the members
    read, the unpacks produce what they produced -- so a planner can check a
    whole sequence of merges before touching any IR.
    """
    members = sorted(members, key=order.index)
    placement = coalesce_placement(order, members)
    if placement is None:
        return None
    unpacks = [u for m in members for u in _unpacks_of(m, order)]
    return _reordered(order, [*members, *unpacks], placement, [*members, *unpacks])


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
    except Exception:  # noqa: BLE001
        return False


def _as_kernel_input(x: ir.IRNode) -> ir.IRNode:
    """Wrap an already-realized kernel input so ``realize_input`` takes it as is.

    A bare ``Buffer`` is not something lowering ever hands ``realize_input``,
    and it answers one with ``copy_input`` -- a fresh buffer that nothing
    schedules, which the wrapper then reads before it exists.
    """
    if isinstance(x, (ir.StorageBox, ir.TensorBox)):
        return x
    if isinstance(x, ir.ReinterpretView):
        data = x.data if isinstance(x.data, (ir.StorageBox, ir.TensorBox)) else ir.StorageBox(x.data)
        return ir.ReinterpretView(data=data, layout=x.layout)
    if isinstance(x, ir.Buffer):
        return ir.StorageBox(x)
    raise ValueError(f"unsupported all-gather input {type(x).__name__}")


def _replace_ir(orig: ir.Buffer, new: ir.Buffer) -> None:
    """Give ``new`` the names and list slots of ``orig`` (see module docstring)."""
    graph = V.graph
    new_buf, orig_buf = new.get_name(), orig.get_name()
    new_op, orig_op = new.get_operation_name(), orig.get_operation_name()

    del graph.name_to_buffer[new_buf]
    new.name = orig_buf
    del graph.name_to_op[new_op]
    new.operation_name = orig_op

    graph.buffers.remove(new)
    graph.buffers[graph.buffers.index(orig)] = new
    graph.name_to_buffer[orig_buf] = new
    graph.operations.remove(new)
    graph.operations[graph.operations.index(orig)] = new
    graph.name_to_op[orig_op] = new

    unaligned = getattr(graph, "unaligned_buffers", None)
    if unaligned is not None and new_buf in unaligned:
        unaligned.discard(new_buf)
        unaligned.add(orig_buf)


def _repoint_inputs(snodes, old: ir.Buffer, new: ir.Buffer) -> None:
    """Swap ``old`` for ``new`` wherever an extern kernel holds it as an input.

    Codegen only needs the name, which is unchanged, but a stale object left in
    ``inputs`` answers ``isinstance`` questions -- ``_WaitKernel`` asks one to
    find the volatile read -- with the replaced node's class.
    """
    for s in snodes:
        for sub in getattr(s, "snodes", None) or (s,):
            node = getattr(sub, "node", None)
            inputs = getattr(node, "inputs", None)
            if not isinstance(node, ir.ExternKernel) or not inputs:
                continue
            for j, inp in enumerate(inputs):
                if inp is old:
                    inputs[j] = new
                elif isinstance(inp, ir.ReinterpretView) and inp.data is old:
                    inp.data = new
                elif isinstance(inp, ir.StorageBox) and inp.data is old:
                    inp.data = new


def _rename_for_mutations(scheduler, olds, new_snode) -> None:
    """Carry mutation renames the replaced snodes' reads were given at init."""
    renames = {}
    for old in olds:
        for dep in itertools.chain(old.read_writes.reads, old.unmet_dependencies):
            real = scheduler.mutation_real_name.get(dep.name)
            if real is not None:
                renames[real] = dep.name
    if not renames:
        return
    new_snode.unmet_dependencies = OrderedSet(d.rename(renames) for d in new_snode.unmet_dependencies)
    new_snode.read_writes.reads = OrderedSet(d.rename(renames) for d in new_snode.read_writes.reads)


def refresh_memory_planning(nodes: list[BaseSchedulerNode]) -> None:
    """Rebuild every buffer's size and successor info for ``nodes``."""
    from torch._inductor.memory import assign_memory_planning_info_for_scheduler_buffers

    if nodes:
        assign_memory_planning_info_for_scheduler_buffers(nodes, nodes[0].scheduler.name_to_buf)


def coalesce_all_gathers(
    order: list[BaseSchedulerNode], members: list[BaseSchedulerNode]
) -> tuple[list[BaseSchedulerNode], BaseSchedulerNode, list[BaseSchedulerNode]]:
    """Merge ``members`` (>= 2 single-weight gathers of one kind, group and dtype) into one.

    Returns ``(new_order, packed, unpacks)``: ``new_order`` is a fresh list --
    ``order`` itself is not mutated -- with the members (and, for copy-engine
    gathers, their unpacks) replaced by the packed launch followed by one
    ``MultiOutput`` per member, in member order.  Raises ``ValueError`` when the
    merge is not legal; the caller is expected to have checked
    ``coalesce_placement`` and ``gather_info`` first, so on a multi-rank run a
    raise here is a divergence and must not be swallowed rank-locally.
    """
    if len(members) < 2:
        raise ValueError("a bucket needs at least two gathers")
    members = sorted(members, key=order.index)
    infos = [gather_info(m) for m in members]
    if any(i is None for i in infos):
        raise ValueError("every member must be a single-weight all-gather")
    if len({(i.kind, i.group_name, i.world, i.dtype) for i in infos}) != 1:
        raise ValueError("members span more than one (kind, group, world, dtype)")
    placement = coalesce_placement(order, members)
    if placement is None:
        raise ValueError(
            "no index follows every member's input and precedes every member's wait: " + _explain_placement(order, members)
        )

    ce = infos[0].kind == "ce"
    scheduler = members[0].scheduler
    launches = [m.node for m in members]
    world, group_name = launches[0].constant_args
    if ce:
        # The names to keep are the unpacks': waits and readers hang off them.
        keep_snodes = []
        for m in members:
            unpacks = _unpacks_of(m, order)
            if len(unpacks) != 1:
                raise ValueError(f"copy-engine gather {m.get_name()} has {len(unpacks)} unpacks, expected one")
            keep_snodes.append(unpacks[0])
        op = _ce_ops()[1]
    else:
        keep_snodes = list(members)
        op = _AG_COALESCED
    olds = [k.node for k in keep_snodes]

    # ``process_kernel`` consults the FX node being lowered for unbacked-symbol
    # bindings; lowering is long over, and a gather introduces no new symbols.
    stand_in = torch.fx.Graph().call_function(op, ())
    n_buffers = len(V.graph.buffers)
    inputs = [_as_kernel_input(o.inputs[0]) for o in launches]
    with V.set_current_node(stand_in):
        if ce:
            unpacks_ir = list(ir.FallbackKernel.create(op, inputs, world, group_name))
        else:
            unpacks_ir = ir._CollectiveKernel.create_out_of_place(op, inputs, world, group_name)
    assert isinstance(unpacks_ir, list) and len(unpacks_ir) == len(olds)
    packed_ir = unpacks_ir[0].inputs[0]
    if packed_ir.get_name() in scheduler.name_to_buf or packed_ir.get_operation_name() in scheduler.name_to_node:
        raise ValueError(f"the coalesced gather was named {packed_ir.get_name()}, which is already in use")
    # Anything beyond the packed launch and its unpacks is a buffer lowering
    # made on the side (a copy of an input) that no scheduler node will run.
    if len(V.graph.buffers) != n_buffers + 1 + len(olds):
        raise ValueError("building the coalesced gather registered buffers it cannot schedule")
    if [i.get_name() for i in packed_ir.inputs] != [o.inputs[0].get_name() for o in launches]:
        raise ValueError("the coalesced gather does not read the members' own inputs")
    packed_ir.origins = OrderedSet(o for launch in launches for o in launch.origins)

    for old, mo in zip(olds, unpacks_ir):
        if tuple(old.get_size()) != tuple(mo.get_size()) or old.get_dtype() != mo.get_dtype():
            raise ValueError(f"coalesced output {mo.get_size()} does not match {old.get_name()} {old.get_size()}")
        mo.origins = OrderedSet(old.origins)
        _replace_ir(old, mo)
    # A copy-engine member's own launch stays in ``V.graph.buffers`` / ``operations``,
    # unscheduled: Inductor names a new node after the lists' lengths, so shrinking
    # them would hand a later merge a name already in use.

    packed = scheduler.create_scheduler_node(packed_ir)
    _rename_for_mutations(scheduler, members, packed)
    packed.ancestors = OrderedSet(a for m in members for a in m.ancestors)
    packed.min_order = min(m.min_order for m in members)
    packed.max_order = packed.min_order
    scheduler.name_to_node[packed.get_name()] = packed
    scheduler.name_to_fused_node[packed.get_name()] = packed
    for buf in packed.get_outputs():
        scheduler.name_to_buf[buf.get_name()] = buf
    if ce:
        for m in members:
            for out in m.get_outputs():
                scheduler.name_to_buf.pop(out.get_name(), None)

    unpacks: list[BaseSchedulerNode] = []
    for kept, mo in zip(keep_snodes, unpacks_ir):
        unpack = scheduler.create_scheduler_node(mo)
        unpack.ancestors = OrderedSet(kept.ancestors) | OrderedSet([packed.get_name()])
        unpack.min_order = kept.min_order
        unpack.max_order = kept.max_order
        scheduler.name_to_node[unpack.get_name()] = unpack
        scheduler.name_to_fused_node[unpack.get_name()] = unpack
        for new_out, old_out in zip(unpack.get_outputs(), kept.get_outputs()):
            new_out.users = list(old_out.users)
            scheduler.name_to_buf[old_out.get_name()] = new_out
        unpacks.append(unpack)

    packed_buf = packed.get_outputs()[0]
    packed_buf.users = [NodeUser(u, can_inplace=False) for u in unpacks]

    member_set = set(members)
    for dep in packed.unmet_dependencies:
        buf = scheduler.name_to_buf.get(dep.name)
        if buf is None:
            continue
        users = [u for u in buf.users if u.node not in member_set]
        users.append(NodeUser(packed, can_inplace=False))
        buf.users = users

    removed = [*members, *keep_snodes] if ce else members
    new_order = _reordered(order, removed, placement, [packed, *unpacks])
    others = [s for s in new_order if s is not packed and s not in unpacks]
    for old, mo in zip(olds, unpacks_ir):
        _repoint_inputs(others, old, mo)
    return new_order, packed, unpacks
