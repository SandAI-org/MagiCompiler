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

"""Latest-safe-launch FSDP all-gather / compute overlap reorder pass.

Installed in ``reorder_for_compute_comm_overlap_passes`` in place of
``raise_comms``/``sink_waits``, right after ``SnodeCostProfile``, whose
``SnodeCostTable`` is this pass's ``cost_fn``; runs on the whole Inductor graph
(``disable_graph_split=True``).  For each FSDP weight all-gather launch, place it
at the LATEST position whose downstream compute still hides the collective::

    sum(compute runtime between launch and first-consumer) >= comm * scale + margin

Not enough upstream compute -> as-early-as-legal (never worse than raise_comms).

What is FSDP-specific lives here: which snodes a gather drags along, and the
cross-rank agreement a collective needs.  Placement is the shared
``alap_schedule``, which runs the one transfer stream on the compute time axis
as the load reorder does.  Rebuild and validation are ``OverlapPass``'s.

Handles both lowering forms: plain all_gather (1 launch / 1 wait) and coalesced
(1 packed launch + N MultiOutput members moved together as one block + N waits).
"""

from __future__ import annotations

import torch
from torch._inductor.comms import _is_fake_dep
from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.utils import contains_wait

from magi_compiler.passes.overlap import (
    DEFAULT_WINDOW_MARGIN_NS,
    CostView,
    OverlapPass,
    SnodeGraph,
    TransferTask,
    alap_schedule,
    enforce_monotone,
    rank_sync,
)
from magi_compiler.passes.snode_utils import (
    is_ce_ag_coalesced,
    is_ce_ag_ir,
    is_multi_output,
    is_weight_gather,
    issues_transfer,
    leaf_collective_node,
)
from magi_compiler.passes.weight_offload.schedule.h2d_snode import is_h2d_load
from magi_compiler.utils import magi_logger

_AG_COALESCED = torch.ops._c10d_functional.all_gather_into_tensor_coalesced.default


class FsdpOverlapReorder(OverlapPass):
    """Callable reorder pass."""

    name = "FSDP overlap reorder"

    def __init__(
        self,
        comm_overlap_window_margin_ns: float = DEFAULT_WINDOW_MARGIN_NS,
        cost_fn=None,
        comm_overlap_window_scale: float = 1.0,
    ) -> None:
        super().__init__(cost_fn)
        self.comm_overlap_window_margin_ns = comm_overlap_window_margin_ns
        # need = comm * scale + margin: collectives are measured in isolation but
        # run concurrent with the compute that hides them.
        self.comm_overlap_window_scale = comm_overlap_window_scale

    # -- OverlapPass hooks ------------------------------------------------
    def wants(self, order: list[BaseSchedulerNode]) -> bool:
        return any(is_weight_gather(s) for s in order)

    def plan(self, graph: SnodeGraph, cost: CostView, ctx: dict) -> list[TransferTask] | None:
        order, index_of = graph.order, graph.index_of
        launches = [s for s in order if is_weight_gather(s)]  # program order
        skel_idx, skel_kinds = rank_sync.collective_skeleton(order)
        mode, sync_group, world = rank_sync.negotiate_mode(anchors=launches, skel_kinds=skel_kinds, costs_ok=cost.ok)
        ctx.update(launches=launches, skel_idx=skel_idx, mode=mode, sync_group=sync_group, world=world)
        if mode == "abort":
            return None

        tasks: list[TransferTask] = []
        for launch in launches:
            group = self._launch_group(launch, graph, cost)
            fc_idx = self._first_consumer_index(group, graph, cost)
            if fc_idx is None:
                continue
            comm = cost(launch)
            lower = graph.earliest_legal_index(group)
            if mode == "pinned":
                # No skeleton to negotiate against: keep the AG between the same two
                # NCCL-issuing snodes it already sat between.
                lower = rank_sync.raise_lower_for_nccl_barriers(lower, index_of[launch], order)
            # A gather is never delayed past where it already sits: its own
            # position is the unhoisted one, so the deadline is the next index.
            tasks.append(
                TransferTask(
                    anchor=launch,
                    group=group,
                    lower=lower,
                    deadline=index_of[launch] + 1,
                    need=comm * self.comm_overlap_window_scale + self.comm_overlap_window_margin_ns,
                    last_user=fc_idx,
                )
            )
        return tasks

    def place(self, graph: SnodeGraph, cost: CostView, tasks: list[TransferTask], ctx: dict) -> None:
        # A group member (the shard prep above all) is rebuilt ahead of its own
        # launch, out of every window it sits in now: it hides no gather.
        moving = {m for t in tasks for m in t.group}
        full = cost
        cost = CostView(lambda s: 0.0 if s in moving else full(s))
        # No in-flight byte budget on gathers.
        alap_schedule(tasks, cost.prefix(graph.order), graph.index_of, float("inf"))
        if ctx["world"] > 1 and ctx["mode"] != "pinned":
            rank_sync.consensus_slot_targets(
                ctx["launches"], {t.anchor: t for t in tasks}, ctx["skel_idx"], graph.index_of, ctx["sync_group"], ctx["world"]
            )
        enforce_monotone(tasks)

    def agree(self, ok: bool, ctx: dict) -> bool:
        committed = rank_sync.agree(ok, ctx["sync_group"], ctx["world"])
        if ok and not committed:
            magi_logger.warning(
                "FSDP overlap reorder: another rank did not commit its rebuilt order; "
                "leaving this rank's graph unchanged too"
            )
        return committed

    def report(self, graph: SnodeGraph, cost: CostView, tasks: list[TransferTask], committed: bool, ctx: dict) -> None:
        moved = sum(1 for t in tasks if graph.index_of[t.anchor] != t.target) if committed else 0
        magi_logger.info("FSDP overlap reorder: repositioned %d/%d weight all-gather launch(es)", moved, len(ctx["launches"]))

    # -- group detection --------------------------------------------------
    def _launch_group(self, launch, graph: SnodeGraph, cost: CostView) -> list[BaseSchedulerNode]:
        """The snodes that must move together with the launch.

        Coalesced: packed collective + its MultiOutput members (they depend on the
        packed buffer and must stay immediately after it, before any wait).
        no-bucket: just the launch (the wait stays put).
        Always plus the upstream shard prep (see _prep_chain).
        """
        order = graph.order
        group = [launch]
        node = leaf_collective_node(launch)
        produced = set(launch.get_buffer_names())
        if node is not None and (node.op_overload is _AG_COALESCED or is_ce_ag_coalesced(node)):
            for s in order:
                if is_multi_output(s) and any((not _is_fake_dep(d)) and d.name in produced for d in s.unmet_dependencies):
                    group.append(s)
            if is_ce_ag_coalesced(node):
                for s in order:
                    if s is launch or s in group or contains_wait(s) or not cost.is_transparent(s):
                        continue
                    deps = [d for d in s.unmet_dependencies if not _is_fake_dep(d)]
                    if deps and all(d.name in produced for d in deps):
                        group.append(s)
        elif is_ce_ag_ir(node):
            for s in order:
                if s is launch or contains_wait(s) or not cost.is_transparent(s):
                    continue
                deps = [d for d in s.unmet_dependencies if not _is_fake_dep(d)]
                if deps and all(d.name in produced for d in deps):
                    group.append(s)
        group.extend(self._prep_chain(group, graph))
        return group

    @staticmethod
    def _prep_chain(group, graph: SnodeGraph) -> list[BaseSchedulerNode]:
        """Hoist shard prep with the gather when it reads only graph inputs or other prep.
        Prep may include H2D loads/waits, casts and pads; real collectives stay out.
        Excluded producers raise the gather's legal floor.
        This pass imposes no byte budget on prep or gathered buffers."""
        buf_to_snode, index_of = graph.buf_to_snode, graph.index_of
        members = set(group)
        stack = list(group)
        reached: set[BaseSchedulerNode] = set()
        while stack:
            for d in stack.pop().unmet_dependencies:
                if _is_fake_dep(d):
                    continue
                prod = buf_to_snode.get(d.name)
                if prod is None or prod in members or prod in reached:
                    continue
                if issues_transfer(prod) and not is_h2d_load(prod):
                    continue  # a real collective is not prep; it has a plan of its own
                reached.add(prod)
                stack.append(prod)

        # Program order, so every producer is classified before its consumers and
        # one pass settles the cascade of a dropped node's dependents.
        keep: set[BaseSchedulerNode] = set()
        for s in sorted(reached, key=lambda n: index_of[n]):
            if all(
                buf_to_snode.get(d.name) in (None, s) or buf_to_snode.get(d.name) in keep
                for d in s.unmet_dependencies
                if not _is_fake_dep(d)
            ):
                keep.add(s)
        return sorted(keep, key=lambda n: index_of[n])

    # -- consumer discovery ----------------------------------------------
    @staticmethod
    def _first_consumer_index(group, graph: SnodeGraph, cost: CostView) -> int | None:
        """Find the earliest real consumer beyond the gather's waits and alias nodes.
        Skip waits inside the moving group, including upstream H2D waits;
        otherwise the gather itself would be mistaken for its first consumer."""
        members = set(group)

        def to_wait(u):
            if contains_wait(u) and u not in members:
                return "wait", False
            return None, u in members or cost.is_transparent(u)

        def to_use(u):
            return (None, True) if cost.is_transparent(u) else ("use", False)

        waits = [u for _, u in graph.walk_users(group, to_wait)]
        if not waits:
            return None
        best = None
        for w in waits:
            uses = [graph.index_of[u] for _, u in graph.walk_users([w], to_use) if u in graph.index_of]
            if uses:
                fc = min(uses)
                best = fc if best is None else min(best, fc)
        return best
