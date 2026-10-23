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

"""Schedule FSDP gathers and shard preparation with shared ALAP placement.

Runs after cost profiling and optional bucketing on the unsplit graph.
Each task uses the original launch's next boundary as its deadline and
``comm * scale + margin`` as its transfer window; gathers are never delayed.
Single and coalesced gathers share dependency validation and rank agreement.
"""

from __future__ import annotations

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
from magi_compiler.passes.snode_utils import AG_COALESCED, is_ce_ag_coalesced_ir, is_ce_ag_ir, leaf_comm_node, weight_gathers
from magi_compiler.utils import magi_logger


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
        # required_overlap_ns = comm * scale + margin: collectives are measured in isolation but
        # run concurrent with the compute that hides them.
        self.comm_overlap_window_scale = comm_overlap_window_scale

    # -- OverlapPass hooks ------------------------------------------------
    def wants(self, order: list[BaseSchedulerNode]) -> bool:
        # plan() negotiates with a collective, so a rank without a weight gather
        # of its own still enters it whenever a peer has one.
        return rank_sync.any_rank(bool(weight_gathers(order)))

    def plan(self, graph: SnodeGraph, cost: CostView, ctx: dict) -> list[TransferTask] | None:
        order, index_of = graph.order, graph.index_of
        launches = weight_gathers(order)
        skel_idx, skel_kinds = rank_sync.collective_skeleton(order)
        mode, sync_group, world = rank_sync.negotiate_mode(n_anchors=len(launches), skel_kinds=skel_kinds, costs_ok=cost.ok)
        ctx.update(launches=launches, skel_idx=skel_idx, mode=mode, sync_group=sync_group, world=world)
        if mode == "abort":
            return None

        tasks: list[TransferTask] = []
        for launch in launches:
            group = self._launch_group(launch, graph, cost)
            if not self._has_real_consumer(group, graph, cost):
                continue
            comm = cost(launch)
            earliest_launch_index = graph.earliest_legal_index(group)
            if mode == "pinned":
                # No skeleton to negotiate against: keep the AG between the same two
                # NCCL-issuing snodes it already sat between.
                earliest_launch_index = rank_sync.raise_lower_for_nccl_barriers(earliest_launch_index, index_of[launch], order)
            # A gather is never delayed past where it already sits: its own
            # position is the unhoisted one, so the deadline is the next index.
            tasks.append(
                TransferTask(
                    anchor=launch,
                    moving_nodes=group,
                    earliest_launch_index=earliest_launch_index,
                    deadline_index=index_of[launch] + 1,
                    required_overlap_ns=comm * self.comm_overlap_window_scale + self.comm_overlap_window_margin_ns,
                )
            )
        return tasks

    def place(self, graph: SnodeGraph, cost: CostView, tasks: list[TransferTask], ctx: dict) -> None:
        # A group member (the shard prep above all) is rebuilt ahead of its own
        # launch, out of every window it sits in now: it hides no gather.
        moving = {m for t in tasks for m in t.moving_nodes}
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
        moved = sum(1 for t in tasks if graph.index_of[t.anchor] != t.launch_index) if committed else 0
        magi_logger.info("FSDP overlap reorder: repositioned %d/%d weight all-gather launch(es)", moved, len(ctx["launches"]))

    # -- group detection --------------------------------------------------
    def _launch_group(self, launch, graph: SnodeGraph, cost: CostView) -> list[BaseSchedulerNode]:
        """The snodes that must move together with the launch.

        Coalesced: packed collective + its MultiOutput members (they depend on the
        packed buffer and must stay immediately after it, before any wait).
        no-bucket: just the launch (the wait stays put).
        Always plus the upstream shard prep (see ``SnodeGraph.prep_chain``).
        """
        group = [launch]
        node = leaf_comm_node(launch)
        if node is not None and (node.op_overload is AG_COALESCED or is_ce_ag_coalesced_ir(node)):
            group.extend(graph.unpacks_of(launch))
        if is_ce_ag_ir(node):
            produced = set(launch.get_buffer_names())
            readers = {u for name in produced for u in graph.users.get(name, ())}
            for s in sorted(readers, key=graph.index_of.__getitem__):
                if s in group or contains_wait(s) or not cost.is_transparent(s):
                    continue
                if all(name in produced for name in (d.name for d in s.unmet_dependencies if not _is_fake_dep(d))):
                    group.append(s)
        group.extend(graph.prep_chain(group))
        return group

    # -- consumer discovery ----------------------------------------------
    @staticmethod
    def _has_real_consumer(group, graph: SnodeGraph, cost: CostView) -> bool:
        """Whether the gather's result reaches a real use beyond its waits and alias nodes.
        Skip waits inside the moving group, including upstream H2D waits;
        otherwise the gather itself would be mistaken for its consumer.

        Gathers carry no byte budget, so only whether a use exists matters, not where."""
        members = set(group)

        def to_wait(u):
            if contains_wait(u) and u not in members:
                return "wait", False
            return None, u in members or cost.is_transparent(u)

        def to_use(u):
            return (None, True) if cost.is_transparent(u) else ("use", False)

        waits = [u for _, u in graph.walk_users(group, to_wait)]
        return any(graph.walk_users([w], to_use) for w in waits)
