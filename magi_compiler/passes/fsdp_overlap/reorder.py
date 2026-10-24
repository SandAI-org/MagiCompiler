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
``transfer_duration_ns * scale + margin`` as its transfer window; gathers are never delayed.
Single and coalesced gathers share dependency validation and rank agreement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from torch._inductor.comms import _is_fake_dep
from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.utils import contains_wait

from magi_compiler.passes.snode_utils import (
    COALESCED_ALL_GATHER_OP,
    is_ce_ag_coalesced_ir,
    is_ce_ag_ir,
    leaf_comm_node,
    weight_gathers,
)
from magi_compiler.passes.transfer_overlap import (
    DEFAULT_WINDOW_MARGIN_NS,
    CostView,
    OverlapCallState,
    OverlapPass,
    SnodeGraph,
    TransferTask,
    enforce_launch_order,
    rank_sync,
    schedule_transfers_alap,
)
from magi_compiler.utils import magi_logger


@dataclass
class FsdpOverlapCallState(OverlapCallState):
    """What ``build_transfer_tasks`` negotiates across ranks, for the later hooks."""

    # Every weight-gather launch in program order, planned or not: ranks align slots on this list.
    launches: list[BaseSchedulerNode] = field(default_factory=list)
    collective_indices: list[int] = field(default_factory=list)  # the collective skeleton, ascending
    placement_mode: str = "abort"  # "slot", "pinned" or "abort"; identical on every rank
    sync_group: object = None
    world_size: int = 1


class FsdpOverlapReorder(OverlapPass):
    """Callable reorder pass."""

    log_prefix = "FSDP overlap reorder"
    call_state_type = FsdpOverlapCallState

    def __init__(
        self,
        comm_overlap_window_margin_ns: float = DEFAULT_WINDOW_MARGIN_NS,
        runtime_ns_fn=None,
        comm_overlap_window_scale: float = 1.0,
    ) -> None:
        super().__init__(runtime_ns_fn)
        self.comm_overlap_window_margin_ns = comm_overlap_window_margin_ns
        # required_overlap_ns = transfer_duration_ns * scale + margin: collectives are
        # measured in isolation but run concurrent with the compute that hides them.
        self.comm_overlap_window_scale = comm_overlap_window_scale

    # -- OverlapPass hooks ------------------------------------------------
    def has_transfer_candidates(self, order: list[BaseSchedulerNode]) -> bool:
        # build_transfer_tasks() negotiates with a collective, so a rank without a
        # weight gather of its own still enters it whenever a peer has one.
        return rank_sync.any_rank_has_flag(bool(weight_gathers(order)))

    def build_transfer_tasks(
        self, graph: SnodeGraph, cost: CostView, call_state: FsdpOverlapCallState
    ) -> list[TransferTask] | None:
        order, original_index_by_node = graph.order, graph.original_index_by_node
        launches = weight_gathers(order)
        collective_indices, collective_signatures = rank_sync.collective_skeleton(order)
        consensus = rank_sync.negotiate_placement_mode(
            candidate_count=len(launches),
            collective_signatures=collective_signatures,
            is_profiling_valid=cost.is_profiling_valid,
        )
        call_state.launches = launches
        call_state.collective_indices = collective_indices
        call_state.placement_mode = consensus.placement_mode
        call_state.sync_group = consensus.sync_group
        call_state.world_size = consensus.world_size
        if consensus.placement_mode == "abort":
            return None

        tasks: list[TransferTask] = []
        for launch in launches:
            moving_nodes = self._collect_moving_nodes(launch, graph, cost)
            if not self._has_nontransparent_consumer(moving_nodes, graph, cost):
                continue
            transfer_duration_ns = cost(launch)
            earliest_launch_index = graph.earliest_legal_index(moving_nodes)
            if consensus.placement_mode == "pinned":
                # No skeleton to negotiate against: keep the AG between the same two
                # NCCL-issuing snodes it already sat between.
                earliest_launch_index = rank_sync.launch_floor_after_collectives(
                    earliest_launch_index, original_index_by_node[launch], order
                )
            # A gather is never delayed past where it already sits: its own
            # position is the unhoisted one, so the deadline is the next index.
            tasks.append(
                TransferTask(
                    anchor=launch,
                    moving_nodes=tuple(moving_nodes),
                    earliest_launch_index=earliest_launch_index,
                    deadline_index=original_index_by_node[launch] + 1,
                    required_overlap_ns=transfer_duration_ns * self.comm_overlap_window_scale
                    + self.comm_overlap_window_margin_ns,
                )
            )
        return tasks

    def assign_launch_indices(
        self, graph: SnodeGraph, cost: CostView, tasks: list[TransferTask], call_state: FsdpOverlapCallState
    ) -> dict:
        # A moving node (the shard prep above all) is rebuilt ahead of its own
        # launch, out of every window it sits in now: it hides no gather.
        moving_nodes = {m for t in tasks for m in t.moving_nodes}
        original_cost_view = cost
        overlap_cost_view = CostView(lambda s: 0.0 if s in moving_nodes else original_cost_view(s))
        # No in-flight byte budget on gathers.
        schedule_result = schedule_transfers_alap(
            tasks, overlap_cost_view.compute_prefix_ns(graph.order), graph.original_index_by_node, float("inf")
        )
        launch_index_by_anchor = schedule_result.launch_index_by_anchor
        if call_state.world_size > 1 and call_state.placement_mode != "pinned":
            launch_index_by_anchor = rank_sync.align_collective_slots(
                anchors_in_order=call_state.launches,
                task_by_anchor={task.anchor: task for task in tasks},
                launch_index_by_anchor=launch_index_by_anchor,
                collective_indices=call_state.collective_indices,
                original_index_by_node=graph.original_index_by_node,
                sync_group=call_state.sync_group,
                world_size=call_state.world_size,
            )
        return enforce_launch_order(tasks, launch_index_by_anchor)

    def can_commit_order(self, is_order_valid: bool, call_state: FsdpOverlapCallState) -> bool:
        committed = rank_sync.all_ranks_agree(is_order_valid, call_state.sync_group, call_state.world_size)
        if is_order_valid and not committed:
            magi_logger.warning(
                "FSDP overlap reorder: another rank did not commit its rebuilt order; "
                "leaving this rank's graph unchanged too"
            )
        return committed

    def log_reorder_summary(
        self,
        graph: SnodeGraph,
        cost: CostView,
        tasks: list[TransferTask],
        launch_index_by_anchor: dict,
        committed: bool,
        call_state: FsdpOverlapCallState,
    ) -> None:
        moved_count = 0
        if committed:
            moved_count = sum(1 for t in tasks if graph.original_index_by_node[t.anchor] != launch_index_by_anchor[t.anchor])
        magi_logger.info(
            "FSDP overlap reorder: repositioned %d/%d weight all-gather launch(es)", moved_count, len(call_state.launches)
        )

    # -- moving-node detection --------------------------------------------
    def _collect_moving_nodes(self, launch, graph: SnodeGraph, cost: CostView) -> list[BaseSchedulerNode]:
        """The snodes that must move together with the launch.

        Coalesced: packed collective + its MultiOutput members (they depend on the
        packed buffer and must stay immediately after it, before any wait).
        no-bucket: just the launch (the wait stays put).
        Always plus the upstream shard prep (see ``SnodeGraph.movable_prep_nodes``).
        """
        moving_nodes = [launch]
        node = leaf_comm_node(launch)
        if node is not None and (node.op_overload is COALESCED_ALL_GATHER_OP or is_ce_ag_coalesced_ir(node)):
            moving_nodes.extend(graph.unpacks_of(launch))
        if is_ce_ag_ir(node):
            produced = set(launch.get_buffer_names())
            readers = {u for name in produced for u in graph.users_by_buffer.get(name, ())}
            for s in sorted(readers, key=graph.original_index_by_node.__getitem__):
                if s in moving_nodes or contains_wait(s) or not cost.is_transparent(s):
                    continue
                if all(name in produced for name in (d.name for d in s.unmet_dependencies if not _is_fake_dep(d))):
                    moving_nodes.append(s)
        moving_nodes.extend(graph.movable_prep_nodes(moving_nodes))
        return moving_nodes

    # -- consumer discovery ----------------------------------------------
    @staticmethod
    def _has_nontransparent_consumer(moving_nodes, graph: SnodeGraph, cost: CostView) -> bool:
        """Whether the gather's result reaches a use that ``CostView.is_transparent``
        does not treat as a forwarder (waits, unpacks, ~zero-cost views).
        Skip waits among the moving nodes, including upstream H2D waits;
        otherwise the gather itself would be mistaken for its consumer.

        Gathers carry no byte budget, so only whether a use exists matters, not where."""
        members = set(moving_nodes)

        def classify_wait(u):
            if contains_wait(u) and u not in members:
                return "wait", False
            return None, u in members or cost.is_transparent(u)

        def classify_consumer(u):
            return (None, True) if cost.is_transparent(u) else ("use", False)

        waits = [u for _, u in graph.collect_classified_users(moving_nodes, classify_wait)]
        return any(graph.collect_classified_users([w], classify_consumer) for w in waits)
