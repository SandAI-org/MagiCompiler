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
cross-rank agreement a collective needs.  Placement is one of the shared
strategies (``placement``): ``index_sweep`` claims compute a snode at a time and
carries a long kernel's remainder to the next gather by hand; ``alap`` schedules
the one transfer stream on the compute time axis, as the load reorder does.
Rebuild and validation are ``OverlapPass``'s.

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
    index_sweep,
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

_PLACEMENTS = ("index_sweep", "alap")


class FsdpOverlapReorder(OverlapPass):
    """Callable reorder pass."""

    name = "FSDP overlap reorder"

    def __init__(
        self,
        comm_overlap_window_margin_ns: float = DEFAULT_WINDOW_MARGIN_NS,
        cost_fn=None,
        comm_overlap_window_scale: float = 1.0,
        move_prep_chain: bool = False,
        placement: str = "index_sweep",
    ) -> None:
        super().__init__(cost_fn)
        if placement not in _PLACEMENTS:
            raise ValueError(f"FsdpOverlapReorder: placement must be one of {_PLACEMENTS}, got {placement!r}")
        self.placement = placement
        # Offload's h2d_load + wait sit between the placeholder and the gather,
        # so the gather cannot move unless that prep chain travels with it.
        # Off by default: without offload this also hoists casts and pads.
        self.move_prep_chain = move_prep_chain
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
        mode, sync_group, world = rank_sync.negotiate_mode(order, launches, skel_kinds, cost.ok)
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
                    meta={"fc_idx": fc_idx, "comm": comm},
                )
            )
        return tasks

    def place(self, graph: SnodeGraph, cost: CostView, tasks: list[TransferTask], ctx: dict) -> None:
        if self.placement == "alap":
            # No in-flight byte budget on gathers, and a gather issued late behind
            # a stalled wait idles the stream: issue from the deadline chain alone.
            alap_schedule(tasks, cost.prefix(graph.order), graph.index_of, float("inf"), follow_resource=False)
            for task in tasks:
                self._log_alap_placement(task, graph)
        else:
            index_sweep(tasks, graph.order, cost)
            for task in reversed(tasks):
                self._log_placement(task, graph)
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
        With ``move_prep_chain``: plus the upstream shard prep (see _prep_chain).
        """
        order = graph.order
        group = [launch]
        node = leaf_collective_node(launch)
        produced = set(launch.get_buffer_names())
        if node is not None and (getattr(node, "op_overload", None) is _AG_COALESCED or is_ce_ag_coalesced(node)):
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
        if self.move_prep_chain:
            group.extend(self._prep_chain(group, graph))
        return group

    @staticmethod
    def _prep_chain(group, graph: SnodeGraph) -> list[BaseSchedulerNode]:
        """The UPSTREAM shard-prep snodes that have to travel with the launch.

        A weight's path from placeholder to gather can hold an ``h2d_load``, its
        wait, a dtype cast and an uneven-shard pad.  Every one of them is a real
        buffer producer, so ``earliest_legal_index`` pins the launch just below
        them: leaving them behind does not make the hoist illegal, it makes it
        impossible.

        Moving a producer earlier is always legal for its readers, so there is no
        "all users inside the group" condition here -- the cost of a longer live
        range is the in-flight budget's business, not correctness'.  The condition
        that does matter is that a traveller reads nothing but graph inputs and
        other travellers: a node that touches an activation would be compute we
        are simultaneously counting as compute that hides this gather.  Anything
        failing that is dropped, and ``earliest_legal_index`` then simply reports
        a higher floor -- a shorter hoist, never a wrong one.
        """
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
        """min over the launch's waits of the earliest real (non-transparent) consumer.

        The waits are reached through any alias layer: a custom-op gather puts an
        alias snode between the launch and its wait, and missing the wait silently
        drops the gather from the placement plan altogether.  A wait that is itself
        a group member is stepped over rather than reported: with host offload the
        group contains the ``h2d_load`` and the wait that guards it, and that wait
        sits UPSTREAM of the gather.  Stopping there would report the gather's own
        launch as its first consumer -- a zero-width overlap window.

        From each wait, a forward walk through transparent forwarders (cost~0
        view / getitem / MultiOutput / split) finds the first genuine consumer.
        """
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

    # -- reporting --------------------------------------------------------
    @staticmethod
    def _log_placement(task: TransferTask, graph: SnodeGraph) -> None:
        """Per-gather placement decision, the record that answers "why didn't
        this gather move earlier":

          cur       = original program index of the launch
          target    = where it was placed (== cur means NOT moved)
          lower     = earliest LEGAL index (real-dep floor) it could move to
          fc_idx    = first real consumer (the wait's user)
          comm      = the gather's runtime it needs to hide
          carry_in  = capacity inherited from the later gather's boundary node
                      (target==cur with a large carry_in means it was already
                      covered and did not have to move at all); 0 when this
                      gather already sat upstream of that boundary node
          carry_idx = boundary node this gather leaves its own remainder in
          acc_upstream = carry_in plus the compute found in [target, cur]
          verdict   = hidden (acc>=need) | COMPUTE-LIMITED (ran out of upstream
                      compute before covering comm -- i.e. hit `lower` or the
                      previous gather's placement first)
        """
        meta = task.meta
        magi_logger.debug(
            "FSDP overlap placement: launch %s(%s) cur=%d -> target=%d fc=%d lower=%d | "
            "comm=%.1fus carry_in=%.1fus carry_idx=%d acc_upstream=%.1fus need=%.1fus %s",
            task.anchor.get_name(),
            getattr(leaf_collective_node(task.anchor), "op_overload", "?"),
            graph.index_of[task.anchor],
            task.target,
            meta["fc_idx"],
            task.lower,
            meta["comm"] / 1e3,
            meta["carry_in"] / 1e3,
            meta["carry_idx"],
            meta["acc"] / 1e3,
            task.need / 1e3,
            "hidden" if meta["acc"] >= task.need else "COMPUTE-LIMITED",
        )

    @staticmethod
    def _log_alap_placement(task: TransferTask, graph: SnodeGraph) -> None:
        """Same record for ``placement='alap'``: exposed is the part of the gather
        the stream could not finish before the launch's own position."""
        magi_logger.debug(
            "FSDP overlap placement (alap): launch %s(%s) cur=%d -> target=%d fc=%d lower=%d | "
            "comm=%.1fus need=%.1fus exposed=%.1fus %s",
            task.anchor.get_name(),
            getattr(leaf_collective_node(task.anchor), "op_overload", "?"),
            graph.index_of[task.anchor],
            task.target,
            task.meta["fc_idx"],
            task.lower,
            task.meta["comm"] / 1e3,
            task.need / 1e3,
            task.exposed / 1e3,
            "hidden" if task.exposed <= 0 else "COMPUTE-LIMITED",
        )
