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

"""Latest-safe-launch FSDP all-gather / compute overlap reorder pass.

Installed into ``torch._inductor.config.reorder_for_compute_comm_overlap_passes``
as a callable ``list[BaseSchedulerNode] -> list[BaseSchedulerNode]``.  Runs on the
WHOLE Inductor graph (MagiCompiler ``disable_graph_split=True``), replacing
PyTorch's builtin ``raise_comms``/``sink_waits``.

Objective (opposite of ``raise_comms``, which schedules comms as EARLY as
possible): for each FSDP weight ``all_gather`` launch, find its wait / first real
consumer and place the launch at the LATEST position whose downstream compute
still hides the collective, i.e. the latest slot where::

    sum(compute runtime between launch and first-consumer) >= comm_runtime + slack

If there is not enough upstream compute, fall back to as-early-as-legal (so we
overlap what we can; never worse than raise_comms).

TWO-POINTER back-to-front sweep (like the offload HeuristicScheduler's reverse
walk for "latest start of transmission", offload/scheduler.py:319):
  * comm pointer  -- weight all-gathers in REVERSE original order (last first);
  * compute pointer -- a SINGLE index that walks backward CONTINUOUSLY over the
    whole graph and is NEVER reset per gather.
Each gather consumes a contiguous run of compute nodes (accumulating their cost)
until it covers its (scaled) comm; its launch target is that stopping point.  The
NEXT (earlier) gather resumes the compute pointer from where it stopped -- NOT
from just before its own consumer -- so no two gathers claim the same compute
(this serializes the single NCCL stream) and the collectives keep their original
relative order automatically (targets only decrease).  All moves are then applied
in one stable-sort rebuild and validated once (``_validate_full``).

The Inductor driver (``comms.reorder_compute_and_comm_for_overlap``) does NOT
validate or repair the returned order and does NOT re-sink collectives, so the
returned list MUST be a valid topological order.  We guarantee that by only ever
inserting a launch group inside ``[earliest-legal, first-consumer)`` and
asserting the move keeps every producer before / every consumer after the group
(else we abort the move and leave the launch in place).

Handles two graph-level forms produced by
``fsdp_overlap.lower_and_bucket.lower_and_bucket_full_graph``:
* no-bucket   : 1 all_gather -> 1 wait -> 1 consumer;
* coalesced   : 1 packed all_gather_into_tensor_coalesced (MultiOutputLayout)
                -> N MultiOutput members -> N waits -> N consumers.  The launch is
                ONE snode with N waits; the packed collective + its N MultiOutput
                members move together as one contiguous block, before any wait.
"""

from collections import defaultdict

import torch
from torch._inductor.comms import _is_fake_dep
from torch._inductor.ir import MultiOutput
from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.utils import contains_collective, contains_wait, is_collective

from magi_compiler.utils import magi_logger


def _debug_enabled() -> bool:
    """MagiLogger has no isEnabledFor; check the underlying std logger."""
    import logging

    return logging.getLogger("magi_compiler").isEnabledFor(logging.DEBUG)


_AG = torch.ops._c10d_functional.all_gather_into_tensor.default
_AG_COALESCED = torch.ops._c10d_functional.all_gather_into_tensor_coalesced.default
_WEIGHT_AG_OPS = (_AG, _AG_COALESCED)

# Default extra headroom (ns) added to each collective's runtime when sizing the
# compute window, absorbing estimator error + kernel-launch latency so the wait
# rarely stalls.  Overridable via the reorder pass constructor.
_DEFAULT_SLACK_NS = 5_000.0


def _leaf_collective_node(snode: BaseSchedulerNode):
    """The underlying collective IR node for a (possibly grouped) snode, or None."""
    node = getattr(snode, "node", None)
    if node is not None and is_collective(node):
        return node
    # GroupedSchedulerNode: find the collective child.
    for child in getattr(snode, "snodes", []) or []:
        cn = getattr(child, "node", None)
        if cn is not None and is_collective(cn):
            return cn
    return None


def _is_weight_gather(snode: BaseSchedulerNode) -> bool:
    node = _leaf_collective_node(snode)
    return node is not None and getattr(node, "op_overload", None) in _WEIGHT_AG_OPS


def _is_multi_output(snode: BaseSchedulerNode) -> bool:
    node = getattr(snode, "node", None)
    return type(node) is MultiOutput


class FsdpOverlapReorder:
    """Callable reorder pass (see module docstring)."""

    def __init__(self, slack_ns: float = _DEFAULT_SLACK_NS, cost_fn=None) -> None:
        self.slack_ns = slack_ns
        # cost_fn: snode -> ns.  Default uses Inductor's estimate_op_runtime hook,
        # which MagiBackend points at the profiling estimator.
        if cost_fn is None:
            from torch._inductor.comms import estimate_op_runtime

            cost_fn = estimate_op_runtime
        self._cost_fn = cost_fn
        # Per-call cost cache keyed by snode.  Reset at the start of every
        # __call__ (snodes are unique per compile).  NEVER let it survive into a
        # deepcopy: Inductor serializes config.reorder_for_compute_comm_overlap_passes
        # into the fx-graph cache key via deepcopy, and snode keys hold FakeTensors
        # whose data_ptr access raises.  __deepcopy__ below returns a clean instance.
        self._cost_cache: dict[BaseSchedulerNode, float] = {}

    def __deepcopy__(self, memo):
        # Return a fresh, cache-free instance so config serialization (fx-graph
        # cache key) never deepcopies snode/FakeTensor state.  cost_fn is shared
        # by reference (it is itself deepcopy-safe: see ProfilingRuntimeEstimator).
        new = FsdpOverlapReorder.__new__(FsdpOverlapReorder)
        new.slack_ns = self.slack_ns
        new._cost_fn = self._cost_fn
        new._cost_cache = {}
        memo[id(self)] = new
        return new

    # -- cost -------------------------------------------------------------
    def _cost(self, snode: BaseSchedulerNode) -> float:
        c = self._cost_cache.get(snode)
        if c is None:
            try:
                c = max(0.0, float(self._cost_fn(snode)))
            except Exception:  # noqa: BLE001
                c = 0.0
            self._cost_cache[snode] = c
        return c

    @staticmethod
    def _is_compute(snode: BaseSchedulerNode) -> bool:
        return not contains_collective(snode) and not contains_wait(snode)

    # -- main -------------------------------------------------------------
    def __call__(self, snodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
        self._cost_cache = {}  # fresh per compile; snodes are unique
        order = list(snodes)
        launches = [s for s in order if _is_weight_gather(s)]
        if not launches:
            return order

        buf_to_snode = {b: s for s in order for b in s.get_buffer_names()}
        op_to_snode: dict[str, BaseSchedulerNode] = {}
        for s in order:
            for op in s.get_operation_names():
                op_to_snode[op] = s
            op_to_snode[s.get_name()] = s
        users: dict[str, set] = defaultdict(set)
        for s in order:
            for d in s.unmet_dependencies:
                if not _is_fake_dep(d):
                    users[d.name].add(s)

        index_of = {s: i for i, s in enumerate(order)}

        # ---- MULTI-RANK DETERMINISM (why this schedule is rank-identical) ----
        # This placement sweep MUST produce an IDENTICAL schedule on every rank, else
        # the weight all-gathers (weight PG) interleave with the eager CP all_to_all
        # (CP PG, inside the attention boundary op) in a rank-divergent order ->
        # cross-PG NCCL collective-order mismatch -> deadlock (verified via flight
        # recorder on 8-GPU gaga4: hang at SeqNum=12, rank0 races weight-PG ahead
        # while peers block on the CP all_to_all).  Rank-consistency needs BOTH:
        #   1. IDENTICAL INPUT GRAPH per rank -- guaranteed by the FSDP redistribute
        #      lowering now emitting the pad node UNCONDITIONALLY (zero-width no-op on
        #      full-chunk ranks), so uneven Shard(0) params no longer give trailing
        #      ranks extra constant_pad_nd nodes (different snode-list length ->
        #      divergent placement).  Verified: stage `0008` byte-identical across ranks.
        #   2. RANK-DETERMINISTIC COSTS -- the sweep accumulates `_cost` to decide how
        #      far to hoist each launch.  Profiling benchmarks PER RANK (timing noise +
        #      replays the attention op's internal CP all_to_all per rank), so with
        #      identical graphs the post-reorder `0037` STILL diverged (rank0=7 vs
        #      rank4=5 gathers before the 1st attention) and hung.  Two ways to make
        #      costs rank-identical, both supported (MagiBackend selects via cost_fn):
        #        (a) ANALYTICAL cost_fn (roofline, shapes+device only) -- deterministic
        #            by construction, zero overhead, but less accurate;
        #        (b) SYNCHRONIZED PROFILING -- the estimator exposes `warm_and_sync`,
        #            which re-measures every op in rank-lockstep (barrier + fixed iters)
        #            and MAX-reduces over gloo, giving REAL costs that are still
        #            identical on every rank.  Safe now that the graph is identical
        #            (same key set on all ranks -> symmetric barriers/all_gather).
        #      Below: if the cost_fn supports warm_and_sync, warm the table on all
        #      compute nodes then sync it; on failure, bail (leave graph unchanged =
        #      overlap off, no hang).  Analytical cost_fn has no warm_and_sync -> skip.
        if hasattr(self._cost_fn, "warm_and_sync") and getattr(self._cost_fn, "_sync_across_ranks", False):
            try:
                for s in order:
                    if self._is_compute(s) or contains_collective(s):
                        self._cost(s)  # populate estimator._table (per structural key)
                n_changed = self._cost_fn.warm_and_sync()
                self._cost_cache = {}  # drop pass-local cache -> re-read synced costs
                magi_logger.info(
                    "FSDP overlap reorder: rank-synchronized profiling done (%d cost entries reconciled)", n_changed
                )
            except Exception as exc:  # noqa: BLE001
                magi_logger.warning("FSDP overlap reorder: synchronized profiling failed (%s); leaving graph unchanged", exc)
                return order

        # ---- TWO-POINTER back-to-front sweep (like offload HeuristicScheduler) ----
        # comm pointer: weight all-gathers in REVERSE original order (last first).
        # compute pointer: a SINGLE index that walks backward CONTINUOUSLY across the
        # whole graph and is NEVER reset per gather -- each gather consumes a
        # contiguous run of compute nodes to cover its (scaled) comm, and the next
        # (earlier) gather resumes from where the pointer stopped (NOT from just
        # before the consumer again).  This (a) serializes the single NCCL stream --
        # no two gathers claim the same compute -- and (b) keeps collectives in their
        # original relative order automatically (targets only decrease).  Mirrors
        # offload/scheduler.py's reverse `i` walk for "latest start of transmission".
        launches_in_order = sorted(launches, key=lambda s: index_of[s])  # original program order

        # Per-gather static facts (on the STABLE original order).
        plans = []  # (launch, group, fc_idx, comm_runtime, lower)
        for launch in launches_in_order:
            group = self._launch_group(launch, order, buf_to_snode, users)
            fc_idx = self._first_consumer_index(launch, group, order, users)
            if fc_idx is None:
                continue
            comm_runtime = self._cost(launch)
            lower = self._earliest_legal_index(group, order, index_of, buf_to_snode, op_to_snode)
            plans.append((launch, group, fc_idx, comm_runtime, lower))

        # ---- Sweep the compute pointer backward (UPSTREAM-of-launch scan) ----
        # For each gather (LATEST first) accumulate the compute that sits IMMEDIATELY
        # BEFORE its launch, walking backward, until the accumulated compute covers
        # comm; move the launch to just before that compute block.  Because this pass
        # moves ONLY the launch (never the wait) and lowering emits ``all_gather;
        # wait`` back-to-back (wait at launch+1), the compute we walk past ends up in
        # the [new_launch, wait] window and therefore actually overlaps the gather.
        # Scanning UPSTREAM (from the launch) -- not downstream from the consumer --
        # is the fix: the old downstream scan counted compute that runs AFTER the
        # (unmoved) wait and can never overlap, so single-weight gathers were judged
        # "covered" and left glued to their wait -> fully exposed.
        #
        # A SINGLE compute pointer walks backward CONTINUOUSLY and is NEVER reset per
        # gather (each gather resumes from where the previous, later gather stopped),
        # so no two gathers claim the same compute -- this serializes the single NCCL
        # stream and keeps collectives in their original relative order (targets only
        # decrease).  A gather with no upstream compute before `lower` (e.g. the
        # graph's first gather) can't move and stays where it is (structural bubble).
        targets: dict = {}  # launch -> target index (in original order space)
        compute_idx = len(order)  # scan compute strictly below this
        for launch, group, fc_idx, comm_runtime, lower in reversed(plans):
            cur = index_of[launch]
            # Start just before the launch, but no later than where the previous
            # (later) gather already consumed compute down to.
            compute_idx = min(compute_idx, cur)
            need = comm_runtime + self.slack_ns
            acc = 0.0
            t = compute_idx
            while t > lower:
                s = order[t - 1]
                if self._is_compute(s):
                    acc += self._cost(s)
                t -= 1
                if acc >= need:
                    break
            # `t` is the earliest position whose downstream compute (up to the launch)
            # covers comm.  target < cur moves the launch earlier; target == cur means
            # no upstream compute was available (graph head / previous gather took it)
            # -> leave in place.  target >= lower always (producers stay before it).
            target = max(lower, t)
            targets[launch] = (target, group)
            compute_idx = target  # next (earlier) gather resumes from actual placement
            if _debug_enabled():
                magi_logger.debug(
                    "FSDP overlap: launch cur=%d -> target=%d fc=%d lower=%d | comm=%.1fus " "acc_upstream=%.1fus %s",
                    cur,
                    target,
                    fc_idx,
                    lower,
                    comm_runtime / 1e3,
                    acc / 1e3,
                    "hidden" if acc >= need else "COMPUTE-LIMITED",
                )

        # ---- CROSS-RANK DETERMINISM: enforce weight-gather relative order ----
        # CRITICAL for multi-rank correctness.  All FSDP weight all-gathers run on
        # the SAME process group, so NCCL matches the Nth all-gather call across
        # ranks positionally -- the ranks MUST issue them in an identical relative
        # order or they deadlock (observed: "Watchdog caught collective operation
        # timeout ... SeqNum=7 COALESCED").  The reverse sweep's ``target =
        # max(lower, t)`` can INVERT two gathers' order: a later gather may move far
        # up (small target, derived from PER-RANK profiled compute costs), while an
        # earlier gather is pinned by its real-dep floor ``lower`` to a LARGER target
        # -> the earlier gather sorts after the later one.  Because the targets
        # depend on per-rank kernel timings (ProfilingRuntimeEstimator benchmarks on
        # each rank independently), whether the inversion happens differs per rank ->
        # divergent collective order -> hang.  Fix: clamp targets to be
        # NON-DECREASING in original program index (a graph property, identical on
        # every rank).  Walk launches in original order with a running max; a gather
        # can never be placed before an earlier-issued gather.  This preserves the
        # original weight-gather subsequence on all ranks regardless of cost jitter,
        # at the cost of occasionally not moving a launch as early as its compute
        # window would allow (it clusters just after the prior gather instead).
        running = -1
        for launch in launches_in_order:
            if launch not in targets:
                continue
            target, group = targets[launch]
            if target < running:
                target = running
            targets[launch] = (target, group)
            running = target

        # ---- apply all moves in ONE rebuild (targets are in the ORIGINAL index
        # space; applying moves incrementally would shift those indices).  Assign a
        # sort key per node: non-group nodes keep their original index; each launch
        # group is inserted just before the node originally at `target` (key
        # target-0.5), members ordered among themselves by original index.  A stable
        # sort then realizes every move at once while preserving all other nodes'
        # relative order and each group's internal order.  Collective order is kept
        # because targets are monotonic (the two-pointer only decreases). ----
        group_members: dict = {}
        for launch, (target, group) in targets.items():
            for m in group:
                group_members[m] = target
        moved = sum(1 for launch, (target, _g) in targets.items() if index_of[launch] != target)

        def _key(s):
            if s in group_members:
                return (group_members[s] - 0.5, index_of[s])
            return (index_of[s], 0.0)

        new_order = sorted(order, key=_key)
        # Validate the rebuilt order is a valid topological order; only commit if so.
        if self._validate_full(new_order, op_to_snode, buf_to_snode, users):
            order[:] = new_order
        else:
            magi_logger.warning("FSDP overlap reorder: rebuilt order failed validation; leaving graph unchanged")
            moved = 0

        measured = getattr(self._cost_fn, "n_measured", None)
        cache_hits = getattr(self._cost_fn, "n_cache_hits", None)
        n_distinct = len(getattr(self._cost_fn, "_table", {}) or {})
        magi_logger.info(
            "FSDP overlap reorder: repositioned %d/%d weight all-gather launch(es) "
            "(cost table: %d distinct ops, measured=%s reused=%s)",
            moved,
            len(launches),
            n_distinct,
            measured,
            cache_hits,
        )
        # Full op->time table at DEBUG, or to a file when MAGI_COMPILE_FSDP_OVERLAP_DUMP
        # is set (so the estimate table can be captured WITHOUT global DEBUG logging,
        # which can trip torch's PT2_COMPILE chromium-event assertion on some builds).
        if hasattr(self._cost_fn, "summary"):
            import os as _os

            if _debug_enabled():
                magi_logger.debug("FSDP overlap %s", self._cost_fn.summary())
            dump = _os.environ.get("MAGI_COMPILE_FSDP_OVERLAP_DUMP")
            if dump:
                try:
                    rank = _os.environ.get("RANK", "0")
                    with open(f"{dump}.rank{rank}", "a") as f:
                        f.write(self._cost_fn.summary() + "\n")
                except Exception as exc:  # noqa: BLE001
                    magi_logger.warning("FSDP overlap: could not write cost dump: %s", exc)
        return order

    # -- group detection --------------------------------------------------
    def _launch_group(self, launch, order, buf_to_snode, users) -> list[BaseSchedulerNode]:
        """The snodes that must move together with the launch.

        Coalesced: packed collective + its MultiOutput members (they depend on the
        packed buffer and must stay immediately after it, before any wait).
        no-bucket: just the launch (the wait stays put).
        """
        group = [launch]
        node = _leaf_collective_node(launch)
        if node is not None and getattr(node, "op_overload", None) is _AG_COALESCED:
            produced = set(launch.get_buffer_names())
            for s in order:
                if _is_multi_output(s) and any((not _is_fake_dep(d)) and d.name in produced for d in s.unmet_dependencies):
                    group.append(s)
        return group

    # -- consumer discovery ----------------------------------------------
    def _wait_snodes(self, group, order, users) -> list[BaseSchedulerNode]:
        produced: set[str] = set()
        for s in group:
            produced |= set(s.get_buffer_names())
        waits = []
        seen = set()
        for b in produced:
            for u in users.get(b, ()):  # readers of the launch/member buffers
                if u in seen:
                    continue
                seen.add(u)
                if contains_wait(u):
                    waits.append(u)
        return waits

    def _first_consumer_index(self, launch, group, order, users) -> int | None:
        """min over all waits of the earliest real (non-transparent) consumer index."""
        index_of = {s: i for i, s in enumerate(order)}
        waits = self._wait_snodes(group, order, users)
        if not waits:
            return None
        best = None
        for w in waits:
            fc = self._first_real_consumer_index(w, index_of, users)
            if fc is not None:
                best = fc if best is None else min(best, fc)
        return best

    def _first_real_consumer_index(self, wait, index_of, users) -> int | None:
        """Forward BFS from a wait through transparent forwarders (cost~0 view /
        getitem / MultiOutput / split) to the first genuine compute consumer."""
        seen = set()
        stack = list(wait.get_buffer_names())
        best = None
        while stack:
            b = stack.pop()
            for u in users.get(b, ()):
                if u in seen:
                    continue
                seen.add(u)
                if self._is_transparent(u):
                    stack.extend(u.get_buffer_names())
                else:
                    idx = index_of.get(u)
                    if idx is not None:
                        best = idx if best is None else min(best, idx)
        return best

    def _is_transparent(self, snode: BaseSchedulerNode) -> bool:
        """A forwarder that doesn't count as the weight's real use: waits,
        MultiOutput unpacks, and ~zero-cost view/reshape/getitem kernels."""
        if contains_wait(snode) or _is_multi_output(snode):
            return True
        # Treat vanishingly cheap nodes (views, getitems, splits) as transparent.
        return self._cost(snode) <= 1.0

    # -- repositioning ----------------------------------------------------
    def _earliest_legal_index(self, group, order, index_of, buf_to_snode, op_to_snode) -> int:
        """1 + max index of any REAL (data-dependency) producer the group needs.

        Uses ONLY non-fake buffer dependencies, walked transitively.  We must NOT
        use ``snode.ancestors``: that transitive-closure set is polluted by the
        artificial ``WeakDep`` edges Inductor inserts between collectives to
        serialize them onto one comm stream (verified: a single-weight qkv/o gather
        lists the PREVIOUS layer's coalesced all-gather as an ancestor via a
        ``WeakDep`` on its output buffer -- but it does NOT read that buffer).  FSDP
        weight all-gathers gather INDEPENDENT param shards; there is no real
        gather->gather data dependency.  Counting the WeakDep pinned ``lower`` right
        after the previous collective, which wrongly forbade moving the launch up
        past the (real, hideable) compute between the two gathers -- the exact reason
        the attention gathers stayed exposed.  A gather's only real producer is its
        own weight-shard placeholder (+ any to_local/pad/cast chain), so real
        ``lower`` is ~0; the launch is then free to move up into earlier compute.
        """
        group_set = set(group)
        lo = 0
        for s in group:
            for d in s.unmet_dependencies:  # buffer names
                if _is_fake_dep(d):  # WeakDep / StarDep -- ordering hint, not data
                    continue
                prod = buf_to_snode.get(d.name)
                if prod is None or prod in group_set:
                    continue
                lo = max(lo, index_of.get(prod, 0) + 1)
        return lo

    def _validate_full(self, new_order, op_to_snode, buf_to_snode, users) -> bool:
        """Check the rebuilt order is a valid topological order w.r.t. REAL data
        dependencies: every node's non-fake buffer producers precede it (the
        Inductor driver does NOT repair the order, so a real-dep violation would
        silently miscompile).  O(nodes * deps).

        We deliberately do NOT validate against ``snode.ancestors``: that set is the
        transitive closure of ALL deps including the artificial ``WeakDep`` ordering
        edges Inductor inserts between collectives (see ``_earliest_legal_index``).
        Our pass intentionally reorders a weight all-gather across such a WeakDep
        (there is no real gather->gather data dependency), so an ancestors-based
        check would false-reject that legal move and no-op the whole reorder.  The
        per-node real-buffer-dep check below is itself a complete topological
        validation over the real data-dependency DAG: if every node's direct real
        producers precede it, the order is a valid real-dep topological order.
        WeakDep ordering is advisory (stream serialization / memory) and is NOT a
        correctness constraint, so violating it is safe."""
        pos = {s: i for i, s in enumerate(new_order)}
        for s in new_order:
            sp = pos[s]
            for d in s.unmet_dependencies:  # buffer names
                if _is_fake_dep(d):  # WeakDep / StarDep -- advisory ordering, not data
                    continue
                prod = buf_to_snode.get(d.name)
                if prod is s:  # fused snode may name its own internal buffers
                    continue
                if prod is not None and pos.get(prod, -1) >= sp:
                    if _debug_enabled():
                        magi_logger.debug(
                            "validate fail: %s@%d needs buffer-dep %s@%d (buf %s)",
                            s.get_name(),
                            sp,
                            prod.get_name(),
                            pos.get(prod, -1),
                            d.name,
                        )
                    return False
        return True
