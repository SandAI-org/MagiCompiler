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

Installed as the only ``reorder_for_compute_comm_overlap_passes`` entry (replaces
``raise_comms``/``sink_waits``); runs on the whole Inductor graph
(``disable_graph_split=True``).  For each FSDP weight all-gather launch, place it
at the LATEST position whose downstream compute still hides the collective::

    sum(compute runtime between launch and first-consumer) >= comm * scale + margin

Not enough upstream compute -> as-early-as-legal (never worse than raise_comms).

Algorithm: two-pointer back-to-front sweep.  Gathers are visited in reverse
program order; a single compute pointer walks backward continuously and is never
reset, so each gather claims a disjoint run of compute (serializing the single
NCCL stream) and targets only decrease.  All moves are applied in one stable-sort
rebuild and validated once (``_validate_full``) -- the Inductor driver does NOT
repair the returned order, so it must be a valid topological order.

Handles both lowering forms: plain all_gather (1 launch / 1 wait) and coalesced
(1 packed launch + N MultiOutput members moved together as one block + N waits).
"""

import hashlib
from collections import defaultdict

import torch
from torch._inductor.comms import _is_fake_dep
from torch._inductor.ir import MultiOutput
from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.utils import contains_collective, contains_wait, is_collective

from magi_compiler.utils import magi_logger

_AG = torch.ops._c10d_functional.all_gather_into_tensor.default
_AG_COALESCED = torch.ops._c10d_functional.all_gather_into_tensor_coalesced.default
_WEIGHT_AG_OPS = (_AG, _AG_COALESCED)

# Default extra headroom (ns) added to each collective's runtime when sizing the
# compute window, absorbing estimator error + kernel-launch latency so the wait
# rarely stalls.  Overridable via the reorder pass constructor.
_DEFAULT_WINDOW_MARGIN_NS = 5_000.0


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


def _graph_fingerprint(order: list[BaseSchedulerNode]) -> str:
    """Rank-comparable digest of the snode sequence: type + op identity + output
    sizes + sorted origin fx TARGETS.  Origins are required -- a fused pointwise
    kernel is one ComputedBuffer whose class/size hide its contents (relu vs
    relu+sin look identical without them).  Targets only, not node names: names
    carry per-rank numbering noise."""
    h = hashlib.sha256()
    for s in order:
        h.update(type(s).__name__.encode())
        for sub in getattr(s, "snodes", None) or (s,):
            n = getattr(sub, "node", None)
            if n is None:
                continue
            op = getattr(n, "op_overload", None) or getattr(n, "python_kernel_name", None) or type(n).__name__
            h.update(str(op).encode())
            try:
                h.update(repr(n.get_size()).encode())
            except Exception:  # noqa: BLE001
                pass
            origins = getattr(n, "origins", None)
            if origins:
                h.update("|".join(sorted(str(getattr(o, "target", o)) for o in origins)).encode())
    return h.hexdigest()


class FsdpOverlapReorder:
    """Callable reorder pass."""

    def __init__(
        self,
        comm_overlap_window_margin_ns: float = _DEFAULT_WINDOW_MARGIN_NS,
        cost_fn=None,
        comm_overlap_window_scale: float = 1.0,
    ) -> None:
        self.comm_overlap_window_margin_ns = comm_overlap_window_margin_ns
        # need = comm * scale + margin: collectives are measured in isolation but
        # run concurrent with the compute that hides them (~1.4-1.5x slower on
        # 8xH100).  See CompileConfig.fsdp_config.comm_overlap_window_scale.
        self.comm_overlap_window_scale = comm_overlap_window_scale
        # cost_fn: snode -> ns (default: Inductor's estimate_op_runtime hook).
        if cost_fn is None:
            from torch._inductor.comms import estimate_op_runtime

            cost_fn = estimate_op_runtime
        self._cost_fn = cost_fn
        # Per-compile cost cache.  Must never survive into a deepcopy: Inductor
        # deepcopies this pass into the fx-graph cache key, and snode keys hold
        # FakeTensors whose data_ptr access raises.
        self._cost_cache: dict[BaseSchedulerNode, float] = {}

    def __deepcopy__(self, memo):
        # Fresh, cache-free instance (see _cost_cache note); cost_fn shared by
        # reference -- it is itself deepcopy-safe.
        new = FsdpOverlapReorder.__new__(FsdpOverlapReorder)
        new.comm_overlap_window_margin_ns = self.comm_overlap_window_margin_ns
        new.comm_overlap_window_scale = self.comm_overlap_window_scale
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

        # Fail-fast: the index-based sweep (and the lockstep profiling below) both
        # require structurally IDENTICAL per-rank graphs, else the weight gathers
        # interleave with other collectives in rank-divergent order -> NCCL
        # deadlock.  Verify with one symmetric all_gather of a graph digest; every
        # rank sees the same result, so all ranks take the same branch.
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            from magi_compiler.profiling.runtime_estimator import _get_cost_sync_group

            fp = (_graph_fingerprint(order), len(order), len(launches))
            world = dist.get_world_size()
            all_fp: list = [None] * world
            dist.all_gather_object(all_fp, fp, group=_get_cost_sync_group())
            if any(f != all_fp[0] for f in all_fp[1:]):
                magi_logger.warning(
                    "FSDP overlap reorder: per-rank graphs are NOT structurally identical "
                    "((digest, n_snodes, n_weight_gathers) per rank: %s). Reordering would "
                    "produce rank-divergent collective order -> NCCL deadlock; leaving the "
                    "graph unchanged (overlap OFF for this graph). Likely cause: uneven "
                    "Shard(0) params -- replicate them or use chunk-padded uniform shards.",
                    [(f[0][:12], f[1], f[2]) if f else None for f in all_fp],
                )
                return order

        # profile_sync: warm the estimator table on every node, then re-measure in
        # rank-lockstep (warm_and_sync) so costs are rank-identical.  On failure,
        # leave the graph unchanged (overlap off, no hang).
        if hasattr(self._cost_fn, "warm_and_sync") and getattr(self._cost_fn, "_sync_across_ranks", False):
            try:
                for s in order:
                    if self._is_compute(s) or contains_collective(s):
                        self._cost(s)
                n_changed = self._cost_fn.warm_and_sync()
                self._cost_cache = {}  # re-read synced costs
                magi_logger.info(
                    "FSDP overlap reorder: rank-synchronized profiling done (%d cost entries reconciled)", n_changed
                )
            except Exception as exc:  # noqa: BLE001
                magi_logger.warning("FSDP overlap reorder: synchronized profiling failed (%s); leaving graph unchanged", exc)
                return order

        # ---- two-pointer back-to-front sweep (see module docstring) ----
        launches_in_order = sorted(launches, key=lambda s: index_of[s])  # original program order

        plans = []  # (launch, group, fc_idx, comm_runtime, lower)
        for launch in launches_in_order:
            group = self._launch_group(launch, order, buf_to_snode, users)
            fc_idx = self._first_consumer_index(launch, group, order, users)
            if fc_idx is None:
                continue
            comm_runtime = self._cost(launch)
            lower = self._earliest_legal_index(group, order, index_of, buf_to_snode, op_to_snode)
            plans.append((launch, group, fc_idx, comm_runtime, lower))

        targets: dict = {}  # launch -> target index (in original order space)
        compute_idx = len(order)  # scan compute strictly below this
        for launch, group, fc_idx, comm_runtime, lower in reversed(plans):
            cur = index_of[launch]
            # Start just before the launch, but no later than where the previous
            # (later) gather already consumed compute down to.
            compute_idx = min(compute_idx, cur)
            need = comm_runtime * self.comm_overlap_window_scale + self.comm_overlap_window_margin_ns
            acc = 0.0
            t = compute_idx
            while t > lower:
                s = order[t - 1]
                if self._is_compute(s):
                    acc += self._cost(s)
                t -= 1
                if acc >= need:
                    break
            # target == cur means no upstream compute left (graph head or previous
            # gather claimed it); target >= lower keeps real producers before it.
            target = max(lower, t)
            targets[launch] = (target, group)
            compute_idx = target  # next (earlier) gather resumes from actual placement
            # Per-gather placement decision, the record that answers "why didn't
            # this gather move earlier":
            #   cur       = original program index of the launch
            #   target    = where it was placed (== cur means NOT moved)
            #   lower     = earliest LEGAL index (real-dep floor) it could move to
            #   fc_idx    = first real consumer (the wait's user)
            #   comm      = the gather's runtime it needs to hide
            #   acc_upstream = compute actually found in [target, cur] to hide it
            #   verdict   = hidden (acc>=need) | COMPUTE-LIMITED (ran out of upstream
            #               compute before covering comm -- i.e. hit `lower` or the
            #               previous gather's placement first)
            magi_logger.debug(
                "FSDP overlap placement: launch cur=%d -> target=%d fc=%d lower=%d | "
                "comm=%.1fus acc_upstream=%.1fus need=%.1fus %s",
                cur,
                target,
                fc_idx,
                lower,
                comm_runtime / 1e3,
                acc / 1e3,
                need / 1e3,
                "hidden" if acc >= need else "COMPUTE-LIMITED",
            )

        # Clamp targets NON-DECREASING in original program order.  NCCL matches the
        # Nth call on a PG positionally across ranks, so the gathers' relative order
        # must be rank-identical; `max(lower, t)` can invert two gathers and whether
        # the inversion happens depends on per-rank cost jitter -> deadlock.  The
        # clamp pins the original subsequence at the cost of occasionally placing a
        # launch later than its compute window would allow.
        running = -1
        for launch in launches_in_order:
            if launch not in targets:
                continue
            target, group = targets[launch]
            if target < running:
                target = running
            targets[launch] = (target, group)
            running = target

        # Apply all moves in ONE stable-sort rebuild (targets live in the original
        # index space; incremental moves would shift them).  Each launch group sorts
        # to key target-0.5 (just before the node originally at `target`), members
        # keep their internal order, everything else keeps its original index.
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
        # Full op->time table at DEBUG.  The guard is load-bearing here: summary()
        # builds the whole table string eagerly, unlike lazy %-format args.
        if hasattr(self._cost_fn, "summary"):
            magi_logger.debug("FSDP overlap %s", self._cost_fn.summary())
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
        """1 + max index of any REAL (non-fake buffer) producer the group needs.

        Deliberately NOT ``snode.ancestors``: that set is polluted by the fake
        ``WeakDep`` edges Inductor inserts between collectives for comm-stream
        serialization.  Weight gathers read independent param shards -- there is no
        real gather->gather dependency -- so counting the WeakDep would pin the
        launch right after the previous collective and forbid the very hoist this
        pass exists for.  A gather's only real producer is its weight-shard
        placeholder (+ to_local/pad/cast chain), so real ``lower`` is ~0."""
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
        """Valid topological order w.r.t. REAL data deps: every node's non-fake
        buffer producers precede it (the driver does not repair the order, so a
        violation would silently miscompile).  Checking direct producers per node
        is a complete validation of the real-dep DAG.  ``snode.ancestors`` is NOT
        used -- it includes the fake WeakDep edges this pass intentionally crosses
        (see ``_earliest_legal_index``); an ancestors check would false-reject
        every legal hoist.  WeakDep is advisory, not a correctness constraint."""
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
