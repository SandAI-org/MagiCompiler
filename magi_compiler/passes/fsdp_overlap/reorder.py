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

import bisect
import hashlib
from collections import defaultdict

import torch
import torch.distributed as dist
from torch._inductor.comms import _is_fake_dep
from torch._inductor.ir import MultiOutput
from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.utils import contains_collective, contains_wait, is_collective

from magi_compiler.utils import magi_logger

_AG = torch.ops._c10d_functional.all_gather_into_tensor.default
_AG_COALESCED = torch.ops._c10d_functional.all_gather_into_tensor_coalesced.default


def _symm_ag_ops():
    """Copy-engine gather ops, imported lazily so this pass stays importable
    without a CUDA build."""
    try:
        from magi_compiler.symm_mem.all_gather import SYMM_ALL_GATHER, SYMM_ALL_GATHER_COALESCED

        return tuple(op for op in (SYMM_ALL_GATHER, SYMM_ALL_GATHER_COALESCED) if op is not None)
    except Exception:  # noqa: BLE001
        return ()


_SYMM_AG_OPS = _symm_ag_ops()
_WEIGHT_AG_OPS = tuple(op for op in (_AG, _AG_COALESCED, *_SYMM_AG_OPS) if op is not None)

# Default extra headroom (ns) added to each collective's runtime when sizing the
# compute window, absorbing estimator error + kernel-launch latency so the wait
# rarely stalls.  Overridable via the reorder pass constructor.
_DEFAULT_WINDOW_MARGIN_NS = 5_000.0


def _is_symm_ag_ir(node) -> bool:
    """
    ``magi::symm_all_gather`` lowers to an ordinary FallbackKernel, so
    Inductor's ``is_collective`` does not recognize it.
    """
    return getattr(node, "op_overload", None) in _SYMM_AG_OPS


def _is_symm_ag_coalesced(node) -> bool:
    try:
        from magi_compiler.symm_mem.all_gather import SYMM_ALL_GATHER_COALESCED
    except Exception:  # noqa: BLE001
        return False
    return SYMM_ALL_GATHER_COALESCED is not None and getattr(node, "op_overload", None) is SYMM_ALL_GATHER_COALESCED


def _is_gather_ir(node) -> bool:
    return node is not None and (is_collective(node) or _is_symm_ag_ir(node))


def _leaf_collective_node(snode: BaseSchedulerNode):
    """The underlying collective IR node for a (possibly grouped) snode, or None."""
    node = getattr(snode, "node", None)
    if _is_gather_ir(node):
        return node
    # GroupedSchedulerNode: find the collective child.
    for child in getattr(snode, "snodes", []) or []:
        cn = getattr(child, "node", None)
        if _is_gather_ir(cn):
            return cn
    return None


def _issues_transfer(snode: BaseSchedulerNode) -> bool:
    return contains_collective(snode) or _leaf_collective_node(snode) is not None


def _is_weight_gather(snode: BaseSchedulerNode) -> bool:
    node = _leaf_collective_node(snode)
    return node is not None and getattr(node, "op_overload", None) in _WEIGHT_AG_OPS


def _is_multi_output(snode: BaseSchedulerNode) -> bool:
    node = getattr(snode, "node", None)
    return type(node) is MultiOutput


def _size_hint_of(sym) -> int:
    """Rank-identical size hint for a sympy symbol (0 if unavailable, e.g. in
    unit tests without a live Inductor graph)."""
    try:
        from torch._inductor.virtualized import V

        return int(V.graph.sizevars.size_hint(sym, fallback=0))
    except Exception:  # noqa: BLE001
        return 0


def _collective_kind_key(snode: BaseSchedulerNode) -> tuple:
    """Coarse, rank-comparable identity of one NCCL-issuing snode."""
    node = _leaf_collective_node(snode) or getattr(snode, "node", None)
    op = getattr(node, "op_overload", None) or getattr(node, "python_kernel_name", None) or type(node).__name__
    dims: tuple = ()
    try:
        dims = tuple("?" if getattr(d, "free_symbols", None) else int(d) for d in node.get_size())
    except Exception:  # noqa: BLE001
        pass
    return (_is_weight_gather(snode), str(op), dims)


def _collective_skeleton(order: list[BaseSchedulerNode]) -> tuple[list[int], list[tuple]]:
    """The graph's collective skeleton: indices (ascending) and rank-comparable
    kinds of every snode that issues a transfer -- functional NCCL collectives,
    custom ops with an internal collective, and copy-engine / symmetric-memory
    gathers.  This sequence is what must stay rank-identical; the compute
    between two consecutive entries is rank-private."""
    from magi_compiler.profiling.runtime_estimator import snode_issues_collective

    idx = [i for i, s in enumerate(order) if snode_issues_collective(s) or _issues_transfer(s)]
    return idx, [_collective_kind_key(order[i]) for i in idx]


def _graph_fingerprint(order: list[BaseSchedulerNode]) -> str:
    """Rank-comparable digest of the snode sequence: type + op identity + output
    sizes + sorted origin fx TARGETS.  Origins are required -- a fused pointwise
    kernel is one ComputedBuffer whose class/size hide its contents (relu vs
    relu+sin look identical without them).  Targets only, not node names: names
    carry per-rank numbering noise.

    """
    import sympy

    h = hashlib.sha256()
    sym_canon: dict = {}  # sympy.Symbol -> canonical sympy.Symbol

    def _canon_size(size) -> str:
        dims = []
        for d in size:
            free = getattr(d, "free_symbols", None)
            if not free:
                dims.append(repr(d))
                continue
            fresh = [sym for sym in free if sym not in sym_canon]
            # Name-free assignment order; symbol name only as the last-resort
            # tie-break (see docstring: that case fails safe).
            fresh.sort(key=lambda sym: (_size_hint_of(sym), d.count(sym), sym.name))
            for sym in fresh:
                sym_canon[sym] = sympy.Symbol(f"c{len(sym_canon):04d}")
            dims.append(repr(d.xreplace(sym_canon)))
        return "[" + ", ".join(dims) + "]"

    for s in order:
        h.update(type(s).__name__.encode())
        for sub in getattr(s, "snodes", None) or (s,):
            n = getattr(sub, "node", None)
            if n is None:
                continue
            op = getattr(n, "op_overload", None) or getattr(n, "python_kernel_name", None) or type(n).__name__
            h.update(str(op).encode())
            try:
                h.update(_canon_size(n.get_size()).encode())
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
        return not _issues_transfer(snode) and not contains_wait(snode)

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

        skel_idx, skel_kinds = _collective_skeleton(order)
        mode, sync_group, world = self._negotiate_mode(order, launches, skel_kinds)
        if mode == "abort":
            return order

        # profile_sync: warm the estimator table on every node, then re-measure in
        # rank-lockstep (warm_and_sync) so shared keys get real, max-reduced costs.
        # On failure, leave the graph unchanged (overlap off, no hang).
        if hasattr(self._cost_fn, "warm_and_sync") and getattr(self._cost_fn, "_sync_across_ranks", False):
            try:
                for s in order:
                    if self._is_compute(s) or _issues_transfer(s):
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
        lowers: dict = {}  # launch -> earliest legal index (real-dep floor)
        for launch in launches_in_order:
            group = self._launch_group(launch, order, buf_to_snode, users)
            fc_idx = self._first_consumer_index(launch, group, order, users)
            if fc_idx is None:
                continue
            comm_runtime = self._cost(launch)
            lower = self._earliest_legal_index(group, order, index_of, buf_to_snode, op_to_snode)
            if mode == "pinned":
                # No skeleton to negotiate against: keep the AG between the same two
                # NCCL-issuing snodes it already sat between.
                lower = self._raise_lower_for_nccl_barriers(lower, index_of[launch], order)
            lowers[launch] = lower
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
                "FSDP overlap placement: launch %s(%s) cur=%d -> target=%d fc=%d lower=%d | "
                "comm=%.1fus acc_upstream=%.1fus need=%.1fus %s",
                launch.get_name(),
                getattr(_leaf_collective_node(launch), "op_overload", "?"),
                cur,
                target,
                fc_idx,
                lower,
                comm_runtime / 1e3,
                acc / 1e3,
                need / 1e3,
                "hidden" if acc >= need else "COMPUTE-LIMITED",
            )

        if world > 1 and mode != "pinned":
            self._consensus_slot_targets(targets, lowers, launches_in_order, skel_idx, index_of, sync_group, world)

        # Keep gather order: clamp targets non-decreasing so cost jitter or
        # same-slot per-rank indices cannot swap two launches.
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
        # The verdict is reduced across ranks: committing on some ranks and not on
        # others is itself a divergent NCCL sequence.
        ok = self._validate_full(new_order, op_to_snode, buf_to_snode, users)
        if not ok:
            magi_logger.warning("FSDP overlap reorder: rebuilt order failed validation; leaving graph unchanged")
        if self._agree(ok, sync_group, world):
            order[:] = new_order
        else:
            if ok:
                magi_logger.warning(
                    "FSDP overlap reorder: another rank did not commit its rebuilt order; "
                    "leaving this rank's graph unchanged too"
                )
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

    # -- multi-rank agreement ---------------------------------------------
    @staticmethod
    def _negotiate_mode(order, launches, skel_kinds) -> tuple[str, object, int]:
        """Rank-identical placement mode: (mode, group, world).

        ``identical`` / ``slot``: skeletons match → consensus slots (in-slot index is per-rank).
        ``pinned``: skeletons differ → keep each AG between its neighboring NCCL snodes.
        ``abort``: weight-AG counts differ → leave the graph unchanged.
        """
        from magi_compiler.profiling.runtime_estimator import _get_cost_sync_group

        group = _get_cost_sync_group()
        world = dist.get_world_size()
        mine = ((_graph_fingerprint(order), len(order), len(launches)), tuple(skel_kinds))
        peers: list = [None] * world
        dist.all_gather_object(peers, mine, group=group)
        if all(p == peers[0] for p in peers[1:]):
            return "identical", group, world

        desc = [(p[0][0][:12], p[0][1], p[0][2], len(p[1])) for p in peers]
        n_ag = [p[0][2] for p in peers]
        if any(g != n_ag[0] for g in n_ag[1:]):
            magi_logger.warning(
                "FSDP overlap reorder: per-rank graphs differ AND weight-AG counts diverge "
                "((digest, n_snodes, n_weight_gathers, n_collectives) per rank: %s). No rank "
                "correspondence to reconcile; leaving the graph unchanged (overlap OFF).",
                desc,
            )
            return "abort", group, world
        if all(p[1] == peers[0][1] for p in peers[1:]):
            magi_logger.warning(
                "FSDP overlap reorder: per-rank graphs are NOT structurally identical "
                "((digest, n_snodes, n_weight_gathers, n_collectives) per rank: %s), but the "
                "collective skeleton matches. Continuing in SLOT-consensus mode: gathers are "
                "placed in a rank-negotiated skeleton slot (they MAY hop CP / EP kernels, as "
                "long as every rank hops the same one).",
                desc,
            )
            return "slot", group, world
        magi_logger.warning(
            "FSDP overlap reorder: per-rank graphs are NOT structurally identical AND their "
            "collective skeletons differ ((digest, n_snodes, n_weight_gathers, n_collectives) "
            "per rank: %s). Continuing in PINNED mode: gathers keep their position relative to "
            "every NCCL-issuing snode (no hop over CP / EP kernels).",
            desc,
        )
        return "pinned", group, world

    @staticmethod
    def _agree(ok: bool, sync_group, world: int) -> bool:
        """Reduce a local yes/no into a rank-identical one (AND over ranks)."""
        if world <= 1:
            return ok

        try:
            t = torch.tensor([1 if ok else 0], dtype=torch.int32)
            dist.all_reduce(t, op=dist.ReduceOp.MIN, group=sync_group)
            return bool(t.item())
        except Exception as exc:  # noqa: BLE001
            magi_logger.warning("FSDP overlap reorder: cross-rank agreement failed (%s); leaving graph unchanged", exc)
            return False

    @staticmethod
    def _consensus_slot_targets(targets, lowers, launches_in_order, skel_idx, index_of, sync_group, world) -> None:
        """Put each gather in the same skeleton slot on every rank (max of desired
        slot and dep floor, then non-decreasing).  Index inside the slot stays local.
        """

        def slot_of(idx: int) -> int:
            return bisect.bisect_left(skel_idx, idx)

        # One entry per launch in program order so skipped gathers stay aligned.
        mine = []
        for launch in launches_in_order:
            own = slot_of(index_of[launch])
            mine.append((slot_of(targets[launch][0]), slot_of(lowers[launch])) if launch in targets else (own, own))
        peers: list = [None] * world
        dist.all_gather_object(peers, mine, group=sync_group)

        running = 0
        for j, launch in enumerate(launches_in_order):
            q = max(max(p[j][0] for p in peers), max(p[j][1] for p in peers))
            q = running = max(q, running)
            if launch not in targets:
                continue
            target, group = targets[launch]
            slot_lo = max(lowers[launch], skel_idx[q - 1] + 1 if q > 0 else 0)
            slot_hi = skel_idx[q] if q < len(skel_idx) else index_of[launch]
            new_target = min(max(target, slot_lo), max(slot_hi, slot_lo))
            targets[launch] = (new_target, group)
            magi_logger.debug(
                "FSDP overlap slot consensus: launch cur=%d slot=%d/%d (mine=%s) target %d -> %d [%d, %d]",
                index_of[launch],
                q,
                slot_of(index_of[launch]),
                mine[j],
                target,
                new_target,
                slot_lo,
                slot_hi,
            )

    # -- group detection --------------------------------------------------
    def _launch_group(self, launch, order, buf_to_snode, users) -> list[BaseSchedulerNode]:
        """The snodes that must move together with the launch.

        Coalesced: packed collective + its MultiOutput members (they depend on the
        packed buffer and must stay immediately after it, before any wait).
        no-bucket: just the launch (the wait stays put).
        """
        group = [launch]
        node = _leaf_collective_node(launch)
        produced = set(launch.get_buffer_names())
        if node is not None and (getattr(node, "op_overload", None) is _AG_COALESCED or _is_symm_ag_coalesced(node)):
            for s in order:
                if _is_multi_output(s) and any((not _is_fake_dep(d)) and d.name in produced for d in s.unmet_dependencies):
                    group.append(s)
            if _is_symm_ag_coalesced(node):
                for s in order:
                    if s is launch or s in group or contains_wait(s) or not self._is_transparent(s):
                        continue
                    deps = [d for d in s.unmet_dependencies if not _is_fake_dep(d)]
                    if deps and all(d.name in produced for d in deps):
                        group.append(s)
        elif _is_symm_ag_ir(node):
            for s in order:
                if s is launch or contains_wait(s) or not self._is_transparent(s):
                    continue
                deps = [d for d in s.unmet_dependencies if not _is_fake_dep(d)]
                if deps and all(d.name in produced for d in deps):
                    group.append(s)
        return group

    # -- consumer discovery ----------------------------------------------
    def _wait_snodes(self, group, order, users) -> list[BaseSchedulerNode]:
        """The waits guarding this launch, reached through any alias layer.

        Searching only the launch's direct readers was enough while every gather
        was an Inductor collective; a custom-op gather puts an alias snode between
        the launch and its wait, and missing the wait silently drops the gather
        from the placement plan altogether.
        """
        stack = [b for s in group for b in s.get_buffer_names()]
        waits: list[BaseSchedulerNode] = []
        seen: set = set()
        while stack:
            for u in users.get(stack.pop(), ()):
                if u in seen:
                    continue
                seen.add(u)
                if contains_wait(u):
                    waits.append(u)
                elif self._is_transparent(u):
                    stack.extend(u.get_buffer_names())
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
    @staticmethod
    def _raise_lower_for_nccl_barriers(lower: int, launch_idx: int, order: list) -> int:
        """Raise ``lower`` so a weight AG cannot hop any NCCL-issuing snode that
        originally precedes it."""
        from magi_compiler.profiling.runtime_estimator import snode_issues_collective

        barrier = lower
        for i in range(lower, launch_idx):
            if snode_issues_collective(order[i]):
                barrier = i + 1
        return barrier

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
