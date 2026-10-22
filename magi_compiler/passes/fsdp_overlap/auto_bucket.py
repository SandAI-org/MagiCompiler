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

"""Scheduler-stage weight all-gather bucketing, before ``FsdpOverlapReorder``.

Fit ``T(S) = alpha + beta * S`` per transport and process group.
The gathered-byte target is ``alpha * (1 - ratio) / (ratio * beta)``.
Gathers at or above the target stay single and separate runs of smaller gathers.
Merge smaller gathers until a bucket reaches the target, so it stays below twice the target;
also respect producer/wait ordering and the configured local-byte cap.

NCCL and copy-engine gathers never share a bucket.
An eight-rank calibration sets default launch-cost ratios to 0.1 and 0.5, respectively.

Every rank joins mode negotiation, including ranks with fewer than two candidates.
Only matching collective skeletons (``slot`` mode) allow bucketing.
All ranks adopt rank 0's plan and validate it against their local dependencies;
IR changes require unanimous legality and matching post-merge collective order.
Local plans separate host-offloaded and resident gathers;
an adopted plan may mix them on another rank.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

import torch.distributed as dist
from torch._inductor.scheduler import BaseSchedulerNode

from magi_compiler.passes.overlap import CostView, SnodeGraph, rank_sync
from magi_compiler.passes.overlap.cost import default_cost_fn
from magi_compiler.passes.snode_cost import SnodeCostTable
from magi_compiler.passes.snode_utils import weight_gathers
from magi_compiler.passes.weight_offload.node_meta import is_host_offloaded
from magi_compiler.utils import magi_logger

from . import ir_coalesce
from .ir_coalesce import GatherInfo, gather_info

DEFAULT_LAUNCH_OVERHEAD_NS = 10_000.0
DEFAULT_OVERHEAD_RATIOS = {"nccl": 0.1, "ce": 0.5}
"""Launch share per transport (``GatherInfo.kind``) when ``overhead_ratio`` is not set."""


# -- the pure planning problem ----------------------------------------------
@dataclass(frozen=True)
class Member:
    """One gather, as the planner sees it (all indices in the fused order)."""

    idx: int  # the launch's own index
    lo: int  # earliest index a launch reading its shard could take
    wait: int  # index of its first wait: a merged launch must come before it
    local_bytes: int  # bytes this rank contributes
    out_bytes: int  # bytes the gather produces


def fit_alpha_beta(
    points: list[tuple[float, float]], alpha_guess_ns: float = DEFAULT_LAUNCH_OVERHEAD_NS
) -> tuple[float, float]:
    """Least-squares ``ns = alpha + beta * bytes`` with both terms kept >= 0.

    With every point at one size the split between the two terms is not
    observable; ``alpha_guess_ns`` (capped at half the cheapest point) is taken
    as the fixed part and the rest is charged per byte.
    """
    pts = [(float(x), float(y)) for x, y in points if x > 0 and y > 0]
    if not pts:
        return alpha_guess_ns, 0.0
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    if sxx <= 1e-9 * max(mx * mx, 1.0):
        alpha = min(alpha_guess_ns, 0.5 * min(y for _, y in pts))
        return alpha, max(0.0, (my - alpha) / mx)
    beta = sum((x - mx) * (y - my) for x, y in pts) / sxx
    alpha = my - beta * mx
    if beta < 0.0:
        return my, 0.0
    if alpha < 0.0:
        return 0.0, sum(x * y for x, y in pts) / sum(x * x for x, _ in pts)
    return alpha, beta


def amortizing_bytes(alpha_ns: float, beta_ns_per_byte: float, ratio: float) -> float:
    """Smallest gathered size whose launch is at most ``ratio`` of its time (inf if bytes cost nothing)."""
    if alpha_ns <= 0.0:
        return 0.0
    if beta_ns_per_byte <= 0.0:
        return float("inf")
    return alpha_ns * (1.0 - ratio) / (ratio * beta_ns_per_byte)


@dataclass
class PlanParams:
    alpha_ns: float
    beta_ns_per_byte: float
    overhead_ratio: float = DEFAULT_OVERHEAD_RATIOS["nccl"]
    max_local_bytes: int = 0  # ``bucket_size_mib``: local-shard bytes per bucket, 0 = no cap

    @property
    def target_bytes(self) -> float:
        return amortizing_bytes(self.alpha_ns, self.beta_ns_per_byte, self.overhead_ratio)


def plan_segments(members: list[Member], params: PlanParams) -> list[tuple[int, int]]:
    """Contiguous ``[start, end)`` segments covering ``members`` (see the module docstring)."""
    target = params.target_bytes
    segs: list[tuple[int, int]] = []
    start = None
    out = local = 0
    lo = wait = 0

    def close(end: int) -> None:
        if start is not None:
            segs.append((start, end))

    for i, m in enumerate(members):
        if m.out_bytes >= target:
            close(i)
            start = None
            segs.append((i, i + 1))
            continue
        if start is not None:
            fits = out < target and (params.max_local_bytes <= 0 or local + m.local_bytes <= params.max_local_bytes)
            if fits and max(members[start].idx, lo, m.lo) <= min(wait, m.wait):
                out += m.out_bytes
                local += m.local_bytes
                lo, wait = max(lo, m.lo), min(wait, m.wait)
                continue
            close(i)
        start, out, local, lo, wait = i, m.out_bytes, m.local_bytes, m.lo, m.wait
    close(len(members))
    return segs


# -- the pass -----------------------------------------------------------------
class FsdpAutoBucket:
    """Reorder pass that merges small weight gathers until their launch is amortized, and never moves anything else."""

    name = "FSDP auto bucket"

    def __init__(self, cost_fn=None, max_bucket_bytes: int = 0, overhead_ratio: float | None = None) -> None:
        self._cost_fn = cost_fn if cost_fn is not None else default_cost_fn()
        self.max_bucket_bytes = int(max_bucket_bytes)
        # None: per transport, from DEFAULT_OVERHEAD_RATIOS.  FSDPConfig bounds it to (0, 1).
        self.overhead_ratio = None if overhead_ratio is None else float(overhead_ratio)

    def __call__(self, snodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
        order = list(snodes)
        cands = self._candidates(order)
        world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        cost = CostView(self._cost_fn)
        group, skeleton = None, []
        if world > 1:
            # Candidate count is rank-local. Every rank joins the exchange below,
            # including one that found nothing to merge. Returning here skips the
            # all_gather and blocks whichever ranks do enter it.
            skel_idx, skel_kinds = rank_sync.collective_skeleton(order)
            mode, group, world = rank_sync.negotiate_mode(
                n_anchors=len(cands), skel_kinds=skel_kinds, costs_ok=cost.ok, who=self.name
            )
            if mode != "slot":
                magi_logger.warning(
                    "%s: per-rank collective skeletons cannot be matched (mode=%s); leaving every weight gather unbucketed",
                    self.name,
                    mode,
                )
                return snodes
            skeleton = [order[i] for i in skel_idx]
        if len(cands) < 2:
            return snodes

        ok, buckets, params_by_group = ir_coalesce.ir_coalesce_supported(), [], {}
        if not ok:
            magi_logger.warning(
                "%s: this torch lacks the Inductor internals scheduler-level coalescing needs; "
                "leaving every weight gather unbucketed",
                self.name,
            )
        else:
            try:
                buckets, params_by_group = self._plan(order, cands, cost)
            except Exception as exc:  # noqa: BLE001
                if world <= 1:
                    raise
                # Only to still enter _adopt's exchange, which every peer does.
                magi_logger.warning(
                    "%s: planning failed (%s); leaving every weight gather unbucketed", self.name, exc, rank="all"
                )
                ok = False
        buckets = self._adopt(ok=ok, buckets=buckets, order=order, cands=cands, group=group, world=world, skeleton=skeleton)
        if buckets is None:
            return snodes
        if not buckets:
            self._report(params_by_group, buckets, n_cands=len(cands))
            return snodes

        for members in buckets:
            out_bytes = sum(gather_info(m).shard_bytes * gather_info(m).world for m in members)
            params = params_by_group[self._fit_key(gather_info(members[0]))]
            order, packed, unpacks = ir_coalesce.coalesce_all_gathers(order, members)
            self._record(packed, unpacks, out_bytes, params)
        snodes[:] = order
        ir_coalesce.refresh_memory_planning(snodes)
        self._report(params_by_group, buckets, n_cands=len(cands))
        return snodes

    # -- candidates -----------------------------------------------------------
    @staticmethod
    def _candidates(order: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
        """The weight gathers (see ``snode_utils.weight_gathers``) in the single-weight form this pass can merge."""
        return [s for s in weight_gathers(order) if gather_info(s) is not None]

    @staticmethod
    def _kind(snode: BaseSchedulerNode, info: GatherInfo) -> tuple:
        offloaded = any(is_host_offloaded(o) for o in snode.node.origins or ())
        return (info.kind, info.dtype, offloaded)

    def _runs(self, cands: list[BaseSchedulerNode]) -> list[list[BaseSchedulerNode]]:
        """Consecutive gathers on one group, split by (transport, dtype, host-offloaded).

        Transport and dtype because one launch is one op.  Host offload only
        because a resident member would wait for its bucket-mates' host loads.
        """
        runs: list[list[BaseSchedulerNode]] = []
        last = object()
        for s in cands:
            info = gather_info(s)
            if info.group_name != last:
                runs.append([])
                last = info.group_name
            runs[-1].append(s)
        out: list[list[BaseSchedulerNode]] = []
        for run in runs:
            by_kind: dict[tuple, list[BaseSchedulerNode]] = {}
            for s in run:
                by_kind.setdefault(self._kind(s, gather_info(s)), []).append(s)
            out.extend(by_kind.values())
        return out

    # -- planning -------------------------------------------------------------
    def _member(self, snode, graph: SnodeGraph) -> Member | None:
        info = gather_info(snode)
        lo = graph.earliest_legal_index({snode, *graph.prep_chain([snode])})
        waits = graph.waits_of(snode)
        if not waits:
            return None
        wait = min(graph.index_of[w] for w in waits)
        return Member(
            idx=graph.index_of[snode], lo=lo, wait=wait, local_bytes=info.shard_bytes, out_bytes=info.shard_bytes * info.world
        )

    @staticmethod
    def _fit_key(info: GatherInfo) -> tuple[str, str, int]:
        return (info.kind, info.group_name, info.world)

    def _params_by_group(self, cands, cost: CostView) -> dict[tuple[str, str, int], PlanParams]:
        """One fit per (transport, process group), and one ratio per transport.

        Per transport because a copy-engine gather's fixed and per-byte costs are
        nothing like an NCCL one's.  Per group because the per-byte cost depends on
        who the bytes come from: a one-rank group's gather is a local copy at device
        bandwidth, and fitted together with an eight-rank group's its gigabyte-sized
        points pull ``beta`` down by an order of magnitude for every other gather.
        """
        points: dict[tuple[str, str, int], list[tuple[float, float]]] = {}
        for s in cands:
            info = gather_info(s)
            points.setdefault(self._fit_key(info), []).append((info.shard_bytes * info.world, cost(s)))
        out = {}
        for key, pts in points.items():
            kind = key[0]
            alpha, beta = fit_alpha_beta(pts)
            out[key] = PlanParams(
                alpha_ns=alpha,
                beta_ns_per_byte=beta,
                overhead_ratio=(self.overhead_ratio if self.overhead_ratio is not None else DEFAULT_OVERHEAD_RATIOS[kind]),
                max_local_bytes=self.max_bucket_bytes,
            )
        return out

    def _plan(self, order, cands, cost: CostView):
        graph = SnodeGraph(order)
        params_by_group = self._params_by_group(cands, cost)
        buckets: list[list[BaseSchedulerNode]] = []
        for run in self._runs(cands):
            params = params_by_group[self._fit_key(gather_info(run[0]))]
            pairs = [(s, self._member(snode=s, graph=graph)) for s in run]
            usable = [(s, m) for s, m in pairs if m is not None]
            if len(usable) < 2:
                continue
            snodes_, members = [s for s, _ in usable], [m for _, m in usable]
            segs = plan_segments(members, params)
            buckets.extend(snodes_[s:e] for s, e in segs if e - s >= 2)

        # Merge in sequence on a stand-in order: each merge moves shard prep,
        # and the next merge must still be legal after it.
        sim, kept = order, []
        for members in buckets:
            nxt = ir_coalesce.simulate_coalesce(sim, members)
            if nxt is None:
                continue
            sim = nxt
            kept.append(members)
        if not graph.validate(sim):
            raise RuntimeError("the merged order is not topological")
        return kept, params_by_group

    def _adopt(self, ok: bool, buckets, order, cands, group, world: int, skeleton):
        """Rank 0's buckets, as this rank's snodes, once every rank has found them legal; None to leave the graph alone.

        Every rank must also merge into the same collective order (see the module docstring).
        """
        if world <= 1:
            return buckets if ok else None
        pos = {s: i for i, s in enumerate(cands)}
        mine = [[pos[s] for s in b] for b in buckets] if ok else None
        peers = rank_sync.exchange((ok, mine), group, world)
        if not all(p[0] for p in peers):
            magi_logger.warning(
                "%s: planning failed on rank(s) %s; leaving every weight gather unbucketed",
                self.name,
                [r for r, p in enumerate(peers) if not p[0]],
            )
            return None
        # Slot mode already matched the skeletons, weight-ness and dtype included,
        # so rank 0's candidate indices name the same collectives here.
        plan = peers[0][1]
        adopted = merged = None
        if all(0 <= i < len(cands) for b in plan for i in b):
            adopted = [[cands[i] for i in b] for b in plan]
            try:
                merged = self._merged(order, adopted)
            except Exception as exc:  # noqa: BLE001 - every peer still enters _commit's exchange
                magi_logger.warning("%s: merging rank 0's buckets failed (%s); not legal here", self.name, exc, rank="all")
        legal = merged is not None
        skeleton_order = self._skeleton_order(merged, skeleton) if legal else None
        return adopted if self._commit(legal, skeleton_order, group, world) else None

    def _commit(self, legal: bool, skeleton_order: str | None, group, world: int) -> bool:
        """Rank-identical go / no-go: every rank found the plan legal and merges into one collective order."""
        peers = rank_sync.exchange((bool(legal), skeleton_order), group, world)
        if not all(p[0] for p in peers):
            magi_logger.warning(
                "%s: rank 0's bucket plan is not legal on rank(s) %s; leaving every weight gather unbucketed",
                self.name,
                [r for r, p in enumerate(peers) if not p[0]],
            )
            return False
        if len({p[1] for p in peers}) > 1:
            magi_logger.warning(
                "%s: rank 0's buckets would land in different collective slots across ranks (per-rank graphs differ); "
                "leaving every weight gather unbucketed",
                self.name,
            )
            return False
        return True

    @staticmethod
    def _skeleton_order(merged, skeleton) -> str:
        """Digest of the order the original collective skeleton takes in ``merged``.

        In a merged stand-in order the members of a bucket sit together where its
        packed launch will go, so equal digests mean the same collective sequence.
        """
        ordinal = {s: j for j, s in enumerate(skeleton)}
        return hashlib.sha256(repr([ordinal[s] for s in merged if s in ordinal]).encode()).hexdigest()

    @staticmethod
    def _merged(order, buckets):
        """``order`` with ``buckets`` merged in sequence (members as stand-ins, see ``_plan``); None if not legal here."""
        sim = order
        for members in buckets:
            infos = [gather_info(m) for m in members]
            if any(i is None for i in infos) or len({(i.kind, i.group_name, i.world, i.dtype) for i in infos}) != 1:
                return None
            sim = ir_coalesce.simulate_coalesce(sim, members)
            if sim is None:
                return None
        return sim if SnodeGraph(order).validate(sim) else None

    # -- after the merge ----------------------------------------------------------
    def _record(self, packed, unpacks, out_bytes: int, params: PlanParams) -> None:
        """Price the new snodes, so the table does not report them as misses."""
        if not isinstance(self._cost_fn, SnodeCostTable):
            return
        self._cost_fn.record(packed, params.alpha_ns + params.beta_ns_per_byte * out_bytes)
        for u in unpacks:
            self._cost_fn.record(u, 0.0)

    def _report(self, params_by_group: dict, buckets: list[list[BaseSchedulerNode]], n_cands: int) -> None:
        sizes = sorted(sum(gather_info(s).shard_bytes for s in b) for b in buckets) or [0]
        mib = [x / 2**20 for x in sizes]
        launches = n_cands - sum(len(b) for b in buckets) + len(buckets)
        magi_logger.info(
            "%s: %d bucket(s) from %d weight gather(s); %d -> %d launch(es); "
            "bucket local MiB min/median/max %.1f/%.1f/%.1f",
            self.name,
            len(buckets),
            n_cands,
            n_cands,
            launches,
            mib[0],
            mib[len(mib) // 2],
            mib[-1],
        )
        if logging.getLogger("magi_compiler").isEnabledFor(logging.DEBUG):
            fits = ", ".join(
                f"{kind} ws={world} group {group}: alpha={p.alpha_ns / 1e3:.1f}us "
                f"beta={p.beta_ns_per_byte * 2**20 / 1e3:.3f}us/MiB, ratio={p.overhead_ratio:.2f}, "
                f"target={p.target_bytes / 2**20:.1f} MiB gathered"
                for (kind, group, world), p in sorted(params_by_group.items())
            )
            magi_logger.debug("%s: %s", self.name, fits)
