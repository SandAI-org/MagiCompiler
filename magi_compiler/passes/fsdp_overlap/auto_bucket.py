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

"""Weight all-gather bucketing sized to amortize the launch, inside the Inductor scheduler.

``bucket_mode="auto"`` is the only weight-gather bucketing there is.  This pass
runs right after ``SnodeCostProfile`` and before ``FsdpOverlapReorder``, so it
sees the fused graph with every gather priced, and the reorder then places the
buckets it built.  NCCL and copy-engine gathers are bucketed alike, never into
one bucket together (see ``ir_coalesce``).

The rule.  A gather of ``S`` gathered bytes costs ``T(S) = alpha + beta * S`` on
its stream, fitted to the measured single gathers.  Merging only ever saves the
fixed ``alpha``; what it costs is memory, because every member after the first
is allocated at the bucket's launch instead of at its own.  So a gather is
merged only while its launch is a large share of its time::

    alpha / T(S) <= ratio   <=>   S >= S_target = alpha * (1 - ratio) / (ratio * beta)

A gather of ``S_target`` or more amortizes its own launch and goes alone; it
also ends the bucket being filled, so a bucket never spans one.  The smaller
ones are merged in program order, while the bucket is under ``S_target`` and
stays under ``2 * S_target`` (and ``bucket_size_mib`` of local bytes, when
set): that bounds what a bucket can allocate early to about one target's worth
of small weights.  A merge also stops at the first member whose shard is not
ready before the bucket's earliest wait, and a last bucket under half the
target folds into the one before it when that fits.

``alpha``/``beta`` are fitted per transport and process group, and ``ratio`` is
chosen per transport: a copy-engine gather pays a much larger fixed cost per
launch and a much smaller one per byte than an NCCL one, so one ratio gives it a
target several times the NCCL one; and a group's per-byte cost depends on how
many ranks it gathers from.
Unless set, ``ratio`` is 0.1 for NCCL and 0.5 for the copy engine -- the values
at which gaga4 400B ran level with the old fixed 512 MiB buckets on each
transport, without allocating more gathered weight early.

Every rank must build the same buckets, or the collective sequence diverges.
The collective skeletons are compared first (``rank_sync.negotiate_mode`` must
answer ``slot``), but the costs are not: most snodes are priced on
each rank from its own measurements, so the fitted ``alpha``/``beta`` can differ
slightly.  So every rank plans, and every rank then takes rank 0's plan, checks
it is legal on its own order, and touches the IR only once every rank has said
yes.  A rank with fewer than two gathers still joins that comparison: the
candidate count is rank-local, and returning before the exchange leaves the
other ranks blocked in it.  Every rank leaves the gathers unbucketed only after
the exchange says so.

Matching collective skeletons align candidate ``j`` across ranks even when
rank-private compute differs.  A bucket's launch depends on local producers, so two
ranks can put it on opposite sides of a CP / EP collective.  So each rank also
reports the order its skeleton takes after the merges, and the plan is applied
only where that order is the same everywhere.  ``pinned`` has no skeleton to
compare against and is left unbucketed.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass

import torch.distributed as dist
from torch._inductor.scheduler import BaseSchedulerNode

from magi_compiler.passes.overlap import CostView, SnodeGraph, rank_sync
from magi_compiler.passes.overlap.cost import default_cost_fn
from magi_compiler.passes.weight_offload.node_meta import is_host_offloaded
from magi_compiler.utils import magi_logger

from . import ir_coalesce
from .ir_coalesce import GatherInfo, gather_info
from .node_meta import is_weight_ag
from .reorder import FsdpOverlapReorder

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
    deadline: int  # first real consumer
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

    @property
    def cap_bytes(self) -> float:
        return 2.0 * self.target_bytes


def plan_segments(members: list[Member], params: PlanParams) -> list[tuple[int, int]]:
    """Contiguous ``[start, end)`` segments covering ``members`` (see the module docstring)."""
    target, cap = params.target_bytes, params.cap_bytes
    segs: list[tuple[int, int]] = []
    small: list[bool] = []  # per segment: a bucket of small gathers (may absorb a tail)
    start = None
    out = local = 0
    lo = wait = 0

    def close(end: int) -> None:
        if start is not None:
            segs.append((start, end))
            small.append(True)

    for i, m in enumerate(members):
        if m.out_bytes >= target:
            close(i)
            start = None
            segs.append((i, i + 1))
            small.append(False)
            continue
        if start is not None:
            fits = out < target and out + m.out_bytes <= cap
            fits = fits and (params.max_local_bytes <= 0 or local + m.local_bytes <= params.max_local_bytes)
            if fits and max(members[start].idx, lo, m.lo) <= min(wait, m.wait):
                out += m.out_bytes
                local += m.local_bytes
                lo, wait = max(lo, m.lo), min(wait, m.wait)
                continue
            close(i)
        start, out, local, lo, wait = i, m.out_bytes, m.local_bytes, m.lo, m.wait
    close(len(members))
    return _fold_tails(members, segs, small, params)


def _fold_tails(members: list[Member], segs, small, params: PlanParams) -> list[tuple[int, int]]:
    """Fold a small last bucket of each run of small gathers into the bucket before it, when that fits."""
    out: list[tuple[int, int]] = []
    out_small: list[bool] = []
    for k, (seg, is_small) in enumerate(zip(segs, small)):
        last_of_run = is_small and (k + 1 == len(segs) or not small[k + 1])
        if last_of_run and out and out_small[-1] and out[-1][1] == seg[0]:
            tail = members[seg[0] : seg[1]]
            merged = members[out[-1][0] : seg[1]]
            tail_bytes = sum(m.out_bytes for m in tail)
            merged_bytes = sum(m.out_bytes for m in merged)
            merged_local = sum(m.local_bytes for m in merged)
            legal = max(merged[0].idx, max(m.lo for m in merged)) <= min(m.wait for m in merged)
            if (
                tail_bytes < 0.5 * params.target_bytes
                and merged_bytes <= params.cap_bytes
                and (params.max_local_bytes <= 0 or merged_local <= params.max_local_bytes)
                and legal
            ):
                out[-1] = (out[-1][0], seg[1])
                continue
        out.append(seg)
        out_small.append(is_small)
    return out


# -- the pass -----------------------------------------------------------------
class FsdpAutoBucket:
    """Reorder pass that merges small weight gathers until their launch is amortized, and never moves anything else."""

    name = "FSDP auto bucket"

    def __init__(self, cost_fn=None, max_bucket_bytes: int = 0, overhead_ratio: float | None = None) -> None:
        if overhead_ratio is not None and not 0.0 < overhead_ratio < 1.0:
            raise ValueError(f"FsdpAutoBucket: overhead_ratio must be in (0, 1), got {overhead_ratio}")
        self._cost_fn = cost_fn if cost_fn is not None else default_cost_fn()
        self.max_bucket_bytes = int(max_bucket_bytes)
        # None: per transport, from DEFAULT_OVERHEAD_RATIOS.
        self.overhead_ratio = None if overhead_ratio is None else float(overhead_ratio)

    def __deepcopy__(self, memo):
        # Through memo, so this copy and the profile pass's copy share one table.
        new = type(self).__new__(type(self))
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            new.__dict__[key] = copy.deepcopy(value, memo) if key == "_cost_fn" else value
        return new

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
                anchors=cands, skel_kinds=skel_kinds, costs_ok=cost.ok, who=self.name
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

        ok, buckets, report = True, [], {}
        try:
            if not ir_coalesce.ir_coalesce_supported():
                raise RuntimeError("this torch lacks the Inductor internals scheduler-level coalescing needs")
            buckets, report = self._plan(order, cands, cost)
        except Exception as exc:  # noqa: BLE001
            magi_logger.warning("%s: planning failed (%s); leaving every weight gather unbucketed", self.name, exc, rank="all")
            ok, buckets = False, []
        adopted = self._adopt(ok=ok, buckets=buckets, order=order, cands=cands, group=group, world=world, skeleton=skeleton)
        if adopted is None:
            return snodes
        if adopted is not buckets:
            buckets = adopted
            report["sizes"] = [sum(gather_info(s).shard_bytes for s in b) for b in buckets]
            report["members"] = sum(len(b) for b in buckets)
        if not buckets:
            self._report(report, 0)
            return snodes

        for members in buckets:
            out_bytes = sum(gather_info(m).shard_bytes * gather_info(m).world for m in members)
            params = report["params"][self._fit_key(gather_info(members[0]))]
            order, packed, unpacks = ir_coalesce.coalesce_all_gathers(order, members)
            self._record(packed, unpacks, out_bytes, params)
        snodes[:] = order
        ir_coalesce.refresh_memory_planning(snodes)
        self._report(report, len(buckets))
        return snodes

    # -- candidates -----------------------------------------------------------
    @staticmethod
    def _candidates(order: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
        """Single-weight gathers (NCCL or copy engine) tagged as weight gathers; every one if none is tagged.

        The untagged fallback matches ``FsdpOverlapReorder``, which treats every
        gather as a weight gather; a graph whose lowering tags weights at all has
        its activation gathers left alone.
        """
        plain = [s for s in order if gather_info(s) is not None]
        tagged = [s for s in plain if any(is_weight_ag(o) for o in getattr(s.node, "origins", ()) or ())]
        return tagged if tagged else plain

    @staticmethod
    def _kind(snode: BaseSchedulerNode, info: GatherInfo) -> tuple:
        offloaded = any(is_host_offloaded(o) for o in getattr(snode.node, "origins", ()) or ())
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
    def _member(self, snode, graph: SnodeGraph, cost: CostView) -> Member | None:
        info = gather_info(snode)
        prep = FsdpOverlapReorder._prep_chain([snode], graph)
        moving = {snode, *prep}
        lo = max(ir_coalesce._producer_index(s, graph.index_of, graph.buf_to_snode, moving) for s in moving)
        waits = ir_coalesce._waits_of(snode, graph.order)
        if not waits:
            return None
        wait = min(graph.index_of[w] for w in waits)
        deadline = FsdpOverlapReorder._first_consumer_index([snode, *prep], graph, cost)
        return Member(
            idx=graph.index_of[snode],
            lo=lo,
            wait=wait,
            deadline=wait if deadline is None else deadline,
            local_bytes=info.shard_bytes,
            out_bytes=info.shard_bytes * info.world,
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
                overhead_ratio=self.overhead_ratio if self.overhead_ratio is not None else DEFAULT_OVERHEAD_RATIOS[kind],
                max_local_bytes=self.max_bucket_bytes,
            )
        return out

    def _plan(self, order, cands, cost: CostView):
        graph = SnodeGraph(order)
        params_by_group = self._params_by_group(cands, cost)
        buckets: list[list[BaseSchedulerNode]] = []
        report = {"n": len(cands), "solo": 0, "share_before": 0.0, "share_after": 0.0, "launches": 0, "sizes": []}
        stream_ns = launch_ns_before = launch_ns_after = 0.0
        for run in self._runs(cands):
            params = params_by_group[self._fit_key(gather_info(run[0]))]
            alpha, beta = params.alpha_ns, params.beta_ns_per_byte
            pairs = [(s, self._member(s, graph, cost)) for s in run]
            usable = [(s, m) for s, m in pairs if m is not None]
            if len(usable) < 2:
                continue
            snodes_, members = [s for s, _ in usable], [m for _, m in usable]
            segs = plan_segments(members, params)
            report["solo"] += sum(1 for m in members if m.out_bytes >= params.target_bytes)
            stream_ns += sum(alpha + beta * m.out_bytes for m in members)
            launch_ns_before += alpha * len(members)
            launch_ns_after += alpha * len(segs)
            buckets.extend(snodes_[s:e] for s, e in segs if e - s >= 2)
        if stream_ns > 0:
            report["share_before"] = launch_ns_before / stream_ns
            report["share_after"] = launch_ns_after / (stream_ns - launch_ns_before + launch_ns_after)

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
        report["sizes"] = [sum(gather_info(s).shard_bytes for s in b) for b in kept]
        report["members"] = sum(len(b) for b in kept)
        report["params"] = params_by_group
        return kept, report

    def _adopt(self, ok: bool, buckets, order, cands, group, world: int, skeleton):
        """Rank 0's buckets, as this rank's snodes, once every rank has found them legal; None to leave the graph alone.

        Returns ``buckets`` itself when this rank's own plan is the one adopted.
        Every rank must also merge into the same collective order (see the module docstring).
        """
        if world <= 1:
            return buckets if ok else None
        pos = {s: i for i, s in enumerate(cands)}
        mine = [[pos[s] for s in b] for b in buckets] if ok else None
        peers: list = [None] * world
        try:
            dist.all_gather_object(peers, (ok, mine, self._signature(cands)), group=group)
        except Exception as exc:  # noqa: BLE001
            magi_logger.warning("%s: cross-rank plan exchange failed (%s); leaving gathers unbucketed", self.name, exc)
            return None
        if not all(p[0] for p in peers):
            magi_logger.warning(
                "%s: planning failed on rank(s) %s; leaving every weight gather unbucketed",
                self.name,
                [r for r, p in enumerate(peers) if not p[0]],
            )
            return None
        if any(p[2] != peers[0][2] for p in peers[1:]):
            magi_logger.warning(
                "%s: the ranks' weight gathers do not correspond one to one (transport, world or dtype "
                "differ at the same position); leaving every weight gather unbucketed",
                self.name,
            )
            return None
        lead = peers[0][1]
        distinct = len({hashlib.sha256(repr(p[1]).encode()).hexdigest() for p in peers})
        if distinct > 1:
            magi_logger.info(
                "%s: ranks planned %d distinct bucket plans from their own measured costs; every rank takes rank 0's",
                self.name,
                distinct,
            )
        try:
            adopted = buckets if lead == mine else [[cands[i] for i in b] for b in lead]
            merged = self._merged(order, adopted)
            legal = merged is not None
        except Exception:  # noqa: BLE001 - an unmappable plan is simply not legal here
            adopted, legal = None, False
        skeleton_order = self._skeleton_order(merged, skeleton) if legal else None
        return adopted if self._commit(legal, skeleton_order, group, world) else None

    def _commit(self, legal: bool, skeleton_order: str | None, group, world: int) -> bool:
        """Rank-identical go / no-go: every rank found the plan legal and merges into one collective order."""
        peers: list = [None] * world
        try:
            dist.all_gather_object(peers, (bool(legal), skeleton_order), group=group)
        except Exception as exc:  # noqa: BLE001
            magi_logger.warning("%s: cross-rank agreement failed (%s); leaving gathers unbucketed", self.name, exc)
            return False
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
    def _signature(cands) -> list[tuple]:
        """What has to match at each candidate position for rank 0's indices to mean the same collectives here.

        The group name is left out: it can be rank-specific for one logical group.
        So is host offload: a shard's load is rank-private, and a bucket mixing
        offloaded and resident members is legal, only slower (see ``_runs``).
        """
        return [(info.kind, info.world, str(info.dtype)) for info in map(gather_info, cands)]

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
        record = getattr(self._cost_fn, "record", None)
        if record is None:
            return
        record(packed, params.alpha_ns + params.beta_ns_per_byte * out_bytes)
        for u in unpacks:
            record(u, 0.0)

    def _report(self, report: dict, n_buckets: int) -> None:
        if not report or "params" not in report:
            return
        fits = ", ".join(
            f"{kind} ws={world} group {group}: alpha={p.alpha_ns / 1e3:.1f}us "
            f"beta={p.beta_ns_per_byte * 2**20 / 1e3:.3f}us/MiB, ratio {p.overhead_ratio:.2f} -> "
            f"target {p.target_bytes / 2**20:.1f} MiB gathered (cap {p.cap_bytes / 2**20:.1f})"
            for (kind, group, world), p in sorted(report["params"].items())
        )
        sizes = sorted(report.get("sizes") or [0])
        mib = [x / 2**20 for x in sizes]
        n = report.get("n", 0)
        launches = n - report.get("members", 0) + n_buckets
        magi_logger.info(
            "%s: %d bucket(s) from %d weight gather(s) (%d merged, %d large enough to go alone); "
            "%d -> %d launch(es); %s; "
            "bucket local MiB min/median/max %.1f/%.1f/%.1f; launch share of stream time %.1f%% -> %.1f%%",
            self.name,
            n_buckets,
            n,
            report.get("members", 0),
            report.get("solo", 0),
            n,
            launches,
            fits,
            mib[0],
            mib[len(mib) // 2],
            mib[-1],
            100.0 * report.get("share_before", 0.0),
            100.0 * report.get("share_after", 0.0),
        )
