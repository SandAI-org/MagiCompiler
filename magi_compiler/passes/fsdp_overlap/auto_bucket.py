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

from magi_compiler.passes.snode_cost import SnodeCostTable
from magi_compiler.passes.snode_utils import weight_gathers
from magi_compiler.passes.transfer_overlap import CostView, SnodeGraph, rank_sync
from magi_compiler.passes.transfer_overlap.cost import default_runtime_ns_fn
from magi_compiler.passes.weight_offload.node_meta import is_host_offloaded
from magi_compiler.utils import magi_logger

from . import ir_coalesce
from .ir_coalesce import GatherInfo, parse_single_gather

DEFAULT_LAUNCH_OVERHEAD_NS = 10_000.0
DEFAULT_OVERHEAD_RATIOS = {"nccl": 0.1, "ce": 0.5}
"""Launch share per transport (``GatherInfo.transport``) when ``overhead_ratio`` is not set."""


# -- the pure planning problem ----------------------------------------------
@dataclass(frozen=True)
class GatherMember:
    """One gather, as the planner sees it (all indices in the fused order)."""

    launch_index: int  # the launch's own index
    earliest_launch_index: int  # earliest index a launch reading its shard could take
    first_wait_index: int  # a merged launch must come before it
    shard_bytes: int  # bytes this rank contributes
    gathered_bytes: int  # bytes the gather produces


def fit_gather_cost_model(
    samples: list[tuple[float, float]], alpha_guess_ns: float = DEFAULT_LAUNCH_OVERHEAD_NS
) -> tuple[float, float]:
    """Least-squares ``ns = alpha + beta * bytes`` with both terms kept >= 0; returns ``(alpha_ns, beta_ns_per_byte)``.

    With every sample at one size the split between the two terms is not
    observable; ``alpha_guess_ns`` (capped at half the cheapest sample) is taken
    as the fixed part and the rest is charged per byte.
    """
    fit_samples = [(float(x), float(y)) for x, y in samples if x > 0 and y > 0]
    if not fit_samples:
        return alpha_guess_ns, 0.0
    sample_count = len(fit_samples)
    mean_bytes = sum(x for x, _ in fit_samples) / sample_count
    mean_ns = sum(y for _, y in fit_samples) / sample_count
    bytes_squared_deviation = sum((x - mean_bytes) ** 2 for x, _ in fit_samples)
    if bytes_squared_deviation <= 1e-9 * max(mean_bytes * mean_bytes, 1.0):
        alpha_ns = min(alpha_guess_ns, 0.5 * min(y for _, y in fit_samples))
        return alpha_ns, max(0.0, (mean_ns - alpha_ns) / mean_bytes)
    beta_ns_per_byte = sum((x - mean_bytes) * (y - mean_ns) for x, y in fit_samples) / bytes_squared_deviation
    alpha_ns = mean_ns - beta_ns_per_byte * mean_bytes
    if beta_ns_per_byte < 0.0:
        return mean_ns, 0.0
    if alpha_ns < 0.0:
        return 0.0, sum(x * y for x, y in fit_samples) / sum(x * x for x, _ in fit_samples)
    return alpha_ns, beta_ns_per_byte


def gather_bytes_for_overhead_ratio(alpha_ns: float, beta_ns_per_byte: float, ratio: float) -> float:
    """Smallest gathered size whose launch is at most ``ratio`` of its time (inf if bytes cost nothing)."""
    if alpha_ns <= 0.0:
        return 0.0
    if beta_ns_per_byte <= 0.0:
        return float("inf")
    return alpha_ns * (1.0 - ratio) / (ratio * beta_ns_per_byte)


@dataclass
class BucketPlanParams:
    alpha_ns: float
    beta_ns_per_byte: float
    overhead_ratio: float = DEFAULT_OVERHEAD_RATIOS["nccl"]
    max_shard_bytes: int = 0  # ``bucket_size_mib``: local-shard bytes per bucket, 0 = no cap

    @property
    def target_gathered_bytes(self) -> float:
        return gather_bytes_for_overhead_ratio(self.alpha_ns, self.beta_ns_per_byte, self.overhead_ratio)


def plan_bucket_segments(members: list[GatherMember], params: BucketPlanParams) -> list[tuple[int, int]]:
    """Contiguous ``[start, end)`` member segments covering ``members`` (see the module docstring)."""
    target_gathered_bytes = params.target_gathered_bytes
    bucket_segments: list[tuple[int, int]] = []
    segment_start = None
    gathered_bytes = shard_bytes = 0
    earliest_launch_index = first_wait_index = 0

    def append_current_segment(segment_end: int) -> None:
        if segment_start is not None:
            bucket_segments.append((segment_start, segment_end))

    for member_position, member in enumerate(members):
        if member.gathered_bytes >= target_gathered_bytes:
            append_current_segment(member_position)
            segment_start = None
            bucket_segments.append((member_position, member_position + 1))
            continue
        if segment_start is not None:
            below_shard_cap = params.max_shard_bytes <= 0 or shard_bytes + member.shard_bytes <= params.max_shard_bytes
            fits_byte_limits = gathered_bytes < target_gathered_bytes and below_shard_cap
            segment_launch_index = max(
                members[segment_start].launch_index, earliest_launch_index, member.earliest_launch_index
            )
            fits_launch_window = segment_launch_index <= min(first_wait_index, member.first_wait_index)
            if fits_byte_limits and fits_launch_window:
                gathered_bytes += member.gathered_bytes
                shard_bytes += member.shard_bytes
                earliest_launch_index = max(earliest_launch_index, member.earliest_launch_index)
                first_wait_index = min(first_wait_index, member.first_wait_index)
                continue
            append_current_segment(member_position)
        segment_start = member_position
        gathered_bytes, shard_bytes = member.gathered_bytes, member.shard_bytes
        earliest_launch_index, first_wait_index = member.earliest_launch_index, member.first_wait_index
    append_current_segment(len(members))
    return bucket_segments


# -- the pass -----------------------------------------------------------------
class FsdpAutoBucket:
    """Reorder pass that merges small weight gathers until their launch is amortized, and never moves anything else."""

    log_prefix = "FSDP auto bucket"

    def __init__(self, runtime_ns_fn=None, max_bucket_bytes: int = 0, overhead_ratio: float | None = None) -> None:
        self._runtime_ns_fn = runtime_ns_fn if runtime_ns_fn is not None else default_runtime_ns_fn()
        self.max_bucket_bytes = int(max_bucket_bytes)
        # None: per transport, from DEFAULT_OVERHEAD_RATIOS.  FSDPConfig bounds it to (0, 1).
        self.overhead_ratio = None if overhead_ratio is None else float(overhead_ratio)

    def __call__(self, snodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
        order = list(snodes)
        candidates = self._collect_candidates(order)
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        cost = CostView(self._runtime_ns_fn)
        sync_group, skeleton_nodes = None, []
        if world_size > 1:
            # Candidate count is rank-local. Every rank joins the exchange below,
            # including one that found nothing to merge. Returning here skips the
            # all_gather and blocks whichever ranks do enter it.
            collective_indices, collective_signatures = rank_sync.collective_skeleton(order)
            placement_mode, sync_group, world_size = rank_sync.negotiate_placement_mode(
                candidate_count=len(candidates),
                collective_signatures=collective_signatures,
                is_profiling_valid=cost.is_profiling_valid,
                pass_name=self.log_prefix,
            )
            if placement_mode != "slot":
                magi_logger.warning(
                    "%s: per-rank collective skeletons cannot be matched (mode=%s); leaving every weight gather unbucketed",
                    self.log_prefix,
                    placement_mode,
                )
                return snodes
            skeleton_nodes = [order[i] for i in collective_indices]
        if len(candidates) < 2:
            return snodes

        is_local_plan_valid, buckets, params_by_cost_model = ir_coalesce.ir_coalesce_supported(), [], {}
        if not is_local_plan_valid:
            magi_logger.warning(
                "%s: this torch lacks the Inductor internals scheduler-level coalescing needs; "
                "leaving every weight gather unbucketed",
                self.log_prefix,
            )
        else:
            try:
                buckets, params_by_cost_model = self._plan_local_buckets(order, candidates, cost)
            except Exception as exc:  # noqa: BLE001
                if world_size <= 1:
                    raise
                # Only to still enter _adopt_rank_zero_buckets' exchange, which every peer does.
                magi_logger.warning(
                    "%s: planning failed (%s); leaving every weight gather unbucketed", self.log_prefix, exc, rank="all"
                )
                is_local_plan_valid = False
        buckets = self._adopt_rank_zero_buckets(
            is_local_plan_valid=is_local_plan_valid,
            buckets=buckets,
            order=order,
            candidates=candidates,
            sync_group=sync_group,
            world_size=world_size,
            skeleton_nodes=skeleton_nodes,
        )
        if buckets is None:
            return snodes
        if not buckets:
            self._log_bucket_summary(params_by_cost_model, buckets, candidate_count=len(candidates))
            return snodes

        for members in buckets:
            gathered_bytes = sum(parse_single_gather(m).shard_bytes * parse_single_gather(m).world_size for m in members)
            params = params_by_cost_model[self._cost_model_key(parse_single_gather(members[0]))]
            order, coalesced_snode, output_snodes = ir_coalesce.coalesce_all_gathers(order, members)
            self._record_coalesced_costs(coalesced_snode, output_snodes, gathered_bytes, params)
        snodes[:] = order
        ir_coalesce.refresh_memory_planning(snodes)
        self._log_bucket_summary(params_by_cost_model, buckets, candidate_count=len(candidates))
        return snodes

    # -- candidates -----------------------------------------------------------
    @staticmethod
    def _collect_candidates(order: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
        """The weight gathers (see ``snode_utils.weight_gathers``) in the single-weight form this pass can merge."""
        return [s for s in weight_gathers(order) if parse_single_gather(s) is not None]

    @staticmethod
    def _bucket_compatibility_key(snode: BaseSchedulerNode, info: GatherInfo) -> tuple:
        is_offloaded = any(is_host_offloaded(o) for o in snode.node.origins or ())
        return (info.transport, info.dtype, is_offloaded)

    def _partition_compatible_gathers(self, candidates: list[BaseSchedulerNode]) -> list[list[BaseSchedulerNode]]:
        """Split into runs of consecutive gathers on one group, then each run by
        (transport, dtype, host-offloaded); a partition need not be contiguous within its run.

        Transport and dtype because one launch is one op.  Host offload only
        because a resident member would wait for its bucket-mates' host loads.
        """
        group_runs: list[list[BaseSchedulerNode]] = []
        previous_group_name = object()
        for s in candidates:
            info = parse_single_gather(s)
            if info.group_name != previous_group_name:
                group_runs.append([])
                previous_group_name = info.group_name
            group_runs[-1].append(s)
        compatible_runs: list[list[BaseSchedulerNode]] = []
        for group_run in group_runs:
            gathers_by_compatibility: dict[tuple, list[BaseSchedulerNode]] = {}
            for s in group_run:
                compatibility_key = self._bucket_compatibility_key(s, parse_single_gather(s))
                gathers_by_compatibility.setdefault(compatibility_key, []).append(s)
            compatible_runs.extend(gathers_by_compatibility.values())
        return compatible_runs

    # -- planning -------------------------------------------------------------
    def _build_gather_member(self, snode, graph: SnodeGraph) -> GatherMember | None:
        info = parse_single_gather(snode)
        earliest_launch_index = graph.earliest_legal_index({snode, *graph.movable_prep_nodes([snode])})
        waits = graph.waits_of(snode)
        if not waits:
            return None
        return GatherMember(
            launch_index=graph.original_index_by_node[snode],
            earliest_launch_index=earliest_launch_index,
            first_wait_index=min(graph.original_index_by_node[w] for w in waits),
            shard_bytes=info.shard_bytes,
            gathered_bytes=info.shard_bytes * info.world_size,
        )

    @staticmethod
    def _cost_model_key(info: GatherInfo) -> tuple[str, str, int]:
        return (info.transport, info.group_name, info.world_size)

    def _fit_bucket_params(self, candidates, cost: CostView) -> dict[tuple[str, str, int], BucketPlanParams]:
        """One fit per (transport, process group), and one ratio per transport.

        Per transport because a copy-engine gather's fixed and per-byte costs are
        nothing like an NCCL one's.  Per group because the per-byte cost depends on
        who the bytes come from: a one-rank group's gather is a local copy at device
        bandwidth, and fitted together with an eight-rank group's its gigabyte-sized
        samples pull ``beta`` down by an order of magnitude for every other gather.
        """
        samples_by_cost_model: dict[tuple[str, str, int], list[tuple[float, float]]] = {}
        for s in candidates:
            info = parse_single_gather(s)
            sample = (info.shard_bytes * info.world_size, cost(s))
            samples_by_cost_model.setdefault(self._cost_model_key(info), []).append(sample)
        params_by_cost_model = {}
        for cost_model_key, fit_samples in samples_by_cost_model.items():
            transport = cost_model_key[0]
            alpha_ns, beta_ns_per_byte = fit_gather_cost_model(fit_samples)
            if self.overhead_ratio is not None:
                overhead_ratio = self.overhead_ratio
            else:
                overhead_ratio = DEFAULT_OVERHEAD_RATIOS[transport]
            params_by_cost_model[cost_model_key] = BucketPlanParams(
                alpha_ns=alpha_ns,
                beta_ns_per_byte=beta_ns_per_byte,
                overhead_ratio=overhead_ratio,
                max_shard_bytes=self.max_bucket_bytes,
            )
        return params_by_cost_model

    def _plan_local_buckets(self, order, candidates, cost: CostView):
        graph = SnodeGraph(order)
        params_by_cost_model = self._fit_bucket_params(candidates, cost)
        buckets: list[list[BaseSchedulerNode]] = []
        for compatible_run in self._partition_compatible_gathers(candidates):
            params = params_by_cost_model[self._cost_model_key(parse_single_gather(compatible_run[0]))]
            candidate_members = [(s, self._build_gather_member(snode=s, graph=graph)) for s in compatible_run]
            schedulable_members = [(s, m) for s, m in candidate_members if m is not None]
            if len(schedulable_members) < 2:
                continue
            gather_nodes = [s for s, _ in schedulable_members]
            members = [m for _, m in schedulable_members]
            for segment_start, segment_end in plan_bucket_segments(members, params):
                if segment_end - segment_start >= 2:
                    buckets.append(gather_nodes[segment_start:segment_end])

        # Merge in sequence on a stand-in order: each merge moves shard prep,
        # and the next merge must still be legal after it.
        simulated_order, legal_buckets = order, []
        for members in buckets:
            next_order = ir_coalesce.simulate_coalesced_order(simulated_order, members)
            if next_order is None:
                continue
            simulated_order = next_order
            legal_buckets.append(members)
        if not graph.is_topological_order(simulated_order):
            raise RuntimeError("the merged order is not topological")
        return legal_buckets, params_by_cost_model

    def _adopt_rank_zero_buckets(
        self, is_local_plan_valid: bool, buckets, order, candidates, sync_group, world_size: int, skeleton_nodes
    ):
        """Rank 0's buckets, as this rank's snodes, once every rank has found them legal; None to leave the graph alone.

        Every rank must also merge into the same collective order (see the module docstring).
        """
        if world_size <= 1:
            return buckets if is_local_plan_valid else None
        candidate_index_by_node = {s: i for i, s in enumerate(candidates)}
        local_bucket_indices = None
        if is_local_plan_valid:
            local_bucket_indices = [[candidate_index_by_node[s] for s in bucket] for bucket in buckets]
        rank_plans = rank_sync.gather_rank_objects((is_local_plan_valid, local_bucket_indices), sync_group, world_size)
        failed_ranks = [rank for rank, (is_valid, _) in enumerate(rank_plans) if not is_valid]
        if failed_ranks:
            magi_logger.warning(
                "%s: planning failed on rank(s) %s; leaving every weight gather unbucketed", self.log_prefix, failed_ranks
            )
            return None
        # Slot mode already matched the skeletons, weight-ness and dtype included,
        # so rank 0's candidate indices name the same collectives here.
        _, rank_zero_plan = rank_plans[0]
        adopted_buckets = merged_order = None
        if all(0 <= i < len(candidates) for bucket in rank_zero_plan for i in bucket):
            adopted_buckets = [[candidates[i] for i in bucket] for bucket in rank_zero_plan]
            try:
                merged_order = self._simulate_bucket_merges(order, adopted_buckets)
            except Exception as exc:  # noqa: BLE001 - every peer still enters _all_ranks_accept_buckets' exchange
                magi_logger.warning(
                    "%s: merging rank 0's buckets failed (%s); not legal here", self.log_prefix, exc, rank="all"
                )
        is_plan_legal = merged_order is not None
        collective_order_digest = None
        if is_plan_legal:
            collective_order_digest = self._collective_order_digest(merged_order, skeleton_nodes)
        if self._all_ranks_accept_buckets(is_plan_legal, collective_order_digest, sync_group, world_size):
            return adopted_buckets
        return None

    def _all_ranks_accept_buckets(
        self, is_plan_legal: bool, collective_order_digest: str | None, sync_group, world_size: int
    ) -> bool:
        """Rank-identical go / no-go: every rank found the plan legal and merges into one collective order.

        Only exchanges verdicts; the IR is rewritten by the caller afterwards.
        """
        rank_votes = rank_sync.gather_rank_objects((bool(is_plan_legal), collective_order_digest), sync_group, world_size)
        illegal_ranks = [rank for rank, (is_legal, _) in enumerate(rank_votes) if not is_legal]
        if illegal_ranks:
            magi_logger.warning(
                "%s: rank 0's bucket plan is not legal on rank(s) %s; leaving every weight gather unbucketed",
                self.log_prefix,
                illegal_ranks,
            )
            return False
        if len({digest for _, digest in rank_votes}) > 1:
            magi_logger.warning(
                "%s: rank 0's buckets would land in different collective slots across ranks (per-rank graphs differ); "
                "leaving every weight gather unbucketed",
                self.log_prefix,
            )
            return False
        return True

    @staticmethod
    def _collective_order_digest(merged_order, skeleton_nodes) -> str:
        """SHA256 of the order the original collective skeleton takes in ``merged_order``.

        In a merged stand-in order the members of a bucket sit together where its
        coalesced launch will go, so equal digests mean the same collective sequence.
        """
        skeleton_index_by_node = {s: j for j, s in enumerate(skeleton_nodes)}
        merged_collective_indices = [skeleton_index_by_node[s] for s in merged_order if s in skeleton_index_by_node]
        return hashlib.sha256(repr(merged_collective_indices).encode()).hexdigest()

    @staticmethod
    def _simulate_bucket_merges(order, buckets):
        """``order`` with ``buckets`` merged in sequence (members as stand-ins, see
        ``_plan_local_buckets``); None if not legal here.  Touches no IR."""
        simulated_order = order
        for members in buckets:
            gather_infos = [parse_single_gather(m) for m in members]
            if any(info is None for info in gather_infos):
                return None
            if len({(i.transport, i.group_name, i.world_size, i.dtype) for i in gather_infos}) != 1:
                return None
            simulated_order = ir_coalesce.simulate_coalesced_order(simulated_order, members)
            if simulated_order is None:
                return None
        return simulated_order if SnodeGraph(order).is_topological_order(simulated_order) else None

    # -- after the merge ----------------------------------------------------------
    def _record_coalesced_costs(self, coalesced_snode, output_snodes, gathered_bytes: int, params: BucketPlanParams) -> None:
        """Price the new snodes, so the table does not report them as misses."""
        if not isinstance(self._runtime_ns_fn, SnodeCostTable):
            return
        self._runtime_ns_fn.record_runtime_ns(
            snode=coalesced_snode, runtime_ns=params.alpha_ns + params.beta_ns_per_byte * gathered_bytes
        )
        for output_snode in output_snodes:
            self._runtime_ns_fn.record_runtime_ns(snode=output_snode, runtime_ns=0.0)

    def _log_bucket_summary(
        self, params_by_cost_model: dict, buckets: list[list[BaseSchedulerNode]], candidate_count: int
    ) -> None:
        bucket_shard_bytes = sorted(sum(parse_single_gather(s).shard_bytes for s in b) for b in buckets) or [0]
        bucket_sizes_mib = [shard_bytes / 2**20 for shard_bytes in bucket_shard_bytes]
        launch_count = candidate_count - sum(len(b) for b in buckets) + len(buckets)
        magi_logger.info(
            "%s: %d bucket(s) from %d weight gather(s); %d -> %d launch(es); "
            "bucket local MiB min/median/max %.1f/%.1f/%.1f",
            self.log_prefix,
            len(buckets),
            candidate_count,
            candidate_count,
            launch_count,
            bucket_sizes_mib[0],
            bucket_sizes_mib[len(bucket_sizes_mib) // 2],
            bucket_sizes_mib[-1],
        )
        if logging.getLogger("magi_compiler").isEnabledFor(logging.DEBUG):
            fit_summary = ", ".join(
                f"{transport} ws={world_size} group {group_name}: alpha={p.alpha_ns / 1e3:.1f}us "
                f"beta={p.beta_ns_per_byte * 2**20 / 1e3:.3f}us/MiB, ratio={p.overhead_ratio:.2f}, "
                f"target={p.target_gathered_bytes / 2**20:.1f} MiB gathered"
                for (transport, group_name, world_size), p in sorted(params_by_cost_model.items())
            )
            magi_logger.debug("%s: %s", self.log_prefix, fit_summary)
