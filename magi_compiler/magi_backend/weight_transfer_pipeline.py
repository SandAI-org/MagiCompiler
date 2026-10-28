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

"""Whole-graph weight rewrites ahead of the graph split, and the scheduler passes they need.

FSDP lowers weight redistributes to explicit gathers; graph weight offload binds
weights to host slots and splices in ``magi::h2d_load``.  Either way the
transfers are then placed behind compute by Inductor reorder passes.
"""

from __future__ import annotations

import functools
from unittest import mock

import torch
import torch.distributed as dist
import torch.fx as fx
from torch.distributed.tensor import DTensor

from magi_compiler.config import CudaGraphMode, FSDPConfig, OffloadConfig
from magi_compiler.passes.fsdp_overlap import (
    FsdpAutoBucket,
    FsdpOverlapReorder,
    bind_weights_for_copy_engine,
    lower_prim_redistribute_to_collectives,
    rewrite_weight_ag_to_copy_engine,
)
from magi_compiler.passes.snode_cost import SnodeCostProfile, SnodeCostTable
from magi_compiler.passes.transfer_overlap import rank_sync
from magi_compiler.passes.weight_offload import (
    FsdpShardSource,
    H2dLoadReorder,
    PlainParamSource,
    bind_host_slots_to_graph,
    get_global_host_weight_store,
    insert_h2d_loads,
)
from magi_compiler.profiling import ProfilingRuntimeEstimator
from magi_compiler.utils import magi_logger

_MAX_LOGGED_WEIGHT_NAMES = 8
"""Arbitrary cap that keeps the restored-weights warning to one readable line."""


def _mib_to_bytes(mib: int) -> int:
    return int(mib) * 1024 * 1024


@functools.cache
def _allow_reorder_without_process_group() -> None:
    """Let Inductor's overlap reorder run in a process that never initialized torch.distributed."""
    from torch._inductor import comms

    reorder = comms.reorder_compute_and_comm_for_overlap

    @functools.wraps(reorder)
    def reorder_as_rank_0(snodes):
        if dist.is_initialized():
            return reorder(snodes)
        # Workaround: torch 2.9's reorder_compute_and_comm_for_overlap gates a debug dump on an
        # unguarded dist.get_rank(); drop once the minimum torch checks is_initialized() first.
        with mock.patch.object(dist, "get_rank", lambda group=None: 0):
            return reorder(snodes)

    comms.reorder_compute_and_comm_for_overlap = reorder_as_rank_0


def _any_rank_inserted_loads(*, offload_config: OffloadConfig, loads_inserted: bool) -> bool:
    if not offload_config.graph_weight_offload:
        return False
    # Overlap reorder must be on for every rank or none: PT 2.12 then gathers a graph hash over WORLD.
    return rank_sync.any_rank_has_flag(loads_inserted)


def build_transfer_overlap_passes(
    *, fsdp_config: FSDPConfig, offload_config: OffloadConfig, loads_inserted: bool, default_passes: list | None
) -> list | None:
    """Inductor's ``reorder_for_compute_comm_overlap_passes`` for this graph; None leaves Inductor's own.

    ``SnodeCostProfile`` runs first and fills the cost table every later pass reads.
    FSDP replaces configured reorders; offload alone keeps only explicitly configured ones.
    Inductor's ``raise_comms``/``sink_waits`` are never added: ``sink_waits`` treats H2D waits
    as collective waits and hoists every dependency-free ``empty`` to the graph start.
    """
    if not (fsdp_config.enable_fsdp or _any_rank_inserted_loads(offload_config=offload_config, loads_inserted=loads_inserted)):
        return None

    costs = SnodeCostTable()
    # Plain-parameter models may compile on only one rank of an initialized WORLD.
    estimator = (
        None if fsdp_config.cost_mode == "analytical" else ProfilingRuntimeEstimator(sync_across_ranks=fsdp_config.enable_fsdp)
    )
    passes: list = [SnodeCostProfile(costs, estimator)]

    if fsdp_config.enable_fsdp:
        if fsdp_config.bucket_mode == "auto":
            # Ahead of the FSDP reorder, which places the merged gathers;
            # bucketing needs profiled costs and prices the nodes it creates.
            passes.append(
                FsdpAutoBucket(
                    runtime_ns_fn=costs,
                    max_bucket_bytes=_mib_to_bytes(fsdp_config.bucket_size_mib),
                    overhead_ratio=fsdp_config.auto_bucket_overhead_ratio,
                )
            )
        passes.append(
            FsdpOverlapReorder(
                comm_overlap_window_margin_ns=fsdp_config.comm_overlap_window_margin_ns,
                runtime_ns_fn=costs,
                comm_overlap_window_scale=fsdp_config.comm_overlap_window_scale,
            )
        )
    else:
        passes.extend(default_passes or ())

    if loads_inserted:
        # Last: loads are hoisted around the gathers' final positions.
        # Rank-local: placing loads issues no collective.
        passes.append(
            H2dLoadReorder(
                bandwidth_bytes_per_ns=get_global_host_weight_store().h2d_bandwidth_bytes_per_ns(
                    offload_config.offload_h2d_bandwidth_gbps
                ),
                weight_budget_bytes=_mib_to_bytes(offload_config.offload_max_device_weight_mib),
                runtime_ns_fn=costs,
            )
        )
    return passes


def _validate_weight_transfer_config(
    *, fsdp_config: FSDPConfig, offload_config: OffloadConfig, cudagraph_mode: CudaGraphMode
) -> None:
    if fsdp_config.enable_fsdp:
        assert cudagraph_mode == CudaGraphMode.NONE, "fsdp_config.enable_fsdp requires cudagraph_mode=NONE"
    if offload_config.graph_weight_offload:
        assert (
            not offload_config.model_cpu_offload
        ), "offload_config.graph_weight_offload and offload_config.model_cpu_offload are mutually exclusive"
        assert cudagraph_mode == CudaGraphMode.NONE, (
            "offload_config.graph_weight_offload requires cudagraph_mode=NONE: "
            "magi::h2d_load uses a separate stream and CUDA events"
        )
        assert not (fsdp_config.enable_fsdp and fsdp_config.transport == "copy_engine"), (
            "offload_config.graph_weight_offload requires fsdp_config.transport='nccl': "
            "copy-engine gathers require device-resident peer shards"
        )


def _restore_weights_without_h2d_load(example_inputs) -> None:
    """Restore this graph's parked weights that the offload rewrite left without loads."""
    # Other models' slots must stay parked until their own graphs compile.
    store = get_global_host_weight_store()
    input_local_tensors = (ex._local_tensor if isinstance(ex, DTensor) else ex for ex in example_inputs or ())
    input_host_slots = []
    for tensor in input_local_tensors:
        if not isinstance(tensor, torch.Tensor):
            continue
        host_slot = store.slot_for_tensor(tensor)
        if host_slot is not None:
            input_host_slots.append(host_slot)
    # Exclude bytes made resident by other calls or passes.
    resident_bytes_before = store.resident_bytes()
    restored_weight_names = store.restore_weights_without_h2d_load(input_host_slots)
    if not restored_weight_names:
        return
    restored_bytes = store.resident_bytes() - resident_bytes_before
    shown_names = ", ".join(restored_weight_names[:_MAX_LOGGED_WEIGHT_NAMES])
    hidden_name_count = len(restored_weight_names) - _MAX_LOGGED_WEIGHT_NAMES
    if hidden_name_count > 0:
        shown_names += f", +{hidden_name_count} more"
    magi_logger.warning(
        "host offload: restored %d weight(s) without graph loads, %.1f MiB; weights: %s",
        len(restored_weight_names),
        restored_bytes / 2**20,
        shown_names,
    )


class WeightTransferPipeline:
    """FSDP and graph-weight-offload rewrites of one whole graph, and the scheduler passes that place them."""

    def __init__(
        self,
        *,
        fsdp_config: FSDPConfig,
        offload_config: OffloadConfig,
        cudagraph_mode: CudaGraphMode,
        disable_graph_split: bool,
    ) -> None:
        if fsdp_config.enable_fsdp and not disable_graph_split:
            # enable_fsdp is process-wide: split-graph models (e.g. a VAE beside an FSDP DiT)
            # share it, but whole-graph FSDP overlap only applies to fullgraph compiles.
            fsdp_config = fsdp_config.model_copy(update={"enable_fsdp": False})
        _validate_weight_transfer_config(fsdp_config=fsdp_config, offload_config=offload_config, cudagraph_mode=cudagraph_mode)
        self._fsdp_config = fsdp_config
        self._offload_config = offload_config

    def rewrite_graph_and_build_scheduler_passes(
        self, graph: fx.GraphModule, example_inputs, *, default_passes: list | None
    ) -> list | None:
        """Rewrite ``graph`` in place; return its scheduler passes, or None to leave Inductor's own."""
        if self._fsdp_config.enable_fsdp:
            lower_prim_redistribute_to_collectives(graph)
            magi_logger.info(
                "FSDP fullgraph overlap: transport=%s bucket_mode=%s",
                self._fsdp_config.transport,
                self._fsdp_config.bucket_mode,
            )
        h2d_loads_inserted = self._rewrite_weight_transfers(graph=graph, example_inputs=example_inputs)
        passes = build_transfer_overlap_passes(
            fsdp_config=self._fsdp_config,
            offload_config=self._offload_config,
            loads_inserted=h2d_loads_inserted,
            default_passes=default_passes,
        )
        if self._offload_config.graph_weight_offload:
            # The compilation interpreter runs next and must not read storage-free weights.
            _restore_weights_without_h2d_load(example_inputs)
        if passes is not None:
            _allow_reorder_without_process_group()
        return passes

    def _rewrite_weight_transfers(self, graph: fx.GraphModule, example_inputs) -> bool:
        """Rewrite copy-engine gathers or host-offload transfers; return True when H2D loads were inserted."""
        fsdp_config = self._fsdp_config
        if fsdp_config.enable_fsdp and fsdp_config.transport == "copy_engine":
            bind_weights_for_copy_engine(
                graph=graph, example_inputs=example_inputs, min_shard_bytes=_mib_to_bytes(fsdp_config.symm_min_shard_mib)
            )
            rewrite_weight_ag_to_copy_engine(graph)
            return False
        if not self._offload_config.graph_weight_offload:
            return False
        bound_weight_count = bind_host_slots_to_graph(
            graph=graph,
            example_inputs=example_inputs,
            source=FsdpShardSource() if fsdp_config.enable_fsdp else PlainParamSource(),
        )
        if bound_weight_count == 0:
            return False
        insert_h2d_loads(graph)
        return True
