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

"""Shared pinned CPU weights backed by anonymous memfd files.

WORLD ranks enter in the same model order, including ranks without eligible weights.
Groups advance serially in weight batches; readers release originals before the next allocation.
Writers retain their fd until all readers map it through /proc/<pid>/fd/<fd>.
No directory entry is created; existing reader mappings survive writer exit.
Allocation-time ENOSPC falls back per host. Fingerprint matches require byte comparison.
NUMA grouping and best-effort placement follow OffloadConfig."""

from __future__ import annotations

import dataclasses
import errno
import os
import shutil
import socket
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Iterator, Literal, Mapping

import torch
import torch.distributed as dist

from magi_compiler.cuda.cudart import pin_memory_in_place
from magi_compiler.utils import envs, magi_logger
from magi_compiler.utils.dist_utils import get_cpu_gloo_group
from magi_compiler.utils.hash import compute_tensor_fingerprint
from magi_compiler.utils.host_memory import get_available_host_memory_bytes
from magi_compiler.utils.numa import NumaPagePolicy, cuda_device_numa_node, numa_page_policy, online_numa_nodes

from .host_memory_pool import ALIGN_BYTES

SharingScope = Literal["per_numa_node", "per_host_interleaved"]
HostWeightSharing = Literal["off", "per_numa_node", "per_host_interleaved"]
"""Mirrors ``OffloadConfig.offload_host_weight_sharing``."""

SHARED_MEMORY_CHUNK_BYTES = 256 << 20
SHARED_MEMORY_HEADROOM_BYTES = 256 << 20
"""Arbitrary reserve for concurrent host-memory and shared-capacity consumers."""


@dataclass(frozen=True)
class _WeightDescriptor:
    """What two ranks compare to decide that they hold the same weight."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    nbytes: int
    fingerprint: bytes


@dataclass(frozen=True)
class _RegionNumaPolicy:
    page_policy: NumaPagePolicy
    numa_nodes: tuple[int, ...]


@dataclass(frozen=True)
class _RankSharingPlanInput:
    """One rank's weights and capacity budget for sharing-group planning."""

    hostname: str
    sharing_group: str
    free_shared_memory_bytes: int
    available_host_memory_bytes: int
    descriptor_by_weight_name: dict[str, _WeightDescriptor]


@dataclass(frozen=True)
class _RegionWeightSlice:
    byte_offset: int
    shape: tuple[int, ...]
    dtype: torch.dtype
    nbytes: int


@dataclass(frozen=True)
class _SharingGroupRegionPlan:
    """A host must fit all its groups; rejected_region_bytes records a rejected group's size."""

    writer_rank: int
    hostname: str
    slice_by_weight_name: Mapping[str, _RegionWeightSlice]
    region_bytes: int
    rejected_region_bytes: int = 0


class RankSharedHostRegion:
    """Empty regions still participate in WORLD publication when other groups share weights."""

    def __init__(
        self,
        *,
        region_plan: _SharingGroupRegionPlan,
        region_label: str,
        is_writer: bool,
        region_numa_policy: _RegionNumaPolicy | None,
        gloo_group: dist.ProcessGroup,
    ) -> None:
        self.is_writer = is_writer
        self._region_plan = region_plan
        self._gloo_group = gloo_group
        self._mapped_region: torch.Tensor | None = None
        self._writer_error: Exception | None = None
        self.compared_bytes = 0
        self.compare_seconds = 0.0
        self._allocation_out_of_space = False
        self._writer_file = ExitStack()
        self._region_path: str | None = None
        if is_writer and region_plan.region_bytes > 0:
            try:
                file_descriptor = self._writer_file.enter_context(
                    _create_shared_memory_file(
                        region_label=region_label, nbytes=region_plan.region_bytes, region_numa_policy=region_numa_policy
                    )
                )
                self._region_path = f"/proc/{os.getpid()}/fd/{file_descriptor}"
            except Exception as error:
                self._writer_error = error
                self._allocation_out_of_space = isinstance(error, OSError) and error.errno == errno.ENOSPC
            if self._writer_error is None:
                try:
                    self._mapped_region = _map_and_pin_shared_memory_file(
                        path=self._region_path, nbytes=region_plan.region_bytes
                    )
                except Exception as error:
                    self._writer_error = error

    def prepare(self) -> set[str]:
        """Every rank: agree on allocation errors before any weight changes ownership."""
        local_error = None if self._allocation_out_of_space else self._writer_error
        status = (self._region_plan.hostname, self._allocation_out_of_space, _error_message(local_error), self._region_path)
        statuses = [None] * dist.get_world_size(group=self._gloo_group)
        dist.all_gather_object(statuses, status, group=self._gloo_group)
        _raise_for_errors(errors=[status[2] for status in statuses], phase="writer preparation", local_error=local_error)
        self._region_path = statuses[self._region_plan.writer_rank][3]
        full_hosts = {hostname for hostname, out_of_space, _, _ in statuses if out_of_space}
        if self._region_plan.hostname in full_hosts:
            self.close_writer_fd()
            self._mapped_region = None
            self._writer_error = None
            self._region_plan = dataclasses.replace(self._region_plan, slice_by_weight_name={}, region_bytes=0)
            magi_logger.warning("host offload: shared allocation exhausted tmpfs; this host keeps private copies")
        return full_hosts

    @property
    def writer_ready(self) -> bool:
        return self._writer_error is None

    @property
    def shared_bytes(self) -> int:
        return sum(weight_slice.nbytes for weight_slice in self._region_plan.slice_by_weight_name.values())

    def holds(self, weight_name: str) -> bool:
        return weight_name in self._region_plan.slice_by_weight_name

    def host_buffer(self, weight_name: str) -> torch.Tensor:
        """The pinned view of ``weight_name`` inside the region; readers have one only after ``publish``."""
        if self._mapped_region is None:
            raise RuntimeError(f"host offload: the shared host region for {weight_name!r} is not mapped on this rank yet")
        weight_slice = self._region_plan.slice_by_weight_name[weight_name]
        weight_bytes = self._mapped_region[weight_slice.byte_offset : weight_slice.byte_offset + weight_slice.nbytes]
        return weight_bytes.view(weight_slice.dtype).view(weight_slice.shape)

    def holds_identical_copy(self, weight_name: str, tensor: torch.Tensor) -> bool:
        start = time.perf_counter()
        is_identical = torch.equal(_as_flat_bytes(self.host_buffer(weight_name)), _as_flat_bytes(tensor))
        self.compare_seconds += time.perf_counter() - start
        self.compared_bytes += tensor.numel() * tensor.element_size()
        return is_identical

    def publish(self, *, writer_error: Exception | None = None) -> None:
        """Every rank: publish filled weights, map readers, then close the writer fd."""
        local_error = self._writer_error if self._writer_error is not None else writer_error
        _exchange_errors(local_error=local_error, phase="writer", gloo_group=self._gloo_group)
        mapping_error = None
        if not self.is_writer and self._region_plan.region_bytes > 0:
            try:
                assert self._region_path is not None
                self._mapped_region = _map_and_pin_shared_memory_file(
                    path=self._region_path, nbytes=self._region_plan.region_bytes
                )
            except Exception as error:
                mapping_error = error
        # Keep the writer fd open until every reader has mapped it; mappings then own the memory.
        _exchange_errors(local_error=mapping_error, phase="reader mapping", gloo_group=self._gloo_group)
        self.close_writer_fd()

    def finish_parking(self, *, local_error: Exception | None) -> None:
        _exchange_errors(local_error=local_error, phase="reader handoff", gloo_group=self._gloo_group)

    def close_writer_fd(self) -> None:
        """Release the writer fd after publication or failure; existing mappings remain valid."""
        self._writer_file.close()


class RankSharedHostParkingPlan:
    def __init__(
        self,
        *,
        local_plan: _SharingGroupRegionPlan,
        local_rank_count: int,
        chunk_plans: tuple[_SharingGroupRegionPlan, ...],
        region_label: str,
        region_numa_policy: _RegionNumaPolicy | None,
        gloo_group: dist.ProcessGroup,
    ) -> None:
        self._local_plan = local_plan
        self._chunk_plans = chunk_plans
        self._region_label = region_label
        self._region_numa_policy = region_numa_policy
        self._gloo_group = gloo_group
        self.shared_bytes = 0
        self.compare_threads = torch.get_num_threads()
        if dist.get_rank() != local_plan.writer_rank and local_plan.slice_by_weight_name:
            rank_thread_budget = max(1, len(os.sched_getaffinity(0)) // local_rank_count)
            self.compare_threads = min(self.compare_threads, rank_thread_budget)

    def holds(self, weight_name: str) -> bool:
        return weight_name in self._local_plan.slice_by_weight_name

    def finish_private_parking(self, *, local_error: Exception | None) -> None:
        _exchange_errors(local_error=local_error, phase="private handoff", gloo_group=self._gloo_group)

    def chunks(self) -> Iterator[tuple[RankSharedHostRegion, tuple[str, ...]]]:
        """WORLD advances one sharing group at a time, after every reader releases the preceding originals."""
        disabled_hosts: set[str] = set()
        for chunk_plan in self._chunk_plans:
            if chunk_plan.hostname not in disabled_hosts and not self._has_chunk_capacity(chunk_plan):
                disabled_hosts.add(chunk_plan.hostname)
            is_local_group = chunk_plan.writer_rank == self._local_plan.writer_rank
            weight_names = tuple(chunk_plan.slice_by_weight_name) if is_local_group else ()
            local_chunk = chunk_plan
            if not is_local_group or chunk_plan.hostname in disabled_hosts:
                local_chunk = dataclasses.replace(chunk_plan, slice_by_weight_name={}, region_bytes=0)
            region = RankSharedHostRegion(
                region_plan=local_chunk,
                region_label=self._region_label,
                is_writer=dist.get_rank() == chunk_plan.writer_rank,
                region_numa_policy=self._region_numa_policy,
                gloo_group=self._gloo_group,
            )
            try:
                disabled_hosts.update(region.prepare())
                yield region, weight_names
                self.shared_bytes += region.shared_bytes
            finally:
                region.close_writer_fd()

    def _has_chunk_capacity(self, chunk_plan: _SharingGroupRegionPlan) -> bool:
        local_error = None
        available_bytes = 0
        try:
            available_bytes = get_available_host_memory_bytes()
        except Exception as error:
            local_error = error
        status = (self._local_plan.hostname, available_bytes, _error_message(local_error))
        statuses = [None] * dist.get_world_size(group=self._gloo_group)
        dist.all_gather_object(statuses, status, group=self._gloo_group)
        _raise_for_errors(errors=[status[2] for status in statuses], phase="chunk capacity", local_error=local_error)
        fits = all(
            available >= chunk_plan.region_bytes + SHARED_MEMORY_HEADROOM_BYTES
            for hostname, available, _ in statuses
            if hostname == chunk_plan.hostname
        )
        if not fits and self._local_plan.hostname == chunk_plan.hostname:
            magi_logger.warning("host offload: insufficient host/cgroup headroom for the next chunk; keeping private copies")
        return fits


def open_rank_shared_host_region(
    *, tensor_by_weight_name: Mapping[str, torch.Tensor], region_label: str, sharing_scope: SharingScope
) -> RankSharedHostParkingPlan | None:
    """Collective over WORLD: call for the same models in the same order and with the same sharing_scope.

    Ranks without eligible CPU weights pass an empty mapping. Returns None everywhere
    when no group can share; allocation-time ENOSPC falls back to private copies per host."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return None
    gloo_group = get_cpu_gloo_group()
    if gloo_group is None:
        magi_logger.warning("host offload: no gloo group, %s keeps a private host copy on every rank", region_label)
        return None

    collection_error = None
    local_plan_input = None
    region_numa_policy = None
    try:
        local_plan_input, region_numa_policy = _collect_local_plan_input(
            tensor_by_weight_name=tensor_by_weight_name, sharing_scope=sharing_scope
        )
    except Exception as error:
        collection_error = error
    _exchange_errors(local_error=collection_error, phase="plan input collection", gloo_group=gloo_group)
    assert local_plan_input is not None
    sharing_group = local_plan_input.sharing_group
    plan_input_by_rank: list[_RankSharingPlanInput | None] = [None] * dist.get_world_size()
    dist.all_gather_object(plan_input_by_rank, local_plan_input, group=gloo_group)

    region_plan_by_sharing_group = _plan_sharing_group_regions(plan_input_by_rank)
    local_region_plan = region_plan_by_sharing_group[sharing_group]
    if local_region_plan.rejected_region_bytes > 0:
        magi_logger.warning(
            "host offload: %s has %.1f GiB free; sharing budget or host/cgroup chunk headroom is insufficient "
            "for %s (%.1f GiB for this rank's group); keeping private copies",
            envs.MAGI_SHARED_BIN_PATH,
            local_plan_input.free_shared_memory_bytes / 2**30,
            region_label,
            local_region_plan.rejected_region_bytes / 2**30,
        )
    if all(region_plan.region_bytes == 0 for region_plan in region_plan_by_sharing_group.values()):
        return None
    return RankSharedHostParkingPlan(
        local_plan=local_region_plan,
        local_rank_count=sum(plan_input.hostname == local_region_plan.hostname for plan_input in plan_input_by_rank),
        chunk_plans=tuple(
            chunk for group_plan in region_plan_by_sharing_group.values() for chunk in _split_region_plan(group_plan)
        ),
        region_label=region_label,
        region_numa_policy=region_numa_policy,
        gloo_group=gloo_group,
    )


def _error_message(error: Exception | None) -> str | None:
    return None if error is None else f"{type(error).__name__}: {error}"


def _raise_for_errors(*, errors: list[str | None], phase: str, local_error: Exception | None) -> None:
    failures = [f"rank {rank}: {error}" for rank, error in enumerate(errors) if error is not None]
    if failures:
        raise RuntimeError(f"host offload: shared host region {phase} failed: {'; '.join(failures)}") from local_error


def _exchange_errors(*, local_error: Exception | None, phase: str, gloo_group: dist.ProcessGroup) -> None:
    errors: list[str | None] = [None] * dist.get_world_size(group=gloo_group)
    dist.all_gather_object(errors, _error_message(local_error), group=gloo_group)
    _raise_for_errors(errors=errors, phase=phase, local_error=local_error)


def _collect_local_plan_input(
    *, tensor_by_weight_name: Mapping[str, torch.Tensor], sharing_scope: SharingScope
) -> tuple[_RankSharingPlanInput, _RegionNumaPolicy | None]:
    hostname = socket.gethostname()
    sharing_group, region_numa_policy = _local_sharing_group_and_region_numa_policy(
        hostname=hostname, sharing_scope=sharing_scope
    )
    return (
        _RankSharingPlanInput(
            hostname=hostname,
            sharing_group=sharing_group,
            free_shared_memory_bytes=shutil.disk_usage(envs.MAGI_SHARED_BIN_PATH).free,
            available_host_memory_bytes=get_available_host_memory_bytes(),
            descriptor_by_weight_name={name: _describe_weight(tensor) for name, tensor in tensor_by_weight_name.items()},
        ),
        region_numa_policy,
    )


def _local_sharing_group_and_region_numa_policy(
    *, hostname: str, sharing_scope: SharingScope
) -> tuple[str, _RegionNumaPolicy | None]:
    """This rank's sharing group, and where its writer should put the region's pages (None: default policy)."""
    if sharing_scope == "per_host_interleaved":
        numa_nodes = tuple(online_numa_nodes())
        region_numa_policy = (
            _RegionNumaPolicy(page_policy="interleave", numa_nodes=numa_nodes) if len(numa_nodes) > 1 else None
        )
        return hostname, region_numa_policy
    gpu_numa_node = cuda_device_numa_node(torch.cuda.current_device())
    if gpu_numa_node is None:
        # Unknown GPU affinity: group by host and retain the default page policy.
        return hostname, None
    return f"{hostname}/numa{gpu_numa_node}", _RegionNumaPolicy(page_policy="preferred", numa_nodes=(gpu_numa_node,))


def _describe_weight(tensor: torch.Tensor) -> _WeightDescriptor:
    return _WeightDescriptor(
        shape=tuple(tensor.shape),
        dtype=tensor.dtype,
        nbytes=tensor.numel() * tensor.element_size(),
        fingerprint=compute_tensor_fingerprint(tensor),
    )


def _plan_sharing_group_regions(plan_input_by_rank: list[_RankSharingPlanInput]) -> dict[str, _SharingGroupRegionPlan]:
    """Every group's region plan, so each rank knows whether any group will synchronize on a region."""
    ranks_by_sharing_group: dict[str, list[int]] = {}
    for rank, plan_input in enumerate(plan_input_by_rank):
        ranks_by_sharing_group.setdefault(plan_input.sharing_group, []).append(rank)
    region_plan_by_sharing_group = {
        sharing_group: _plan_sharing_group_region(
            group_plan_input_by_rank={rank: plan_input_by_rank[rank] for rank in group_ranks}
        )
        for sharing_group, group_ranks in ranks_by_sharing_group.items()
    }
    return _drop_region_plans_on_hosts_without_room(
        region_plan_by_sharing_group=region_plan_by_sharing_group, plan_input_by_rank=plan_input_by_rank
    )


def _plan_sharing_group_region(group_plan_input_by_rank: dict[int, _RankSharingPlanInput]) -> _SharingGroupRegionPlan:
    """Lay out, in the writer's order, the weights every rank of one group holds identically."""
    writer_rank, *reader_ranks = group_plan_input_by_rank
    writer_plan_input = group_plan_input_by_rank[writer_rank]
    reader_plan_inputs = [group_plan_input_by_rank[rank] for rank in reader_ranks]
    shared_descriptor_by_weight_name = {
        name: descriptor
        for name, descriptor in writer_plan_input.descriptor_by_weight_name.items()
        if reader_plan_inputs
        and all(plan_input.descriptor_by_weight_name.get(name) == descriptor for plan_input in reader_plan_inputs)
    }

    slice_by_weight_name = {}
    region_bytes = 0
    for name, descriptor in shared_descriptor_by_weight_name.items():
        slice_by_weight_name[name] = _RegionWeightSlice(
            byte_offset=region_bytes, shape=descriptor.shape, dtype=descriptor.dtype, nbytes=descriptor.nbytes
        )
        # Aligned like the private pool's regions, which also keeps every dtype view legal.
        region_bytes += (descriptor.nbytes + ALIGN_BYTES - 1) // ALIGN_BYTES * ALIGN_BYTES
    return _SharingGroupRegionPlan(
        writer_rank=writer_rank,
        hostname=writer_plan_input.hostname,
        slice_by_weight_name=slice_by_weight_name,
        region_bytes=region_bytes,
    )


def _split_region_plan(region_plan: _SharingGroupRegionPlan) -> Iterator[_SharingGroupRegionPlan]:
    slices = {}
    chunk_bytes = 0
    for name, weight_slice in region_plan.slice_by_weight_name.items():
        aligned_bytes = (weight_slice.nbytes + ALIGN_BYTES - 1) // ALIGN_BYTES * ALIGN_BYTES
        if slices and chunk_bytes + aligned_bytes > SHARED_MEMORY_CHUNK_BYTES:
            yield dataclasses.replace(region_plan, slice_by_weight_name=slices, region_bytes=chunk_bytes)
            slices = {}
            chunk_bytes = 0
        slices[name] = dataclasses.replace(weight_slice, byte_offset=chunk_bytes)
        chunk_bytes += aligned_bytes
    if slices:
        yield dataclasses.replace(region_plan, slice_by_weight_name=slices, region_bytes=chunk_bytes)


def _drop_region_plans_on_hosts_without_room(
    *, region_plan_by_sharing_group: dict[str, _SharingGroupRegionPlan], plan_input_by_rank: list[_RankSharingPlanInput]
) -> dict[str, _SharingGroupRegionPlan]:
    """Check total sharing budget and the transient chunk against host/cgroup headroom."""
    free_bytes_by_hostname: dict[str, int] = {}
    available_bytes_by_hostname: dict[str, int] = {}
    for plan_input in plan_input_by_rank:
        free_bytes_by_hostname[plan_input.hostname] = min(
            plan_input.free_shared_memory_bytes,
            free_bytes_by_hostname.get(plan_input.hostname, plan_input.free_shared_memory_bytes),
        )
        available_bytes_by_hostname[plan_input.hostname] = min(
            plan_input.available_host_memory_bytes,
            available_bytes_by_hostname.get(plan_input.hostname, plan_input.available_host_memory_bytes),
        )
    peak_chunk_by_hostname: dict[str, int] = {}
    needed_bytes_by_hostname: dict[str, int] = {}
    for region_plan in region_plan_by_sharing_group.values():
        for chunk in _split_region_plan(region_plan):
            peak_chunk_by_hostname[region_plan.hostname] = max(
                peak_chunk_by_hostname.get(region_plan.hostname, 0), chunk.region_bytes
            )
        needed_bytes_by_hostname[region_plan.hostname] = (
            needed_bytes_by_hostname.get(region_plan.hostname, 0) + region_plan.region_bytes
        )

    def fits(hostname: str) -> bool:
        return (
            needed_bytes_by_hostname[hostname] + SHARED_MEMORY_HEADROOM_BYTES <= free_bytes_by_hostname[hostname]
            and peak_chunk_by_hostname.get(hostname, 0) + SHARED_MEMORY_HEADROOM_BYTES <= available_bytes_by_hostname[hostname]
        )

    return {
        sharing_group: (
            region_plan
            if fits(region_plan.hostname)
            else dataclasses.replace(
                region_plan, slice_by_weight_name={}, region_bytes=0, rejected_region_bytes=region_plan.region_bytes
            )
        )
        for sharing_group, region_plan in region_plan_by_sharing_group.items()
    }


@contextmanager
def _create_shared_memory_file(
    *, region_label: str, nbytes: int, region_numa_policy: _RegionNumaPolicy | None
) -> Iterator[int]:
    with os.fdopen(os.memfd_create(f"magi_offload_{region_label}", flags=os.MFD_CLOEXEC), "w+b") as region_file:
        file_descriptor = region_file.fileno()
        # Allocate before mapping to report capacity failure instead of SIGBUS during weight copies.
        if region_numa_policy is None:
            os.posix_fallocate(file_descriptor, 0, nbytes)
        else:
            with numa_page_policy(policy=region_numa_policy.page_policy, nodes=region_numa_policy.numa_nodes):
                os.posix_fallocate(file_descriptor, 0, nbytes)
        yield file_descriptor


def _map_and_pin_shared_memory_file(*, path: str, nbytes: int) -> torch.Tensor:
    mapped_region = torch.from_file(path, shared=True, size=nbytes, dtype=torch.uint8)
    pin_error_message = (
        f"host offload: could not pin the shared host region {path}; magi::h2d_load needs pinned memory to copy "
        "asynchronously. Anonymous memfd mappings require CUDA host registration support and loadable libcudart; set "
        "offload_config.offload_host_weight_sharing='off'."
    )
    try:
        pin_memory_in_place(mapped_region)
    except RuntimeError as error:
        raise RuntimeError(pin_error_message) from error
    if not mapped_region.is_pinned():
        raise RuntimeError(pin_error_message)
    return mapped_region


def _as_flat_bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(-1).view(torch.uint8)
