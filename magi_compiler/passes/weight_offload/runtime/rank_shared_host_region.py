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

"""Shared pinned copies of identical CPU weights, backed by MAGI_SHARED_BIN_PATH files.

WORLD ranks enter in the same model order, including ranks without eligible weights.
Writers allocate before handoff and fill before readers map; files outlive all
reader mappings. Allocation-time ENOSPC falls back per host. Sampled fingerprints
only select candidates: readers must compare bytes before adopting shared weights.
NUMA grouping and best-effort placement follow OffloadConfig."""

from __future__ import annotations

import dataclasses
import errno
import os
import shutil
import socket
import uuid
from dataclasses import dataclass
from typing import Literal, Mapping

import torch
import torch.distributed as dist

from magi_compiler.cuda.cudart import pin_memory_in_place
from magi_compiler.utils import envs, magi_logger
from magi_compiler.utils.dist_utils import get_cpu_gloo_group
from magi_compiler.utils.hash import compute_tensor_fingerprint
from magi_compiler.utils.numa import NumaPagePolicy, cuda_device_numa_node, numa_page_policy, online_numa_nodes

from .host_memory_pool import ALIGN_BYTES

SharingScope = Literal["per_numa_node", "per_host_interleaved"]
HostWeightSharing = Literal["off", "per_numa_node", "per_host_interleaved"]
"""Mirrors ``OffloadConfig.offload_host_weight_sharing``."""

SHARED_MEMORY_HEADROOM_BYTES = 256 << 20
"""Arbitrary margin for unrelated tmpfs users; concurrent allocations can still exhaust it."""


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
    """One rank's contribution to the gather; ``region_path`` is used if it becomes its group's writer."""

    hostname: str
    sharing_group: str
    region_path: str
    free_shared_memory_bytes: int
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
    region_path: str
    slice_by_weight_name: Mapping[str, _RegionWeightSlice]
    region_bytes: int
    rejected_region_bytes: int = 0


class RankSharedHostRegion:
    """Empty regions still participate in WORLD publication when other groups share weights."""

    def __init__(
        self,
        *,
        region_plan: _SharingGroupRegionPlan,
        is_writer: bool,
        region_numa_policy: _RegionNumaPolicy | None,
        gloo_group: dist.ProcessGroup,
    ) -> None:
        self.is_writer = is_writer
        self._region_plan = region_plan
        self._gloo_group = gloo_group
        self._mapped_region: torch.Tensor | None = None
        self._writer_error: Exception | None = None
        self._allocation_out_of_space = False
        if is_writer and region_plan.region_bytes > 0:
            try:
                _create_shared_memory_file(
                    path=region_plan.region_path, nbytes=region_plan.region_bytes, region_numa_policy=region_numa_policy
                )
            except Exception as error:
                self._writer_error = error
                self._allocation_out_of_space = isinstance(error, OSError) and error.errno == errno.ENOSPC
            if self._writer_error is None:
                try:
                    self._mapped_region = _map_and_pin_shared_memory_file(
                        path=region_plan.region_path, nbytes=region_plan.region_bytes
                    )
                except Exception as error:
                    self._writer_error = error

    def prepare(self) -> None:
        """Every rank: agree on allocation errors before any weight changes ownership."""
        local_error = None if self._allocation_out_of_space else self._writer_error
        status = (self._region_plan.hostname, self._allocation_out_of_space, _error_message(local_error))
        statuses = [None] * dist.get_world_size(group=self._gloo_group)
        dist.all_gather_object(statuses, status, group=self._gloo_group)
        _raise_for_errors(errors=[status[2] for status in statuses], phase="writer preparation", local_error=local_error)
        full_hosts = {hostname for hostname, out_of_space, _ in statuses if out_of_space}
        if self._region_plan.hostname in full_hosts:
            self.remove_file()
            self._mapped_region = None
            self._writer_error = None
            self._region_plan = dataclasses.replace(self._region_plan, slice_by_weight_name={}, region_bytes=0)
            magi_logger.warning("host offload: shared allocation exhausted tmpfs; this host keeps private copies")

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
        return torch.equal(_as_flat_bytes(self.host_buffer(weight_name)), _as_flat_bytes(tensor))

    def publish(self, *, writer_error: Exception | None = None) -> None:
        """Every rank: wait until the writers have filled their regions, map this group's, then drop its file."""
        local_error = self._writer_error if self._writer_error is not None else writer_error
        _exchange_errors(local_error=local_error, phase="writer", gloo_group=self._gloo_group)
        mapping_error = None
        if not self.is_writer and self._region_plan.region_bytes > 0:
            try:
                self._mapped_region = _map_and_pin_shared_memory_file(
                    path=self._region_plan.region_path, nbytes=self._region_plan.region_bytes
                )
            except Exception as error:
                mapping_error = error
        # The file must outlive every reader's mapping call; the exchange also synchronizes successful mappings.
        _exchange_errors(local_error=mapping_error, phase="reader mapping", gloo_group=self._gloo_group)
        self.remove_file()

    def remove_file(self) -> None:
        """Delete the writer's file; idempotent, so a failed park can call it too."""
        if self.is_writer and os.path.exists(self._region_plan.region_path):
            os.remove(self._region_plan.region_path)


def open_rank_shared_host_region(
    *, tensor_by_weight_name: Mapping[str, torch.Tensor], region_label: str, sharing_scope: SharingScope
) -> RankSharedHostRegion | None:
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
            tensor_by_weight_name=tensor_by_weight_name, region_label=region_label, sharing_scope=sharing_scope
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
            "host offload: %s has %.1f GiB free, too little for this host's shared %s regions (%.1f GiB for this "
            "rank's group); every rank keeps a private copy",
            envs.MAGI_SHARED_BIN_PATH,
            local_plan_input.free_shared_memory_bytes / 2**30,
            region_label,
            local_region_plan.rejected_region_bytes / 2**30,
        )
    if all(region_plan.region_bytes == 0 for region_plan in region_plan_by_sharing_group.values()):
        return None
    shared_region = RankSharedHostRegion(
        region_plan=local_region_plan,
        is_writer=dist.get_rank() == local_region_plan.writer_rank,
        region_numa_policy=region_numa_policy,
        gloo_group=gloo_group,
    )
    try:
        shared_region.prepare()
    except Exception:
        shared_region.remove_file()
        raise
    return shared_region


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
    *, tensor_by_weight_name: Mapping[str, torch.Tensor], region_label: str, sharing_scope: SharingScope
) -> tuple[_RankSharingPlanInput, _RegionNumaPolicy | None]:
    hostname = socket.gethostname()
    sharing_group, region_numa_policy = _local_sharing_group_and_region_numa_policy(
        hostname=hostname, sharing_scope=sharing_scope
    )
    return (
        _RankSharingPlanInput(
            hostname=hostname,
            sharing_group=sharing_group,
            region_path=os.path.join(envs.MAGI_SHARED_BIN_PATH, f"magi_offload_{region_label}_{uuid.uuid4().hex}.bin"),
            free_shared_memory_bytes=shutil.disk_usage(envs.MAGI_SHARED_BIN_PATH).free,
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
        region_path=writer_plan_input.region_path,
        slice_by_weight_name=slice_by_weight_name,
        region_bytes=region_bytes,
    )


def _drop_region_plans_on_hosts_without_room(
    *, region_plan_by_sharing_group: dict[str, _SharingGroupRegionPlan], plan_input_by_rank: list[_RankSharingPlanInput]
) -> dict[str, _SharingGroupRegionPlan]:
    """Groups on one host compete for the same tmpfs; writing beyond capacity causes SIGBUS."""
    free_bytes_by_hostname: dict[str, int] = {}
    for plan_input in plan_input_by_rank:
        free_bytes_by_hostname[plan_input.hostname] = min(
            plan_input.free_shared_memory_bytes,
            free_bytes_by_hostname.get(plan_input.hostname, plan_input.free_shared_memory_bytes),
        )
    needed_bytes_by_hostname: dict[str, int] = {}
    for region_plan in region_plan_by_sharing_group.values():
        needed_bytes_by_hostname[region_plan.hostname] = (
            needed_bytes_by_hostname.get(region_plan.hostname, 0) + region_plan.region_bytes
        )

    def fits(hostname: str) -> bool:
        return needed_bytes_by_hostname[hostname] + SHARED_MEMORY_HEADROOM_BYTES <= free_bytes_by_hostname[hostname]

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


def _create_shared_memory_file(*, path: str, nbytes: int, region_numa_policy: _RegionNumaPolicy | None) -> None:
    file_descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    try:
        # tmpfs allocates the pages here, under this thread's NUMA policy. Reserving them now also
        # turns running out of tmpfs into an error instead of a SIGBUS while writing the mapping.
        if region_numa_policy is None:
            os.posix_fallocate(file_descriptor, 0, nbytes)
        else:
            with numa_page_policy(policy=region_numa_policy.page_policy, nodes=region_numa_policy.numa_nodes):
                os.posix_fallocate(file_descriptor, 0, nbytes)
    except OSError:
        os.remove(path)
        raise
    finally:
        os.close(file_descriptor)


def _map_and_pin_shared_memory_file(*, path: str, nbytes: int) -> torch.Tensor:
    mapped_region = torch.from_file(path, shared=True, size=nbytes, dtype=torch.uint8)
    pin_error_message = (
        f"host offload: could not pin the shared host region {path}; magi::h2d_load needs pinned memory to copy "
        "asynchronously. cudaHostRegister refuses mappings of files outside tmpfs, so MAGI_SHARED_BIN_PATH must "
        "be a tmpfs such as /dev/shm, and libcudart must be loadable; or set "
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
