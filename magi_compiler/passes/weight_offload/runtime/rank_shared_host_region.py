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

"""Rank-shared pinned CPU weights backed by anonymous memfd regions.

Parking is a WORLD collective: every rank enters for the same models in the same order,
including ranks without eligible weights. One gather plans the sharing groups (per NUMA node
or per host) and splits each group's identical weights into chunks. Round k parks chunk k of
every group at once: writers allocate and fill, one exchange publishes them, readers map and
byte-compare, and a second exchange confirms every handoff before writers close their memfd.
An allocation-time ENOSPC keeps that chunk's weights private."""

from __future__ import annotations

import errno
import mmap
import os
import socket
import weakref
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Literal, Mapping

import torch
import torch.distributed as dist

from magi_compiler.utils import magi_logger
from magi_compiler.utils.dist_utils import get_cpu_gloo_group
from magi_compiler.utils.hash import as_flat_bytes, compute_tensor_fingerprint
from magi_compiler.utils.numa import NumaPagePolicy, cuda_device_numa_node, numa_page_policy, online_numa_nodes

from .host_memory_pool import ALIGN_BYTES, _register_host_memory, _unregister_host_memory

SharingScope = Literal["per_numa_node", "per_host_interleaved"]
HostWeightSharing = Literal["off", SharingScope]

# Arbitrary; a host's transient pageable-plus-shared peak grows by one chunk per sharing group.
SHARED_MEMORY_CHUNK_BYTES = 256 << 20


@dataclass(frozen=True)
class _WeightDescriptor:
    """What two ranks compare to decide that they hold the same weight."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    nbytes: int
    fingerprint: bytes


@dataclass(frozen=True)
class _RankPlanInput:
    hostname: str
    sharing_group: str
    descriptor_by_weight_name: dict[str, _WeightDescriptor]


@dataclass(frozen=True)
class _WeightSlice:
    byte_offset: int
    descriptor: _WeightDescriptor


@dataclass(frozen=True)
class _Chunk:
    slice_by_weight_name: Mapping[str, _WeightSlice]
    nbytes: int


_EMPTY_CHUNK = _Chunk(slice_by_weight_name={}, nbytes=0)


@dataclass(frozen=True)
class _SharingGroupPlan:
    # The first rank writes every chunk; the others read it.
    ranks: tuple[int, ...]
    chunks: tuple[_Chunk, ...]


@dataclass(frozen=True)
class _NumaPolicy:
    page_policy: NumaPagePolicy
    numa_nodes: tuple[int, ...]


class SharedHostChunk:
    """One parking round; leaving the context closes the writer's memfd, so enter it around ``finish``."""

    def __init__(self, *, plan: SharedHostParkingPlan, chunk: _Chunk) -> None:
        self.is_writer = dist.get_rank() == plan.writer_rank
        self.is_shared = chunk.nbytes > 0
        self.weight_names = tuple(chunk.slice_by_weight_name)
        self._plan = plan
        self._chunk = chunk
        self._writer_fd: int | None = None
        self._writer_error: Exception | None = None
        self._region_path: str | None = None
        self._mapped_region: torch.Tensor | None = None

    def __enter__(self) -> SharedHostChunk:
        if not (self.is_writer and self.is_shared):
            return self
        try:
            self._writer_fd = _create_shared_memory_file(
                region_label=self._plan.region_label, nbytes=self._chunk.nbytes, numa_policy=self._plan.numa_policy
            )
            # Readers open the memfd through procfs, so the ranks of a group must share a PID namespace.
            self._region_path = f"/proc/{os.getpid()}/fd/{self._writer_fd}"
            self._mapped_region = _map_and_pin_shared_memory_file(path=self._region_path, nbytes=self._chunk.nbytes)
        except Exception as error:
            if isinstance(error, OSError) and error.errno == errno.ENOSPC:
                magi_logger.warning(
                    "host offload: a %s shared chunk hit ENOSPC; its weights stay private", self._plan.region_label
                )
                self.is_shared = False
            else:
                self._writer_error = error
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._writer_fd is not None:
            os.close(self._writer_fd)
            self._writer_fd = None

    @property
    def shared_bytes(self) -> int:
        if not self.is_shared:
            return 0
        return sum(weight_slice.descriptor.nbytes for weight_slice in self._chunk.slice_by_weight_name.values())

    def host_buffer(self, weight_name: str) -> torch.Tensor:
        """The pinned view of ``weight_name`` in the region; readers have one only after ``map_writer_region``."""
        assert self._mapped_region is not None, f"host offload: the shared region of {weight_name!r} is not mapped"
        weight_slice = self._chunk.slice_by_weight_name[weight_name]
        descriptor = weight_slice.descriptor
        weight_bytes = self._mapped_region[weight_slice.byte_offset : weight_slice.byte_offset + descriptor.nbytes]
        return weight_bytes.view(descriptor.dtype).view(descriptor.shape)

    def holds_identical_copy(self, weight_name: str, tensor: torch.Tensor) -> bool:
        return torch.equal(as_flat_bytes(self.host_buffer(weight_name)), as_flat_bytes(tensor))

    def publish(self, *, writer_error: Exception | None) -> None:
        """Every rank: agree on the writers' allocation and fill; readers learn their writer's path and fallback."""
        local_error = self._writer_error if self._writer_error is not None else writer_error
        statuses = _all_gather(
            (self.is_shared, self._region_path, _error_message(local_error)), gloo_group=self._plan.gloo_group
        )
        _raise_for_errors(errors=[status[2] for status in statuses], phase="writer", local_error=local_error)
        self.is_shared, self._region_path, _ = statuses[self._plan.writer_rank]

    def map_writer_region(self) -> None:
        if not self.is_writer and self.is_shared:
            self._mapped_region = _map_and_pin_shared_memory_file(path=self._region_path, nbytes=self._chunk.nbytes)

    def finish(self, *, reader_error: Exception | None) -> None:
        """Every rank: the writer's memfd must stay open until this exchange confirms every reader mapped it."""
        _exchange_errors(local_error=reader_error, phase="reader handoff", gloo_group=self._plan.gloo_group)


@dataclass(frozen=True)
class SharedHostParkingPlan:
    writer_rank: int
    chunks: tuple[_Chunk, ...]
    round_count: int
    region_label: str
    numa_policy: _NumaPolicy | None
    parking_threads: int
    gloo_group: dist.ProcessGroup

    def open_chunk(self, round_index: int) -> SharedHostChunk:
        """Every rank, every round: groups with fewer chunks than ``round_count`` take part with an empty one."""
        return SharedHostChunk(plan=self, chunk=self.chunks[round_index] if round_index < len(self.chunks) else _EMPTY_CHUNK)


def plan_shared_host_parking(
    *, tensor_by_weight_name: Mapping[str, torch.Tensor], region_label: str, sharing_scope: SharingScope
) -> SharedHostParkingPlan | None:
    """Collective over WORLD: call for the same models in the same order and with the same sharing_scope.

    Ranks without eligible CPU weights pass an empty mapping. Returns None everywhere when no group can share."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return None
    gloo_group = get_cpu_gloo_group()
    if gloo_group is None:
        magi_logger.warning("host offload: no gloo group, %s keeps a private host copy on every rank", region_label)
        return None

    local_plan_input, numa_policy, collection_error = None, None, None
    try:
        local_plan_input, numa_policy = _collect_plan_input(
            tensor_by_weight_name=tensor_by_weight_name, sharing_scope=sharing_scope
        )
    except Exception as error:
        collection_error = error
    gathered = _all_gather((local_plan_input, _error_message(collection_error)), gloo_group=gloo_group)
    _raise_for_errors(errors=[error for _, error in gathered], phase="plan input collection", local_error=collection_error)
    plan_input_by_rank = [plan_input for plan_input, _ in gathered]

    group_plan_by_sharing_group = _plan_sharing_groups(plan_input_by_rank)
    round_count = max(len(group_plan.chunks) for group_plan in group_plan_by_sharing_group.values())
    if round_count == 0:
        return None
    local_group_plan = group_plan_by_sharing_group[local_plan_input.sharing_group]
    host_rank_count = sum(plan_input.hostname == local_plan_input.hostname for plan_input in plan_input_by_rank)
    return SharedHostParkingPlan(
        writer_rank=local_group_plan.ranks[0],
        chunks=local_group_plan.chunks,
        round_count=round_count,
        region_label=region_label,
        numa_policy=numa_policy,
        # Every rank of a host copies or compares at once; split its CPUs instead of oversubscribing them.
        parking_threads=min(torch.get_num_threads(), max(1, len(os.sched_getaffinity(0)) // host_rank_count)),
        gloo_group=gloo_group,
    )


def _all_gather(value: object, *, gloo_group: dist.ProcessGroup) -> list:
    gathered = [None] * dist.get_world_size(group=gloo_group)
    dist.all_gather_object(gathered, value, group=gloo_group)
    return gathered


def _error_message(error: Exception | None) -> str | None:
    return None if error is None else f"{type(error).__name__}: {error}"


def _raise_for_errors(*, errors: list[str | None], phase: str, local_error: Exception | None) -> None:
    failures = [f"rank {rank}: {error}" for rank, error in enumerate(errors) if error is not None]
    if failures:
        raise RuntimeError(f"host offload: shared host region {phase} failed: {'; '.join(failures)}") from local_error


def _exchange_errors(*, local_error: Exception | None, phase: str, gloo_group: dist.ProcessGroup) -> None:
    errors = _all_gather(_error_message(local_error), gloo_group=gloo_group)
    _raise_for_errors(errors=errors, phase=phase, local_error=local_error)


def _collect_plan_input(
    *, tensor_by_weight_name: Mapping[str, torch.Tensor], sharing_scope: SharingScope
) -> tuple[_RankPlanInput, _NumaPolicy | None]:
    hostname = socket.gethostname()
    sharing_group, numa_policy = _sharing_group_and_numa_policy(hostname=hostname, sharing_scope=sharing_scope)
    descriptor_by_weight_name = {name: _describe_weight(tensor) for name, tensor in tensor_by_weight_name.items()}
    return (
        _RankPlanInput(hostname=hostname, sharing_group=sharing_group, descriptor_by_weight_name=descriptor_by_weight_name),
        numa_policy,
    )


def _sharing_group_and_numa_policy(*, hostname: str, sharing_scope: SharingScope) -> tuple[str, _NumaPolicy | None]:
    """This rank's sharing group, and where its writer should put the region's pages (None: default policy)."""
    if sharing_scope == "per_host_interleaved":
        numa_nodes = tuple(online_numa_nodes())
        return hostname, _NumaPolicy(page_policy="interleave", numa_nodes=numa_nodes) if len(numa_nodes) > 1 else None
    gpu_numa_node = cuda_device_numa_node(torch.cuda.current_device())
    if gpu_numa_node is None:
        # Unknown GPU affinity: group by host and retain the default page policy.
        return hostname, None
    return f"{hostname}/numa{gpu_numa_node}", _NumaPolicy(page_policy="preferred", numa_nodes=(gpu_numa_node,))


def _describe_weight(tensor: torch.Tensor) -> _WeightDescriptor:
    return _WeightDescriptor(
        shape=tuple(tensor.shape),
        dtype=tensor.dtype,
        nbytes=tensor.numel() * tensor.element_size(),
        fingerprint=compute_tensor_fingerprint(tensor),
    )


def _plan_sharing_groups(plan_input_by_rank: list[_RankPlanInput]) -> dict[str, _SharingGroupPlan]:
    """Every group's plan, so each rank knows how many WORLD rounds parking takes."""
    ranks_by_sharing_group: dict[str, list[int]] = {}
    for rank, plan_input in enumerate(plan_input_by_rank):
        ranks_by_sharing_group.setdefault(plan_input.sharing_group, []).append(rank)
    return {
        sharing_group: _SharingGroupPlan(
            ranks=tuple(ranks), chunks=_plan_group_chunks([plan_input_by_rank[rank] for rank in ranks])
        )
        for sharing_group, ranks in ranks_by_sharing_group.items()
    }


def _plan_group_chunks(group_plan_inputs: list[_RankPlanInput]) -> tuple[_Chunk, ...]:
    """Chunk, in the writer's order, the weights every rank of the group holds identically; larger weights stand alone."""
    writer_plan_input, *reader_plan_inputs = group_plan_inputs
    if not reader_plan_inputs:
        return ()
    chunks = []
    slice_by_weight_name: dict[str, _WeightSlice] = {}
    chunk_bytes = 0
    for name, descriptor in writer_plan_input.descriptor_by_weight_name.items():
        if any(plan_input.descriptor_by_weight_name.get(name) != descriptor for plan_input in reader_plan_inputs):
            continue
        # Aligned like the private pool's regions, which also keeps every dtype view legal.
        aligned_bytes = (descriptor.nbytes + ALIGN_BYTES - 1) // ALIGN_BYTES * ALIGN_BYTES
        if slice_by_weight_name and chunk_bytes + aligned_bytes > SHARED_MEMORY_CHUNK_BYTES:
            chunks.append(_Chunk(slice_by_weight_name=slice_by_weight_name, nbytes=chunk_bytes))
            slice_by_weight_name, chunk_bytes = {}, 0
        slice_by_weight_name[name] = _WeightSlice(byte_offset=chunk_bytes, descriptor=descriptor)
        chunk_bytes += aligned_bytes
    if slice_by_weight_name:
        chunks.append(_Chunk(slice_by_weight_name=slice_by_weight_name, nbytes=chunk_bytes))
    return tuple(chunks)


def _create_shared_memory_file(*, region_label: str, nbytes: int, numa_policy: _NumaPolicy | None) -> int:
    file_descriptor = os.memfd_create(f"magi_offload_{region_label}", os.MFD_CLOEXEC)
    try:
        page_policy = (
            nullcontext()
            if numa_policy is None
            else numa_page_policy(policy=numa_policy.page_policy, nodes=numa_policy.numa_nodes)
        )
        # Allocate every page now, under the NUMA policy, so a capacity failure is ENOSPC here, not SIGBUS on fill.
        with page_policy:
            os.posix_fallocate(file_descriptor, 0, nbytes)
    except BaseException:
        os.close(file_descriptor)
        raise
    return file_descriptor


def _map_and_pin_shared_memory_file(*, path: str, nbytes: int) -> torch.Tensor:
    """The mapping stays registered while any view of it lives and is unregistered before it is unmapped.

    A rank that adopts no weight of a chunk drops its mapping right away; a registration left behind
    would make a later mapping that reuses those addresses fail to register."""
    with open(path, "r+b") as region_file:
        mapping = mmap.mmap(region_file.fileno(), nbytes, flags=mmap.MAP_SHARED)
    mapped_region = torch.frombuffer(mapping, dtype=torch.uint8)
    pin_error_message = (
        f"host offload: could not pin the shared host region {path}, which magi::h2d_load needs; this requires CUDA "
        "host registration of memfd mappings, else set offload_config.offload_host_weight_sharing='off'."
    )
    try:
        _register_host_memory(mapped_region)
    except RuntimeError as error:
        raise RuntimeError(pin_error_message) from error
    if not mapped_region.is_pinned():
        raise RuntimeError(pin_error_message)
    weakref.finalize(mapping, _unregister_host_memory, mapped_region.data_ptr())
    return mapped_region
