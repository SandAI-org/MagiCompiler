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

"""One pinned host copy of the weights that a group of ranks on one machine holds identically.

Each rank parks its own pinned copy of an unsharded weight, so a machine running N
ranks of a replicated model holds N copies. A shared region keeps one per sharing
group: the group's lowest rank (the writer) copies the weights into a file under
``MAGI_SHARED_BIN_PATH``, and every rank of the group maps the file and pins its own
mapping, so ``magi::h2d_load`` still copies asynchronously.

A sharing group, chosen by ``SharingScope``, is either

* ``per_numa_node``: the ranks whose GPUs attach to one NUMA node of a machine; the
  region's pages stay on that node, so every load reads local memory, and a machine
  keeps one copy per NUMA node.
* ``per_host_interleaved``: all ranks of a machine; the region's pages alternate over
  its NUMA nodes, so no node serves every GPU, and a machine keeps one copy.

Where the pages land matters once every GPU loads at once: on a 2-socket 8x H200 host,
one region on a single node cut the concurrent H2D bandwidth of the far socket's GPUs
from 55 to 26 GB/s; ``per_host_interleaved`` measured 53 GB/s, ``per_numa_node`` 55 GB/s.

The protocol is a collective over WORLD, entered by every rank in the same order:

1. ``open_rank_shared_host_region`` gathers every rank's sharing group, free shared
   memory and weight fingerprints. A weight is shared in a group when every rank there
   holds it with the same name, shape, dtype and fingerprint, and the machine has room
   for all of its groups' regions; otherwise it stays private.
2. The writer fills its host buffers, then ``publish`` lets the other ranks map the
   file and removes it; the mappings keep the memory alive.
3. A fingerprint only samples a weight, so readers compare the mapped copy with their
   own byte for byte (``holds_identical_copy``) and keep a private copy on mismatch.
"""

from __future__ import annotations

import dataclasses
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
"""Free space a host's shared-memory directory keeps beyond its regions; arbitrary."""


@dataclass(frozen=True)
class _WeightDescriptor:
    """What two ranks compare to decide that they hold the same weight."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    nbytes: int
    fingerprint: bytes


@dataclass(frozen=True)
class _PagePlacement:
    policy: NumaPagePolicy
    numa_nodes: tuple[int, ...]


@dataclass(frozen=True)
class _RankOffer:
    """One rank's contribution to the gather; ``region_path`` is used if it becomes its group's writer."""

    hostname: str
    sharing_group: str
    region_path: str
    free_shared_memory_bytes: int
    descriptor_by_weight_name: dict[str, _WeightDescriptor]


@dataclass(frozen=True)
class _SharedWeightPlacement:
    byte_offset: int
    shape: tuple[int, ...]
    dtype: torch.dtype
    nbytes: int


@dataclass(frozen=True)
class _GroupLayout:
    """Where each weight shared by one sharing group lives in that group's region.

    A group on a machine without room for all of its groups' regions shares nothing;
    ``rejected_region_bytes`` is what its own region needed.
    """

    writer_rank: int
    hostname: str
    region_path: str
    placement_by_weight_name: Mapping[str, _SharedWeightPlacement]
    region_bytes: int
    rejected_region_bytes: int = 0


class RankSharedHostRegion:
    """This group's shared weight copies, written by one rank and mapped by all of them.

    A rank whose group shares nothing still gets a region, with no weights, because
    ``publish`` synchronizes WORLD and every rank must join it.
    """

    def __init__(
        self, *, layout: _GroupLayout, is_writer: bool, page_placement: _PagePlacement | None, gloo_group: dist.ProcessGroup
    ) -> None:
        self.is_writer = is_writer
        self._layout = layout
        self._gloo_group = gloo_group
        self._mapped_region: torch.Tensor | None = None
        self._writer_error: Exception | None = None
        if is_writer and layout.region_bytes > 0:
            try:
                _create_shared_memory_file(path=layout.region_path, nbytes=layout.region_bytes, page_placement=page_placement)
                self._mapped_region = _map_and_pin_shared_memory_file(path=layout.region_path, nbytes=layout.region_bytes)
            except Exception as error:
                self._writer_error = error

    @property
    def writer_ready(self) -> bool:
        return self._writer_error is None

    @property
    def shared_bytes(self) -> int:
        return sum(placement.nbytes for placement in self._layout.placement_by_weight_name.values())

    def holds(self, weight_name: str) -> bool:
        return weight_name in self._layout.placement_by_weight_name

    def host_buffer(self, weight_name: str) -> torch.Tensor:
        """The pinned view of ``weight_name`` inside the region; readers have one only after ``publish``."""
        if self._mapped_region is None:
            raise RuntimeError(f"host offload: the shared host region for {weight_name!r} is not mapped on this rank yet")
        placement = self._layout.placement_by_weight_name[weight_name]
        region_slice = self._mapped_region[placement.byte_offset : placement.byte_offset + placement.nbytes]
        return region_slice.view(placement.dtype).view(placement.shape)

    def holds_identical_copy(self, weight_name: str, tensor: torch.Tensor) -> bool:
        return torch.equal(_as_flat_bytes(self.host_buffer(weight_name)), _as_flat_bytes(tensor))

    def publish(self, *, writer_error: Exception | None = None) -> None:
        """Every rank: wait until the writers have filled their regions, map this group's, then drop its file."""
        local_error = self._writer_error if self._writer_error is not None else writer_error
        error_message = None if local_error is None else f"{type(local_error).__name__}: {local_error}"
        errors: list[str | None] = [None] * dist.get_world_size(group=self._gloo_group)
        dist.all_gather_object(errors, error_message, group=self._gloo_group)
        failures = [f"rank {rank}: {error}" for rank, error in enumerate(errors) if error is not None]
        if failures:
            raise RuntimeError(f"host offload: shared host region writer failed: {'; '.join(failures)}") from local_error
        if not self.is_writer and self._layout.region_bytes > 0:
            self._mapped_region = _map_and_pin_shared_memory_file(
                path=self._layout.region_path, nbytes=self._layout.region_bytes
            )
        # The file must outlive every reader's mapping call; the mappings then keep the memory alive.
        dist.barrier(group=self._gloo_group)
        self.remove_file()

    def remove_file(self) -> None:
        """Delete the writer's file; idempotent, so a failed park can call it too."""
        if self.is_writer and os.path.exists(self._layout.region_path):
            os.remove(self._layout.region_path)


def open_rank_shared_host_region(
    *, tensor_by_weight_name: Mapping[str, torch.Tensor], region_label: str, sharing_scope: SharingScope
) -> RankSharedHostRegion | None:
    """Agree with every rank on which of ``tensor_by_weight_name`` its sharing group shares.

    Collective over WORLD: every rank must call it, with its own CPU tensors and the same
    ``sharing_scope``, for the same module in the same order. Returns None on every rank
    when no group shares any weight.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return None
    gloo_group = get_cpu_gloo_group()
    if gloo_group is None:
        magi_logger.warning("host offload: no gloo group, %s keeps a private host copy on every rank", region_label)
        return None

    hostname = socket.gethostname()
    sharing_group, page_placement = _local_sharing_group_and_page_placement(hostname=hostname, sharing_scope=sharing_scope)
    local_offer = _RankOffer(
        hostname=hostname,
        sharing_group=sharing_group,
        region_path=os.path.join(envs.MAGI_SHARED_BIN_PATH, f"magi_offload_{region_label}_{uuid.uuid4().hex}.bin"),
        free_shared_memory_bytes=shutil.disk_usage(envs.MAGI_SHARED_BIN_PATH).free,
        descriptor_by_weight_name={name: _describe_weight(tensor) for name, tensor in tensor_by_weight_name.items()},
    )
    offers_by_rank: list[_RankOffer | None] = [None] * dist.get_world_size()
    dist.all_gather_object(offers_by_rank, local_offer, group=gloo_group)

    layout_by_sharing_group = _plan_group_layouts(offers_by_rank)
    local_layout = layout_by_sharing_group[sharing_group]
    if local_layout.rejected_region_bytes > 0:
        magi_logger.warning(
            "host offload: %s has %.1f GiB free, too little for this host's shared %s regions (%.1f GiB for this "
            "rank's group); every rank keeps a private copy",
            envs.MAGI_SHARED_BIN_PATH,
            local_offer.free_shared_memory_bytes / 2**30,
            region_label,
            local_layout.rejected_region_bytes / 2**30,
        )
    if all(layout.region_bytes == 0 for layout in layout_by_sharing_group.values()):
        return None
    return RankSharedHostRegion(
        layout=local_layout,
        is_writer=dist.get_rank() == local_layout.writer_rank,
        page_placement=page_placement,
        gloo_group=gloo_group,
    )


def _local_sharing_group_and_page_placement(
    *, hostname: str, sharing_scope: SharingScope
) -> tuple[str, _PagePlacement | None]:
    """This rank's sharing group, and where its writer should put the region's pages (None: default policy)."""
    if sharing_scope == "per_host_interleaved":
        numa_nodes = tuple(online_numa_nodes())
        page_placement = _PagePlacement(policy="interleave", numa_nodes=numa_nodes) if len(numa_nodes) > 1 else None
        return hostname, page_placement
    gpu_numa_node = cuda_device_numa_node(torch.cuda.current_device())
    if gpu_numa_node is None:
        # Without NUMA affinity every GPU is equally far from every page, so the host is one group.
        return hostname, None
    return f"{hostname}/numa{gpu_numa_node}", _PagePlacement(policy="preferred", numa_nodes=(gpu_numa_node,))


def _describe_weight(tensor: torch.Tensor) -> _WeightDescriptor:
    return _WeightDescriptor(
        shape=tuple(tensor.shape),
        dtype=tensor.dtype,
        nbytes=tensor.numel() * tensor.element_size(),
        fingerprint=compute_tensor_fingerprint(tensor),
    )


def _plan_group_layouts(offers_by_rank: list[_RankOffer]) -> dict[str, _GroupLayout]:
    """Every group's layout, so each rank knows whether any group will synchronize on a region."""
    ranks_by_sharing_group: dict[str, list[int]] = {}
    for rank, offer in enumerate(offers_by_rank):
        ranks_by_sharing_group.setdefault(offer.sharing_group, []).append(rank)
    layout_by_sharing_group = {
        sharing_group: _plan_group_layout(group_offers_by_rank={rank: offers_by_rank[rank] for rank in group_ranks})
        for sharing_group, group_ranks in ranks_by_sharing_group.items()
    }
    return _drop_layouts_on_hosts_without_room(layout_by_sharing_group=layout_by_sharing_group, offers_by_rank=offers_by_rank)


def _plan_group_layout(group_offers_by_rank: dict[int, _RankOffer]) -> _GroupLayout:
    """Lay out, in the writer's order, the weights every rank of one group holds identically."""
    writer_rank, *reader_ranks = group_offers_by_rank
    writer_offer = group_offers_by_rank[writer_rank]
    reader_offers = [group_offers_by_rank[rank] for rank in reader_ranks]
    shared_descriptor_by_weight_name = {
        name: descriptor
        for name, descriptor in writer_offer.descriptor_by_weight_name.items()
        if reader_offers and all(offer.descriptor_by_weight_name.get(name) == descriptor for offer in reader_offers)
    }

    placement_by_weight_name = {}
    region_bytes = 0
    for name, descriptor in shared_descriptor_by_weight_name.items():
        placement_by_weight_name[name] = _SharedWeightPlacement(
            byte_offset=region_bytes, shape=descriptor.shape, dtype=descriptor.dtype, nbytes=descriptor.nbytes
        )
        # Aligned like the private pool's regions, which also keeps every dtype view legal.
        region_bytes += (descriptor.nbytes + ALIGN_BYTES - 1) // ALIGN_BYTES * ALIGN_BYTES
    return _GroupLayout(
        writer_rank=writer_rank,
        hostname=writer_offer.hostname,
        region_path=writer_offer.region_path,
        placement_by_weight_name=placement_by_weight_name,
        region_bytes=region_bytes,
    )


def _drop_layouts_on_hosts_without_room(
    *, layout_by_sharing_group: dict[str, _GroupLayout], offers_by_rank: list[_RankOffer]
) -> dict[str, _GroupLayout]:
    """Empty every layout of a machine whose shared memory cannot hold all of its groups' regions.

    The groups of one machine share its tmpfs, and writing past a full tmpfs is SIGBUS.
    """
    free_bytes_by_hostname: dict[str, int] = {}
    for offer in offers_by_rank:
        free_bytes_by_hostname[offer.hostname] = min(
            offer.free_shared_memory_bytes, free_bytes_by_hostname.get(offer.hostname, offer.free_shared_memory_bytes)
        )
    needed_bytes_by_hostname: dict[str, int] = {}
    for layout in layout_by_sharing_group.values():
        needed_bytes_by_hostname[layout.hostname] = needed_bytes_by_hostname.get(layout.hostname, 0) + layout.region_bytes

    def fits(hostname: str) -> bool:
        return needed_bytes_by_hostname[hostname] + SHARED_MEMORY_HEADROOM_BYTES <= free_bytes_by_hostname[hostname]

    return {
        sharing_group: (
            layout
            if fits(layout.hostname)
            else dataclasses.replace(
                layout, placement_by_weight_name={}, region_bytes=0, rejected_region_bytes=layout.region_bytes
            )
        )
        for sharing_group, layout in layout_by_sharing_group.items()
    }


def _create_shared_memory_file(*, path: str, nbytes: int, page_placement: _PagePlacement | None) -> None:
    file_descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    try:
        # tmpfs allocates the pages here, under this thread's NUMA policy. Reserving them now also
        # turns running out of tmpfs into an error instead of a SIGBUS while writing the mapping.
        if page_placement is None:
            os.posix_fallocate(file_descriptor, 0, nbytes)
        else:
            with numa_page_policy(policy=page_placement.policy, nodes=page_placement.numa_nodes):
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
