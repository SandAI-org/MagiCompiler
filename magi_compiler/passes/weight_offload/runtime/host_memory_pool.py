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

"""Allocate host tensor views from dtype-specific slabs.

A dtype may occupy multiple slabs. Existing views retain their storage after pool reset.

A pool is either pinned or paged.  Pinned is what makes a host-to-device copy
asynchronous at all -- a pageable source forces ``cudaMemcpyAsync`` to stage
through a driver bounce buffer synchronously -- but it is a hard, unswappable
reservation of host RAM.  Paged suits data that is only staged on the host.

The pool hands out views and never frees one individually; ``reset`` drops
everything at once.  What a region holds, and who reads it, is the caller's.
"""

from __future__ import annotations

import ctypes
import mmap
import weakref
from dataclasses import dataclass
from typing import Sequence

import torch

from magi_compiler.cuda.cudart import init_cudart, pin_memory_in_place

SLAB_BYTES = 1 << 30
"""Default cap on one slab (1 GiB).

A smaller cap bounds the over-allocation of the last, partly used slab -- which
for a pinned pool is host RAM nobody else can have.  A request larger than this
gets a slab of its own.
"""

ALIGN_BYTES = 512
"""Default start alignment of every region: what the DMA engine wants for peak
host-to-device throughput."""

EXACT_PINNED_MIN_BYTES = 64 << 20
"""Pinned requests from this size get their own exact-size region registered in place.

torch's pinned allocator rounds every block up to a power of two, so a 960 MiB weight
reserves 1 GiB and a pinned slab packs 480 MiB weights with 6% left over; measured
4 GiB per rank on the 400B W8A8 Gaga4 Base. Below this size slabs keep the count of
cudaHostRegister calls low.
"""


def _unregister_host_memory(address: int) -> None:
    cudart = init_cudart()
    if cudart is not None:
        cudart.cudaHostUnregister(ctypes.c_void_p(address))


def _exact_pinned_region(nbytes: int) -> torch.Tensor:
    """``nbytes`` of page-aligned anonymous memory, page-locked with cudaHostRegister.

    The mapping stays registered while any view of it lives and is unregistered before it is unmapped.
    """
    mapping = mmap.mmap(-1, nbytes)
    region = torch.frombuffer(mapping, dtype=torch.uint8)
    pin_memory_in_place(region)
    if not region.is_pinned():
        raise RuntimeError(f"host pool: could not page-lock a {nbytes / 2**20:.1f} MiB region with cudaHostRegister")
    weakref.finalize(mapping, _unregister_host_memory, region.data_ptr())
    return region


@dataclass
class _Slab:
    storage: torch.Tensor
    used_numel: int  # elements handed out, alignment padding included


class HostMemPool:
    """Bump allocator over host slabs, one dtype per slab, pinned or paged.

    Pinned allocation failures propagate; pageable storage requires ``pinned=False``.
    """

    def __init__(self, pinned: bool = True, slab_bytes: int = SLAB_BYTES, align_bytes: int = ALIGN_BYTES) -> None:
        self._pinned = pinned
        self._slab_bytes = slab_bytes
        self._align_bytes = align_bytes
        self._slabs: list[_Slab] = []

    @property
    def pinned(self) -> bool:
        return self._pinned

    def reset(self) -> None:
        """Forget every slab.  Regions already handed out keep theirs alive."""
        self._slabs.clear()

    def alloc(self, shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
        """An uninitialized host region of ``shape`` and ``dtype``.

        Returns a *view* into a slab.  Views keep the slab alive, so the slab
        list is only there to make the reservation explicit and resettable.
        """
        shape = tuple(int(d) for d in shape)
        numel = 1
        for d in shape:
            numel *= d
        return self._allocate_slab_region(numel=numel, dtype=dtype).view(shape)

    def _allocate_slab_region(self, numel: int, dtype: torch.dtype) -> torch.Tensor:
        align_numel = max(1, self._align_bytes // dtype.itemsize)
        padded_numel = (numel + align_numel - 1) // align_numel * align_numel

        if self._pinned and padded_numel * dtype.itemsize >= EXACT_PINNED_MIN_BYTES:
            storage = _exact_pinned_region(padded_numel * dtype.itemsize).view(dtype)
            self._slabs.append(_Slab(storage=storage, used_numel=padded_numel))
            return storage[:numel]

        for slab in self._slabs:
            if slab.storage.dtype is not dtype:
                continue
            if slab.used_numel + padded_numel <= slab.storage.numel():
                offset_numel = slab.used_numel
                slab.used_numel += padded_numel
                return slab.storage[offset_numel : offset_numel + numel]

        storage = torch.empty(
            max(padded_numel, self._slab_bytes // dtype.itemsize), dtype=dtype, device="cpu", pin_memory=self._pinned
        )
        self._slabs.append(_Slab(storage=storage, used_numel=padded_numel))
        return storage[:numel]
