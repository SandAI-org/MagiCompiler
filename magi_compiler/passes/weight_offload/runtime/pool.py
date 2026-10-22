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

"""Host memory, bump-allocated out of a few large slabs.

One slab per dtype holds many tensors: a caller such as weight offload asks
for thousands of small regions, and one allocation per region would pay the
allocator -- for pinned memory, a ``cudaHostAlloc`` driver round trip that also
pauses every other stream -- thousands of times.

A pool is either pinned or paged.  Pinned is what makes a host-to-device copy
asynchronous at all -- a pageable source forces ``cudaMemcpyAsync`` to stage
through a driver bounce buffer synchronously -- but it is a hard, unswappable
reservation of host RAM.  Paged suits data that is only staged on the host.

The pool hands out views and never frees one individually; ``reset`` drops
everything at once.  What a region holds, and who reads it, is the caller's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

SLAB_BYTES = 1 << 30
"""Default cap on one slab (1 GiB).

A smaller cap bounds the over-allocation of the last, partly used slab -- which
for a pinned pool is host RAM nobody else can have.  A request larger than this
gets a slab of its own.
"""

ALIGN_BYTES = 512
"""Default start alignment of every region: what the DMA engine wants for peak
host-to-device throughput."""


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
        return self._carve(numel=numel, dtype=dtype).view(shape)

    def _carve(self, numel: int, dtype: torch.dtype) -> torch.Tensor:
        align_numel = max(1, self._align_bytes // dtype.itemsize)
        padded_numel = (numel + align_numel - 1) // align_numel * align_numel

        for slab in self._slabs:
            if slab.storage.dtype is not dtype:
                continue
            if slab.used_numel + padded_numel <= slab.storage.numel():
                offset = slab.used_numel
                slab.used_numel += padded_numel
                return slab.storage[offset : offset + numel]

        storage = torch.empty(
            max(padded_numel, self._slab_bytes // dtype.itemsize), dtype=dtype, device="cpu", pin_memory=self._pinned
        )
        self._slabs.append(_Slab(storage=storage, used_numel=padded_numel))
        return storage[:numel]
