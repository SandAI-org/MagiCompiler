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

"""Baked artifact slot -> this process's host-pool slot.

A cached kernel bakes the slot integers the compiling process minted.  A later
process replays it under ``using_slot_remap`` (see the ``host_slots.py``
sidecar next to the piecewise cache) rather than compiling the loads again, and
``magi::h2d_load`` translates every slot through ``resolve_slot``.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

# Installed only around a loaded compiled graph, so two models in one process
# cannot clobber each other.
_SLOT_REMAP: ContextVar[dict[int, int] | None] = ContextVar("magi_host_slot_remap", default=None)


def resolve_slot(slot: int) -> int:
    """Translate a baked artifact slot onto this process's pool.

    Identity when no remap is installed -- compile-time loads and a cache miss
    both mint and bake the same integers.
    """
    remap = _SLOT_REMAP.get()
    if remap is None:
        return slot
    try:
        return remap[slot]
    except KeyError:
        raise RuntimeError(
            f"magi::h2d_load got baked slot {slot} which is not in this artifact's "
            f"remap {sorted(remap)}. The compiled graph's slot ids do not match the sidecar."
        ) from None


@contextmanager
def using_slot_remap(remap: dict[int, int] | None) -> Iterator[None]:
    """Install ``baked -> current`` for the duration of a loaded compiled graph."""
    token = _SLOT_REMAP.set(remap)
    try:
        yield
    finally:
        _SLOT_REMAP.reset(token)
