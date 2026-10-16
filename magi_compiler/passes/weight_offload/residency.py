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

"""What a device move means for a compiled module whose weights live in host memory.

Every offloaded shard keeps its bytes in the host pool; placement promotes some of
them back onto the device. Moving the module to the host frees those device copies.
Moving it back promotes exactly them again; the rest are loaded by the graph as usual.

Slots are tracked per compiled instance: one process may compile several models
against the same pool, and moving one must not touch another's residency.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from magi_compiler.utils import magi_logger

from .runtime import host_weights

_SLOTS_ATTR = "_magi_host_slots"
"""Slots ``handoff`` adopted for this instance; present, possibly empty, once it ran."""

_RELEASED_ATTR = "_magi_released_slots"
"""Promoted slots a move to the host freed, to be promoted again on the way back."""


def record_slots(module: nn.Module, slots: Sequence[int]) -> None:
    setattr(module, _SLOTS_ATTR, tuple(slots))


def has_slots(module: nn.Module) -> bool:
    return hasattr(module, _SLOTS_ATTR)


def move(module: nn.Module, *, to_host: bool) -> None:
    """Apply a device move of ``module`` to the residency of its host-offloaded weights."""
    if to_host:
        _release(module)
    else:
        _promote_released(module)


def _subtree_slots(module: nn.Module) -> list[int]:
    # A nested compiled instance may have parked its own weights.
    return [slot for m in module.modules() for slot in getattr(m, _SLOTS_ATTR, ())]


def _release(module: nn.Module) -> None:
    resident = [slot for slot in _subtree_slots(module) if host_weights.is_resident(slot)]
    if not resident:
        return
    torch.cuda.synchronize()
    nbytes = host_weights.release_resident_many(resident)
    setattr(module, _RELEASED_ATTR, getattr(module, _RELEASED_ATTR, ()) + tuple(resident))
    magi_logger.info(
        "host offload: %s moved to host; freed %d promoted weight(s) (%.1f MiB) on the device",
        type(module).__name__,
        len(resident),
        nbytes / 2**20,
    )


def _promote_released(module: nn.Module) -> None:
    released = getattr(module, _RELEASED_ATTR, ())
    if not released:
        return
    nbytes = host_weights.make_resident_many(released)
    setattr(module, _RELEASED_ATTR, ())
    magi_logger.info(
        "host offload: %s moved to device; promoted %d weight(s) (%.1f MiB) back, the rest load in the graph",
        type(module).__name__,
        len(released),
        nbytes / 2**20,
    )
