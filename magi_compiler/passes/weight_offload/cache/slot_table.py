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

"""Host-slot tables persisted next to the piecewise compile cache.

The compiled artifact bakes process-local slot integers into ``magi::h2d_load``.
A sidecar records which weight each baked integer referred to, so a later
process can remap those integers onto the slots it minted.
"""

from __future__ import annotations

from typing import Mapping, TypedDict

import torch.fx as fx
import torch.nn as nn


class HostSlotMetadata(TypedDict):
    name: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    resident: bool


def parse_host_slot_table(serialized_slots: dict) -> dict[int, HostSlotMetadata]:
    """Normalize the literal sidecar at the file boundary; invalid entries raise a cache-miss error."""
    slot_table: dict[int, HostSlotMetadata] = {}
    for serialized_slot, metadata in serialized_slots.items():
        if not isinstance(serialized_slot, (int, str)) or not isinstance(metadata, dict):
            raise ValueError("each host slot must map an integer id to a metadata dict")
        slot = int(serialized_slot)
        if slot < 0 or slot in slot_table:
            raise ValueError(f"invalid or duplicate host slot {slot}")
        name, shape, dtype = metadata["name"], metadata["shape"], metadata["dtype"]
        nbytes, resident = metadata["nbytes"], metadata.get("resident", False)
        if not isinstance(name, str) or not isinstance(dtype, str):
            raise ValueError(f"host slot {slot} name and dtype must be strings")
        if not isinstance(shape, (tuple, list)) or any(type(dim) is not int or dim < 0 for dim in shape):
            raise ValueError(f"host slot {slot} shape must contain nonnegative integers")
        if type(nbytes) is not int or nbytes < 0 or not isinstance(resident, bool):
            raise ValueError(f"host slot {slot} needs a nonnegative byte count and a boolean resident flag")
        slot_table[slot] = HostSlotMetadata(name=name, shape=tuple(shape), dtype=dtype, nbytes=nbytes, resident=resident)
    return slot_table


def slot_identity(slot_metadata: HostSlotMetadata) -> tuple[str, tuple[int, ...], str]:
    """Name + layout: what a peer process can match without sharing slot ints."""
    return (slot_metadata["name"], slot_metadata["shape"], slot_metadata["dtype"])


def collect_host_slot_table(module: nn.Module) -> dict[int, HostSlotMetadata]:
    """Every host-offload slot tagged on ``module``'s FX graphs, keyed by slot.

    Walks child GraphModules as well: after a split the tags live on the
    submods, not on the root.
    """
    from ..node_meta import host_slot
    from ..runtime.host_weights import get_global_host_weight_store

    store = get_global_host_weight_store()
    slot_table: dict[int, HostSlotMetadata] = {}
    for _, child in module.named_modules():
        if not isinstance(child, fx.GraphModule):
            continue
        for node in child.graph.nodes:
            slot = host_slot(node)
            if slot is None or slot in slot_table:
                continue
            host_buffer = store.host_buffer(slot)
            slot_table[int(slot)] = {
                "name": store.parameter_name(slot),
                "shape": tuple(int(d) for d in host_buffer.shape),
                "dtype": str(host_buffer.dtype),
                "nbytes": int(store.slot_bytes(slot)),
                "resident": bool(store.is_resident(slot)),
            }
    return slot_table


def _slot_by_weight_identity(slot_table: Mapping[int, HostSlotMetadata]) -> dict[tuple, int] | None:
    """``identity -> slot``, or None if two slots share an identity."""
    slot_by_weight_identity: dict[tuple, int] = {}
    for slot, slot_metadata in slot_table.items():
        weight_identity = slot_identity(slot_metadata)
        if weight_identity in slot_by_weight_identity:
            return None
        slot_by_weight_identity[weight_identity] = slot
    return slot_by_weight_identity


def match_host_slot_tables(
    cached_slots: Mapping[int, HostSlotMetadata], current_slots: Mapping[int, HostSlotMetadata]
) -> dict[int, int] | None:
    """Baked slot -> current slot, or None if the identity sets differ.

    A mismatch is a cache miss: the artifact's loads are not the weights this
    process bound, so replaying it would copy the wrong bytes.
    """
    cached_slot_by_identity = _slot_by_weight_identity(cached_slots)
    current_slot_by_identity = _slot_by_weight_identity(current_slots)
    if cached_slot_by_identity is None or current_slot_by_identity is None:
        return None
    if set(cached_slot_by_identity) != set(current_slot_by_identity):
        return None
    return {
        cached_slot: current_slot_by_identity[weight_identity]
        for weight_identity, cached_slot in cached_slot_by_identity.items()
    }


def refresh_resident_flags(slot_table: Mapping[int, HostSlotMetadata]) -> dict[int, HostSlotMetadata]:
    """Copy ``slot_table`` with ``resident`` taken from the live pool.

    Placement promotes slots during Inductor compile, after the table was first
    collected.  The sidecar must record that final set so a cache hit can
    replay it.
    """
    from ..runtime.host_weights import get_global_host_weight_store

    store = get_global_host_weight_store()
    return {slot: {**slot_metadata, "resident": store.is_resident(slot)} for slot, slot_metadata in slot_table.items()}
