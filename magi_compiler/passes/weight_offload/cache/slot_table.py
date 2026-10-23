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

from typing import Any, Mapping

import torch.fx as fx
import torch.nn as nn


def slot_identity(slot_metadata: Mapping[str, Any]) -> tuple:
    """Name + layout: what a peer process can match without sharing slot ints."""
    return (slot_metadata["name"], tuple(slot_metadata["shape"]), str(slot_metadata["dtype"]))


def collect_host_slot_table(module: nn.Module) -> dict[int, dict[str, Any]]:
    """Every host-offload slot tagged on ``module``'s FX graphs, keyed by slot.

    Walks child GraphModules as well: after a split the tags live on the
    submods, not on the root.
    """
    from ..node_meta import host_slot
    from ..runtime.host_weights import get_global_host_weight_store

    store = get_global_host_weight_store()
    slot_table: dict[int, dict[str, Any]] = {}
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


def _slot_by_weight_identity(slot_table: Mapping[int, Mapping[str, Any]]) -> dict[tuple, int] | None:
    """``identity -> slot``, or None if two slots share an identity."""
    slot_by_weight_identity: dict[tuple, int] = {}
    for slot, slot_metadata in slot_table.items():
        weight_identity = slot_identity(slot_metadata)
        if weight_identity in slot_by_weight_identity:
            return None
        slot_by_weight_identity[weight_identity] = int(slot)
    return slot_by_weight_identity


def match_host_slot_tables(
    cached_slots: Mapping[int, Mapping[str, Any]], current_slots: Mapping[int, Mapping[str, Any]]
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


def refresh_resident_flags(slot_table: Mapping[int, Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    """Copy ``slot_table`` with ``resident`` taken from the live pool.

    Placement promotes slots during Inductor compile, after the table was first
    collected.  The sidecar must record that final set so a cache hit can
    replay it.
    """
    from ..runtime.host_weights import get_global_host_weight_store

    store = get_global_host_weight_store()
    return {
        int(slot): {**dict(slot_metadata), "resident": bool(store.is_resident(int(slot)))}
        for slot, slot_metadata in slot_table.items()
    }
