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

import torch.nn as nn


def slot_identity(info: Mapping[str, Any]) -> tuple:
    """Name + layout: what a peer process can match without sharing slot ints."""
    return (info["name"], tuple(info["shape"]), str(info["dtype"]))


def collect_host_slot_table(module: nn.Module) -> dict[int, dict[str, Any]]:
    """Every host-offload slot tagged on ``module``'s FX graphs, keyed by slot.

    Walks child GraphModules as well: after a split the tags live on the
    submods, not on the root.
    """
    from . import host_pool
    from .node_meta import host_slot

    table: dict[int, dict[str, Any]] = {}
    for _, child in module.named_modules():
        graph = getattr(child, "graph", None)
        if graph is None:
            continue
        for node in graph.nodes:
            slot = host_slot(node)
            if slot is None or slot in table:
                continue
            host = host_pool.get(slot)
            table[int(slot)] = {
                "name": host_pool.name_of(slot),
                "shape": tuple(int(d) for d in host.shape),
                "dtype": str(host.dtype),
                "nbytes": int(host_pool.slot_bytes(slot)),
                "resident": bool(host_pool.is_resident(slot)),
            }
    return table


def _index_by_identity(table: Mapping[int, Mapping[str, Any]]) -> dict[tuple, int] | None:
    """``identity -> slot``, or None if two slots share an identity."""
    by_id: dict[tuple, int] = {}
    for slot, info in table.items():
        ident = slot_identity(info)
        if ident in by_id:
            return None
        by_id[ident] = int(slot)
    return by_id


def match_host_slot_tables(
    sidecar: Mapping[int, Mapping[str, Any]], current: Mapping[int, Mapping[str, Any]]
) -> dict[int, int] | None:
    """Baked slot -> current slot, or None if the identity sets differ.

    A mismatch is a cache miss: the artifact's loads are not the weights this
    process bound, so replaying it would copy the wrong bytes.
    """
    baked_by_id = _index_by_identity(sidecar)
    current_by_id = _index_by_identity(current)
    if baked_by_id is None or current_by_id is None:
        return None
    if set(baked_by_id) != set(current_by_id):
        return None
    return {baked: current_by_id[ident] for ident, baked in baked_by_id.items()}


def refresh_resident_flags(table: Mapping[int, Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    """Copy ``table`` with ``resident`` taken from the live pool.

    Placement promotes slots during Inductor compile, after the table was first
    collected.  The sidecar must record that final set so a cache hit can
    replay it.
    """
    from . import host_pool

    return {int(slot): {**dict(info), "resident": bool(host_pool.is_resident(int(slot)))} for slot, info in table.items()}
