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

"""Copy deferred weights to their final place once compilation has chosen the resident ones.

With ``offload_config.defer_host_materialization`` a host-built model's first CUDA move only
defers its offloadable weights: each slot reads a zero placeholder and keeps its CPU original.
Compilation reads nothing else, and its placement makes some slots resident. Before the compiled
graph first runs, every deferred weight of the models it reads is copied once: a resident one
into its device storage with no host copy, any other into pinned host memory, shared with the
other ranks of its sharing group when they hold it identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from magi_compiler.utils import magi_logger
from magi_compiler.utils.dist_utils import get_cpu_gloo_group
from magi_compiler.utils.torch_threads import intra_op_threads

from ..transfer_overlap import rank_sync
from .runtime.host_weights import HostWeightStore, get_global_host_weight_store
from .runtime.rank_shared_host_region import HostWeightSharing, SharedHostChunk, plan_shared_host_parking


@dataclass(frozen=True)
class _HostCopyBytes:
    private_bytes: int
    shared_bytes: int


def materialize_deferred_host_weights(
    example_inputs: Sequence[Any], *, host_weight_sharing: HostWeightSharing, region_label: str
) -> None:
    """Complete every deferred slot of the models whose weights this compiled graph reads.

    Unless ``host_weight_sharing`` is "off", this is a collective over WORLD on every compile.
    """
    store = get_global_host_weight_store()
    graph_input_slots = _host_slots_of_graph_inputs(example_inputs=example_inputs, store=store)
    pending_slots = store.pending_slots_in_groups_of(graph_input_slots)
    if not _any_rank_has_pending_slots(has_pending_slots=bool(pending_slots), host_weight_sharing=host_weight_sharing):
        return
    resident_slots = [slot for slot in pending_slots if store.is_resident(slot)]
    offloaded_slots = [slot for slot in pending_slots if not store.is_resident(slot)]
    if host_weight_sharing == "off":
        host_copy_bytes = _complete_with_private_host_copies(store=store, offloaded_slots=offloaded_slots)
    else:
        host_copy_bytes = _complete_with_shared_host_copies(
            store=store,
            weight_key_by_slot=_shared_weight_key_by_slot(
                store=store, graph_input_slots=graph_input_slots, pending_slots=pending_slots
            ),
            offloaded_slots=offloaded_slots,
            host_weight_sharing=host_weight_sharing,
            region_label=region_label,
        )
    # Last: a shared region's writer may still have needed these CPU originals.
    for slot in resident_slots:
        store.complete_pending_slot_on_device(slot)
    magi_logger.info(
        "host offload: %s materialized %d deferred weight(s): %.1f MiB resident on the device only, "
        "%.1f MiB in private pinned host memory, %.1f MiB in a rank-shared region",
        region_label,
        len(pending_slots),
        sum(store.slot_bytes(slot) for slot in resident_slots) / 2**20,
        host_copy_bytes.private_bytes / 2**20,
        host_copy_bytes.shared_bytes / 2**20,
    )


def _host_slots_of_graph_inputs(*, example_inputs: Sequence[Any], store: HostWeightStore) -> list[int]:
    host_slots = []
    for graph_input in example_inputs or ():
        local_tensor = graph_input._local_tensor if isinstance(graph_input, DTensor) else graph_input
        if not isinstance(local_tensor, torch.Tensor):
            continue
        host_slot = store.slot_for_tensor(local_tensor)
        if host_slot is not None:
            host_slots.append(host_slot)
    return host_slots


def _any_rank_has_pending_slots(*, has_pending_slots: bool, host_weight_sharing: HostWeightSharing) -> bool:
    if host_weight_sharing == "off":
        return has_pending_slots
    # Every rank must take the same branch into the shared region's collectives.
    return rank_sync.any_rank_has_flag(has_pending_slots)


def _copy_into_private_host_buffer(*, store: HostWeightStore, slot: int) -> torch.Tensor:
    cpu_original = store.pending_source(slot)
    host_buffer = store.reserve_host_buffer(
        shape=cpu_original.shape, dtype=cpu_original.dtype, parameter_name=store.parameter_name(slot)
    )
    host_buffer.copy_(cpu_original)
    return host_buffer


def _complete_with_private_host_copies(*, store: HostWeightStore, offloaded_slots: Sequence[int]) -> _HostCopyBytes:
    # One slot at a time, so a CPU original in anonymous memory is freed before the next copy.
    for slot in offloaded_slots:
        store.complete_pending_slot_in_host_buffer(slot, _copy_into_private_host_buffer(store=store, slot=slot))
    return _HostCopyBytes(private_bytes=sum(store.slot_bytes(slot) for slot in offloaded_slots), shared_bytes=0)


def _shared_weight_key_by_slot(
    *, store: HostWeightStore, graph_input_slots: Sequence[int], pending_slots: Sequence[int]
) -> dict[int, str]:
    """Name each pending slot the same way on every rank, so ranks match their copies of one weight by it.

    A parameter name is unique only within its deferred group, the module whose first CUDA move deferred
    it, and the group id is process-local; so the name is prefixed with the order in which this graph's
    inputs first read the group. A slot whose key is still not unique is left out and keeps a private copy:
    sharing it could complete it with the bytes of the other slot under that key.
    """
    group_order_by_group_id: dict[int | None, int] = {}
    for slot in graph_input_slots:
        if store.is_pending(slot):
            group_order_by_group_id.setdefault(store.deferred_group_id(slot), len(group_order_by_group_id))
    slots_by_key: dict[str, list[int]] = {}
    for slot in pending_slots:
        group_order = group_order_by_group_id[store.deferred_group_id(slot)]
        slots_by_key.setdefault(f"{group_order}/{store.parameter_name(slot)}", []).append(slot)
    return {slots[0]: key for key, slots in slots_by_key.items() if len(slots) == 1}


def _keys_resident_on_every_rank(local_resident_keys: set[str]) -> set[str]:
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return local_resident_keys
    resident_keys_by_rank: list[set[str] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(resident_keys_by_rank, local_resident_keys, group=get_cpu_gloo_group())
    return set.intersection(*resident_keys_by_rank)


def _complete_with_shared_host_copies(
    *,
    store: HostWeightStore,
    weight_key_by_slot: dict[int, str],
    offloaded_slots: Sequence[int],
    host_weight_sharing: HostWeightSharing,
    region_label: str,
) -> _HostCopyBytes:
    """Offer every weight some rank keeps on the host, so a group shares it even where it is resident on a few ranks.
    Private copies follow the last round: no collective comes after them, so a local failure cannot strand peers."""
    local_resident_keys = {key for slot, key in weight_key_by_slot.items() if store.is_resident(slot)}
    resident_everywhere_keys = _keys_resident_on_every_rank(local_resident_keys)
    cpu_original_by_key = {
        key: store.pending_source(slot) for slot, key in weight_key_by_slot.items() if key not in resident_everywhere_keys
    }
    parking_plan = plan_shared_host_parking(
        tensor_by_weight_name=cpu_original_by_key, region_label=region_label, sharing_scope=host_weight_sharing
    )
    if parking_plan is None:
        return _complete_with_private_host_copies(store=store, offloaded_slots=offloaded_slots)
    offloaded_slot_by_key = {weight_key_by_slot[slot]: slot for slot in offloaded_slots if slot in weight_key_by_slot}
    shared_bytes = 0
    with intra_op_threads(parking_plan.parking_threads):
        for round_index in range(parking_plan.round_count):
            with parking_plan.open_chunk(round_index) as chunk:
                shared_bytes += _complete_shared_chunk(
                    store=store,
                    chunk=chunk,
                    cpu_original_by_key=cpu_original_by_key,
                    offloaded_slot_by_key=offloaded_slot_by_key,
                )
    still_pending_slots = [slot for slot in offloaded_slots if store.is_pending(slot)]
    private_bytes = _complete_with_private_host_copies(store=store, offloaded_slots=still_pending_slots).private_bytes
    return _HostCopyBytes(private_bytes=private_bytes, shared_bytes=shared_bytes)


def _complete_shared_chunk(
    *,
    store: HostWeightStore,
    chunk: SharedHostChunk,
    cpu_original_by_key: dict[str, torch.Tensor],
    offloaded_slot_by_key: dict[str, int],
) -> int:
    """Writers fill the whole chunk, their own resident weights included, before ``publish``;
    readers adopt the writer's pages before ``finish``. Returns the bytes this rank now reads from the chunk."""
    adopted_bytes = 0
    writer_error = None
    if chunk.is_writer and chunk.is_shared:
        try:
            for key in chunk.weight_names:
                chunk.host_buffer(key).copy_(cpu_original_by_key[key])
            adopted_bytes = _adopt_shared_copies(store=store, chunk=chunk, offloaded_slot_by_key=offloaded_slot_by_key)
        except Exception as error:
            writer_error = error
    chunk.publish(writer_error=writer_error)
    reader_error = None
    if not chunk.is_writer and chunk.is_shared:
        try:
            chunk.map_writer_region()
            adopted_bytes = _adopt_shared_copies(store=store, chunk=chunk, offloaded_slot_by_key=offloaded_slot_by_key)
        except Exception as error:
            reader_error = error
    chunk.finish(reader_error=reader_error)
    return adopted_bytes


def _adopt_shared_copies(*, store: HostWeightStore, chunk: SharedHostChunk, offloaded_slot_by_key: dict[str, int]) -> int:
    """Complete this rank's offloaded slots in the chunk; one whose bytes differ stays pending for a private copy."""
    adopted_bytes = 0
    for key in chunk.weight_names:
        slot = offloaded_slot_by_key.get(key)
        # None: resident here, though another rank of the group keeps it on the host.
        if slot is None:
            continue
        if not chunk.is_writer and not chunk.holds_identical_copy(key, store.pending_source(slot)):
            magi_logger.warning(
                "host offload: %s matched the shared copy's fingerprint but not its bytes; this rank keeps a private copy", key
            )
            continue
        store.complete_pending_slot_in_host_buffer(slot, chunk.host_buffer(key))
        adopted_bytes += store.slot_bytes(slot)
    return adopted_bytes
