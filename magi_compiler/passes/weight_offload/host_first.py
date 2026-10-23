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

"""Park eligible weights in pinned memory on meta to_empty or the first CPU-to-CUDA move.

After initialization, handoff installs storage-free CUDA stand-ins before tracing;
it temporarily allocates one local shard. Tied registrations share a reservation
and slot. Externally materialized weights stay resident; reclaimed weights are reported.
Device moves update residency without replacing traced stand-ins. Other tensors
use the class's _apply; DTensor movement stays with the sharding owner.

Dtype contract: finish every dtype conversion before parking, i.e. before the
first CUDA move of a host-built model or before ``to_empty`` of a meta-built one.
Weights are parked in the dtype they have at that moment and stay in it.
Supported later calls are pure device moves (``.cuda()``, ``.cpu()``, ``.to(device)``).
A cast that visibly changes a handed-over stand-in's dtype is refused; anything
else, e.g. ``.to(device, dtype)``, is unsupported and not reliably detected.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, Replicate, Shard

from magi_compiler.utils import magi_logger

from . import residency
from .runtime.host_weights import get_global_host_weight_store

_TO_EMPTY_QUALNAME = "Module.to_empty.<locals>.<lambda>"
"""Only ``to_empty`` materializes; dtype casts and other ``_apply`` uses keep torch behavior."""

_HANDOFF_PENDING_ATTR = "_magi_host_first_pending"
"""Set on an instance whose weights are waiting in host memory for their handoff."""

_PARKED_NAMES_ATTR = "_magi_host_first_parked"
"""Names ``install_host_first_hooks`` put in host memory, so the handoff can tell a
weight it was never given from one that was taken back out from under it."""

_PARKING_DECIDED_ATTR = "_magi_host_first_settled"
"""Set once ``to_empty`` or the first move to CUDA decided what to park, even nothing."""


def _local_weight_tensor(param: torch.Tensor) -> torch.Tensor:
    """The tensor whose bytes we park: a DTensor's local shard, or the Parameter itself."""
    return param._local_tensor if isinstance(param, DTensor) else param


def _is_offloadable(param: object) -> bool:
    """Must agree with the weight sources: plain Parameters or single Shard(0)/Replicate DTensors.
    ``restore_unclaimed_weights`` restores parked weights that no graph loads."""
    if isinstance(param, DTensor):
        placements = param.placements
        if len(placements) != 1:
            return False
        placement = placements[0]
        return isinstance(placement, Replicate) or (isinstance(placement, Shard) and placement.dim == 0)
    return isinstance(param, nn.Parameter)


def _normalize_parameter_name(name: str) -> str:
    """``layers.0.attn.wq.parametrizations.weight.original`` -> ``layers.0.attn.wq.weight``."""
    parts = [p for p in name.split(".") if p not in ("parametrizations", "original")]
    return ".".join(parts) or name


def _rebuild_dtensor(param: DTensor, local: torch.Tensor) -> DTensor:
    """Preserve DTensor metadata without moving ``local`` to the mesh device.
    ``DTensor.from_local`` would copy a host reservation onto CUDA."""
    from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta

    meta = param._spec.tensor_meta
    spec = DTensorSpec(param._spec.mesh, param._spec.placements, tensor_meta=TensorMeta(meta.shape, meta.stride, local.dtype))
    return DTensor(local, spec, requires_grad=False)


def _swap_parameter_payload(param: nn.Parameter, new_data) -> None:
    """Swap ``new_data`` into ``param`` in place before tracing.
    Identity must survive: tied registrations, optimizers and hooks hold this object."""
    torch.utils.swap_tensors(param, nn.Parameter(new_data, requires_grad=param.requires_grad))


def _collect_parked_parameters(module: nn.Module) -> list[tuple[list[str], nn.Parameter]]:
    """Group CPU payloads by Parameter identity, retaining every tied registration.
    After CUDA ``to_empty``, these are the weights redirected into host slabs."""
    parameters_by_identity: dict[int, tuple[list[str], nn.Parameter]] = {}
    for name, param in module.named_parameters(recurse=True, remove_duplicate=False):
        if isinstance(param, DTensor):
            is_parked = param._local_tensor.device.type == "cpu" and param.device_mesh.device_type != "cpu"
        else:
            is_parked = param.device.type == "cpu"
        if is_parked:
            parameters_by_identity.setdefault(id(param), ([], param))[0].append(name)
    return list(parameters_by_identity.values())


def _warn_missing_parked_weights(module: nn.Module, parked_parameters: list[tuple[list[str], nn.Parameter]]) -> None:
    """Report weights parked here but re-materialized by an external ``_apply``.
    They remain readable on the device but no longer save offload memory."""
    expected_names = getattr(module, _PARKED_NAMES_ATTR, None)
    if not expected_names:
        return
    # Consume the record once; a repeated handoff would otherwise report every name missing.
    setattr(module, _PARKED_NAMES_ATTR, [])
    parked_names = {_normalize_parameter_name(name) for names, _ in parked_parameters for name in names}
    missing_names = [name for name in expected_names if name not in parked_names]
    if missing_names:
        magi_logger.warning(
            "host offload: %d weight(s) reclaimed before handoff (%s); offload skipped, check ties outside %s",
            len(missing_names),
            ", ".join(missing_names[:8]) + (" ..." if len(missing_names) > 8 else ""),
            type(module).__name__,
        )


def _probe_move_device(fn) -> torch.device | None:
    """Where ``fn`` moves a tensor between host and CUDA, or None if it does not or also casts it.
    Judged by effect, not name, so a caller's own move function is recognized too.

    The probe has the default dtype, so ``.to(device, dtype)`` with that same dtype
    reads as a pure move and parked weights skip its cast; see the module's dtype contract."""
    try:
        probe = torch.empty(0)
        moved = fn(probe)
        if not isinstance(moved, torch.Tensor) or moved.dtype != probe.dtype:
            return None
        if moved.is_cuda:
            return moved.device
        moved_back = fn(torch.empty(0, device="cuda"))
    except Exception:  # noqa: BLE001 - a function that cannot take a probe is not a device move
        return None
    return moved_back.device if moved_back.dtype == probe.dtype and moved_back.device.type == "cpu" else None


def _reject_cast_after_handoff(module: nn.Module, fn) -> None:
    """Raise if ``fn`` would change a stand-in's dtype: torch would read its empty
    CUDA storage, and the host slot and any compiled load keep the old dtype.

    A guard for the common ``model.cuda().half()``, not a complete classifier;
    the supported path is to cast before parking (see the module's dtype contract)."""
    probed_dtype_devices = set()
    for name, param in module.named_parameters():
        local_weight = _local_weight_tensor(param)
        dtype_device = (local_weight.dtype, local_weight.device)
        if dtype_device in probed_dtype_devices or get_global_host_weight_store().slot_for_tensor(local_weight) is None:
            continue
        probed_dtype_devices.add(dtype_device)
        try:
            converted_probe = fn(torch.empty(0, dtype=local_weight.dtype, device=local_weight.device))
        except Exception:  # noqa: BLE001 - a function that cannot take a probe is not a cast
            continue
        if isinstance(converted_probe, torch.Tensor) and converted_probe.dtype != local_weight.dtype:
            raise RuntimeError(
                f"host offload: cannot cast {type(module).__name__}.{_normalize_parameter_name(name)} from {local_weight.dtype} to "
                f"{converted_probe.dtype}: its weights were already handed to host memory. Finish every dtype "
                "conversion before the first CUDA move (host-built model) or before to_empty (meta-built "
                "model), e.g. model.half().cuda() instead of model.cuda().half(); or disable "
                "offload_config.graph_weight_offload."
            )


def _manages_device_moves(instance: nn.Module) -> bool:
    return getattr(instance, _HANDOFF_PENDING_ATTR, False) or residency.is_handoff_complete(instance)


def _has_dtensor_parameters(module: nn.Module) -> bool:
    return any(isinstance(p, DTensor) for p in module.parameters())


def _can_park_on_first_cuda_move(module: nn.Module) -> bool:
    """Unsharded and fully materialized, with at least one CPU parameter."""
    params = list(module.parameters())
    if any(isinstance(p, DTensor) or p.is_meta for p in params):
        return False
    return any(p.device.type == "cpu" for p in params)


def _skip_managed_weights(fn):
    """``fn`` for everything but the stand-ins, whose placement is the pool's."""

    def apply_to_unmanaged_tensor(tensor):
        return tensor if get_global_host_weight_store().slot_for_tensor(tensor) is not None else fn(tensor)

    return apply_to_unmanaged_tensor


# A move may arrive under inference mode, and placement later writes these stand-ins outside it.
@torch.inference_mode(False)
def _install_weight_standins(module: nn.Module, parked_parameters: list[tuple[list[str], nn.Parameter, torch.Tensor]]) -> int:
    """Swap each weight for a storage-free CUDA stand-in and adopt its filled host buffer.
    Returns the bytes handed over. Stand-in construction temporarily allocates one full shard."""
    device = torch.device("cuda", torch.cuda.current_device())
    handed_off_bytes = 0
    host_slots = []
    for names, param, host_buffer in parked_parameters:
        device_standin_buffer = torch.empty(host_buffer.shape, dtype=host_buffer.dtype, device=device)
        device_standin_buffer.untyped_storage().resize_(0)
        new_data = (
            _rebuild_dtensor(param=param, local=device_standin_buffer) if isinstance(param, DTensor) else device_standin_buffer
        )
        _swap_parameter_payload(param=param, new_data=new_data)
        # After, not before: the slot is keyed on the tensor the graph will be traced against.
        device_standin = _local_weight_tensor(param)
        host_slots.append(
            get_global_host_weight_store().bind_host_buffer(
                host_buffer=host_buffer, device_standin=device_standin, parameter_name=_normalize_parameter_name(names[0])
            )
        )
        handed_off_bytes += host_buffer.numel() * host_buffer.element_size()
    residency.record_handoff_slots(module=module, slots=host_slots)
    return handed_off_bytes


@torch.inference_mode(False)
def _park_cpu_parameters(module: nn.Module, min_weight_bytes: int) -> int:
    """Copy a host-built module's offloadable weights into the pool and hand them over.
    Returns the number parked; the rest keep their normal placement."""
    parameters_by_identity: dict[int, tuple[list[str], nn.Parameter]] = {}
    for name, param in module.named_parameters(remove_duplicate=False):
        if param.device.type != "cpu" or param.numel() * param.element_size() < min_weight_bytes:
            continue
        if param.is_contiguous():
            parameters_by_identity.setdefault(id(param), ([], param))[0].append(name)
    parked_parameters = []
    for names, param in parameters_by_identity.values():
        host_buffer = get_global_host_weight_store().reserve_host_buffer(
            shape=param.shape, dtype=param.dtype, parameter_name=_normalize_parameter_name(names[0])
        )
        host_buffer.copy_(param.detach())
        parked_parameters.append((names, param, host_buffer))
    if not parked_parameters:
        return 0
    handed_off_bytes = _install_weight_standins(module=module, parked_parameters=parked_parameters)
    magi_logger.info(
        "host offload: %s parked %d CPU weight(s), %.1f MiB, on first CUDA move",
        type(module).__name__,
        len(parked_parameters),
        handed_off_bytes / 2**20,
    )
    return len(parked_parameters)


def install_host_first_hooks(instance: nn.Module, conf) -> None:
    """Install per-instance host-first materialization and residency-aware device moves.

    Redirect eligible CUDA ``to_empty`` allocations to pinned host memory.
    A host-built plain module parks eligible weights on its first CUDA move.
    Later device moves release or restore promoted slots. Callers must finish dtype
    conversion before parking; see the module's dtype contract."""
    if getattr(instance, "_magi_host_first_installed", False):
        return
    min_weight_bytes = int(conf.offload_config.offload_min_shard_mib * 1024 * 1024)

    def call_class_apply(self, fn, recurse):
        # The class's, looked up per call: the instance's is this wrapper, and a
        # sharding owner may swap the class after compile-time patching.
        return type(self)._apply(self, fn, recurse=recurse)

    def _apply_with_host_offload(self, fn, recurse=True):
        # Meta-built (e.g. SimpleFSDP): to_empty(cuda) puts eligible weights in the pool, handoff pending.
        if getattr(fn, "__qualname__", "") == _TO_EMPTY_QUALNAME:
            return _materialize_host_weights(self=self, fn=fn, recurse=recurse)
        # None unless fn is a pure host<->CUDA move: dtype casts and other conversions get None.
        # Combined device+dtype moves are unsupported once weights are parked (module docstring).
        target = _probe_move_device(fn)
        # Decided once per instance. Weights on the host at the first CUDA move are parked: built on
        # the host, or built on CUDA and moved to the host first. Weights already on CUDA never are.
        if target is not None and target.type == "cuda" and not getattr(self, _PARKING_DECIDED_ATTR, False):
            setattr(self, _PARKING_DECIDED_ATTR, True)
            if _can_park_on_first_cuda_move(self):
                _park_cpu_parameters(module=self, min_weight_bytes=min_weight_bytes)
        if target is None:
            _reject_cast_after_handoff(module=self, fn=fn)
        # Plain torch: not a device move, or no handoff is pending and none ever ran; e.g. a move to
        # the host before the first CUDA move, or a module already on CUDA at that move.
        if target is None or not _manages_device_moves(self):
            return call_class_apply(self=self, fn=fn, recurse=recurse)
        # A device move of a module that owns host slots: hand off first if to_empty left it pending.
        handoff_host_weights_if_pending(self)
        residency.update_residency_for_device_move(module=self, to_host=target.type == "cpu")
        # Sharded: the sharding owner moves its DTensors in place, so nothing else moves here.
        # Unsharded: the remaining tensors move normally, the stand-ins stay where the pool put them.
        return (
            self
            if _has_dtensor_parameters(self)
            else call_class_apply(self=self, fn=_skip_managed_weights(fn), recurse=recurse)
        )

    def _materialize_host_weights(self, fn, recurse):
        # fn is to_empty's own empty_like, recognized by qualname above.
        target = fn(torch.empty(0, device="meta")).device
        if target.type != "cuda":
            return call_class_apply(self=self, fn=fn, recurse=recurse)
        setattr(self, _PARKING_DECIDED_ATTR, True)

        # Resolve candidates before _apply starts swapping parameter objects.
        selected_names_by_param_id = {}
        materialized_weight_names = []
        noncontiguous_weight_names = []
        for name, param in self.named_parameters(recurse=recurse):
            if not _is_offloadable(param):
                continue
            local_weight = _local_weight_tensor(param)
            # Before the two refusals below, so neither reports a weight that was
            # never a candidate in the first place.
            if local_weight.numel() * local_weight.element_size() < min_weight_bytes:
                continue
            if not local_weight.is_contiguous():
                # Flat host reservations and H2D outputs cannot preserve this stride.
                noncontiguous_weight_names.append(_normalize_parameter_name(name))
                continue
            if not local_weight.is_meta:
                # An eager module may share this already materialized weight;
                # emptying it at handoff would leave that reader without an H2D load.
                materialized_weight_names.append(_normalize_parameter_name(name))
                continue
            selected_names_by_param_id[id(param)] = _normalize_parameter_name(name)

        if noncontiguous_weight_names:
            magi_logger.warning(
                "host offload: %d non-contiguous weight(s) of %s stay on device to preserve layout (%s)",
                len(noncontiguous_weight_names),
                type(self).__name__,
                ", ".join(noncontiguous_weight_names[:8]) + (" ..." if len(noncontiguous_weight_names) > 8 else ""),
            )

        if materialized_weight_names:
            magi_logger.warning(
                "host offload: %d already-materialized weight(s) of %s stay on device for possible external readers (%s)",
                len(materialized_weight_names),
                type(self).__name__,
                ", ".join(materialized_weight_names[:8]) + (" ..." if len(materialized_weight_names) > 8 else ""),
            )

        if not selected_names_by_param_id:
            return call_class_apply(self=self, fn=fn, recurse=recurse)

        parked_sizes_bytes = []
        # _apply visits every tied registration; reuse one host buffer for all of them.
        host_buffers_by_param_id: dict[int, torch.Tensor] = {}

        def allocate_parameter_storage(tensor):
            name = selected_names_by_param_id.get(id(tensor))
            if name is None:
                return fn(tensor)
            host_buffer = host_buffers_by_param_id.get(id(tensor))
            if host_buffer is None:
                local_weight = _local_weight_tensor(tensor)
                host_buffer = get_global_host_weight_store().reserve_host_buffer(
                    shape=local_weight.shape, dtype=local_weight.dtype, parameter_name=name
                )
                host_buffers_by_param_id[id(tensor)] = host_buffer
                parked_sizes_bytes.append(host_buffer.numel() * host_buffer.element_size())
            return _rebuild_dtensor(param=tensor, local=host_buffer) if isinstance(tensor, DTensor) else host_buffer

        result = call_class_apply(self=self, fn=allocate_parameter_storage, recurse=recurse)
        setattr(self, _HANDOFF_PENDING_ATTR, True)
        setattr(self, _PARKED_NAMES_ATTR, list(selected_names_by_param_id.values()))
        magi_logger.info(
            "host offload: %s materialized %d weight(s), %.1f MiB, in pinned host memory for %s",
            type(self).__name__,
            len(parked_sizes_bytes),
            sum(parked_sizes_bytes) / 2**20,
            target,
        )
        return result

    instance._apply = _apply_with_host_offload.__get__(instance, type(instance))
    instance._magi_host_first_installed = True
    magi_logger.info("host offload: %s will materialize its offloadable weights in host memory", type(instance).__name__)


def handoff_host_weights_if_pending(instance: object) -> int:
    """Complete the handoff for ``instance`` if its weights are still on the host."""
    if not getattr(instance, _HANDOFF_PENDING_ATTR, False):
        return 0
    return handoff_host_weights(instance)


def handoff_host_weights(module: nn.Module) -> int:
    """Install storage-free CUDA stand-ins and return the number of weights handed off.

    Run after checkpoint loading and weight post-processing, but before Dynamo
    traces devices."""
    parked_parameters = _collect_parked_parameters(module)
    _warn_missing_parked_weights(module=module, parked_parameters=parked_parameters)
    if not parked_parameters:
        setattr(module, _HANDOFF_PENDING_ATTR, False)
        residency.record_handoff_slots(module=module, slots=())
        return 0

    # Detach plain Parameters before swap_tensors replaces their host payload.
    handed_off_bytes = _install_weight_standins(
        module=module,
        parked_parameters=[(names, param, _local_weight_tensor(param).detach()) for names, param in parked_parameters],
    )
    setattr(module, _HANDOFF_PENDING_ATTR, False)
    magi_logger.info(
        "host offload: handed off %d weight(s), %.1f MiB, for %s; their CUDA stand-ins have no storage",
        len(parked_parameters),
        handed_off_bytes / 2**20,
        type(module).__name__,
    )
    return len(parked_parameters)
