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
from .runtime import host_weights

_TO_EMPTY = "Module.to_empty.<locals>.<lambda>"
"""Only ``to_empty`` materializes; dtype casts and other ``_apply`` uses keep torch behavior."""

_PENDING_ATTR = "_magi_host_first_pending"
"""Set on an instance whose weights are waiting in host memory for their handoff."""

_PARKED_ATTR = "_magi_host_first_parked"
"""Names ``patch_materialize`` put in host memory, so the handoff can tell a
weight it was never given from one that was taken back out from under it."""

_SETTLED_ATTR = "_magi_host_first_settled"
"""Set once ``to_empty`` or the first move to CUDA decided what to park, even nothing."""


def _payload(param: torch.Tensor) -> torch.Tensor:
    """The tensor whose bytes we park: a DTensor's local shard, or the Parameter itself."""
    return param._local_tensor if isinstance(param, DTensor) else param


def _is_offloadable(param: object) -> bool:
    """Must agree with the weight sources: plain Parameters or single Shard(0)/Replicate DTensors.
    ``restore_unclaimed`` restores parked weights that no graph loads."""
    if isinstance(param, DTensor):
        placements = param.placements
        if len(placements) != 1:
            return False
        placement = placements[0]
        return isinstance(placement, Replicate) or (isinstance(placement, Shard) and placement.dim == 0)
    return isinstance(param, nn.Parameter)


def _pretty(name: str) -> str:
    """``layers.0.attn.wq.parametrizations.weight.original`` -> ``layers.0.attn.wq.weight``."""
    parts = [p for p in name.split(".") if p not in ("parametrizations", "original")]
    return ".".join(parts) or name


def _rebuild(param: DTensor, local: torch.Tensor) -> DTensor:
    """Preserve DTensor metadata without moving ``local`` to the mesh device.
    ``DTensor.from_local`` would copy a host reservation onto CUDA."""
    from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta

    meta = param._spec.tensor_meta
    spec = DTensorSpec(param._spec.mesh, param._spec.placements, tensor_meta=TensorMeta(meta.shape, meta.stride, local.dtype))
    return DTensor(local, spec, requires_grad=False)


def _replace_param(module: nn.Module, names: list[str], param: nn.Parameter, new_data) -> nn.Parameter:
    """Replace before tracing, updating every tied registration if ``swap_tensors`` fails.
    Return the registered Parameter: construction can detach the input, and the
    host store must key its slot on the object the graph will actually use."""
    replacement = nn.Parameter(new_data, requires_grad=param.requires_grad)
    try:
        torch.utils.swap_tensors(param, replacement)
        return param
    except Exception as exc:  # noqa: BLE001
        # Re-registration must update every tied path when identity cannot be preserved.
        magi_logger.warning(
            "host offload: swap_tensors on %s failed (%s); re-registering %d path(s) instead", names[0], exc, len(names)
        )

    for name in names:
        parent_path, _, attr = name.rpartition(".")
        parent = module.get_submodule(parent_path) if parent_path else module
        parent.register_parameter(attr, replacement)
    return replacement


def _parked_params(module: nn.Module) -> list[tuple[list[str], nn.Parameter]]:
    """Group CPU payloads by Parameter identity, retaining every tied registration.
    After CUDA ``to_empty``, these are the weights redirected into host slabs;
    all names are needed if handoff falls back to re-registration."""
    groups: dict[int, tuple[list[str], nn.Parameter]] = {}
    for name, param in module.named_parameters(recurse=True, remove_duplicate=False):
        if isinstance(param, DTensor):
            parked = param._local_tensor.device.type == "cpu" and param.device_mesh.device_type != "cpu"
        else:
            parked = param.device.type == "cpu"
        if parked:
            groups.setdefault(id(param), ([], param))[0].append(name)
    return list(groups.values())


def _warn_about_reclaimed(module: nn.Module, parked: list[tuple[list[str], nn.Parameter]]) -> None:
    """Report weights parked here but re-materialized by an external ``_apply``.
    They remain readable on the device but no longer save offload memory."""
    expected = getattr(module, _PARKED_ATTR, None)
    if not expected:
        return
    # Consume the record once; a repeated handoff would otherwise report every name missing.
    setattr(module, _PARKED_ATTR, [])
    found = {_pretty(name) for names, _ in parked for name in names}
    missing = [name for name in expected if name not in found]
    if missing:
        magi_logger.warning(
            "host offload: %d weight(s) reclaimed before handoff (%s); offload skipped, check ties outside %s",
            len(missing),
            ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else ""),
            type(module).__name__,
        )


def _move_target(fn) -> torch.device | None:
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


def _refuse_casting_handed_over(module: nn.Module, fn) -> None:
    """Raise if ``fn`` would change a stand-in's dtype: torch would read its empty
    CUDA storage, and the host slot and any compiled load keep the old dtype.

    A guard for the common ``model.cuda().half()``, not a complete classifier;
    the supported path is to cast before parking (see the module's dtype contract)."""
    probed = set()
    for name, param in module.named_parameters():
        payload = _payload(param)
        layout = (payload.dtype, payload.device)
        if layout in probed or host_weights.default_store().slot_of(payload) is None:
            continue
        probed.add(layout)
        try:
            cast = fn(torch.empty(0, dtype=payload.dtype, device=payload.device))
        except Exception:  # noqa: BLE001 - a function that cannot take a probe is not a cast
            continue
        if isinstance(cast, torch.Tensor) and cast.dtype != payload.dtype:
            raise RuntimeError(
                f"host offload: cannot cast {type(module).__name__}.{_pretty(name)} from {payload.dtype} to "
                f"{cast.dtype}: its weights were already handed to host memory. Finish every dtype "
                "conversion before the first CUDA move (host-built model) or before to_empty (meta-built "
                "model), e.g. model.half().cuda() instead of model.cuda().half(); or disable "
                "offload_config.graph_weight_offload."
            )


def _manages_device_moves(instance: nn.Module) -> bool:
    return getattr(instance, _PENDING_ATTR, False) or residency.has_slots(instance)


def _is_sharded(module: nn.Module) -> bool:
    return any(isinstance(p, DTensor) for p in module.parameters())


def _built_on_host(module: nn.Module) -> bool:
    """Unsharded, fully materialized, with weights on the host: never went through ``to_empty``."""
    params = list(module.parameters())
    if any(isinstance(p, DTensor) or p.is_meta for p in params):
        return False
    return any(p.device.type == "cpu" for p in params)


def _leaving_parked(fn):
    """``fn`` for everything but the stand-ins, whose placement is the pool's."""

    def apply(t):
        return t if host_weights.default_store().slot_of(t) is not None else fn(t)

    return apply


# A move may arrive under inference mode, and placement later writes these stand-ins outside it.
@torch.inference_mode(False)
def _hand_over(module: nn.Module, parked: list[tuple[list[str], nn.Parameter, torch.Tensor]]) -> int:
    """Swap each weight for a storage-free CUDA stand-in and adopt its filled host buffer.
    Returns the bytes handed over. Stand-in construction temporarily allocates one full shard."""
    device = torch.device("cuda", torch.cuda.current_device())
    nbytes = 0
    slots = []
    for names, param, host in parked:
        stand_in = torch.empty(host.shape, dtype=host.dtype, device=device)
        stand_in.untyped_storage().resize_(0)
        new_data = _rebuild(param, stand_in) if isinstance(param, DTensor) else stand_in
        installed = _replace_param(module, names, param, new_data)
        # After, not before: registration is what decides which tensor object
        # the graph will be traced against, and the slot is keyed on it.
        device_tensor = _payload(installed)
        slots.append(host_weights.default_store().adopt(host, device_tensor, name=_pretty(names[0])))
        nbytes += host.numel() * host.element_size()
    residency.record_slots(module, slots)
    return nbytes


@torch.inference_mode(False)
def _park_host_built(module: nn.Module, min_bytes: int) -> int:
    """Copy a host-built module's offloadable weights into the pool and hand them over.
    Returns the number parked; the rest keep their normal placement."""
    groups: dict[int, tuple[list[str], nn.Parameter]] = {}
    for name, param in module.named_parameters(remove_duplicate=False):
        if param.device.type != "cpu" or param.numel() * param.element_size() < min_bytes:
            continue
        if param.is_contiguous():
            groups.setdefault(id(param), ([], param))[0].append(name)
    parked = []
    for names, param in groups.values():
        host = host_weights.default_store().reserve(param.shape, param.dtype, name=_pretty(names[0]))
        host.copy_(param.detach())
        parked.append((names, param, host))
    if not parked:
        return 0
    nbytes = _hand_over(module, parked)
    magi_logger.info(
        "host offload: %s parked %d CPU weight(s), %.1f MiB, on first CUDA move",
        type(module).__name__,
        len(parked),
        nbytes / 2**20,
    )
    return len(parked)


def patch_materialize(instance: nn.Module, conf) -> None:
    """Install per-instance host-first materialization and residency-aware device moves.

    Redirect eligible CUDA ``to_empty`` allocations to pinned host memory.
    A host-built plain module parks eligible weights on its first CUDA move.
    Later device moves release or restore promoted slots. Callers must finish dtype
    conversion before parking; see the module's dtype contract."""
    if getattr(instance, "_magi_host_first_installed", False):
        return
    min_bytes = int(conf.offload_config.offload_min_shard_mib * 1024 * 1024)

    def orig_apply(self, fn, recurse):
        # The class's, looked up per call: the instance's is this wrapper, and a
        # sharding owner may swap the class after compile-time patching.
        return type(self)._apply(self, fn, recurse=recurse)

    def _host_first_apply(self, fn, recurse=True):
        # Meta-built (e.g. SimpleFSDP): to_empty(cuda) puts eligible weights in the pool, handoff pending.
        if getattr(fn, "__qualname__", "") == _TO_EMPTY:
            return _materialize_on_host(self, fn, recurse)
        # None unless fn is a pure host<->CUDA move: dtype casts and other conversions get None.
        # Combined device+dtype moves are unsupported once weights are parked (module docstring).
        target = _move_target(fn)
        # Decided once per instance. Weights on the host at the first CUDA move are parked: built on
        # the host, or built on CUDA and moved to the host first. Weights already on CUDA never are.
        if target is not None and target.type == "cuda" and not getattr(self, _SETTLED_ATTR, False):
            setattr(self, _SETTLED_ATTR, True)
            if _built_on_host(self):
                _park_host_built(self, min_bytes)
        if target is None:
            _refuse_casting_handed_over(self, fn)
        # Plain torch: not a device move, or no handoff is pending and none ever ran; e.g. a move to
        # the host before the first CUDA move, or a module already on CUDA at that move.
        if target is None or not _manages_device_moves(self):
            return orig_apply(self, fn, recurse=recurse)
        # A device move of a module that owns host slots: hand off first if to_empty left it pending.
        handoff_if_pending(self)
        residency.move(self, to_host=target.type == "cpu")
        # Sharded: the sharding owner moves its DTensors in place, so nothing else moves here.
        # Unsharded: the remaining tensors move normally, the stand-ins stay where the pool put them.
        return self if _is_sharded(self) else orig_apply(self, _leaving_parked(fn), recurse=recurse)

    def _materialize_on_host(self, fn, recurse):
        # fn is to_empty's own empty_like, recognized by qualname above.
        target = fn(torch.empty(0, device="meta")).device
        if target.type != "cuda":
            return orig_apply(self, fn, recurse=recurse)
        setattr(self, _SETTLED_ATTR, True)

        # Resolve candidates before _apply starts swapping parameter objects.
        chosen = {}
        shared = []
        strided = []
        for name, param in self.named_parameters(recurse=recurse):
            if not _is_offloadable(param):
                continue
            payload = _payload(param)
            # Before the two refusals below, so neither reports a weight that was
            # never a candidate in the first place.
            if payload.numel() * payload.element_size() < min_bytes:
                continue
            if not payload.is_contiguous():
                # Flat host reservations and H2D outputs cannot preserve this stride.
                strided.append(_pretty(name))
                continue
            if not payload.is_meta:
                # An eager module may share this already materialized weight;
                # emptying it at handoff would leave that reader without an H2D load.
                shared.append(_pretty(name))
                continue
            chosen[id(param)] = _pretty(name)

        if strided:
            magi_logger.warning(
                "host offload: %d non-contiguous weight(s) of %s stay on device to preserve layout (%s)",
                len(strided),
                type(self).__name__,
                ", ".join(strided[:8]) + (" ..." if len(strided) > 8 else ""),
            )

        if shared:
            magi_logger.warning(
                "host offload: %d already-materialized weight(s) of %s stay on device for possible external readers (%s)",
                len(shared),
                type(self).__name__,
                ", ".join(shared[:8]) + (" ..." if len(shared) > 8 else ""),
            )

        if not chosen:
            return orig_apply(self, fn, recurse=recurse)

        parked = []
        # _apply visits every tied registration; reuse one host buffer for all of them.
        hosts: dict[int, torch.Tensor] = {}

        def materialize(t):
            name = chosen.get(id(t))
            if name is None:
                return fn(t)
            host = hosts.get(id(t))
            if host is None:
                payload = _payload(t)
                host = host_weights.default_store().reserve(payload.shape, payload.dtype, name=name)
                hosts[id(t)] = host
                parked.append(host.numel() * host.element_size())
            return _rebuild(t, host) if isinstance(t, DTensor) else host

        result = orig_apply(self, materialize, recurse=recurse)
        setattr(self, _PENDING_ATTR, True)
        setattr(self, _PARKED_ATTR, list(chosen.values()))
        magi_logger.info(
            "host offload: %s materialized %d weight(s), %.1f MiB, in pinned host memory for %s",
            type(self).__name__,
            len(parked),
            sum(parked) / 2**20,
            target,
        )
        return result

    instance._apply = _host_first_apply.__get__(instance, type(instance))
    instance._magi_host_first_installed = True
    magi_logger.info("host offload: %s will materialize its offloadable weights in host memory", type(instance).__name__)


def handoff_if_pending(instance: object) -> int:
    """Complete the handoff for ``instance`` if its weights are still on the host."""
    if not getattr(instance, _PENDING_ATTR, False):
        return 0
    return handoff(instance)


def handoff(module: nn.Module) -> int:
    """Install storage-free CUDA stand-ins and return the number of weights handed off.

    Run after checkpoint loading and weight post-processing, but before Dynamo
    traces devices."""
    parked = _parked_params(module)
    _warn_about_reclaimed(module, parked)
    if not parked:
        setattr(module, _PENDING_ATTR, False)
        residency.record_slots(module, ())
        return 0

    # Detach plain Parameters before swap_tensors replaces their host payload.
    nbytes = _hand_over(module, [(names, param, _payload(param).detach()) for names, param in parked])
    setattr(module, _PENDING_ATTR, False)
    magi_logger.info(
        "host offload: handed off %d weight(s), %.1f MiB, for %s; their CUDA stand-ins have no storage",
        len(parked),
        nbytes / 2**20,
        type(module).__name__,
    )
    return len(parked)
