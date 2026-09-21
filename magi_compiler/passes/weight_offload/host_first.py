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

"""Build a compiled module's weights in host memory, so they are never on the device.

``to_empty`` is the single place a meta-built model turns into real storage, so a
patched ``_apply`` hands back pinned host memory for the weights that offload is
going to want.  The checkpoint reads straight into it.  Nothing about the loader
changes: it still sees DTensors of the right global shape, dtype and sharding,
and ``dcp.load`` still writes them in place.  Peak device memory stays at the
resident set instead of the whole model.

Two properties make this safe rather than clever:

* **Scope comes from the module tree, not a list.**  ``nn.Module._apply``
  recurses into children, so patching the class that ``@magi_compile`` decorates
  intercepts exactly the subtree that will be compiled.  Weights outside it --
  an embedding, a final projection, anything the eager prologue touches -- never
  reach this code and keep their device storage.

  The boundary is drawn around modules, and a weight tied across it belongs to
  both sides: one Parameter object, registered inside the subtree and out.
  ``_apply`` swaps the object rather than the registration, so whichever side
  runs last decides what the object holds, and neither side can see the other.
  Both orders are handled here instead of left to chance -- materialize declines
  a weight that already has storage, and the handoff says so when one it parked
  has been given storage again -- because the outcome of guessing is an illegal
  access in a module that was never compiled.
* **The graph never sees a host tensor.**  :func:`handoff` runs on the first
  call, before Dynamo traces: each parked weight becomes a CUDA tensor with
  zero-length storage, and the host buffer is adopted into the pool under its
  slot.  From there ``slot_of`` finds the shard, ``h2d_load`` fetches it, and
  ``H2dLoadReorder`` decides which ones go back to the device because nothing
  could hide their transfer.

The same path covers an unsharded ``nn.Parameter``.  There is no local shard
and no mesh: the parameter itself is what gets reserved, adopted, and later
looked up by ``PlainParamSource``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from magi_compiler.utils import magi_logger

from . import host_pool

_TO_EMPTY = "Module.to_empty.<locals>.<lambda>"
"""Qualname of the callable ``nn.Module.to_empty`` hands to ``_apply``.

Matching on it rather than on every ``_apply`` is the difference between
"materialize this meta model" and the dozen other things ``_apply`` does -- a
dtype cast, a device move, pinning -- none of which should quietly redirect a
weight into host memory.
"""

_PENDING_ATTR = "_magi_host_first_pending"
"""Set on an instance whose weights are waiting in host memory for their handoff."""

_PARKED_ATTR = "_magi_host_first_parked"
"""Names ``patch_materialize`` put in host memory, so the handoff can tell a
weight it was never given from one that was taken back out from under it."""


def _payload(param) -> torch.Tensor:
    """The tensor whose bytes we park: a DTensor's local shard, or the Parameter itself."""
    local = getattr(param, "_local_tensor", None)
    return local if local is not None else param


def _is_dtensor(param: object) -> bool:
    return getattr(param, "_local_tensor", None) is not None


def _is_offloadable(param: object) -> bool:
    """True for a weight a later source can be made to load back.

    Two shapes, matching the two ``WeightSource`` implementations:

    * A ``Shard(0)`` or ``Replicate`` DTensor -- ``FsdpShardSource``.  Replicate
      is the one worth spelling out: SimpleFSDP falls back to it for a weight
      whose dim0 does not divide the mesh, so every rank holds a FULL copy, and
      it is the last one to want left resident.
    * An unsharded ``nn.Parameter`` -- ``PlainParamSource``.  No mesh, no local
      shard; the parameter is the weight.

    A weight parked here that the graph then declines to load would be read as
    freed storage, so this has to agree with the source.  ``restore_unclaimed``
    catches what they still miss.
    """
    from torch.distributed.tensor import DTensor, Replicate, Shard

    if isinstance(param, DTensor):
        placements = param.placements
        if len(placements) != 1:
            return False
        placement = placements[0]
        return isinstance(placement, Replicate) or (isinstance(placement, Shard) and placement.dim == 0)
    return isinstance(param, nn.Parameter)


def _pretty(name: str) -> str:
    """``layers.0.attn.wq.parametrizations.weight.original`` -> ``layers.0.attn.wq.weight``.

    SimpleFSDP's parametrization buries every weight two levels down, and the
    placement log prints these next to each other at forty layers; ``param_name``
    does the same tidying for the names Dynamo produces.
    """
    parts = [p for p in name.split(".") if p not in ("parametrizations", "original")]
    return ".".join(parts) or name


def _rebuild(param, local: torch.Tensor):
    """``param`` with ``local`` behind it, same mesh, placements and global shape.

    ``DTensor.from_local`` cannot be used: it moves the local tensor to the
    mesh's device type, which for a CUDA mesh would pull the host buffer onto the
    GPU and undo the whole point.  The low-level constructor takes the tensor as
    given, which is what both directions of the swap need -- host on the way in,
    storage-free CUDA on the way out.
    """
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta

    meta = param._spec.tensor_meta
    spec = DTensorSpec(param._spec.mesh, param._spec.placements, tensor_meta=TensorMeta(meta.shape, meta.stride, local.dtype))
    return DTensor(local, spec, requires_grad=False)


def _replace_param(module: nn.Module, names: list[str], param: nn.Parameter, new_data) -> nn.Parameter:
    """Point every path in ``names`` at ``new_data``.  Returns the parameter that ended up registered.

    ``swap_tensors`` first, because it keeps the very object SimpleFSDP's
    parametrization registered and every other reference already points at.  It
    refuses a tensor anything holds a weakref to, so re-registration by path is
    the fallback; both are only legal before the first trace, which is where
    this runs.

    The two are not interchangeable, and a tied weight is where the difference
    shows: ``swap_tensors`` rewrites the object, so one call reaches every name
    that shares it, while re-registration writes one entry of one module's
    ``_parameters``.  Walking ``names`` is what keeps the fallback equivalent --
    a path left behind would still hold the host tensor and enter compilation as
    a CPU weight, which nothing downstream can detect: the pool knows the shard
    by the stand-in it minted, not by the bytes, so ``parked_slot`` would report
    the weight as never materialized while it sits in a pinned slab.

    Which object comes back matters, and not only for tidiness: ``nn.Parameter``
    detaches what it is given, so the local tensor behind the registered
    parameter is a *different object* from the one handed in, sharing its
    storage.  The pool keys slots by object identity, so it has to be told about
    the one that survived.
    """
    replacement = nn.Parameter(new_data, requires_grad=param.requires_grad)
    try:
        torch.utils.swap_tensors(param, replacement)
        return param
    except Exception as exc:  # noqa: BLE001
        # Warning, not debug: this is the branch where the in-place guarantee the
        # rest of the handoff leans on is gone, and the re-registration below has
        # to stand in for it.
        magi_logger.warning(
            "host offload: swap_tensors on %s failed (%s); re-registering %d path(s) instead", names[0], exc, len(names)
        )

    for name in names:
        parent_path, _, attr = name.rpartition(".")
        parent = module.get_submodule(parent_path) if parent_path else module
        parent.register_parameter(attr, replacement)
    return replacement


def _parked_params(module: nn.Module) -> list[tuple[list[str], nn.Parameter]]:
    """Every weight of ``module`` living in the host pool's slabs, with every path it answers to.

    Identified by where the bytes are rather than by a side table: only this
    file redirects a compiled module's weights onto the host, so after
    ``to_empty(cuda)`` a CPU payload is unambiguous -- a DTensor local under a
    CUDA mesh, or an unsharded Parameter that is itself on CPU.  Reading it off
    the tensor cannot go stale the way a registry keyed on objects ``_apply``
    swaps would.

    A tied weight is one Parameter object registered under several names, and
    the handoff needs both halves of that fact: grouping on identity is what
    keeps the shard on a single slot, and keeping the names is what lets
    :func:`_replace_param` reach every registration site.  ``named_parameters``
    would hand back only the first name -- it dedupes by default -- so the walk
    asks for the duplicates and does the grouping itself.
    """
    groups: dict[int, tuple[list[str], nn.Parameter]] = {}
    for name, param in module.named_parameters(recurse=True, remove_duplicate=False):
        if _is_dtensor(param):
            parked = param._local_tensor.device.type == "cpu" and param._spec.mesh.device_type != "cpu"
        else:
            parked = param.device.type == "cpu"
        if parked:
            groups.setdefault(id(param), ([], param))[0].append(name)
    return list(groups.values())


def _warn_about_reclaimed(module: nn.Module, parked: list[tuple[list[str], nn.Parameter]]) -> None:
    """Say so when a weight materialize parked is no longer on the host.

    One thing takes a parked weight back: an ``_apply`` outside the compiled
    subtree reaching the same Parameter object, which is what a weight tied
    across the compile boundary looks like from in here.  ``_apply`` swaps the
    object rather than the registration, so the sibling's ``to_empty`` gives the
    shared object device storage again and the host buffer is orphaned.

    Nothing is broken afterwards -- the weight has real storage and every reader
    of it works -- but the pinned reservation behind it will never be loaded and
    the device memory the offload was asked to save is still spent, while the
    materialize log has already claimed the opposite.  Silence is the wrong
    default for that: an offload that quietly does nothing reads exactly like one
    that worked, and the only visible difference is a high-water mark that did
    not move.
    """
    expected = getattr(module, _PARKED_ATTR, None)
    if not expected:
        return
    # One reconciliation per materialize: the record describes a single
    # to_empty, and a second handoff over an already handed-off module would
    # otherwise find every name missing and say so.
    setattr(module, _PARKED_ATTR, [])
    found = {_pretty(name) for names, _ in parked for name in names}
    missing = [name for name in expected if name not in found]
    if missing:
        magi_logger.warning(
            "host offload: %d weight(s) materialized in host memory are back on the device before the "
            "handoff and will not be offloaded (%s); the usual cause is a weight tied to a module outside "
            "%s, whose own to_empty re-materialized the shared Parameter",
            len(missing),
            ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else ""),
            type(module).__name__,
        )


def patch_materialize(instance: nn.Module, conf) -> None:
    """Make ``to_empty`` on ``instance`` build its offloadable weights in host memory.

    Per instance rather than per class, because that is where the compile
    decision is: ``@magi_compile`` on a class and ``magi_compile(module)`` on an
    object both funnel through the same per-instance hook, and an instance of the
    decorated class that is never compiled has no business losing its weights.
    Every use of ``_apply`` other than ``to_empty`` passes straight through.
    """
    if getattr(instance, "_magi_host_first_installed", False):
        return
    min_bytes = int(conf.offload_config.offload_min_shard_mib * 1024 * 1024)
    # The class's, not the instance's: taking it back off the instance would find
    # this wrapper and recurse.
    orig_apply = type(instance)._apply

    def _host_first_apply(self, fn, recurse=True):
        if getattr(fn, "__qualname__", "") != _TO_EMPTY:
            return orig_apply(self, fn, recurse=recurse)
        try:
            target = fn(torch.empty(0, device="meta")).device
        except Exception:  # noqa: BLE001 - an unreadable target is simply not ours to redirect
            return orig_apply(self, fn, recurse=recurse)
        if target.type != "cuda":
            return orig_apply(self, fn, recurse=recurse)

        # Resolved before _apply runs: it swaps the parameter objects as it goes,
        # so a predicate evaluated inside the callback would be reading tensors
        # that are halfway through being replaced.
        chosen = {}
        shared = []
        for name, param in self.named_parameters(recurse=recurse):
            if not _is_offloadable(param):
                continue
            payload = _payload(param)
            if not payload.is_meta:
                # Host-first can only redirect a weight it materializes itself,
                # and this one already has storage.  The reading worth guarding
                # against is that another _apply got to the same Parameter object
                # first, which is what a weight tied to a module outside the
                # compiled subtree looks like from here: parking it would redirect
                # a weight the eager prologue reads, and the handoff would then
                # empty it with no load in that module's graph to put the bytes
                # back.  Leave it alone either way -- the cost is one weight that
                # is not offloaded, against an illegal access nothing traces here.
                shared.append(_pretty(name))
                continue
            if payload.numel() * payload.element_size() < min_bytes:
                continue
            chosen[id(param)] = _pretty(name)

        if shared:
            magi_logger.warning(
                "host offload: %d weight(s) of %s already have storage and stay on the device (%s); host-first "
                "only redirects a weight it materializes itself, so one arriving with storage was either not "
                "built on meta or is shared with a module outside the compiled subtree whose to_empty reached "
                "it first -- and that one must keep its bytes, because the graph that would load them back is "
                "not the graph that reads it",
                len(shared),
                type(self).__name__,
                ", ".join(shared[:8]) + (" ..." if len(shared) > 8 else ""),
            )

        if not chosen:
            return orig_apply(self, fn, recurse=recurse)

        parked = []
        # A tied weight reaches _apply once per module that registers it.  Both
        # sites have to end up over the same bytes, which is what tying means,
        # so the second one gets a fresh wrapper over the first one's buffer
        # rather than a second reservation.
        hosts: dict[int, torch.Tensor] = {}

        def materialize(t):
            name = chosen.get(id(t))
            if name is None:
                return fn(t)
            host = hosts.get(id(t))
            if host is None:
                payload = _payload(t)
                host = host_pool.reserve(payload.shape, payload.dtype, name=name)
                hosts[id(t)] = host
                parked.append(host.numel() * host.element_size())
            return _rebuild(t, host) if _is_dtensor(t) else host

        result = orig_apply(self, materialize, recurse=recurse)
        setattr(self, _PENDING_ATTR, True)
        setattr(self, _PARKED_ATTR, list(chosen.values()))
        magi_logger.info(
            "host offload: %s materialized %d weight(s) (%.1f MiB) in pinned host memory instead of on %s; "
            "the checkpoint loads straight into them and no device byte is spent until the graph asks",
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
    """Complete the handoff for ``instance`` if its weights are still on the host.

    On the call path, so it is a single attribute read once the first call is
    past.
    """
    if not getattr(instance, _PENDING_ATTR, False):
        return 0
    return handoff(instance)


def handoff(module: nn.Module) -> int:
    """Swap each host-parked weight for a storage-free CUDA stand-in.  Returns the count.

    This is the moment the model stops being loadable and starts being
    compilable.  It has to happen before Dynamo traces -- the graph is lowered
    against the parameter's device, and a CPU one would compile CPU kernels --
    and it cannot happen any earlier, because everything between the two points
    (``dcp.load``, ``reset_parameters``, the quantization post-hooks that rewrite
    a weight's layout in place) is code that wants to read and write the bytes.

    Peak cost is one shard: the stand-in is allocated at full size so its sizes
    and strides are the real ones, then immediately emptied.
    """
    parked = _parked_params(module)
    _warn_about_reclaimed(module, parked)
    if not parked:
        setattr(module, _PENDING_ATTR, False)
        return 0

    device = torch.device("cuda", torch.cuda.current_device())
    nbytes = 0
    for names, param in parked:
        # Snapshot the host tensor before replace: for an unsharded Parameter
        # the payload IS param, and swap_tensors would otherwise turn that
        # reference into the CUDA stand-in.
        host = _payload(param)
        if host is param:
            host = param.detach()
        stand_in = torch.empty(host.shape, dtype=host.dtype, device=device)
        stand_in.untyped_storage().resize_(0)
        new_data = _rebuild(param, stand_in) if _is_dtensor(param) else stand_in
        installed = _replace_param(module, names, param, new_data)
        # After, not before: registration is what decides which tensor object
        # the graph will be traced against, and the slot is keyed on it.
        device_tensor = installed._local_tensor if _is_dtensor(installed) else installed
        host_pool.adopt(host, device_tensor, name=_pretty(names[0]))
        nbytes += host.numel() * host.element_size()

    setattr(module, _PENDING_ATTR, False)
    magi_logger.info(
        "host offload: handed %d weight(s) (%.1f MiB) over to the host pool; %s enters compilation with "
        "no weight storage on the device",
        len(parked),
        nbytes / 2**20,
        type(module).__name__,
    )
    return len(parked)
