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

from __future__ import annotations

import ctypes
from functools import lru_cache

import torch
import torch._C._distributed_c10d as _c10d

from magi_compiler.utils import magi_logger

from .symm_arena import lookup_shard

_LIB = torch.library.Library("magi", "FRAGMENT")
# Signatures mirror ``_c10d_functional::all_gather_into_tensor`` / ``_coalesced`` so
# the rewrite pass can retarget a node without rebuilding its args.  That is also the
# only reason ``group_name`` is here: the copy engine reads peers from the arena.
_SCHEMA = "symm_all_gather(Tensor local, int group_size, str group_name) -> Tensor"
_SCHEMA_COALESCED = "symm_all_gather_coalesced(Tensor[] shards, int group_size, str group_name) -> Tensor[]"

# One gather: where it lands, the local shard, and every rank's view of that shard.
_Gather = tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]
# ``cudaMemcpyBatchAsync`` arguments frozen for one submission: dsts, srcs, sizes, count.
_Plan = tuple[ctypes.Array, ctypes.Array, ctypes.Array, int]


class _cudaMemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _cudaMemcpyAttributes(ctypes.Structure):
    _fields_ = [
        ("srcAccessOrder", ctypes.c_int),
        ("srcLocHint", _cudaMemLocation),
        ("dstLocHint", _cudaMemLocation),
        ("flags", ctypes.c_uint),
    ]


class BatchMemcpy:
    """``cudaMemcpyBatchAsync``: one submission for a whole layer's copies.

    Runtime signature is 8 args with no ``failIdx`` (that's driver-only
    ``cuMemcpyBatchAsync``), and it rejects the legacy null stream.
    """

    _SRC_ACCESS_ORDER_STREAM = 1

    def __init__(self) -> None:
        lib = ctypes.CDLL(None)
        lib.cudaGetErrorString.restype = ctypes.c_char_p
        self._strerror = lib.cudaGetErrorString
        self._fn = lib.cudaMemcpyBatchAsync
        self._fn.restype = ctypes.c_int
        self._fn.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
            ctypes.POINTER(_cudaMemcpyAttributes),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        self._attrs = (_cudaMemcpyAttributes * 1)()
        self._attrs[0].srcAccessOrder = self._SRC_ACCESS_ORDER_STREAM
        self._attr_idxs = (ctypes.c_size_t * 1)(0)

    @staticmethod
    def plan(triples: list[tuple[int, int, int]]) -> _Plan:
        """Freeze ``(dst_ptr, src_ptr, nbytes)`` for one submission.

        Raw pointers, not tensor views: a per-peer view is ~1.5us of CPU.
        Valid only while both sides stay alive; rebuilt per call.
        """
        n = len(triples)
        return (
            (ctypes.c_void_p * n)(*[d for d, _s, _b in triples]),
            (ctypes.c_void_p * n)(*[s for _d, s, _b in triples]),
            (ctypes.c_size_t * n)(*[b for _d, _s, b in triples]),
            n,
        )

    def run(self, plan: _Plan, stream: torch.cuda.Stream) -> None:
        dsts, srcs, sizes, n = plan
        rc = self._fn(dsts, srcs, sizes, n, self._attrs, self._attr_idxs, 1, ctypes.c_void_p(stream.cuda_stream))
        if rc:
            raise RuntimeError(f"cudaMemcpyBatchAsync failed: {self._strerror(rc).decode()}")


@lru_cache(maxsize=1)
def _batcher() -> BatchMemcpy | None:
    try:
        return BatchMemcpy()
    except (AttributeError, OSError) as exc:
        magi_logger.warning("cudaMemcpyBatchAsync unavailable (%s); falling back to per-copy submission", exc)
        return None


class _EventWork(_c10d.Work):
    """c10d Work whose ``wait()`` is a stream wait on the copy-engine event."""

    def __init__(self, event: torch.cuda.Event) -> None:
        super().__init__()
        self._event = event

    def wait(self, timeout=None) -> bool:  # noqa: ARG002 - c10d's signature
        torch.cuda.current_stream().wait_event(self._event)
        return True


@lru_cache(maxsize=1)
def _copy_stream() -> torch.cuda.Stream:
    """The one stream every copy-engine gather is submitted on."""
    return torch.cuda.Stream()


def _shard_peers(local: torch.Tensor, group_size: int) -> tuple[torch.Tensor, ...]:
    """The registered peer views of a local shard, validated."""
    entry = lookup_shard(local.data_ptr())
    if entry is None:
        raise RuntimeError(
            "magi::symm_all_gather got a tensor that is not a registered symmetric-memory shard. "
            "Only weights materialized through the arena can be gathered by the copy engine; "
            "the rewrite pass should have left this gather on NCCL."
        )
    peers = entry.peer_views
    if len(peers) != group_size:
        raise RuntimeError(f"shard has {len(peers)} peers but the gather asks for group_size={group_size}")
    return peers


def _copy_triples(gathers: list[_Gather]) -> list[tuple[int, int, int]]:
    """Flatten to one ``(dst_ptr, src_ptr, nbytes)`` per (member, peer) pair."""
    triples: list[tuple[int, int, int]] = []
    for out, local, peers in gathers:
        nbytes = local.numel() * local.element_size()  # dest contiguous; rank r at r*nbytes
        base = out.data_ptr()
        triples.extend((base + r * nbytes, p.data_ptr(), nbytes) for r, p in enumerate(peers))
    return triples


def _copy_per_peer(gathers: list[_Gather]) -> None:
    """Fallback for runtimes without ``cudaMemcpyBatchAsync``: one ``copy_`` per peer."""
    for out, local, peers in gathers:
        rows = local.shape[0]
        for r, p in enumerate(peers):
            out[r * rows : (r + 1) * rows].copy_(p, non_blocking=True)


def _issue_gathers(gathers: list[_Gather]) -> torch.cuda.Event:
    """Submit every gather as one batch; return the event that completes them all.

    Stream sync, submission and event are paid once for the whole call, not
    once per member -- that fixed CPU cost otherwise inflates the overlap window.
    """
    batcher = _batcher()
    stream = _copy_stream()
    stream.wait_stream(torch.cuda.current_stream())  # copies after compute-stream writes to the shards
    with torch.cuda.stream(stream):
        if batcher is not None:
            batcher.run(BatchMemcpy.plan(_copy_triples(gathers)), stream)
        else:
            _copy_per_peer(gathers)
        event = torch.cuda.Event()
        event.record(stream)
    return event


def _gather_dest(local: torch.Tensor, group_size: int) -> torch.Tensor:
    """Destination for gathering ``local``: rank r's shard lands at row ``r * rows``."""
    return local.new_empty((local.shape[0] * group_size, *local.shape[1:]))


def _symm_all_gather(local: torch.Tensor, group_size: int, group_name: str) -> torch.Tensor:
    peers = _shard_peers(local, group_size)
    out = _gather_dest(local, group_size)
    event = _issue_gathers([(out, local, peers)])
    _c10d._register_work(out, _EventWork(event))
    return out


def _symm_all_gather_meta(local: torch.Tensor, group_size: int, group_name: str) -> torch.Tensor:
    return _gather_dest(local, group_size)


def _symm_all_gather_coalesced(shards: list[torch.Tensor], group_size: int, group_name: str) -> list[torch.Tensor]:
    """One stream sync, one batch, one event for the whole bucket -- not one per member."""
    gathers: list[_Gather] = []
    for local in shards:
        peers = _shard_peers(local, group_size)  # validates before allocating
        gathers.append((_gather_dest(local, group_size), local, peers))
    event = _issue_gathers(gathers)
    outs = [out for out, _local, _peers in gathers]
    # Registry takes ownership of each Work; members share the event, not the wrapper.
    for out in outs:
        _c10d._register_work(out, _EventWork(event))
    return outs


def _symm_all_gather_coalesced_meta(shards: list[torch.Tensor], group_size: int, group_name: str) -> list[torch.Tensor]:
    return [_gather_dest(local, group_size) for local in shards]


def _register() -> None:
    _LIB.define(_SCHEMA)
    _LIB.impl("symm_all_gather", _symm_all_gather, "CUDA")
    _LIB.impl("symm_all_gather", _symm_all_gather_meta, "Meta")

    _LIB.define(_SCHEMA_COALESCED)
    _LIB.impl("symm_all_gather_coalesced", _symm_all_gather_coalesced, "CUDA")
    _LIB.impl("symm_all_gather_coalesced", _symm_all_gather_coalesced_meta, "Meta")


_register()

# Importing this module is what makes the ops exist, so these are always bound --
# callers guard the import, not the value.
SYMM_ALL_GATHER = torch.ops.magi.symm_all_gather.default
SYMM_ALL_GATHER_COALESCED = torch.ops.magi.symm_all_gather_coalesced.default
