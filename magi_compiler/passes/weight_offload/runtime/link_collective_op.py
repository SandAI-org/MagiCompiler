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

"""NCCL collectives queued on a link stream, so the transfers on one physical link run one at a time.

Each ``magi::link_*`` op takes the schema of the ``_c10d_functional`` collective it
replaces at codegen (see ``RouteCollectivesToLinkStream``), with one overload per link:

* ``.pcie``   -- queued on the H2D load stream: without NVLink, loads, gathers and all-to-alls share PCIe;
* ``.nvlink`` -- queued on one stream all collectives share: they share NVLink, loads do not.

``magi::link_pairwise_all_to_all_single`` runs an all-to-all as pairwise send/recv steps
(``CompileConfig.pairwise_all_to_all``); its ``.off`` overload queues on a stream of its own.

The collective is issued while the link stream is current, so NCCL starts only after
every transfer queued before it, and the link stream then waits for it, so transfers
queued after it start once it is done.  The result is published through an
``EventWork``: the graph's ``wait_tensor`` is unchanged.
"""

from __future__ import annotations

import itertools
from functools import lru_cache
from typing import Callable

import torch
import torch._C._distributed_c10d as _c10d
import torch.distributed as dist
from torch.distributed.distributed_c10d import _resolve_process_group

from magi_compiler.cuda.event_work import EventWork

from ...snode_utils import ALL_GATHER_OP, ALL_TO_ALL_OP, COALESCED_ALL_GATHER_OP
from ...transfer_overlap import LinkSharing
from .h2d_op import get_h2d_stream

_LINK_COLLECTIVE_LIBRARY = torch.library.Library("magi", "FRAGMENT")

_FUNCTIONAL_ALL_GATHER_OUT = torch.ops._c10d_functional.all_gather_into_tensor_out.default
_FUNCTIONAL_ALL_GATHER_COALESCED = torch.ops._c10d_functional.all_gather_into_tensor_coalesced.default
_FUNCTIONAL_WAIT = torch.ops._c10d_functional.wait_tensor.default


@lru_cache(maxsize=1)
def get_nvlink_stream() -> torch.cuda.Stream:
    """The stream every collective queues on when the collectives share NVLink."""
    return torch.cuda.Stream()


@lru_cache(maxsize=1)
def get_pairwise_all_to_all_stream() -> torch.cuda.Stream:
    """The stream pairwise all-to-alls queue on when link sharing is off."""
    return torch.cuda.Stream()


_LINK_STREAM_GETTER_BY_LINK: dict[LinkSharing, Callable[[], torch.cuda.Stream]] = {
    LinkSharing.PCIE: get_h2d_stream,
    LinkSharing.NVLINK: get_nvlink_stream,
}
_PAIRWISE_STREAM_GETTER_BY_LINK: dict[LinkSharing, Callable[[], torch.cuda.Stream]] = {
    **_LINK_STREAM_GETTER_BY_LINK,
    LinkSharing.OFF: get_pairwise_all_to_all_stream,
}

_LINK_OP_NAME_BY_COLLECTIVE = {
    ALL_GATHER_OP: "link_all_gather",
    COALESCED_ALL_GATHER_OP: "link_all_gather_coalesced",
    ALL_TO_ALL_OP: "link_all_to_all_single",
}
_PAIRWISE_ALL_TO_ALL_OP_NAME = "link_pairwise_all_to_all_single"


def link_kernel_name(
    collective_op: torch._ops.OpOverload, link_sharing: LinkSharing, *, pairwise_all_to_all: bool = False
) -> str:
    """The Python kernel name of the link op that queues ``collective_op`` on ``link_sharing``'s link stream.

    A pairwise all-to-all also routes with link sharing off, onto ``get_pairwise_all_to_all_stream``."""
    if pairwise_all_to_all and collective_op is ALL_TO_ALL_OP:
        return f"torch.ops.magi.{_PAIRWISE_ALL_TO_ALL_OP_NAME}.{link_sharing.value}"
    assert link_sharing.collectives_take_turns, "collectives keep NCCL's own streams when link sharing is off"
    return f"torch.ops.magi.{_LINK_OP_NAME_BY_COLLECTIVE[collective_op]}.{link_sharing.value}"


def _queue_behind_compute(link_stream: torch.cuda.Stream) -> None:
    # Inputs come from the compute stream, and an output allocated there may reuse a block it just freed.
    link_stream.wait_stream(torch.cuda.current_stream())


def _publish_link_completion(
    outputs: list[torch.Tensor], link_stream: torch.cuda.Stream, wait_for_collective: Callable[[], object]
) -> None:
    """Hold the link stream behind the collective, and hand the graph's ``wait_tensor`` an event after it."""
    with torch.cuda.stream(link_stream):
        # The link stream, not compute, waits for NCCL here.
        wait_for_collective()
        collective_done_event = torch.cuda.Event()
        collective_done_event.record(link_stream)
    for tensor in outputs:
        _c10d._register_work(tensor, EventWork(collective_done_event))


def _link_all_gather(link_stream: torch.cuda.Stream, shard: torch.Tensor, group_size: int, group_name: str) -> torch.Tensor:
    # Allocated on the compute stream, like the load outputs; record_stream keeps it until the link is past it.
    gathered = torch.empty(_gathered_shape(shard, group_size), dtype=shard.dtype, device=shard.device)
    _queue_behind_compute(link_stream)
    with torch.cuda.stream(link_stream):
        _FUNCTIONAL_ALL_GATHER_OUT(shard, group_size, group_name, out=gathered)
    shard.record_stream(link_stream)
    gathered.record_stream(link_stream)
    _publish_link_completion([gathered], link_stream, wait_for_collective=lambda: _FUNCTIONAL_WAIT(gathered))
    return gathered


def _link_all_gather_coalesced(
    link_stream: torch.cuda.Stream, shards: list[torch.Tensor], group_size: int, group_name: str
) -> list[torch.Tensor]:
    compute_stream = torch.cuda.current_stream()
    _queue_behind_compute(link_stream)
    with torch.cuda.stream(link_stream):
        # No out variant exists: outputs come from the link stream's pool and must outlive their compute readers.
        gathered = list(_FUNCTIONAL_ALL_GATHER_COALESCED(shards, group_size, group_name))
    for shard in shards:
        shard.record_stream(link_stream)
    for tensor in gathered:
        tensor.record_stream(compute_stream)
    _publish_link_completion(
        gathered, link_stream, wait_for_collective=lambda: [_FUNCTIONAL_WAIT(tensor) for tensor in gathered]
    )
    return gathered


def _link_all_to_all_single(
    link_stream: torch.cuda.Stream,
    send_tensor: torch.Tensor,
    output_split_sizes: list[int],
    input_split_sizes: list[int],
    group_name: str,
) -> torch.Tensor:
    send_tensor = send_tensor.contiguous()
    # Allocated on the compute stream for the same reason as a gather's output.
    received = send_tensor.new_empty(_all_to_all_output_shape(send_tensor, output_split_sizes))
    _queue_behind_compute(link_stream)
    with torch.cuda.stream(link_stream):
        # c10d rather than the functional op: it fills an output the compute stream owns.
        work = dist.all_to_all_single(
            received,
            send_tensor,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=_resolve_process_group(group_name),
            async_op=True,
        )
    send_tensor.record_stream(link_stream)
    received.record_stream(link_stream)
    _publish_link_completion([received], link_stream, wait_for_collective=work.wait)
    return received


def _link_pairwise_all_to_all_single(
    link_stream: torch.cuda.Stream,
    send_tensor: torch.Tensor,
    output_split_sizes: list[int],
    input_split_sizes: list[int],
    group_name: str,
) -> torch.Tensor:
    send_tensor = send_tensor.contiguous()
    received = send_tensor.new_empty(_all_to_all_output_shape(send_tensor, output_split_sizes))
    _queue_behind_compute(link_stream)
    with torch.cuda.stream(link_stream):
        _pairwise_exchange(
            received=received,
            send_tensor=send_tensor,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=_resolve_process_group(group_name),
        )
    send_tensor.record_stream(link_stream)
    received.record_stream(link_stream)
    # Every step already held the link stream behind its send/recv.
    _publish_link_completion([received], link_stream, wait_for_collective=lambda: None)
    return received


def _pairwise_exchange(
    *,
    received: torch.Tensor,
    send_tensor: torch.Tensor,
    output_split_sizes: list[int],
    input_split_sizes: list[int],
    group: dist.ProcessGroup,
) -> None:
    """``world_size - 1`` steps on the current stream: at step k rank r sends to r + k and receives from r - k."""
    rank, world_size = group.rank(), group.size()
    input_offsets = [0, *itertools.accumulate(input_split_sizes)]
    output_offsets = [0, *itertools.accumulate(output_split_sizes)]
    received[output_offsets[rank] : output_offsets[rank + 1]].copy_(send_tensor[input_offsets[rank] : input_offsets[rank + 1]])
    for step in range(1, world_size):
        destination_rank, source_rank = (rank + step) % world_size, (rank - step) % world_size
        send_recv_ops = [
            dist.P2POp(
                dist.isend,
                send_tensor[input_offsets[destination_rank] : input_offsets[destination_rank + 1]],
                dist.get_global_rank(group, destination_rank),
                group,
            ),
            dist.P2POp(
                dist.irecv,
                received[output_offsets[source_rank] : output_offsets[source_rank + 1]],
                dist.get_global_rank(group, source_rank),
                group,
            ),
        ]
        # The next step queues behind this wait on the stream, so each rank has one peer in flight at a time.
        for work in dist.batch_isend_irecv(send_recv_ops):
            work.wait()


def _gathered_shape(shard: torch.Tensor, group_size: int) -> list[int]:
    return [shard.shape[0] * group_size, *shard.shape[1:]]


def _all_to_all_output_shape(send_tensor: torch.Tensor, output_split_sizes: list[int]) -> list[int]:
    return [sum(output_split_sizes), *send_tensor.shape[1:]]


def _link_all_gather_meta(shard: torch.Tensor, group_size: int, group_name: str) -> torch.Tensor:
    return shard.new_empty(_gathered_shape(shard, group_size))


def _link_all_gather_coalesced_meta(shards: list[torch.Tensor], group_size: int, group_name: str) -> list[torch.Tensor]:
    return [shard.new_empty(_gathered_shape(shard, group_size)) for shard in shards]


def _link_all_to_all_single_meta(
    send_tensor: torch.Tensor, output_split_sizes: list[int], input_split_sizes: list[int], group_name: str
) -> torch.Tensor:
    return send_tensor.new_empty(_all_to_all_output_shape(send_tensor, output_split_sizes))


_ALL_TO_ALL_SCHEMA = "(Tensor input, SymInt[] output_split_sizes, SymInt[] input_split_sizes, str group_name) -> Tensor"

_LINK_OP_DEFINITIONS = {
    # op name: (schema after the name, CUDA kernel taking the link stream first, Meta kernel, stream getter by link)
    "link_all_gather": (
        "(Tensor input, int group_size, str group_name) -> Tensor",
        _link_all_gather,
        _link_all_gather_meta,
        _LINK_STREAM_GETTER_BY_LINK,
    ),
    "link_all_gather_coalesced": (
        "(Tensor[] inputs, int group_size, str group_name) -> Tensor[]",
        _link_all_gather_coalesced,
        _link_all_gather_coalesced_meta,
        _LINK_STREAM_GETTER_BY_LINK,
    ),
    "link_all_to_all_single": (
        _ALL_TO_ALL_SCHEMA,
        _link_all_to_all_single,
        _link_all_to_all_single_meta,
        _LINK_STREAM_GETTER_BY_LINK,
    ),
    _PAIRWISE_ALL_TO_ALL_OP_NAME: (
        _ALL_TO_ALL_SCHEMA,
        _link_pairwise_all_to_all_single,
        _link_all_to_all_single_meta,
        _PAIRWISE_STREAM_GETTER_BY_LINK,
    ),
}


def _on_link_stream(cuda_kernel: Callable, get_link_stream: Callable[[], torch.cuda.Stream]) -> Callable:
    def kernel(*args, **kwargs):
        return cuda_kernel(get_link_stream(), *args, **kwargs)

    return kernel


def _register_link_collective_ops() -> None:
    for op_name, (schema, cuda_kernel, meta_kernel, stream_getter_by_link) in _LINK_OP_DEFINITIONS.items():
        for link_sharing, get_link_stream in stream_getter_by_link.items():
            overload_name = f"{op_name}.{link_sharing.value}"
            _LINK_COLLECTIVE_LIBRARY.define(overload_name + schema)
            _LINK_COLLECTIVE_LIBRARY.impl(overload_name, _on_link_stream(cuda_kernel, get_link_stream), "CUDA")
            _LINK_COLLECTIVE_LIBRARY.impl(overload_name, meta_kernel, "Meta")


_register_link_collective_ops()
