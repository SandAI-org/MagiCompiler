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

"""torchrun entrypoint: two graph all-to-alls whose split sizes cross between the ranks.

Both exchanges send a (3, 4) tensor.  Rank 0 uses splits [1, 2] in the first exchange and
[2, 1] in the second; rank 1 uses them the other way round.  Each exchange is valid, but a
profiling key that held the rank-local split sizes would name the first exchange on rank 0
and the second on rank 1, so ``warm_and_sync`` would replay two different exchanges together
and NCCL would see mismatched send and receive sizes.  Every key shared by the ranks must
therefore keep the same exchange as its representative on every rank; only then does the
helper run ``warm_and_sync``, which must measure the all-to-all entry.

Run: torchrun --nproc_per_node=2 tests/feature_tests/profiling/profiling_helper/estimator_all_to_all_pairing_helper.py

Markers (rank 0): A2A_PAIRED ok=<bool>, A2A_WARMSYNC ok=<bool>, A2A_PASS / A2A_FAIL
"""

from __future__ import annotations

import os

import torch
import torch._inductor.config as inductor_config
import torch.distributed as dist
from torch._inductor.utils import contains_collective

from magi_compiler.profiling import ProfilingRuntimeEstimator

_ALL_TO_ALL_OP = torch.ops._c10d_functional.all_to_all_single.default
_WAIT_TENSOR_OP = torch.ops._c10d_functional.wait_tensor.default
_ROWS, _COLS = 3, 4
# rank -> (split sizes of the first exchange, split sizes of the second exchange)
_SPLIT_SIZES_BY_RANK = {0: ([1, 2], [2, 1]), 1: ([2, 1], [1, 2])}


class _CaptureCollectives:
    def __init__(self) -> None:
        self.collective_snodes = []

    def __call__(self, snodes):
        self.collective_snodes.extend(snode for snode in snodes if contains_collective(snode))
        return snodes


def _capture_collective_snodes(fn, x: torch.Tensor) -> list:
    capture = _CaptureCollectives()
    saved_config = (
        inductor_config.reorder_for_compute_comm_overlap,
        inductor_config.reorder_for_compute_comm_overlap_passes,
        inductor_config.force_disable_caches,
    )
    inductor_config.reorder_for_compute_comm_overlap = True
    inductor_config.reorder_for_compute_comm_overlap_passes = [capture]
    inductor_config.force_disable_caches = True
    try:
        torch._dynamo.reset()
        torch.compile(fn, dynamic=False)(x)
        torch.cuda.synchronize()
    finally:
        (
            inductor_config.reorder_for_compute_comm_overlap,
            inductor_config.reorder_for_compute_comm_overlap_passes,
            inductor_config.force_disable_caches,
        ) = saved_config
    return capture.collective_snodes


def main() -> None:
    dist.init_process_group("cpu:gloo,cuda:nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, "the crossed split sizes are written for two ranks"
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    group_name = dist.group.WORLD.group_name
    first_split_sizes, second_split_sizes = _SPLIT_SIZES_BY_RANK[rank]

    def two_exchanges(x: torch.Tensor) -> torch.Tensor:
        first = _WAIT_TENSOR_OP(_ALL_TO_ALL_OP(x, first_split_sizes, first_split_sizes, group_name))
        second = _WAIT_TENSOR_OP(_ALL_TO_ALL_OP(x, second_split_sizes, second_split_sizes, group_name))
        return first + second

    x = torch.randn(_ROWS, _COLS, device="cuda")
    collective_snodes = _capture_collective_snodes(two_exchanges, x)
    assert len(collective_snodes) == 2, f"expected two all-to-all snodes, got {len(collective_snodes)}"

    estimator = ProfilingRuntimeEstimator(sync_across_ranks=True)
    for snode in collective_snodes:
        estimator(snode)
    exchange_index_by_key = {
        repr(key): next(index for index, captured in enumerate(collective_snodes) if captured is snode)
        for key, snode in estimator._key_snode.items()
    }
    exchange_index_by_key_per_rank: list = [None] * world_size
    dist.all_gather_object(exchange_index_by_key_per_rank, exchange_index_by_key)
    shared_keys = set.intersection(*(set(index_by_key) for index_by_key in exchange_index_by_key_per_rank))
    paired = bool(shared_keys) and all(
        len({index_by_key[key] for index_by_key in exchange_index_by_key_per_rank}) == 1 for key in shared_keys
    )

    # A mispaired key would make warm_and_sync replay mismatched exchanges and hang in NCCL.
    warm_sync_ok = False
    if paired:
        estimator.warm_and_sync()
        collective_entries = [entry for entry in estimator.table.values() if entry.kind == "collective"]
        warm_sync_ok = bool(collective_entries) and all(entry.measured for entry in collective_entries)

    passed_per_rank: list = [None] * world_size
    dist.all_gather_object(passed_per_rank, paired and warm_sync_ok)
    if rank == 0:
        print(f"A2A_PAIRED ok={paired} exchange_index_by_key_per_rank={exchange_index_by_key_per_rank}", flush=True)
        print(f"A2A_WARMSYNC ok={warm_sync_ok}", flush=True)
        print("A2A_PASS" if all(passed_per_rank) else "A2A_FAIL", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    raise SystemExit(0 if all(passed_per_rank) else 1)


if __name__ == "__main__":
    main()
