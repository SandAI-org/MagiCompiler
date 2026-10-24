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

"""torchrun entrypoint: an internal-collective extern whose fx shapes differ per rank
(an uneven split: rank 0 holds one extra row) is keyed by its replay inputs, so it must
still be lockstep-measured by ``warm_and_sync`` when its materialize hook replays
rank-identical inputs, and must fall back to analytical -- without hanging -- when the
hook does not.

Run: torchrun --nproc_per_node=2 tests/feature_tests/profiling/profiling_helper/estimator_rank_local_helper.py

Markers (rank 0): RANK_LOCAL_MATCHED ok=<bool>, RANK_LOCAL_UNMATCHED ok=<bool>, RANK_LOCAL_PASS / RANK_LOCAL_FAIL
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.fx as fx
from torch._inductor.scheduler import ExternKernelSchedulerNode

from magi_compiler.profiling import ProfilingRuntimeEstimator, register_materialize_inputs

_ANALYTICAL_NS = 123.0
_ROWS, _COLS = 4096, 1024


def _gather_rows(x: torch.Tensor, splits: list[int]) -> torch.Tensor:
    outs = [x.new_empty(s, x.shape[1]) for s in splits]
    dist.all_gather(outs, x.contiguous())
    return torch.cat(outs)


@torch.library.custom_op("magi_test::gather_rows_even", mutates_args=())
def gather_rows_even(x: torch.Tensor, splits: list[int]) -> torch.Tensor:
    return _gather_rows(x, splits)


@torch.library.custom_op("magi_test::gather_rows_local", mutates_args=())
def gather_rows_local(x: torch.Tensor, splits: list[int]) -> torch.Tensor:
    return _gather_rows(x, splits)


def _even_replay(x, splits):
    rows = max(splits)
    return torch.randn(rows, x.shape[1], device=x.device, dtype=x.dtype), [rows] * len(splits)


def _local_replay(x, splits):
    return x, [x.shape[0]] * len(splits)


class FakeExternIR:
    def __init__(self, fx_node):
        self._fx = fx_node

    def get_origin_node(self):
        return self._fx


class _Extern(ExternKernelSchedulerNode):
    def __init__(self, fx_node):
        self.node = FakeExternIR(fx_node)

    def get_name(self):
        return "op_rank_local"

    def get_nodes(self):
        return [self]

    def get_estimated_runtime(self):
        return _ANALYTICAL_NS


def _snode(op, rows: int, splits: list[int], dev) -> _Extern:
    g = fx.Graph()
    x = g.placeholder("x")
    x.meta["val"] = torch.empty(rows, _COLS, device=dev, dtype=torch.bfloat16)
    out = g.call_function(op, (x, splits))
    out.meta["val"] = torch.empty(sum(splits), _COLS, device=dev, dtype=torch.bfloat16)
    return _Extern(out)


def _entry_after_sync(op, rank: int, dev):
    splits = [_ROWS + 1, _ROWS]
    est = ProfilingRuntimeEstimator(sync_across_ranks=True)
    est(_snode(op, rows=splits[rank], splits=splits, dev=dev))
    est.warm_and_sync()
    [entry] = est.table.values()
    return entry


def main() -> None:
    dist.init_process_group("cpu:gloo,cuda:nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.cuda.current_device()
    register_materialize_inputs("magi_test::gather_rows_even", _even_replay)
    register_materialize_inputs("magi_test::gather_rows_local", _local_replay)

    matched = _entry_after_sync(torch.ops.magi_test.gather_rows_even.default, rank, dev)
    matched_ns = [None] * dist.get_world_size()
    dist.all_gather_object(matched_ns, matched.ns)
    matched_ok = matched.measured and matched.ns != _ANALYTICAL_NS and len(set(matched_ns)) == 1

    unmatched = _entry_after_sync(torch.ops.magi_test.gather_rows_local.default, rank, dev)
    unmatched_ok = not unmatched.measured and unmatched.ns == _ANALYTICAL_NS

    t = torch.tensor([int(matched_ok and unmatched_ok)], device=dev)
    dist.all_reduce(t)
    all_ok = int(t.item()) == dist.get_world_size()
    if rank == 0:
        print(f"RANK_LOCAL_MATCHED ok={matched_ok} ns={matched.ns:.0f}", flush=True)
        print(f"RANK_LOCAL_UNMATCHED ok={unmatched_ok}", flush=True)
        print("RANK_LOCAL_PASS" if all_ok else "RANK_LOCAL_FAIL", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
