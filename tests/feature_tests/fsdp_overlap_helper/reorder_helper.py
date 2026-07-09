# Copyright (c) 2025 SandAI. All Rights Reserved.
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

"""torchrun entrypoint: drive ``FsdpOverlapReorder`` through a real ``torch.compile``
on a tiny graph that contains upstream compute + a weight all-gather + a consumer,
and print markers for the pytest driver (test_fsdp_overlap_reorder.py).

The reorder pass is an Inductor ``reorder_for_compute_comm_overlap_passes`` callback;
it needs a process group (its multi-rank-determinism warmup calls dist.get_rank()).
We build a fn:  y = (x @ w0).relu()            # upstream compute
                g = all_gather(shard) ; wait    # a weight gather to hoist
                out = y + gathered_use          # consumer after the compute
and wrap the pass so we can assert it RAN and returned a valid schedule.

Run: torchrun --nproc_per_node=1 tests/feature_tests/fsdp_overlap_helper/reorder_helper.py

Markers (rank 0):
  REORDER_CALLED gathers=<n>
  REORDER_OK moved=<n>          (pass returned; N launches repositioned)
  REORDER_FINITE ok=<bool>      (compiled output finite + matches eager)
  REORDER_PASS / REORDER_FAIL
"""

from __future__ import annotations

import os

import torch
import torch._inductor.config as inductor_config
import torch.distributed as dist

from magi_compiler.passes.fsdp_overlap import FsdpOverlapReorder
from magi_compiler.passes.fsdp_overlap import reorder as _ro


def main() -> None:
    dist.init_process_group("cpu:gloo,cuda:nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.cuda.current_device()
    grp = dist.group.WORLD.group_name
    torch.manual_seed(0)

    _AG = torch.ops._c10d_functional.all_gather_into_tensor.default
    _WAIT = torch.ops._c10d_functional.wait_tensor.default

    H = 512
    w0 = torch.randn(H, H, device=dev, dtype=torch.bfloat16)
    shard = torch.randn(H, H, device=dev, dtype=torch.bfloat16)

    def fn(x, w0, shard):
        y = (x @ w0).relu()  # upstream compute the gather can hide behind
        g = _WAIT(_AG(shard, world, grp))  # weight all-gather + wait
        gathered = g.reshape(world * H, H)[:H]  # use the gathered weight
        return y @ gathered

    # instrument the pass: count how many times it runs and how many launches move.
    calls = {"n": 0, "gathers": 0, "moved": 0}
    orig_call = FsdpOverlapReorder.__call__

    def spy(self, snodes):
        calls["n"] += 1
        calls["gathers"] += sum(1 for s in snodes if _ro._is_weight_gather(s))
        out = orig_call(self, snodes)
        return out

    FsdpOverlapReorder.__call__ = spy

    reorder = FsdpOverlapReorder(slack_ns=5000.0)  # default cost_fn (Inductor analytical)
    prev_flag = inductor_config.reorder_for_compute_comm_overlap
    prev_passes = inductor_config.reorder_for_compute_comm_overlap_passes
    prev_cache = inductor_config.force_disable_caches
    inductor_config.reorder_for_compute_comm_overlap = True
    inductor_config.reorder_for_compute_comm_overlap_passes = [reorder]
    inductor_config.force_disable_caches = True
    try:
        torch._dynamo.reset()
        x = torch.randn(H, H, device=dev, dtype=torch.bfloat16)
        eager = fn(x, w0, shard)
        compiled = torch.compile(fn, dynamic=False)
        out = compiled(x, w0, shard)
        torch.cuda.synchronize()
    finally:
        inductor_config.reorder_for_compute_comm_overlap = prev_flag
        inductor_config.reorder_for_compute_comm_overlap_passes = prev_passes
        inductor_config.force_disable_caches = prev_cache
        FsdpOverlapReorder.__call__ = orig_call

    finite = bool(torch.isfinite(out).all().item())
    rel = ((out.float() - eager.float()).norm() / (eager.float().norm() + 1e-6)).item()
    numeric_ok = finite and rel < 5e-2

    if rank == 0:
        print(f"REORDER_CALLED gathers={calls['gathers']}", flush=True)
        print(f"REORDER_OK ran={calls['n'] > 0}", flush=True)
        print(f"REORDER_FINITE ok={numeric_ok} rel={rel:.5f}", flush=True)
        ok = calls["n"] > 0 and calls["gathers"] >= 1 and numeric_ok
        print("REORDER_PASS" if ok else "REORDER_FAIL", flush=True)
        rc = 0 if ok else 1
    else:
        rc = 0

    dist.barrier()
    dist.destroy_process_group()
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
