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

"""torchrun entrypoint: compile-time weight offload, end to end, on a SimpleFSDP model.

Chain under test (magi_backend._apply_fsdp_fullgraph_overlap with
``offload_config.graph_weight_offload``)::

  lowering -> host binding -> h2d_load insertion -> bucketing
           -> FsdpOverlapReorder (phase 1) -> H2dLoadReorder (phase 2)

The load-bearing question this answers is whether a shard whose CUDA storage has
been freed survives as an Inductor graph input: the graph still carries it, for
its shape and for the data edge from the parameter, but there are no bytes behind
it until ``magi::h2d_load`` puts some there.  Nothing downstream of Dynamo has an
opinion about that in theory; this is where we find out in practice.

Run: torchrun --nproc_per_node=N .../offload_e2e_helper.py [--bucket-mode ...]

Markers printed on rank 0 (grepped by the test):
  OFFLOAD_CONFIG world=<n> bucket_mode=<m>
  OFFLOAD_FREED mib=<f>  shards=<n>
  OFFLOAD_COMPILED
  OFFLOAD_NUMERIC rel=<f> ok=<bool>
  OFFLOAD_PASS / OFFLOAD_FAIL
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh

from magi_compiler import magi_compile
from magi_compiler.config import CompileMode, CudaGraphMode


class Block(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class TinyModel(nn.Module):
    def __init__(self, hidden: int, n_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList(Block(hidden) for _ in range(n_layers))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket-mode", default="none", choices=["none", "coalesced"])
    ap.add_argument("--bucket-size-mib", type=int, default=0)
    ap.add_argument("--cost-mode", default="analytical", choices=["analytical", "profile_sync"])
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--min-shard-mib", type=float, default=0.0)
    # Counter-intuitive but load-bearing: SimpleFSDP reads ac_mode="none" as
    # "apply REGIONAL activation checkpointing", which wraps every weight access
    # in a checkpoint HOP and hides the redistribute inside a dynamo subgraph
    # where the lowering pass cannot see it.  Anything else leaves the
    # redistribute in the top-level graph, which is what this chain needs.
    ap.add_argument("--ac-mode", default="full")
    # 0 = take the fastest schedule and accept its residency; a cap pulls weights
    # back into the offload plan, into windows the schedule already left idle.
    ap.add_argument("--max-resident-mib", type=int, default=0)
    args = ap.parse_args()

    dist.init_process_group("cpu:gloo,cuda:nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.cuda.current_device()
    torch.manual_seed(0)
    os.environ.setdefault("MAGI_LOGGING_LEVEL", "INFO")

    if rank == 0:
        print(f"OFFLOAD_CONFIG world={world} bucket_mode={args.bucket_mode}", flush=True)

    from torchtitan.experiments.simple_fsdp.simple_fsdp import data_parallel

    mesh = init_device_mesh("cuda", (world,))

    hidden = args.hidden
    ref = TinyModel(hidden, n_layers=args.n_layers).to(dev).to(torch.bfloat16)
    x = torch.randn(64, hidden, device=dev, dtype=torch.bfloat16)
    with torch.no_grad():
        eager_out = ref(x)

    model = TinyModel(hidden, n_layers=args.n_layers).to(dev).to(torch.bfloat16)
    with torch.no_grad():
        for (_, dst), (_, src) in zip(model.named_parameters(), ref.named_parameters()):
            dst.copy_(src)
    model = data_parallel(model, mesh, mode="fully_shard", ac_mode=args.ac_mode)

    def _patch(cfg):
        cfg.compile_mode = CompileMode.MAGI_COMPILE
        cfg.cudagraph_mode = CudaGraphMode.NONE
        cfg.disable_graph_split = True
        cfg.fsdp_config.enable_fsdp = True
        cfg.fsdp_config.bucket_mode = args.bucket_mode
        cfg.fsdp_config.bucket_size_mib = args.bucket_size_mib
        cfg.fsdp_config.cost_mode = args.cost_mode
        cfg.offload_config.graph_weight_offload = True
        cfg.offload_config.offload_min_shard_mib = args.min_shard_mib
        cfg.offload_config.offload_max_resident_mib = args.max_resident_mib
        return cfg

    compiled = magi_compile(model, config_patch=_patch, dynamic_arg_dims={"x": 0})

    with torch.no_grad():
        out = compiled(x)
        torch.cuda.synchronize()

    from magi_compiler.offload import host_pool

    if rank == 0:
        print(
            f"OFFLOAD_FREED mib={host_pool.bound_bytes() / 2**20:.2f} shards={host_pool.num_bound()} "
            f"promoted_mib={host_pool.resident_bytes() / 2**20:.2f}",
            flush=True,
        )
        if host_pool.num_bound() == 0:
            # Offload binds what the redistribute lowering exposed, so zero shards
            # means the lowering found nothing -- which is a property of the
            # installed SimpleFSDP, not of this run.  Say so explicitly: a numeric
            # check on an un-offloaded graph passes for the wrong reason.
            print("OFFLOAD_SKIPPED reason=no_weight_gather_lowered", flush=True)
        print("OFFLOAD_COMPILED", flush=True)

    out_f = out.float()
    ref_f = eager_out.float()
    rel = ((out_f - ref_f).norm() / (ref_f.norm() + 1e-6)).item()
    ok = bool(torch.isfinite(out_f).all().item()) and rel < 5e-2

    # A second call: the host pool must still back the compiled artifact, and the
    # loads must be idempotent (nothing consumed the host copy on the first pass).
    with torch.no_grad():
        again = compiled(x)
        torch.cuda.synchronize()
    ok = ok and bool(torch.allclose(again.float(), out_f, atol=1e-3))

    ok_t = torch.tensor([1 if ok else 0], device=dev)
    dist.all_reduce(ok_t)
    all_ok = int(ok_t.item()) == world

    if rank == 0:
        print(f"OFFLOAD_NUMERIC rel={rel:.5f} ok={ok}", flush=True)
        print("OFFLOAD_PASS" if all_ok else "OFFLOAD_FAIL", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
