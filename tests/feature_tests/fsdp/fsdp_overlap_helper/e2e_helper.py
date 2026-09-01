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

"""torchrun entrypoint: exercise the WHOLE FSDP-overlap chain end to end on a small
SimpleFSDP-style model, and print machine-checkable markers to stdout for the pytest
driver (tests/feature_tests/test_fsdp_overlap_e2e.py).

Chain under test (magi_backend._apply_fsdp_fullgraph_overlap):
  redistribute lowering  ->  whole-graph bucketing (coalesced)  ->  FsdpOverlapReorder

Model: two Linear layers whose params are Shard(0) DTensors (via torchtitan
``data_parallel(mode="fully_shard")``), with an
opaque ``@magi_register_custom_op(is_subgraph_boundary=True)`` op between them so the
graph contains an opaque extern call alongside the
weight ``prim_redistribute`` nodes the lowering pass rewrites.  Bucketing is
whole-graph: the boundary op does NOT split buckets.

Run: torchrun --nproc_per_node=N tests/feature_tests/fsdp_overlap_helper/e2e_helper.py
     [--cost-mode analytical|profile_sync]

Markers printed on rank 0 (grepped by the test):
  E2E_CONFIG world=<n> cost_mode=<m>
  E2E_COMPILED            (compiled callable produced, forward ran)
  E2E_NUMERIC rel=<f> ok=<bool>   (compiled vs eager output)
  E2E_PASS / E2E_FAIL
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh

from magi_compiler import magi_compile, magi_register_custom_op
from magi_compiler.config import CompileMode, CudaGraphMode


# An opaque boundary op: stays in the graph as an opaque extern call
# Elementwise so it is trivially correct.
@magi_register_custom_op(name="fsdp_e2e::boundary_gelu", is_subgraph_boundary=True)
def boundary_gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x)


class Block(nn.Module):
    """One transformer-ish block: two Linears with an opaque extern op between them.
    Multiple blocks give the graph many Shard(0) weight all-gathers interleaved with
    compute -- so the bucketing pass coalesces them whole-graph and the reorder pass
    has real compute to hide gathers behind."""

    def __init__(self, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = boundary_gelu(x)
        x = self.fc2(x)
        return x


class TinyModel(nn.Module):
    """A multi-layer stack of Blocks (default 4) -> many Shard(0) weights,
    exercising the full lower->bucket->reorder chain at scale."""

    def __init__(self, hidden: int, n_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList(Block(hidden) for _ in range(n_layers))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-mode", default="analytical", choices=["analytical", "profile_sync"])
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    args = ap.parse_args()

    dist.init_process_group("cpu:gloo,cuda:nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.cuda.current_device()
    torch.manual_seed(0)

    # Surface the backend's INFO logs (redistribute lowering / bucketing / reorder
    # "repositioned N/M") to stderr so the pytest driver can assert the chain ran.
    # These come back through torch's logging even from standalone_compile.
    os.environ.setdefault("MAGI_LOGGING_LEVEL", "INFO")

    if rank == 0:
        print(f"E2E_CONFIG world={world} cost_mode={args.cost_mode}", flush=True)

    from torchtitan.experiments.simple_fsdp.simple_fsdp import data_parallel

    mesh = init_device_mesh("cuda", (world,))

    # --- eager reference (plain replicated weights) ---
    hidden = args.hidden
    ref = TinyModel(hidden, n_layers=args.n_layers).to(dev).to(torch.bfloat16)
    x = torch.randn(64, hidden, device=dev, dtype=torch.bfloat16)
    with torch.no_grad():
        eager_out = ref(x)

    # --- SimpleFSDP-sharded + magi_compile with the overlap chain ---
    model = TinyModel(hidden, n_layers=args.n_layers).to(dev).to(torch.bfloat16)
    # copy the reference weights so outputs are comparable (all layers)
    with torch.no_grad():
        for (_, dst), (_, src) in zip(model.named_parameters(), ref.named_parameters()):
            dst.copy_(src)
    model = data_parallel(model, mesh, mode="fully_shard")

    def _patch(cfg):
        cfg.compile_mode = CompileMode.MAGI_COMPILE
        cfg.cudagraph_mode = CudaGraphMode.NONE
        cfg.disable_graph_split = True
        cfg.fsdp_config.enable_fsdp = True
        cfg.fsdp_config.bucket_mode = "coalesced"
        cfg.fsdp_config.cost_mode = args.cost_mode
        return cfg

    # dim 0 of the input (token count) is the dynamic dim.
    compiled = magi_compile(model, config_patch=_patch, dynamic_arg_dims={"x": 0})

    with torch.no_grad():
        out = compiled(x)
        torch.cuda.synchronize()
    if rank == 0:
        print("E2E_COMPILED", flush=True)

    # numeric check vs eager (bf16 tolerance; world=1 all_gather is identity, world>1
    # gathers the sharded weight back -> same math).
    out_f = out.float()
    ref_f = eager_out.float()
    rel = ((out_f - ref_f).norm() / (ref_f.norm() + 1e-6)).item()
    ok = bool(torch.isfinite(out_f).all().item()) and rel < 5e-2

    # confirm every rank agrees (no rank-divergent hang/output)
    ok_t = torch.tensor([1 if ok else 0], device=dev)
    dist.all_reduce(ok_t)
    all_ok = int(ok_t.item()) == world

    if rank == 0:
        print(f"E2E_NUMERIC rel={rel:.5f} ok={ok}", flush=True)
        print("E2E_PASS" if all_ok else "E2E_FAIL", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
