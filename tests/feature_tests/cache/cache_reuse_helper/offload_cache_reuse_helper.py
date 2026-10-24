# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import argparse
import json
import os
from datetime import timedelta
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._dynamo.utils import counters

from magi_compiler import magi_compile
from magi_compiler.config import CudaGraphMode
from magi_compiler.passes.weight_offload import get_global_host_weight_store
from magi_compiler.utils import compilation_counter

DEVICE = "cuda"
DTYPE = torch.bfloat16
HIDDEN = 256
BATCH = 4


class TinyOffloadNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(HIDDEN, HIDDEN, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _patch_offload(cfg, *, fsdp: bool = False):
    cfg.offload_config.graph_weight_offload = True
    cfg.offload_config.offload_min_shard_mib = 0.0
    cfg.offload_config.offload_max_device_weight_mib = 4
    cfg.offload_config.offload_h2d_bandwidth_gbps = 20.0
    cfg.fsdp_config.enable_fsdp = fsdp
    cfg.fsdp_config.cost_mode = "profile_sync" if fsdp else "analytical"
    cfg.cudagraph_mode = CudaGraphMode.NONE
    cfg.disable_graph_split = True
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--fsdp", action="store_true")
    args = parser.parse_args()

    if args.fsdp:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("cpu:gloo,cuda:nccl", timeout=timedelta(seconds=60))
    else:
        # A plain single-GPU model: offload scheduling must not need a process group.
        assert not dist.is_initialized()

    torch._dynamo.reset()
    torch.manual_seed(2026)
    weight = torch.randn(HIDDEN, HIDDEN, dtype=DTYPE)
    torch.manual_seed(7)
    x_cpu = torch.randn(BATCH, HIDDEN, dtype=DTYPE)

    with torch.device("meta"):
        model = TinyOffloadNet()
        model.to(DTYPE)
    if args.fsdp:
        from torch.distributed.device_mesh import init_device_mesh
        from torchtitan.experiments.simple_fsdp.simple_fsdp import data_parallel

        mesh = init_device_mesh("cuda", (dist.get_world_size(),))
        model = data_parallel(model, mesh, mode="fully_shard", ac_mode="full")
    model = magi_compile(
        model, dynamic_arg_dims={"x": 0}, config_patch=partial(_patch_offload, fsdp=args.fsdp), model_tag="offload_cache"
    )
    model.to_empty(device=DEVICE)
    with torch.no_grad():
        if args.fsdp:
            local_weight = next(model.parameters())._local_tensor
            local_weight.copy_(weight.chunk(dist.get_world_size())[dist.get_rank()])
        else:
            model.fc.weight.copy_(weight)

    x = x_cpu.to(device=DEVICE)
    with torch.no_grad():
        out = model(x)
        expected = torch.nn.functional.linear(x, weight.to(device=DEVICE))
    torch.testing.assert_close(out, expected)

    payload = {
        "shape": list(out.shape),
        "sum": float(out.float().sum().item()),
        "mean": float(out.float().mean().item()),
        "weight_count": get_global_host_weight_store().weight_count(),
        "resident_bytes": get_global_host_weight_store().resident_bytes(),
        "fx_graph_cache_hits": counters["inductor"]["fxgraph_cache_hit"],
        "fx_graph_cache_misses": counters["inductor"]["fxgraph_cache_miss"],
        "autograd_cache_hits": counters["aot_autograd"]["autograd_cache_hit"],
        "inductor_compiles": compilation_counter.num_inductor_compiles,
        "artifacts_saved": compilation_counter.num_compiled_artifacts_saved,
    }
    output_path = Path(args.output)
    if args.fsdp:
        output_path = output_path.with_suffix(f".rank{dist.get_rank()}.json")
    with output_path.open("w") as f:
        json.dump(payload, f)
    if args.fsdp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
