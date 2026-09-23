# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import argparse
import json

import torch
import torch.distributed as dist
import torch.nn as nn

from magi_compiler import magi_compile
from magi_compiler.config import CudaGraphMode
from magi_compiler.passes.weight_offload import host_pool

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


def _patch_offload(cfg):
    cfg.offload_config.graph_weight_offload = True
    cfg.offload_config.offload_min_shard_mib = 0.0
    cfg.cudagraph_mode = CudaGraphMode.NONE
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not dist.is_initialized():
        # Inductor's compute/comm overlap pass calls get_rank(); a 1-rank
        # HashStore group avoids a TCP port and is enough for that check.
        dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)

    torch._dynamo.reset()
    torch.manual_seed(2026)
    weight = torch.randn(HIDDEN, HIDDEN, dtype=DTYPE)
    torch.manual_seed(7)
    x_cpu = torch.randn(BATCH, HIDDEN, dtype=DTYPE)

    with torch.device("meta"):
        model = TinyOffloadNet()
        model.to(DTYPE)
    model = magi_compile(model, dynamic_arg_dims={"x": 0}, config_patch=_patch_offload, model_tag="offload_cache")
    model.to_empty(device=DEVICE)
    with torch.no_grad():
        model.fc.weight.copy_(weight)

    x = x_cpu.to(device=DEVICE)
    with torch.no_grad():
        out = model(x)

    payload = {
        "shape": list(out.shape),
        "sum": float(out.float().sum().item()),
        "mean": float(out.float().mean().item()),
        "num_bound": host_pool.num_bound(),
    }
    with open(args.output, "w") as f:
        json.dump(payload, f)


if __name__ == "__main__":
    main()
