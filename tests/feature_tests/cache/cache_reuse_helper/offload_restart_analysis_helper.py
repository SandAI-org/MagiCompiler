# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""One graph-weight-offload run whose first trace raises TensorifyScalarRestartAnalysis; writes the output to
``--output``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from magi_compiler import magi_compile
from magi_compiler.config import CudaGraphMode

DEVICE = "cuda"
DTYPE = torch.bfloat16
HIDDEN = 256
SEQ_LEN = 8


def _patch_offload(cfg):
    cfg.offload_config.graph_weight_offload = True
    cfg.offload_config.offload_min_shard_mib = 0
    cfg.cudagraph_mode = CudaGraphMode.NONE
    cfg.disable_graph_split = True
    return cfg


@magi_compile(dynamic_arg_dims={"x": 0}, config_patch=_patch_offload, model_tag="offload_restart")
class OffloadBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # LayerNorm's eps is a Python float that Dynamo lifts as a scalar input: tensorifying it restarts the trace.
        self.norm = nn.LayerNorm(HIDDEN, dtype=DTYPE)
        self.fc = nn.Linear(HIDDEN, HIDDEN, bias=False, dtype=DTYPE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc(self.norm(x))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch._dynamo.reset()
    torch.manual_seed(2026)
    model = OffloadBlock().eval()
    norm_weight, norm_bias, fc_weight = (p.detach().to(DEVICE) for p in (model.norm.weight, model.norm.bias, model.fc.weight))
    # The first CUDA move parks the weights in host memory, as a deploy's model load does.
    model.to(DEVICE)
    x = torch.randn(SEQ_LEN, HIDDEN, device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        expected = x + nn.functional.linear(nn.functional.layer_norm(x, (HIDDEN,), norm_weight, norm_bias), fc_weight)
        out = model(x)
    torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
    args.output.write_text(json.dumps({"shape": list(out.shape), "sum": float(out.float().sum().item())}))


if __name__ == "__main__":
    main()
