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

"""torchrun helper for test_placement_cache_key.py.

Compiles a SimpleFSDP MLP wrapped with ``data_parallel(mode=--mode)`` and writes
``<output-dir>/rank<r>.json`` with ``num_inductor_compiles`` and ``max_abs_diff``.

``--disable-fix`` drops the graph signature from the magi cache key.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh

import magi_compiler.magi_backend.magi_backend as magi_backend
from magi_compiler import magi_compile
from magi_compiler.utils import compilation_counter

HIDDEN = 64


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(HIDDEN, 4 * HIDDEN, bias=False)
        self.fc2 = nn.Linear(4 * HIDDEN, HIDDEN, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["fully_shard", "replicate"], required=True)
    parser.add_argument("--disable-fix", action="store_true")
    args = parser.parse_args()

    if args.disable_fix:
        magi_backend.compute_graph_hash = lambda graph, example_inputs: ""

    dist.init_process_group("cpu:gloo,cuda:nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))

    from torchtitan.experiments.simple_fsdp.simple_fsdp import data_parallel

    torch.manual_seed(0)
    reference = MLP().cuda().to(torch.bfloat16)
    torch.manual_seed(0)
    model = data_parallel(MLP().cuda().to(torch.bfloat16), mesh, mode=args.mode, ac_mode="full")
    compiled = magi_compile(model, dynamic_arg_dims={"x": 0})

    x = torch.randn(8, HIDDEN, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        out = compiled(x)
        expected = reference(x)

    payload = {
        "num_inductor_compiles": compilation_counter.num_inductor_compiles,
        "max_abs_diff": (out.float() - expected.float()).abs().max().item(),
    }
    Path(args.output_dir, f"rank{rank}.json").write_text(json.dumps(payload))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
