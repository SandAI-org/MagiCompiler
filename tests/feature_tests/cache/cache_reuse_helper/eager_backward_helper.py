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

"""Helper script for test_aot_autograd_fallback.py.

Runs a real inference step (forward only under torch.no_grad) through
magi_compile, with a spy on standalone_compile that records
AOTConfig.force_non_lazy_backward_lowering at each call.

Supports --disable-fix to reproduce the bug by replacing
_force_eager_backward_lowering with a no-op context manager.

Output JSON payload
-------------------
- backward_flag_during_compile: list of bool, one per standalone_compile call
- all_flags_true: True iff every call saw force_non_lazy_backward_lowering=True
- num_standalone_compile_calls: len(backward_flag_during_compile)
- num_compiled_artifacts_saved: from compilation_counter
- num_inductor_compiles: from compilation_counter
- output_value: scalar output value
"""
from __future__ import annotations

import argparse
import contextlib
import json
from unittest.mock import patch

import torch
import torch._inductor as _inductor_mod
import torch.nn as nn

from magi_compiler import magi_compile
from magi_compiler.utils import compilation_counter

DEVICE = "cuda"
DTYPE = torch.bfloat16
HIDDEN = 16


@magi_compile(dynamic_arg_dims={"x": 0})
class InferenceModel(nn.Module):
    """Minimal inference model (forward only)."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(HIDDEN, HIDDEN, dtype=DTYPE, device=DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--disable-fix", action="store_true", help="Replace _force_eager_backward_lowering with no-op to reproduce bug"
    )
    args = parser.parse_args()

    torch._dynamo.reset()
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    if args.disable_fix:
        import magi_compiler.magi_backend.piecewise_compiler as _pc_mod

        _pc_mod._force_eager_backward_lowering = contextlib.nullcontext

    _real_standalone_compile = _inductor_mod.standalone_compile
    backward_flag_during_compile: list[bool] = []

    def _spy_standalone_compile(graph, example_inputs, **kwargs):
        from torch._functorch._aot_autograd.schemas import AOTConfig

        probe = AOTConfig(
            fw_compiler=lambda *a, **k: None,
            bw_compiler=lambda *a, **k: None,
            partition_fn=lambda *a, **k: None,
            decompositions={},
            num_params_buffers=0,
            aot_id=-1,
            keep_inference_input_mutations=False,
        )
        backward_flag_during_compile.append(probe.force_non_lazy_backward_lowering)
        return _real_standalone_compile(graph, example_inputs, **kwargs)

    with patch("torch._inductor.standalone_compile", side_effect=_spy_standalone_compile):
        model = InferenceModel()
        x = torch.randn(4, HIDDEN, device=DEVICE, dtype=DTYPE)
        with torch.no_grad():
            output = model(x)
        output_value = float(output.float().sum().item())

    payload = {
        "backward_flag_during_compile": backward_flag_during_compile,
        "all_flags_true": all(backward_flag_during_compile),
        "num_standalone_compile_calls": len(backward_flag_during_compile),
        "num_compiled_artifacts_saved": compilation_counter.num_compiled_artifacts_saved,
        "num_inductor_compiles": compilation_counter.num_inductor_compiles,
        "output_value": output_value,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f)


if __name__ == "__main__":
    main()
