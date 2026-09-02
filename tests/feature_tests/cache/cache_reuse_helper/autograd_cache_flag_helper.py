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

"""Helper script for test_autograd_function_cache_flag.py.

Each subprocess invocation runs one full training step (forward + backward +
optimizer) on a model that contains a torch.autograd.Function subclass, causing
Dynamo to emit an autograd_function_apply HigherOrderOperator node.

A spy is installed on torch._inductor.standalone_compile to record the value of
autograd_cache_allow_custom_autograd_functions at the moment standalone_compile
is called. Because piecewise_compiler.py imports standalone_compile inside its
compile() method body (``from torch._inductor import standalone_compile``), the
import resolves to the patched spy while the patch is active.

The spy delegates to the real standalone_compile so that compilation and artifact
saving proceed normally.

Output JSON payload
-------------------
- flag_during_compile: list of bool, one entry per standalone_compile call
- all_flags_true: True iff every entry in flag_during_compile is True
- num_standalone_compile_calls: len(flag_during_compile)
- num_compiled_artifacts_saved: from compilation_counter
- num_inductor_compiles: from compilation_counter
- loss: scalar training loss value
"""

from __future__ import annotations

import argparse
import json
from unittest.mock import patch

import torch
import torch._functorch.config as functorch_config
import torch._inductor as _inductor_mod
import torch.nn as nn

from magi_compiler import magi_compile
from magi_compiler.utils import compilation_counter

DEVICE = "cuda"
DTYPE = torch.bfloat16
HIDDEN = 16


class _ScaledSigmoid(torch.autograd.Function):
    """A custom autograd function.

    When Dynamo traces ``_ScaledSigmoid.apply(x)`` it emits an
    ``autograd_function_apply`` HigherOrderOperator node — the node that
    previously caused AOTAutograd caching to be bypassed.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return torch.sigmoid(x) * 2.0

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        sig = torch.sigmoid(x)
        return grad_output * sig * (1.0 - sig) * 2.0


@magi_compile(dynamic_arg_dims={"x": 0})
class TrainingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(HIDDEN, HIDDEN, dtype=DTYPE, device=DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _ScaledSigmoid.apply(self.linear(x))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch._dynamo.reset()
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    # Install a spy on standalone_compile to record the flag state at call time.
    # The real implementation is captured before patching so compilation proceeds
    # normally and artifacts are saved as usual.
    _real_standalone_compile = _inductor_mod.standalone_compile
    flag_during_compile: list[bool] = []

    def _spy_standalone_compile(graph, example_inputs, **kwargs):
        flag_during_compile.append(functorch_config.autograd_cache_allow_custom_autograd_functions)
        return _real_standalone_compile(graph, example_inputs, **kwargs)

    with patch("torch._inductor.standalone_compile", side_effect=_spy_standalone_compile):
        model = TrainingModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        x = torch.randn(4, HIDDEN, device=DEVICE, dtype=DTYPE)

        optimizer.zero_grad()
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

    payload = {
        "flag_during_compile": flag_during_compile,
        "all_flags_true": all(flag_during_compile),
        "num_standalone_compile_calls": len(flag_during_compile),
        "num_compiled_artifacts_saved": compilation_counter.num_compiled_artifacts_saved,
        "num_inductor_compiles": compilation_counter.num_inductor_compiles,
        "loss": float(loss.float().item()),
    }
    with open(args.output, "w") as f:
        json.dump(payload, f)


if __name__ == "__main__":
    main()
