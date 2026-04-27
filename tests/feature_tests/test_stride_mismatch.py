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

"""
Test: non-contiguous view crossing a piecewise boundary.

When ``split`` + ``unsqueeze`` produces a non-contiguous view and that tensor
crosses a piecewise boundary (custom op registered as subgraph boundary),
Inductor may change its stride (e.g. mm padding, kernel fusion).

The framework fix in ``PiecewiseCompileInterpreter._restride_outputs``
captures Inductor's actual output strides via
``TracingContext.report_output_strides`` and updates the FakeTensor
metadata before it flows into the next subgraph's compilation.
This ensures ``assert_size_stride`` in the downstream subgraph matches
the runtime stride.
"""

import pytest
import torch
import torch.nn as nn

from magi_compiler import magi_compile, magi_register_custom_op
from magi_compiler.config import get_compile_config


@magi_register_custom_op(name="test_stride::boundary_op", is_subgraph_boundary=True)
def boundary_op(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


class SplitGateModel(nn.Module):
    """Linear -> split -> unsqueeze(non-contiguous gate) -> boundary_op
    -> use gate after boundary."""

    def __init__(self, hidden: int, main_dim: int, gate_dim: int):
        super().__init__()
        self.main_dim = main_dim
        self.gate_dim = gate_dim
        self.proj = nn.Linear(hidden, main_dim + gate_dim, bias=False)
        self.out_proj = nn.Linear(main_dim, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.proj(x)
        main, gate = projected.split([self.main_dim, self.gate_dim], dim=1)
        gate = gate.unsqueeze(-1)  # non-contiguous view, stride=(total, 1, 1)

        main = main.reshape(x.shape[0], self.gate_dim, self.main_dim // self.gate_dim)
        main = torch.ops.test_stride.boundary_op(main)

        out = main * torch.sigmoid(gate)
        return self.out_proj(out.reshape(x.shape[0], self.main_dim))


def _run():
    torch._dynamo.reset()
    get_compile_config().splitting_ops.clear()
    get_compile_config().splitting_ops.append("test_stride::boundary_op")

    device = "cuda"
    dtype = torch.bfloat16
    hidden, main_dim, gate_dim = 5120, 5120, 40

    model = SplitGateModel(hidden, main_dim, gate_dim).to(device, dtype).eval()
    compiled = magi_compile(model, dynamic_arg_dims={"x": 0})

    for seq_len in [32, 64, 17]:
        x = torch.randn(seq_len, hidden, device=device, dtype=dtype)
        with torch.no_grad():
            ref = model(x)
            out = compiled(x)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        print(f"  seq_len={seq_len}: PASS")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_non_contiguous_view_across_piecewise_boundary():
    """Non-contiguous gate view should work without .contiguous() thanks to
    _restride_outputs aligning FakeTensor strides with Inductor output."""
    _run()


if __name__ == "__main__":
    _run()
    print("PASS: non-contiguous view across piecewise boundary handled correctly")
