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

"""AOT compile failure must fall back to JIT instead of aborting the first request."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from magi_compiler import magi_compile
from magi_compiler.magi_backend.magi_compiler_base import MagiCompileState


class _Tiny(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_aot_failure_falls_back_to_jit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the fallback, aot_compile raising aborts warmup for every model in the process."""

    def _boom(self, *args, **kwargs):
        raise RuntimeError("Attempted to read undefined local variable")

    monkeypatch.setattr(MagiCompileState, "aot_compile", _boom)
    model = magi_compile(
        _Tiny().cuda().eval(), dynamic_arg_dims={"x": 0}, config_patch=lambda c: c.model_copy(update={"aot": True})
    )
    x = torch.randn(4, 4, device="cuda")
    with torch.no_grad():
        out = model(x)
    torch.testing.assert_close(out, x + 1)
