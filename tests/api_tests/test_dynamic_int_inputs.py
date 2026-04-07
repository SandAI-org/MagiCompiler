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

"""End-to-end tests for dynamic int graph inputs in piecewise compilation.

Reproduces the scenario where ``dynamic=True`` + ``assume_static_by_default=False``
causes data-dependent Python ints (e.g. split sizes derived from
``torch.bincount`` / ``tolist``) to become symbolic graph inputs.
Verifies that piecewise compilation handles non-tensor graph inputs correctly.
"""

import os

import pytest
import torch
import torch.nn as nn

from magi_compiler import magi_compile
from magi_compiler.config import CompileMode, get_compile_config

DEVICE = "cuda"
HIDDEN = 64
SEQ_LEN = 128
TOLERANCE = 1e-3


@pytest.fixture(autouse=True)
def _magi_compile_mode():
    config = get_compile_config()
    old_mode = config.compile_mode
    config.compile_mode = CompileMode.MAGI_COMPILE
    config.cache_root_dir = os.environ.get("MAGI_COMPILE_CACHE_ROOT_DIR", config.cache_root_dir)
    torch._dynamo.reset()
    yield
    config.compile_mode = old_mode
    torch._dynamo.reset()


# ========================= Helpers =========================


class SimpleDispatcher:
    """Minimal reproduction of ModalityDispatcher:
    compute split sizes from a tensor, then use them in ``torch.split``.
    """

    def __init__(self, modality_mapping: torch.Tensor, num_groups: int):
        self.group_size = torch.bincount(modality_mapping, minlength=num_groups).to(torch.int32)
        self.group_size_cpu: list[int] = [int(x) for x in self.group_size.to("cpu").tolist()]
        self.permute_mapping = torch.argsort(modality_mapping)
        self.inv_permute_mapping = torch.argsort(self.permute_mapping)

    def dispatch(self, x: torch.Tensor) -> list[torch.Tensor]:
        return list(torch.split(x, self.group_size_cpu, dim=0))

    def undispatch(self, *groups: torch.Tensor) -> torch.Tensor:
        return torch.cat(groups, dim=0)


def _make_inputs(seq_len: int, hidden: int, num_groups: int):
    x = torch.randn(seq_len, hidden, device=DEVICE)
    modality = torch.randint(0, num_groups, (seq_len,), device=DEVICE)
    return x, modality


# ========================= Tests =========================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestDynamicIntInputs:
    """Regression tests for int graph inputs in piecewise-compiled subgraphs."""

    def test_jit_compile_with_split_sizes(self):
        """JIT compile succeeds when ``torch.split`` receives data-dependent
        int sizes that become symbolic graph inputs.
        """

        @magi_compile(dynamic_arg_dims={"x": 0, "permute_mapping": 0})
        class _Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.linears = nn.ModuleList([nn.Linear(HIDDEN, HIDDEN) for _ in range(3)])

            def forward(self, x: torch.Tensor, permute_mapping: torch.Tensor, dispatcher: SimpleDispatcher):
                x = x[permute_mapping]
                groups = dispatcher.dispatch(x)
                processed = [self.linears[i](g) for i, g in enumerate(groups)]
                return dispatcher.undispatch(*processed)[dispatcher.inv_permute_mapping]

        class _Outer(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = _Block()

            def forward(self, x, modality):
                d = SimpleDispatcher(modality, 3)
                return self.block(x, d.permute_mapping, d)

        model = _Outer().to(DEVICE)
        x, modality = _make_inputs(SEQ_LEN, HIDDEN, 3)

        with torch.no_grad():
            out = model(x, modality)

        assert out.shape == x.shape

    def test_jit_compile_numerical_correctness(self):
        """Compiled output matches eager forward output."""

        @magi_compile(dynamic_arg_dims={"x": 0, "permute_mapping": 0})
        class _Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.linears = nn.ModuleList([nn.Linear(HIDDEN, HIDDEN) for _ in range(3)])

            def forward(self, x: torch.Tensor, permute_mapping: torch.Tensor, dispatcher: SimpleDispatcher):
                x = x[permute_mapping]
                groups = dispatcher.dispatch(x)
                processed = [self.linears[i](g) for i, g in enumerate(groups)]
                return dispatcher.undispatch(*processed)[dispatcher.inv_permute_mapping]

        class _Outer(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = _Block()

            def forward(self, x, modality):
                d = SimpleDispatcher(modality, 3)
                return self.block(x, d.permute_mapping, d)

        model = _Outer().to(DEVICE)
        x, modality = _make_inputs(SEQ_LEN, HIDDEN, 3)

        # Eager: call block.forward() directly, bypassing @magi_compile wrapper
        dispatcher = SimpleDispatcher(modality, 3)
        with torch.no_grad():
            eager_out = model.block.forward(x, dispatcher.permute_mapping, dispatcher)

        # Compiled: goes through @magi_compile
        with torch.no_grad():
            compiled_out = model(x, modality)

        assert model.block._magi_state_for_forward is not None
        assert model.block._magi_state_for_forward.jit_compiled_code is not None
        assert torch.allclose(
            eager_out, compiled_out, atol=TOLERANCE, rtol=TOLERANCE
        ), f"max diff = {(eager_out - compiled_out).abs().max().item()}"

    def test_jit_compile_different_distributions(self):
        """Same sequence length but different modality distributions:
        the symbolic int split sizes vary across calls.
        """

        @magi_compile(dynamic_arg_dims={"x": 0, "permute_mapping": 0})
        class _Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.linears = nn.ModuleList([nn.Linear(HIDDEN, HIDDEN) for _ in range(3)])

            def forward(self, x: torch.Tensor, permute_mapping: torch.Tensor, dispatcher: SimpleDispatcher):
                x = x[permute_mapping]
                groups = dispatcher.dispatch(x)
                processed = [self.linears[i](g) for i, g in enumerate(groups)]
                return dispatcher.undispatch(*processed)[dispatcher.inv_permute_mapping]

        class _Outer(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = _Block()

            def forward(self, x, modality):
                d = SimpleDispatcher(modality, 3)
                return self.block(x, d.permute_mapping, d)

        model = _Outer().to(DEVICE)
        x = torch.randn(SEQ_LEN, HIDDEN, device=DEVICE)

        for seed in range(3):
            torch.manual_seed(seed)
            modality = torch.randint(0, 3, (SEQ_LEN,), device=DEVICE)
            with torch.no_grad():
                out = model(x, modality)
            assert out.shape == x.shape, f"shape mismatch at seed={seed}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
