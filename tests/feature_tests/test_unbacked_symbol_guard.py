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
Test for GuardOnDataDependentSymNode triggered by mark_unbacked + view with -1.

When a dispatcher marks per-group token counts as unbacked symbols
(u0, u1, u2) via mark_unbacked, downstream reshape operations that infer
a dimension via -1 force Inductor to guard on expressions like
`u0+u1+u2 > 0`, which is forbidden for unbacked symbolic integers.
"""

pass

import pytest
import torch
import torch._dynamo.decorators

HIDDEN_SIZE = 64
NUM_HEADS = 4
HEAD_DIM = 16


class ModalityDispatcherMinimal:
    """Stripped-down ModalityDispatcher that reproduces the mark_unbacked pattern."""

    def __init__(self, modality_mapping: torch.Tensor, num_modalities: int):
        self.num_modalities = num_modalities
        permute_mapping = torch.argsort(modality_mapping)
        permuted = modality_mapping[permute_mapping]
        group_size = torch.bincount(permuted, minlength=num_modalities).to(torch.int32)
        group_size_cpu = [int(x) for x in group_size.to("cpu").tolist()]

        self._size_carrier = torch.empty(*group_size_cpu)
        if not torch.compiler.is_compiling():
            for i in range(num_modalities):
                torch._dynamo.decorators.mark_unbacked(self._size_carrier, i)

    @property
    def group_size_cpu(self) -> list[int]:
        return [self._size_carrier.shape[i] for i in range(self.num_modalities)]

    def dispatch(self, x: torch.Tensor) -> list[torch.Tensor]:
        return list(torch.split(x, self.group_size_cpu, dim=0))

    def undispatch(self, *parts: torch.Tensor) -> torch.Tensor:
        return torch.cat(parts, dim=0)


class BuggyModel(torch.nn.Module):
    """Reproduces the original bug: view(k.shape[0], num_heads, -1)."""

    def __init__(self):
        super().__init__()
        qkv_out = NUM_HEADS * HEAD_DIM * 3 + NUM_HEADS
        self.linear = torch.nn.Linear(HIDDEN_SIZE, qkv_out, bias=False, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor, modality_mapping: torch.Tensor):
        md = ModalityDispatcherMinimal(modality_mapping, 3)
        parts = md.dispatch(x)
        x = md.undispatch(*parts)

        qkv = self.linear(x)
        q_size = NUM_HEADS * HEAD_DIM
        kv_size = NUM_HEADS * HEAD_DIM
        _, k, _, g = torch.split(qkv, [q_size, kv_size, kv_size, NUM_HEADS], dim=1)
        k = k.view(-1, NUM_HEADS, HEAD_DIM)
        # BUG: view with -1 on unbacked-symbol seq_len triggers guard error
        g = g.view(k.shape[0], NUM_HEADS, -1)
        return g.sum()


class FixedModel(torch.nn.Module):
    """Fixed version: unsqueeze(-1) avoids the problematic -1 inference."""

    def __init__(self):
        super().__init__()
        qkv_out = NUM_HEADS * HEAD_DIM * 3 + NUM_HEADS
        self.linear = torch.nn.Linear(HIDDEN_SIZE, qkv_out, bias=False, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor, modality_mapping: torch.Tensor):
        md = ModalityDispatcherMinimal(modality_mapping, 3)
        parts = md.dispatch(x)
        x = md.undispatch(*parts)

        qkv = self.linear(x)
        q_size = NUM_HEADS * HEAD_DIM
        kv_size = NUM_HEADS * HEAD_DIM
        _, k, _, g = torch.split(qkv, [q_size, kv_size, kv_size, NUM_HEADS], dim=1)
        k = k.view(-1, NUM_HEADS, HEAD_DIM)
        # FIX: unsqueeze avoids -1 dimension inference on unbacked symbols
        g = g.unsqueeze(-1)
        return g.sum()


def _make_inputs(seq_len: int, modality_sizes: list[int], device: str):
    assert sum(modality_sizes) == seq_len
    parts = []
    for mod_id, size in enumerate(modality_sizes):
        parts.append(torch.full((size,), mod_id, dtype=torch.long, device=device))
    modality_mapping = torch.cat(parts)
    x = torch.randn(seq_len, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    return x, modality_mapping


def test_unbacked_symbol_guard_error():
    """The original view(k.shape[0], NUM_HEADS, -1) MUST raise GuardOnDataDependentSymNode."""
    torch._dynamo.reset()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BuggyModel().to(device)
    compiled = torch.compile(model, dynamic=True, fullgraph=False)

    x, mm = _make_inputs(150, [100, 30, 20], device)
    with pytest.raises(torch._inductor.exc.InductorError, match="GuardOnDataDependentSymNode"):
        compiled(x, mm)


def test_unbacked_symbol_guard_fixed():
    """The fixed unsqueeze(-1) version must succeed for multiple dynamic shapes."""
    torch._dynamo.reset()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FixedModel().to(device)
    compiled = torch.compile(model, dynamic=True, fullgraph=False)

    x1, mm1 = _make_inputs(150, [100, 30, 20], device)
    out1 = compiled(x1, mm1)
    assert out1.shape == ()

    x2, mm2 = _make_inputs(200, [120, 50, 30], device)
    out2 = compiled(x2, mm2)
    assert out2.shape == ()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
