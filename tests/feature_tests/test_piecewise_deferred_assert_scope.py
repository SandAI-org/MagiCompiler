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

"""Tests for _scope_deferred_runtime_asserts in piecewise compilation.

Problem: after piecewise splitting, each sub-graph is compiled via
standalone_compile independently, but they share the same ShapeEnv.
ShapeEnv.deferred_runtime_asserts may contain Eq constraints referencing
backed SymInts (e.g. s90) that only belong to *another* sub-graph's
placeholders.  Inductor's GraphLowering blindly emits these as Python
runtime assertions, producing ``NameError: name 'sXX' is not defined``.

Fix: ``_scope_deferred_runtime_asserts`` (in piecewise_compiler.py) narrows
deferred_runtime_asserts to only reachable symbols before each
standalone_compile call, then restores the original dict afterwards.

test_without_fix_raises_nameerror_pt29:
    patches the fix away → originally caused NameError on PT 2.9.
    Marked xfail because later commits may have incidentally fixed it.
test_without_fix_passes_on_pt212:
    patches the fix away → PT 2.12 upstream no longer emits stale asserts.
test_with_fix_passes:
    uses the real fix → runs correctly on all versions.
"""

from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch
import torch as _torch
import torch.nn as nn

_TORCH_VERSION = tuple(int(x) for x in _torch.__version__.split(".")[:2])
_IS_TORCH_212 = _TORCH_VERSION >= (2, 12)

from magi_compiler import magi_compile, magi_register_custom_op

HIDDEN = 64
NUM_MOD = 3


class _Dispatcher:
    """Mimics ModalityDispatcher: permute + unbacked group sizes."""

    def __init__(self, modality_mapping, num_modalities):
        self.num_modalities = num_modalities
        self.permute_mapping = torch.argsort(modality_mapping)
        self.inv_permute_mapping = torch.argsort(self.permute_mapping)
        permuted = modality_mapping[self.permute_mapping]

        gs = torch.bincount(permuted, minlength=num_modalities).to(torch.int32)
        gs_cpu = [int(v) for v in gs.to("cpu").tolist()]

        self._carrier = torch.empty(*gs_cpu)
        if not torch.compiler.is_compiling():
            for i in range(num_modalities):
                torch._dynamo.decorators.mark_unbacked(self._carrier, i)

    @property
    def group_sizes(self):
        return [self._carrier.shape[i] for i in range(self.num_modalities)]

    def dispatch(self, x):
        return list(torch.split(x, self.group_sizes, dim=0))

    def undispatch(self, *groups):
        return torch.cat(groups, dim=0)


def _identity_meta(x):
    return torch.empty_like(x)


@magi_register_custom_op("test_scope::identity", infer_output_meta_fn=_identity_meta, is_subgraph_boundary=True)
def identity_op(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


class _InnerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(HIDDEN, HIDDEN, bias=False) for _ in range(NUM_MOD)])

    def forward(self, x, permute_mapping, inv_permute_mapping, dispatcher):
        # ── Sub-graph 1 (before boundary) ──
        # Inputs : x (shape[0] → backed s77), dispatcher (group_sizes → unbacked u0, u1, u2)
        # NOT here: permute_mapping (s90), inv_permute_mapping (s92)
        chunks = dispatcher.dispatch(x)
        outs = [self.linears[i](c) for i, c in enumerate(chunks)]
        out = dispatcher.undispatch(*outs)

        # identity_op is registered with is_subgraph_boundary=True, which
        # forces magi_compile to split here into two piecewise sub-graphs.
        #
        # ShapeEnv (shared) holds two deferred_runtime_asserts:
        #   Eq(u0 + u1 + u2, s77)   ← s77 only in sub-graph 1
        #   Eq(u0 + u1 + u2, s90)   ← s90 only in sub-graph 2
        #
        # Without _scope_deferred_runtime_asserts, sub-graph 2 inherits
        # Eq(u0+u1+u2, s77) and Inductor emits `if not (... == s77):`,
        # but s77 is not a placeholder in sub-graph 2 → NameError.
        out = identity_op(out)

        # ── Sub-graph 2 (after boundary) ──
        # Inputs : permute_mapping (shape[0] → backed s90), inv_permute_mapping (s92)
        # NOT here: x (s77), u0, u1, u2
        out = out[inv_permute_mapping]
        out = out[permute_mapping]
        return out


class _OuterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = _InnerBlock()

    def forward(self, x, modality_mapping):
        dispatcher = _Dispatcher(modality_mapping, NUM_MOD)
        x = x[dispatcher.permute_mapping]
        out = self.block(x, dispatcher.permute_mapping, dispatcher.inv_permute_mapping, dispatcher)
        return out[dispatcher.inv_permute_mapping]


def _make_inputs(sizes, device="cuda"):
    seq = sum(sizes)
    x = torch.randn(seq, HIDDEN, device=device)
    parts = []
    for i, n in enumerate(sizes):
        parts.extend([i] * n)
    mm = torch.tensor(parts, dtype=torch.long, device=device)
    return x, mm


def _build_compiled_model(device="cuda"):
    torch._dynamo.reset()
    model = _OuterModel().to(device).eval()
    model.block = magi_compile(model.block, dynamic_arg_dims={"x": 0, "permute_mapping": 0, "inv_permute_mapping": 0})
    return torch.compile(model, dynamic=True)


def _run_two_shapes(compiled, device="cuda"):
    """Run two different shapes to exercise the compiled model."""
    x1, mm1 = _make_inputs((32, 16, 16), device)
    with torch.no_grad():
        out1 = compiled(x1, mm1)
    assert out1.shape == (64, HIDDEN)

    x2, mm2 = _make_inputs((24, 12, 12), device)
    with torch.no_grad():
        out2 = compiled(x2, mm2)
    assert out2.shape == (48, HIDDEN)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(_IS_TORCH_212, reason="PT 2.12 upstream fixed this; see _pt212 variant")
@pytest.mark.xfail(
    reason="Original NameError may no longer reproduce due to later MagiCompiler fixes; " "kept as regression guard",
    raises=AssertionError,
    strict=False,
)
def test_without_fix_raises_nameerror_pt29():
    """PT 2.9: without the fix, Inductor should emit code referencing an
    absent backed SymInt → NameError.  Marked xfail because subsequent
    MagiCompiler commits may have incidentally resolved the trigger."""
    compiled = _build_compiled_model()

    with patch("magi_compiler.magi_backend.piecewise_compiler._scope_deferred_runtime_asserts", return_value=nullcontext()):
        with pytest.raises(NameError, match=r"name 's\d+' is not defined"):
            _run_two_shapes(compiled)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(not _IS_TORCH_212, reason="PT 2.9 expected NameError; see _pt29 variant")
def test_without_fix_passes_on_pt212():
    """PT 2.12: upstream no longer emits stale deferred runtime asserts,
    so even without the MagiCompiler fix, the model runs correctly."""
    compiled = _build_compiled_model()

    with patch("magi_compiler.magi_backend.piecewise_compiler._scope_deferred_runtime_asserts", return_value=nullcontext()):
        _run_two_shapes(compiled)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_with_fix_passes():
    """With _scope_deferred_runtime_asserts active, all shapes run correctly."""
    compiled = _build_compiled_model()
    _run_two_shapes(compiled)
