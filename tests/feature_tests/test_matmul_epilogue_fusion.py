# Copyright (c) 2025 SandAI. All Rights Reserved.
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

"""Tests for the CUTLASS Sm80EVT matmul-epilogue fusion path on RTX 5090.

Three families of checks:

  1. Positive numerical equivalence: every supported epilogue (the 7 athena
     activations + binary ops + 1-D bias) must match eager within bf16 tol.
  2. Fusion-actually-fired: the emitted graph must contain a
     ``magi_epilogue.matmul_custom_evt`` node — a green numerical test alone
     would silently pass even if fusion was skipped (eager == "compiled").
  3. Negative fallback: shapes / dtypes / chains the EVT pass does NOT
     support must keep the original ``aten.mm`` and run through cuBLAS.
     Catches over-eager fusion that would corrupt downstream consumers.
"""

from typing import Optional

import pytest
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F

from magi_compiler.api import magi_compile
from magi_compiler.config import get_compile_config

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

_SM120_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 12,
    reason="CUTLASS EVT path targets sm_120 (Blackwell consumer)",
)

_SM90_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="SM90 multi-AuxLoad EVT path targets Hopper (H100)",
)


# ── Activations from athena/performer_v16/activation.py (verbatim) ────────────


def high_precision_silu(x, out_dtype: Optional[torch.dtype] = None):
    out_dtype = x.dtype if out_dtype is None else out_dtype
    return F.silu(x.to(torch.float32)).to(out_dtype)


def high_precision_sigmoid(x, out_dtype: Optional[torch.dtype] = None):
    out_dtype = x.dtype if out_dtype is None else out_dtype
    return F.sigmoid(x.to(torch.float32)).to(out_dtype)


def high_precision_gelu(x, out_dtype: Optional[torch.dtype] = None):
    out_dtype = x.dtype if out_dtype is None else out_dtype
    return F.gelu(x.to(torch.float32)).to(out_dtype)


def swiglu(x, alpha: float = 1.702, limit: float = 7.0, out_dtype: Optional[torch.dtype] = None):
    out_dtype = x.dtype if out_dtype is None else out_dtype
    x = x.to(torch.float32)
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return (out_glu * (x_linear + 1)).to(out_dtype)


def gelu7(x, alpha: float = 1.702, limit: float = 7.0, out_dtype: Optional[torch.dtype] = None):
    out_dtype = x.dtype if out_dtype is None else out_dtype
    x = x.to(torch.float32)
    x_glu = x.clamp(min=None, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return out_glu.to(out_dtype)


def relu_square(x, out_dtype: Optional[torch.dtype] = None):
    out_dtype = x.dtype if out_dtype is None else out_dtype
    return torch.square(F.relu(x.to(torch.float32))).to(out_dtype)


# ── Compile + fusion-side instrumentation ────────────────────────────────────


class _FusionStats:
    """Records what the EVT pass did to the graph during one ``magi_compile``.

    Captured by patching ``MatmulEvtEpilogueFusionPass.__call__`` for the scope
    of a test. We track:
      * mm_before    — count of ``aten.mm`` nodes seen on entry
      * mm_after     — same after the pass
      * fused_count  — number of ``magi_epilogue.matmul_custom_evt`` nodes
                       inserted (i.e. how many mm sites the pass actually
                       replaced; ``mm_before - mm_after`` only matches when
                       fusion never aborts mid-walk).
      * kinds        — the ``kind`` arg of each emitted op, e.g.
                       ["evt_row", "swiglu_dual"].

    Tests assert against these to prove the pass made the right choice — a
    purely numerical comparison against eager would silently pass even when
    fusion was skipped (because both paths fall back to cuBLAS).
    """

    def __init__(self) -> None:
        self.mm_before = 0
        self.mm_after = 0
        self.fused_count = 0
        self.kinds: list = []
        # out_dtype_id of each emitted op (args[6]). Encoded as
        #   bf16 → 0, fp16 → 1, fp32 → 2 (see evt_runtime._OUT_DTYPE_ID).
        # Tests assert against this to catch silent dtype regressions in the
        # FX pass's last-node meta lookup or codegen's ElementC typedef.
        self.out_dtype_ids: list = []
        # ir_json strings (args[3]) of each emitted op. Used to verify
        # per-node compute_dtype propagation through the walker.
        self.ir_jsons: list = []


def _install_pass_instrument():
    """Returns (stats, restore_fn). Wraps the FX pass to record per-call deltas."""
    from magi_compiler.passes.piecewise_graph.fusion import matmul_epilogue_fusion as P

    stats = _FusionStats()
    original = P.MatmulEvtEpilogueFusionPass.__call__
    evt_op = torch.ops.magi_epilogue.matmul_custom_evt.default
    mm_targets = (torch.ops.aten.mm.default, torch.ops.aten.mm)

    def _instrumented(self, graph: fx.Graph):
        before = sum(1 for n in graph.nodes if n.op == "call_function" and n.target in mm_targets)
        result = original(self, graph)
        after = sum(1 for n in graph.nodes if n.op == "call_function" and n.target in mm_targets)
        emitted_kinds = []
        emitted_out_dtype_ids = []
        emitted_ir_jsons = []
        for n in graph.nodes:
            if n.op == "call_function" and n.target is evt_op:
                # signature: (A, B, extras, ir_json, kind, n_out, out_dtype_id)
                if len(n.args) >= 4:
                    emitted_ir_jsons.append(n.args[3])
                if len(n.args) >= 5:
                    emitted_kinds.append(n.args[4])
                if len(n.args) >= 7:
                    emitted_out_dtype_ids.append(n.args[6])
        stats.mm_before += before
        stats.mm_after += after
        stats.fused_count += len(emitted_kinds)
        stats.kinds.extend(emitted_kinds)
        stats.out_dtype_ids.extend(emitted_out_dtype_ids)
        stats.ir_jsons.extend(emitted_ir_jsons)
        return result

    P.MatmulEvtEpilogueFusionPass.__call__ = _instrumented

    def restore():
        P.MatmulEvtEpilogueFusionPass.__call__ = original

    return stats, restore


def _compile_and_check(
    model: nn.Module,
    inputs,
    *,
    atol: float = 0.5,
    rtol: float = 0.0,
    expect_fused: int = -1,
    expect_kinds: Optional[list] = None,
    expect_out_dtype: Optional[torch.dtype] = None,
    expect_actual_dtype: Optional[torch.dtype] = None,
    dynamic_arg_dims=None,
    cast_model_to_bf16: bool = True,
):
    """Compile ``model``, run it on ``inputs``, compare against eager.

    Parameters
    ----------
    model, inputs
        ``inputs`` is a tuple/list passed positionally to forward.
    atol, rtol
        Numerical tolerance: ``|actual - expected| <= atol + rtol*|expected|``.
    expect_fused
        Number of mm sites the pass MUST have replaced. Use 0 for negative
        tests (fusion must NOT fire). -1 disables the check.
    expect_kinds
        If set, the multiset of emitted op ``kind`` args must equal this list.
        E.g. ``["swiglu_dual"]`` for the swiglu special-case path.
    expect_out_dtype
        If set, every emitted op's ``out_dtype_id`` (args[6]) MUST decode to
        this dtype. Catches silent regressions where the FX pass picks the
        wrong terminal-node dtype, or where Inductor inserts an extra cast
        that the IR walker wasn't expecting.
    expect_actual_dtype
        If set, the runtime result tensor MUST have this dtype. Independent
        check from ``expect_out_dtype`` — they should agree but a mismatch
        between them would mean the codegen's StoreD typedef diverged from
        the op's declared out_dtype_id.
    dynamic_arg_dims
        Forwarded to magi_compile. Defaults to making the first arg's M
        dynamic (matches our fusion guards).
    cast_model_to_bf16
        Default True (mirrors the standard test setup). Pass False when the
        model already has the dtype mix you want (e.g. fp16-only or mixed
        bf16 / fp16 weights).
    """
    if dynamic_arg_dims is None:
        # Use the model's forward signature to pick the first arg name.
        import inspect

        params = list(inspect.signature(model.forward).parameters)
        if not params:
            dynamic_arg_dims = {}
        else:
            dynamic_arg_dims = {params[0]: 0}

    model = model.cuda()
    # Use bfloat16 by default so the EVT pass actually fires (the pass
    # requires bf16/fp16). Skip the auto-cast for tests that explicitly
    # set up a different dtype mix.
    if cast_model_to_bf16 and any(p.dtype.is_floating_point for p in model.parameters()):
        model = model.bfloat16()
    # Disable gradients on parameters; otherwise magi_compile / aot_autograd
    # produces a forward+backward joint graph and the mm node has an extra
    # user (the saved tensor for backward), which the EVT escape detector
    # correctly refuses to fuse.
    for p in model.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        expected = model(*inputs)

    get_compile_config().disable_cache = True
    stats, restore = _install_pass_instrument()
    try:
        compiled_model = magi_compile(model, dynamic_arg_dims=dynamic_arg_dims)
        with torch.no_grad():
            actual = compiled_model(*inputs)
    finally:
        restore()

    # Numerical check.
    abs_diff = (actual - expected).abs()
    tol = atol + rtol * expected.abs()
    max_violation = (abs_diff - tol).max().item()
    assert max_violation <= 0, (
        f"Fused result outside tolerance: "
        f"max(|diff| - tol) = {max_violation:.4f}, "
        f"max |diff| = {abs_diff.max().item():.4f}, "
        f"fusion stats: fused={stats.fused_count} kinds={stats.kinds}"
    )

    # Fusion-actually-fired check.
    if expect_fused >= 0:
        assert stats.fused_count == expect_fused, (
            f"Expected {expect_fused} fused mm sites, got {stats.fused_count}. "
            f"mm_before={stats.mm_before} mm_after={stats.mm_after} "
            f"emitted kinds={stats.kinds}"
        )
    if expect_kinds is not None:
        assert sorted(stats.kinds) == sorted(expect_kinds), (
            f"Expected emitted kinds {sorted(expect_kinds)}, " f"got {sorted(stats.kinds)}"
        )
    if expect_out_dtype is not None:
        from magi_compiler.passes.piecewise_graph.fusion.evt_runtime import out_dtype_from_id

        assert stats.out_dtype_ids, (
            f"expect_out_dtype={expect_out_dtype} but no fusion fired " f"(out_dtype_ids list is empty)"
        )
        decoded = [out_dtype_from_id(i) for i in stats.out_dtype_ids]
        for got in decoded:
            assert got == expect_out_dtype, (
                f"Emitted out_dtype mismatch: expected {expect_out_dtype}, " f"got {got} (full list: {decoded})"
            )
    if expect_actual_dtype is not None:
        assert actual.dtype == expect_actual_dtype, (
            f"Runtime result dtype mismatch: expected {expect_actual_dtype}, " f"got {actual.dtype}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Positive tests — every athena activation must fuse and stay numerically OK
# ─────────────────────────────────────────────────────────────────────────────


class _Bf16MmModel(nn.Module):
    """All positive activation models share this skeleton: bf16 mm followed
    by an epilogue fn that returns bf16. Weight is held in (N, K) row-major
    form and accessed via ``permute([1, 0])`` to mirror the real GAGA2 graph."""

    def __init__(self, k: int, n: int, epilogue):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n, k))
        self._epi = epilogue

    def forward(self, a):
        y = torch.mm(a, self.weight.permute(1, 0))
        return self._epi(y, out_dtype=torch.bfloat16)


_M, _K, _N = 1024, 1024, 1024


def _input_a():
    return torch.randn(_M, _K, device="cuda", dtype=torch.bfloat16)


@_SM120_ONLY
@pytest.mark.parametrize(
    "epi_name,epi_fn,atol,rtol",
    [
        ("silu", high_precision_silu, 0.5, 0.0),
        ("sigmoid", high_precision_sigmoid, 0.5, 0.0),
        ("gelu", high_precision_gelu, 0.5, 0.0),
        ("gelu7", gelu7, 0.5, 0.0),
        ("relu_square", relu_square, 0.0, 0.2),
    ],
)
def test_evt_unary_activations_fuse(epi_name, epi_fn, atol, rtol):
    """All unary activations must fuse to a single ``evt_col`` op."""
    model = _Bf16MmModel(_K, _N, epi_fn)
    _compile_and_check(model, (_input_a(),), atol=atol, rtol=rtol, expect_fused=1, expect_kinds=["evt_col"])


@_SM120_ONLY
def test_evt_relu_native():
    """Plain ``aten.relu`` (no fp32 cast) — exercises the built-in CUTLASS
    ReLu functor mapping in the IR."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(_N, _K))

        def forward(self, a):
            return torch.relu(torch.mm(a, self.weight.permute(1, 0))).to(torch.bfloat16)

    _compile_and_check(M(), (_input_a(),), expect_fused=1, expect_kinds=["evt_col"])


@_SM120_ONLY
def test_evt_swiglu_dispatches_to_dualgemm():
    """SwiGLU7 must take the dedicated DualGemm one-stage path, not generic EVT."""
    model = _Bf16MmModel(_K, _N, swiglu)
    _compile_and_check(model, (_input_a(),), atol=0.5, rtol=0.05, expect_fused=1, expect_kinds=["swiglu_dual"])


@_SM120_ONLY
def test_evt_swiglu_custom_constants():
    """SwiGLU7 with non-default alpha/limit/one still fuses and computes correctly."""

    def swiglu_custom(x, out_dtype=None):
        out_dtype = x.dtype if out_dtype is None else out_dtype
        x = x.to(torch.float32)
        x_glu, x_linear = x[..., ::2], x[..., 1::2]
        x_glu = x_glu.clamp(max=5.0)
        x_linear = x_linear.clamp(min=-5.0, max=5.0)
        out_glu = x_glu * torch.sigmoid(2.0 * x_glu)
        return (out_glu * (x_linear + 1)).to(out_dtype)

    model = _Bf16MmModel(_K, _N, swiglu_custom)
    _compile_and_check(model, (_input_a(),), atol=0.5, rtol=0.05, expect_fused=1, expect_kinds=["swiglu_dual"])


@_SM120_ONLY
def test_evt_swiglu_constants_roundtrip_in_ir_json():
    """Verify that swiglu constant values are captured in ir_json."""
    import json as _json

    def swiglu_custom(x, out_dtype=None):
        out_dtype = x.dtype if out_dtype is None else out_dtype
        x = x.to(torch.float32)
        x_glu, x_linear = x[..., ::2], x[..., 1::2]
        x_glu = x_glu.clamp(max=3.0)
        x_linear = x_linear.clamp(min=-3.0, max=3.0)
        out_glu = x_glu * torch.sigmoid(1.5 * x_glu)
        return (out_glu * (x_linear + 1)).to(out_dtype)

    model = _Bf16MmModel(_K, _N, swiglu_custom).cuda().bfloat16()
    for p in model.parameters():
        p.requires_grad_(False)

    a = _input_a()
    with torch.no_grad():
        expected = model(a)

    get_compile_config().disable_cache = True
    stats, restore = _install_pass_instrument()
    try:
        compiled = magi_compile(model, dynamic_arg_dims={"a": 0})
        with torch.no_grad():
            actual = compiled(a)
    finally:
        restore()

    diff = (actual.float() - expected.float()).abs().max().item()
    assert diff <= 0.5, f"swiglu custom constants max|diff|={diff}"

    assert stats.fused_count == 1
    assert stats.kinds == ["swiglu_dual"]
    assert len(stats.ir_jsons) == 1
    sw7 = _json.loads(stats.ir_jsons[0])
    assert sw7["alpha"] == 1.5, f"Expected alpha=1.5, got {sw7['alpha']}"
    assert sw7["limit"] == 3.0, f"Expected limit=3.0, got {sw7['limit']}"
    assert sw7["one"] == 1.0, f"Expected one=1.0, got {sw7['one']}"


@_SM90_ONLY
def test_evt_sm90_swiglu_custom_constants():
    """SM90: SwiGLU7 with non-default alpha/limit still fuses correctly."""

    def swiglu_custom(x, out_dtype=None):
        out_dtype = x.dtype if out_dtype is None else out_dtype
        x = x.to(torch.float32)
        x_glu, x_linear = x[..., ::2], x[..., 1::2]
        x_glu = x_glu.clamp(max=5.0)
        x_linear = x_linear.clamp(min=-5.0, max=5.0)
        out_glu = x_glu * torch.sigmoid(2.0 * x_glu)
        return (out_glu * (x_linear + 1)).to(out_dtype)

    model = _Bf16MmModel(_K, _N, swiglu_custom)
    _compile_and_check(model, (_input_a(),), atol=0.5, rtol=0.05, expect_fused=1, expect_kinds=["swiglu_dual"])


@_SM90_ONLY
def test_evt_sm90_swiglu_constants_roundtrip_in_ir_json():
    """SM90: Verify that swiglu constant values are captured in ir_json."""
    import json as _json

    def swiglu_custom(x, out_dtype=None):
        out_dtype = x.dtype if out_dtype is None else out_dtype
        x = x.to(torch.float32)
        x_glu, x_linear = x[..., ::2], x[..., 1::2]
        x_glu = x_glu.clamp(max=3.0)
        x_linear = x_linear.clamp(min=-3.0, max=3.0)
        out_glu = x_glu * torch.sigmoid(1.5 * x_glu)
        return (out_glu * (x_linear + 1)).to(out_dtype)

    model = _Bf16MmModel(_K, _N, swiglu_custom).cuda().bfloat16()
    for p in model.parameters():
        p.requires_grad_(False)

    a = _input_a()
    with torch.no_grad():
        expected = model(a)

    get_compile_config().disable_cache = True
    stats, restore = _install_pass_instrument()
    try:
        compiled = magi_compile(model, dynamic_arg_dims={"a": 0})
        with torch.no_grad():
            actual = compiled(a)
    finally:
        restore()

    diff = (actual.float() - expected.float()).abs().max().item()
    assert diff <= 0.5, f"SM90 swiglu custom constants max|diff|={diff}"

    assert stats.fused_count == 1
    assert stats.kinds == ["swiglu_dual"]
    assert len(stats.ir_jsons) == 1
    sw7 = _json.loads(stats.ir_jsons[0])
    assert sw7["alpha"] == 1.5, f"Expected alpha=1.5, got {sw7['alpha']}"
    assert sw7["limit"] == 3.0, f"Expected limit=3.0, got {sw7['limit']}"
    assert sw7["one"] == 1.0, f"Expected one=1.0, got {sw7['one']}"


# ─────────────────────────────────────────────────────────────────────────────
# Binary-op positive tests — chains containing add/sub/mul/div on the mm output
# ─────────────────────────────────────────────────────────────────────────────


@_SM120_ONLY
def test_evt_mm_plus_scalar():
    """``mm + 0.5`` — scalar add absorbs into ``add_scalar`` IR node.

    Tolerance: eager runs the add in bf16 (lossy ulp at ±0.5); CUTLASS runs
    the add in fp32 then casts. The ~1.0 absolute diff observed is bf16
    rounding noise on the eager side, not a CUTLASS bug.
    """

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(_N, _K))

        def forward(self, a):
            return (torch.mm(a, self.weight.permute(1, 0)) + 0.5).to(torch.bfloat16)

    _compile_and_check(M(), (_input_a(),), atol=1.5, expect_fused=1, expect_kinds=["evt_col"])


@_SM120_ONLY
def test_evt_mm_times_scalar():
    """``mm * 0.25`` — scalar mul (mul_scalar IR)."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(_N, _K))

        def forward(self, a):
            return (torch.mm(a, self.weight.permute(1, 0)) * 0.25).to(torch.bfloat16)

    _compile_and_check(M(), (_input_a(),), expect_fused=1, expect_kinds=["evt_col"])


@_SM120_ONLY
def test_evt_mm_div_scalar_then_silu():
    """``silu(mm / 8)`` — scalar div + activation chain."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(_N, _K))

        def forward(self, a):
            y = torch.mm(a, self.weight.permute(1, 0)) / 8.0
            return high_precision_silu(y, out_dtype=torch.bfloat16)

    _compile_and_check(M(), (_input_a(),), expect_fused=1, expect_kinds=["evt_col"])


@_SM120_ONLY
def test_evt_mm_minus_scalar_then_relu():
    """``relu(mm - 2.0)``."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(_N, _K))

        def forward(self, a):
            y = torch.mm(a, self.weight.permute(1, 0)) - 2.0
            return torch.relu(y).to(torch.bfloat16)

    _compile_and_check(M(), (_input_a(),), expect_fused=1, expect_kinds=["evt_col"])


@_SM120_ONLY
def test_evt_mm_plus_1d_bias():
    """``silu(mm + bias_N)`` — 1-D bias as RowBroadcast extras."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(_N, _K))
            self.bias = nn.Parameter(torch.randn(_N))

        def forward(self, a):
            y = torch.mm(a, self.weight.permute(1, 0)) + self.bias
            return high_precision_silu(y, out_dtype=torch.bfloat16)

    # atol=1.5: eager does the bias-add in bf16 (lossy), CUTLASS in fp32 —
    # the ~1.0 abs diff is bf16 ulp noise on the eager side.
    _compile_and_check(M(), (_input_a(),), atol=1.5, expect_fused=1, expect_kinds=["evt_col"])


@_SM120_ONLY
def test_evt_mm_times_aux_load():
    """``(mm * gate_MxN)`` — full (M, N) auxiliary tensor multiply.

    The gate must be supplied as a regular forward arg (not a model parameter)
    because magi_compile doesn't trace through Parameters of dynamic shape.
    """

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(_N, _K))

        def forward(self, a, gate):
            y = torch.mm(a, self.weight.permute(1, 0)) * gate
            return y.to(torch.bfloat16)

    a = _input_a()
    gate = torch.randn(_M, _N, device="cuda", dtype=torch.bfloat16)
    _compile_and_check(
        M(), (a, gate), atol=0.0, rtol=0.1, expect_fused=1, expect_kinds=["evt_col"], dynamic_arg_dims={"a": 0, "gate": 0}
    )


# ─────────────────────────────────────────────────────────────────────────────
# Negative tests — fusion must NOT fire and the chain must fall back to cuBLAS
# ─────────────────────────────────────────────────────────────────────────────


@_SM120_ONLY
def test_evt_no_fuse_intermediate_escapes():
    """Attention → residual → RMSNorm pattern: ``add(residual, mm)`` is
    consumed both by ``square(...)`` (would-be-fused) AND by ``mul(_, rsqrt)``
    later. The pass MUST refuse — fusing would silently drop the value the
    rest of RMSNorm needs."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(5120, _K))
            self.gamma = nn.Parameter(torch.randn(5120))

        def forward(self, a, residual):
            y = torch.mm(a, self.weight.permute(1, 0)).float()
            x = residual + y
            var = x.pow(2).mean(-1, keepdim=True)
            rsqrt = torch.rsqrt(var + 1e-6)
            return (x * rsqrt * (self.gamma + 1)).to(torch.bfloat16)

    a = _input_a()
    residual = torch.randn(_M, 5120, device="cuda", dtype=torch.float32)
    # `residual + y` couples a's M to residual's M; mark both dynamic so
    # Dynamo doesn't specialize a's declared dynamic dim → ConstraintViolation.
    _compile_and_check(M(), (a, residual), atol=2.0, rtol=0.1, expect_fused=0, dynamic_arg_dims={"a": 0, "residual": 0})


@_SM120_ONLY
def test_evt_no_fuse_bare_mm():
    """A bare ``mm`` with no epilogue at all — Store(Accum) is trivial.
    Replacing cuBLAS with a CUTLASS GEMM that does identical work is strictly
    slower, so the pass must skip."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(_N, _K))

        def forward(self, a):
            return torch.mm(a, self.weight.permute(1, 0))

    _compile_and_check(M(), (_input_a(),), atol=0.5, expect_fused=0)


@_SM120_ONLY
def test_evt_no_fuse_k_misaligned():
    """K not divisible by 8 fails the bf16 alignment guard — cuBLAS path."""

    class M(nn.Module):
        def __init__(self, k, n):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(n, k))

        def forward(self, a):
            y = torch.mm(a, self.weight.permute(1, 0))
            return high_precision_silu(y, out_dtype=torch.bfloat16)

    K = 1023  # 1023 % 8 = 7 → should NOT fuse
    N = 1024
    a = torch.randn(_M, K, device="cuda", dtype=torch.bfloat16)
    _compile_and_check(M(K, N), (a,), expect_fused=0)


@_SM120_ONLY
def test_evt_col_n_misaligned_still_fuses():
    """N=1026 is not 128-bit aligned for bf16 but the runtime pads the
    output stride to a 128-byte boundary, so fusion should still fire."""

    class M(nn.Module):
        def __init__(self, k, n):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(n, k))

        def forward(self, a):
            y = torch.mm(a, self.weight.permute(1, 0))
            return high_precision_silu(y, out_dtype=torch.bfloat16)

    K = 1024
    N = 1026
    a = torch.randn(_M, K, device="cuda", dtype=torch.bfloat16)
    _compile_and_check(M(K, N), (a,), expect_fused=1)


@_SM120_ONLY
def test_evt_swiglu_small_n_still_fuses():
    """N=12: n_out=6 is not 128-bit aligned for bf16 but the runtime pads
    the output stride, so swiglu fusion should still fire."""

    class M(nn.Module):
        def __init__(self, k, n):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(n, k))

        def forward(self, a):
            y = torch.mm(a, self.weight.permute(1, 0))
            return swiglu(y, out_dtype=torch.bfloat16)

    K = 1024
    N = 12
    a = torch.randn(_M, K, device="cuda", dtype=torch.bfloat16)
    _compile_and_check(M(K, N), (a,), expect_fused=1)


# ─────────────────────────────────────────────────────────────────────────────
# IR / cache key invariants
# ─────────────────────────────────────────────────────────────────────────────


@_SM120_ONLY
def test_evt_ir_canonical_determinism():
    """Same IR built twice → identical canonical JSON. If this regresses, the
    .cu module disk cache silently misses and recompiles every run."""
    from magi_compiler.passes.piecewise_graph.fusion.evt_ir import Accum, Compute, Store, cache_key, to_canonical_json

    a = Store(Compute("silu", (Compute("add", (Accum(), Accum())),)), "bfloat16")
    b = Store(Compute("silu", (Compute("add", (Accum(), Accum())),)), "bfloat16")
    assert to_canonical_json(a) == to_canonical_json(b)
    assert cache_key(a, "bfloat16", "bfloat16") == cache_key(b, "bfloat16", "bfloat16")


# ─────────────────────────────────────────────────────────────────────────────
# out_dtype correctness — verify the EVT pass picks the right Store dtype +
# the codegen's ElementC matches + the runtime returns a tensor of that dtype.
#
# Matrix:
#   input dtype | epilogue compute | output dtype | expected out_dtype_id
#   ─────────────────────────────────────────────────────────────────────
#   bf16        | bf16             | bf16         | 0                   (default)
#   bf16        | fp32             | bf16         | 0                   (high_precision_silu)
#   bf16        | fp32             | fp32         | 2                   (no final cast)
#   bf16        | bf16             | fp16         | 1                   (cross-precision)
#   fp16        | fp16             | fp16         | 1                   (fp16-only path)
#   fp32 input  | —                | —            | not fused (negative)
# ─────────────────────────────────────────────────────────────────────────────


@_SM120_ONLY
def test_evt_out_dtype_bf16_native():
    """bf16 mm → bf16 silu → bf16 output (no fp32 promotion). Pure-bf16 chain.
    out_dtype_id MUST be 0 (bf16) and the runtime tensor MUST be bf16."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(_N, _K))

        def forward(self, a):
            return F.silu(torch.mm(a, self.weight.permute(1, 0)))  # bf16 → bf16

    _compile_and_check(
        M(),
        (_input_a(),),
        expect_fused=1,
        expect_kinds=["evt_col"],
        expect_out_dtype=torch.bfloat16,
        expect_actual_dtype=torch.bfloat16,
    )


@_SM120_ONLY
def test_evt_out_dtype_bf16_via_high_precision():
    """The athena ``high_precision_silu`` pattern: bf16 → cast(fp32) → silu →
    cast(bf16). The IR walker absorbs both casts; final output is bf16 even
    though the compute went through fp32 internally.

    This is the most common athena pattern — a regression here means the
    inner-cast handling broke and out_dtype is silently wrong."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(_N, _K))

        def forward(self, a):
            y = torch.mm(a, self.weight.permute(1, 0))
            return high_precision_silu(y, out_dtype=torch.bfloat16)

    _compile_and_check(
        M(),
        (_input_a(),),
        expect_fused=1,
        expect_kinds=["evt_col"],
        expect_out_dtype=torch.bfloat16,
        expect_actual_dtype=torch.bfloat16,
    )


@_SM120_ONLY
def test_evt_out_dtype_fp32_no_final_cast():
    """bf16 mm → fp32 cast → silu → keep fp32 (no final cast back).

    out_dtype_id MUST be 2 (fp32). Exercises codegen's ``ElementC = float``
    path + the runtime D allocator with fp32 row-stride alignment (4 elements
    = 16 bytes — different vector size than bf16's 8 bytes).
    """

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(_N, _K))

        def forward(self, a):
            y = torch.mm(a, self.weight.permute(1, 0)).float()
            return F.silu(y)  # stays fp32

    _compile_and_check(
        M(),
        (_input_a(),),
        expect_fused=1,
        expect_kinds=["evt_col"],
        expect_out_dtype=torch.float32,
        expect_actual_dtype=torch.float32,
    )


@_SM120_ONLY
def test_evt_out_dtype_bf16_to_fp16():
    """bf16 mm → silu → cast(fp16). Cross-precision: bf16 inputs but fp16
    output. out_dtype_id MUST be 1 (fp16). Exercises the codegen's
    ``ElementA = bfloat16_t`` + ``ElementC = half_t`` mixed instantiation."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(_N, _K))

        def forward(self, a):
            return F.silu(torch.mm(a, self.weight.permute(1, 0))).half()

    _compile_and_check(
        M(),
        (_input_a(),),
        atol=0.5,
        expect_fused=1,
        expect_kinds=["evt_col"],
        expect_out_dtype=torch.float16,
        expect_actual_dtype=torch.float16,
    )


@_SM120_ONLY
def test_evt_out_dtype_fp16_native():
    """fp16 mm + fp16 silu → fp16 output. Pure-fp16 path — exercises the
    pass's bf16/fp16 branch in the input-dtype check, plus the codegen's
    ``cutlass::half_t`` ElementA/B/C path end-to-end."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(_N, _K))

        def forward(self, a):
            return F.silu(torch.mm(a, self.weight.permute(1, 0)))  # fp16 → fp16

    a = torch.randn(_M, _K, device="cuda", dtype=torch.float16)
    # Cast model to fp16 (not bf16) so all parameters match A's dtype.
    model = M().cuda().half()
    for p in model.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        expected = model(a)

    get_compile_config().disable_cache = True
    stats, restore = _install_pass_instrument()
    try:
        compiled = magi_compile(model, dynamic_arg_dims={"a": 0})
        with torch.no_grad():
            actual = compiled(a)
    finally:
        restore()

    diff = (actual.float() - expected.float()).abs().max().item()
    assert diff <= 0.5, f"fp16 silu max|diff|={diff}"
    assert stats.fused_count == 1, f"fp16 path should fuse but got fused_count={stats.fused_count}"
    assert stats.kinds == ["evt_col"], stats.kinds
    assert stats.out_dtype_ids == [1], f"Expected out_dtype_id=[1] (fp16), got {stats.out_dtype_ids}"
    assert actual.dtype == torch.float16, actual.dtype


@_SM120_ONLY
def test_evt_no_fuse_fp32_mm():
    """fp32 mm — pass requires bf16 (or fp16); fp32 must skip."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(_N, _K))

        def forward(self, a):
            y = torch.mm(a, self.weight.permute(1, 0))
            return F.silu(y)

    a = torch.randn(_M, _K, device="cuda", dtype=torch.float32)

    model = M().cuda()  # fp32 — do NOT bfloat16() the model
    with torch.no_grad():
        expected = model(a)

    get_compile_config().disable_cache = True
    stats, restore = _install_pass_instrument()
    try:
        compiled_model = magi_compile(model, dynamic_arg_dims={"a": 0})
        with torch.no_grad():
            actual = compiled_model(a)
    finally:
        restore()

    diff = (actual - expected).abs().max().item()
    assert diff <= 1.0, f"fp32 mm result diverged: {diff}"
    assert stats.fused_count == 0, (
        f"fp32 mm should NOT fuse, but pass emitted {stats.fused_count} ops " f"(kinds={stats.kinds})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SM90 AuxLoad — all AuxLoad nodes use ``Sm90AuxLoad<0>`` (inline ld.global,
# no SMEM staging). The C-operand TMA channel is left unused. Tests below
# exercise single and multi-AuxLoad paths on H100.
# ─────────────────────────────────────────────────────────────────────────────


@_SM90_ONLY
def test_evt_sm90_single_aux_load_fuse():
    """``(mm * gate)`` — single (M, N) auxiliary via Sm90AuxLoad<0> (ld.global).

    We use ``*`` instead of ``+`` because Inductor folds ``mm + tensor`` into
    ``aten.addmm`` (which the EVT pass doesn't recognise), but ``mm * tensor``
    stays as separate mm + mul nodes.
    """

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(_N, _K))

        def forward(self, a, gate):
            y = torch.mm(a, self.weight.permute(1, 0)) * gate
            return y.to(torch.bfloat16)

    a = _input_a()
    gate = torch.randn(_M, _N, device="cuda", dtype=torch.bfloat16)
    _compile_and_check(
        M(), (a, gate), atol=0.0, rtol=0.1, expect_fused=1, expect_kinds=["evt_col"], dynamic_arg_dims={"a": 0, "gate": 0}
    )


@_SM90_ONLY
def test_evt_sm90_two_aux_loads_fuse():
    """``(mm + R1 + R2)`` — two (M, N) residuals fuse into one EVT op.

    Both AuxLoad nodes use Sm90AuxLoad<0> (inline ld.global). Validates the
    multi-AuxLoad path end-to-end: the kernel compiles, runs, and matches
    eager within bf16 tolerance.
    """

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(_N, _K))

        def forward(self, a, r1, r2):
            y = torch.mm(a, self.weight.permute(1, 0)) + r1 + r2
            return y.to(torch.bfloat16)

    a = _input_a()
    r1 = torch.randn(_M, _N, device="cuda", dtype=torch.bfloat16)
    r2 = torch.randn(_M, _N, device="cuda", dtype=torch.bfloat16)
    _compile_and_check(
        M(),
        (a, r1, r2),
        atol=2.0,
        rtol=0.05,
        expect_fused=1,
        expect_kinds=["evt_col"],
        dynamic_arg_dims={"a": 0, "r1": 0, "r2": 0},
    )


@_SM90_ONLY
def test_evt_sm90_three_aux_loads_fuse():
    """``(mm + R1 + R2 + R3)`` — three (M, N) residuals.

    All three AuxLoad nodes use Sm90AuxLoad<0> (inline ld.global). Confirms
    ≥3 aux can compile / run on the SM90 path.
    """

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(_N, _K))

        def forward(self, a, r1, r2, r3):
            y = torch.mm(a, self.weight.permute(1, 0)) + r1 + r2 + r3
            return y.to(torch.bfloat16)

    a = _input_a()
    r1 = torch.randn(_M, _N, device="cuda", dtype=torch.bfloat16)
    r2 = torch.randn(_M, _N, device="cuda", dtype=torch.bfloat16)
    r3 = torch.randn(_M, _N, device="cuda", dtype=torch.bfloat16)
    _compile_and_check(
        M(),
        (a, r1, r2, r3),
        atol=3.0,
        rtol=0.05,
        expect_fused=1,
        expect_kinds=["evt_col"],
        dynamic_arg_dims={"a": 0, "r1": 0, "r2": 0, "r3": 0},
    )


# ── can_render unit tests — exercise the SM90 gate directly, no GPU needed ────


def test_can_render_accepts_multi_aux():
    """SM90 ``can_render`` accepts IR trees with multiple AuxLoad nodes
    (one per distinct input_idx). This is the constraint we relaxed.
    """
    from magi_compiler.passes.piecewise_graph.fusion.evt_ir import Accum, AuxLoad, Compute, Store
    from magi_compiler.passes.piecewise_graph.fusion.sm90.evt_codegen import can_render

    # D = (acc + R1) + R2
    ir = Store(
        child=Compute(
            op="add",
            children=(
                Compute(op="add", children=(Accum(), AuxLoad(input_idx=0, dtype="bfloat16"))),
                AuxLoad(input_idx=1, dtype="bfloat16"),
            ),
        ),
        out_dtype="bfloat16",
    )
    assert can_render(ir) is True

    # Single AuxLoad still works (preserved single-aux path).
    ir_one = Store(child=Compute(op="add", children=(Accum(), AuxLoad(input_idx=0, dtype="bfloat16"))), out_dtype="bfloat16")
    assert can_render(ir_one) is True

    # 3 distinct AuxLoad — confirm ≥3 isn't capped.
    ir_three = Store(
        child=Compute(
            op="add",
            children=(
                Compute(
                    op="add",
                    children=(
                        Compute(op="add", children=(Accum(), AuxLoad(input_idx=0, dtype="bfloat16"))),
                        AuxLoad(input_idx=1, dtype="bfloat16"),
                    ),
                ),
                AuxLoad(input_idx=2, dtype="bfloat16"),
            ),
        ),
        out_dtype="bfloat16",
    )
    assert can_render(ir_three) is True


def test_can_render_rejects_repeated_aux_idx():
    """Same external tensor (same input_idx) reused at multiple AuxLoad
    positions in the IR is rejected — the SM90 codegen's leaf_args dict is
    keyed by input_idx and would clash. FX pass falls back to Inductor lower
    for such cases.
    """
    from magi_compiler.passes.piecewise_graph.fusion.evt_ir import Accum, AuxLoad, Compute, Store
    from magi_compiler.passes.piecewise_graph.fusion.sm90.evt_codegen import can_render

    # D = (acc * gate) + gate  — same AuxLoad(input_idx=0) appears twice.
    ir_dup = Store(
        child=Compute(
            op="add",
            children=(
                Compute(op="mul", children=(Accum(), AuxLoad(input_idx=0, dtype="bfloat16"))),
                AuxLoad(input_idx=0, dtype="bfloat16"),
            ),
        ),
        out_dtype="bfloat16",
    )
    assert can_render(ir_dup) is False


# ─────────────────────────────────────────────────────────────────────────────
# Per-node compute_dtype — verify the IR, walker, codegen, and end-to-end
# behaviour when type-conversion ops (to(fp32), to(bf16)) change the compute
# precision of subsequent fused ops.
# ─────────────────────────────────────────────────────────────────────────────


def test_evt_ir_compute_dtype_roundtrip():
    """Compute with non-default compute_dtype serialises and round-trips."""
    import json

    from magi_compiler.passes.piecewise_graph.fusion.evt_ir import Accum, Compute, Store, to_canonical_json
    from magi_compiler.passes.piecewise_graph.fusion.evt_runtime import _ir_from_json

    # bf16 compute_dtype → must appear in JSON
    ir_bf16 = Store(Compute("silu", (Accum(),), compute_dtype="bfloat16"), "bfloat16")
    j_bf16 = to_canonical_json(ir_bf16)
    parsed = json.loads(j_bf16)
    assert parsed["child"]["compute_dtype"] == "bfloat16"

    # Default fp32 → must NOT appear in JSON (backward compat)
    ir_default = Store(Compute("silu", (Accum(),)), "bfloat16")
    j_default = to_canonical_json(ir_default)
    assert "compute_dtype" not in j_default

    # Round-trip: bf16 survives
    restored = _ir_from_json(j_bf16)
    assert restored.child.compute_dtype == "bfloat16"

    # Round-trip: old JSON without compute_dtype → defaults to fp32
    restored_default = _ir_from_json(j_default)
    assert restored_default.child.compute_dtype == "float32"

    # Mixed chain: two Compute nodes with different compute_dtype
    ir_mixed = Store(
        Compute(
            "add",
            (Compute("silu", (Accum(),), compute_dtype="float32"), Compute("neg", (Accum(),), compute_dtype="bfloat16")),
            compute_dtype="bfloat16",
        ),
        "bfloat16",
    )
    j_mixed = to_canonical_json(ir_mixed)
    p = json.loads(j_mixed)
    # root add → bfloat16
    assert p["child"]["compute_dtype"] == "bfloat16"
    # silu child → float32 (default, NOT in JSON)
    silu_child = p["child"]["children"][0]
    assert "compute_dtype" not in silu_child
    # neg child → bfloat16
    neg_child = p["child"]["children"][1]
    assert neg_child["compute_dtype"] == "bfloat16"


def test_evt_ir_compute_dtype_cache_key_differs():
    """Same op tree with different compute_dtype MUST produce different cache keys."""
    from magi_compiler.passes.piecewise_graph.fusion.evt_ir import Accum, Compute, Store, to_canonical_json

    ir_fp32 = Store(Compute("silu", (Accum(),), compute_dtype="float32"), "bfloat16")
    ir_bf16 = Store(Compute("silu", (Accum(),), compute_dtype="bfloat16"), "bfloat16")
    assert to_canonical_json(ir_fp32) != to_canonical_json(ir_bf16)


def test_evt_ir_compute_dtype_valid_types():
    """All hardware-supported floating-point ALU types are accepted as compute_dtype.

    H100 (sm_90) and RTX 5090 (sm_120) natively support FP32, FP16, BF16 at
    full ALU speed. FP64 is full-speed on H100 but extremely slow on 5090;
    INT64/32/16/8 are ALU-supported but CUTLASS VisitorCompute only templates
    over floating-point. The EVT path therefore restricts compute_dtype to
    {float32, float16, bfloat16}.
    """
    from magi_compiler.passes.piecewise_graph.fusion.evt_ir import Accum, Compute

    # These must all succeed without raising.
    for dt in ("float32", "float16", "bfloat16"):
        node = Compute("silu", (Accum(),), compute_dtype=dt)
        assert node.compute_dtype == dt


def test_evt_ir_compute_dtype_rejects_unsupported():
    """compute_dtype values outside the CUTLASS-supported set must raise.

    FP64: full-speed on H100 but too slow on 5090 to be useful in epilogues.
    INT types (int8/16/32/64): hardware ALU supports them but CUTLASS
    VisitorCompute / Sm90Compute are floating-point-only templates.
    """
    from magi_compiler.passes.piecewise_graph.fusion.evt_ir import Accum, Compute

    for bad_dt in ("float64", "int8", "int16", "int32", "int64"):
        with pytest.raises(ValueError, match="Unsupported compute_dtype"):
            Compute("silu", (Accum(),), compute_dtype=bad_dt)


def test_evt_codegen_sm80_per_node_compute_dtype():
    """SM80 codegen emits per-node element types in VisitorCompute."""
    from magi_compiler.passes.piecewise_graph.fusion.evt_ir import Accum, Compute, Store
    from magi_compiler.passes.piecewise_graph.fusion.sm80.evt_codegen import render_evt_cu

    ir = Store(
        Compute(
            "add",
            (Compute("silu", (Accum(),), compute_dtype="float32"), Compute("neg", (Accum(),), compute_dtype="bfloat16")),
            compute_dtype="bfloat16",
        ),
        "bfloat16",
    )
    src = render_evt_cu(ir, "bfloat16", "bfloat16")

    # The silu node should use float, float (default)
    assert "VisitorCompute<" in src
    # The neg and add nodes should use cutlass::bfloat16_t
    assert "cutlass::bfloat16_t, cutlass::bfloat16_t" in src
    # The silu node should use float, float
    assert "float, float" in src


def test_evt_codegen_sm90_per_node_compute_dtype():
    """SM90 codegen emits per-node element types in Sm90Compute."""
    from magi_compiler.passes.piecewise_graph.fusion.evt_ir import Accum, Compute, Store
    from magi_compiler.passes.piecewise_graph.fusion.sm90.evt_codegen import can_render, render_evt_cu

    ir = Store(
        Compute(
            "add",
            (Compute("silu", (Accum(),), compute_dtype="float32"), Compute("neg", (Accum(),), compute_dtype="bfloat16")),
            compute_dtype="bfloat16",
        ),
        "bfloat16",
    )
    assert can_render(ir) is True
    src = render_evt_cu(ir, "bfloat16", "bfloat16")

    assert "Sm90Compute<" in src
    # bfloat16_t appears in at least one Sm90Compute (neg and add nodes)
    assert "cutlass::bfloat16_t, cutlass::bfloat16_t" in src
    # float appears in at least one Sm90Compute (silu node)
    assert "float, float" in src


def _parse_ir_compute_dtypes(ir_json_str: str) -> list:
    """Extract all compute_dtype values from Compute nodes in an IR JSON string."""
    import json

    dtypes = []

    def _walk(d):
        if not isinstance(d, dict):
            return
        if d.get("kind") == "compute":
            dtypes.append(d.get("compute_dtype", "float32"))
            for c in d.get("children", []):
                _walk(c)
        elif d.get("kind") == "store":
            _walk(d.get("child"))

    _walk(json.loads(ir_json_str))
    return dtypes


@_SM120_ONLY
def test_evt_mixed_compute_dtype_chain():
    """mm → to(fp32) → silu → to(bf16) → add_scalar(0.5).

    silu must have compute_dtype=float32 (fp32 region).
    add_scalar must have compute_dtype=bfloat16 (bf16 region after cast).
    Verifies: (1) fusion fires, (2) IR carries correct per-node dtypes,
    (3) numerical result matches eager.
    """

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(_N, _K))

        def forward(self, a):
            y = torch.mm(a, self.weight.permute(1, 0))
            y = y.float()
            y = F.silu(y)
            y = y.bfloat16()
            y = y + 0.5
            return y

    model = M().cuda().bfloat16()
    for p in model.parameters():
        p.requires_grad_(False)
    a = _input_a()

    with torch.no_grad():
        expected = model(a)

    get_compile_config().disable_cache = True
    stats, restore = _install_pass_instrument()
    try:
        compiled = magi_compile(model, dynamic_arg_dims={"a": 0})
        with torch.no_grad():
            actual = compiled(a)
    finally:
        restore()

    # Numerical check
    diff = (actual.float() - expected.float()).abs().max().item()
    assert diff <= 1.5, f"Mixed compute_dtype chain max|diff|={diff}"

    # Fusion must have fired
    assert stats.fused_count == 1, f"Expected 1 fusion, got {stats.fused_count}"

    # Verify per-node compute_dtype in the emitted IR
    assert len(stats.ir_jsons) == 1, f"Expected 1 ir_json, got {len(stats.ir_jsons)}"
    compute_dtypes = _parse_ir_compute_dtypes(stats.ir_jsons[0])
    assert "bfloat16" in compute_dtypes, f"Expected at least one bfloat16 compute_dtype in IR, " f"got {compute_dtypes}"
    assert "float32" in compute_dtypes, f"Expected at least one float32 compute_dtype in IR, " f"got {compute_dtypes}"


@_SM120_ONLY
def test_evt_default_compute_dtype_stays_fp32():
    """mm → silu (no explicit cast) → to(bf16).

    Without an explicit to(fp32) or to(bf16) before the silu, the walker's
    current_compute_dtype stays at its default "float32" (the GEMM accumulator
    precision). The silu Compute node must have compute_dtype=float32.
    """

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(_N, _K))

        def forward(self, a):
            y = torch.mm(a, self.weight.permute(1, 0))
            return F.silu(y).to(torch.bfloat16)

    model = M().cuda().bfloat16()
    for p in model.parameters():
        p.requires_grad_(False)
    a = _input_a()

    with torch.no_grad():
        expected = model(a)

    get_compile_config().disable_cache = True
    stats, restore = _install_pass_instrument()
    try:
        compiled = magi_compile(model, dynamic_arg_dims={"a": 0})
        with torch.no_grad():
            actual = compiled(a)
    finally:
        restore()

    diff = (actual.float() - expected.float()).abs().max().item()
    assert diff <= 0.5, f"Default fp32 compute_dtype chain max|diff|={diff}"
    assert stats.fused_count == 1, f"Expected 1 fusion, got {stats.fused_count}"

    # All Compute nodes should be float32 (default — no cast in chain)
    assert len(stats.ir_jsons) == 1
    compute_dtypes = _parse_ir_compute_dtypes(stats.ir_jsons[0])
    assert all(dt == "float32" for dt in compute_dtypes), f"Expected all compute_dtype=float32 (no cast), got {compute_dtypes}"


@_SM90_ONLY
def test_evt_sm90_mixed_compute_dtype_chain():
    """SM90 variant of the mixed compute_dtype chain test.

    mm → to(fp32) → silu → to(bf16) → add_scalar(0.5).
    Same assertions as the SM120 test but exercises the Sm90Compute codegen path.
    """

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(_N, _K))

        def forward(self, a):
            y = torch.mm(a, self.weight.permute(1, 0))
            y = y.float()
            y = F.silu(y)
            y = y.bfloat16()
            y = y + 0.5
            return y

    model = M().cuda().bfloat16()
    for p in model.parameters():
        p.requires_grad_(False)
    a = _input_a()

    with torch.no_grad():
        expected = model(a)

    get_compile_config().disable_cache = True
    stats, restore = _install_pass_instrument()
    try:
        compiled = magi_compile(model, dynamic_arg_dims={"a": 0})
        with torch.no_grad():
            actual = compiled(a)
    finally:
        restore()

    diff = (actual.float() - expected.float()).abs().max().item()
    assert diff <= 1.5, f"SM90 mixed compute_dtype chain max|diff|={diff}"
    assert stats.fused_count == 1, f"Expected 1 fusion, got {stats.fused_count}"

    assert len(stats.ir_jsons) == 1
    compute_dtypes = _parse_ir_compute_dtypes(stats.ir_jsons[0])
    assert "bfloat16" in compute_dtypes, f"Expected at least one bfloat16 compute_dtype in IR, " f"got {compute_dtypes}"
    assert "float32" in compute_dtypes, f"Expected at least one float32 compute_dtype in IR, " f"got {compute_dtypes}"


# ─────────────────────────────────────────────────────────────────────────────
# SM90 unary activation + scalar / bias tests — parity with SM120 positive
# tests, exercising the TMA-based Sm90EVT codegen + runtime end-to-end.
# ─────────────────────────────────────────────────────────────────────────────


@_SM90_ONLY
@pytest.mark.parametrize(
    "epi_name,epi_fn,atol,rtol",
    [
        ("silu", high_precision_silu, 0.5, 0.0),
        ("sigmoid", high_precision_sigmoid, 0.5, 0.0),
        ("gelu", high_precision_gelu, 0.5, 0.0),
        ("gelu7", gelu7, 0.5, 0.0),
        ("relu_square", relu_square, 0.0, 0.2),
    ],
)
def test_evt_sm90_unary_activations_fuse(epi_name, epi_fn, atol, rtol):
    """SM90: all unary activations must fuse and match eager."""
    model = _Bf16MmModel(_K, _N, epi_fn)
    _compile_and_check(model, (_input_a(),), atol=atol, rtol=rtol, expect_fused=1, expect_kinds=["evt_col"])


@_SM90_ONLY
def test_evt_sm90_swiglu_dispatches_to_dualgemm():
    """SM90: SwiGLU7 must take the dedicated DualGemm path."""
    model = _Bf16MmModel(_K, _N, swiglu)
    _compile_and_check(model, (_input_a(),), atol=0.5, rtol=0.05, expect_fused=1, expect_kinds=["swiglu_dual"])


@_SM90_ONLY
def test_evt_sm90_mm_plus_scalar():
    """SM90: ``mm + 0.5`` scalar add."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(_N, _K))

        def forward(self, a):
            return (torch.mm(a, self.weight.permute(1, 0)) + 0.5).to(torch.bfloat16)

    _compile_and_check(M(), (_input_a(),), atol=1.5, expect_fused=1, expect_kinds=["evt_col"])


@_SM90_ONLY
def test_evt_sm90_mm_plus_1d_bias():
    """SM90: ``silu(mm + bias_N)`` — 1-D bias as RowBroadcast."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(_N, _K))
            self.bias = nn.Parameter(torch.randn(_N))

        def forward(self, a):
            y = torch.mm(a, self.weight.permute(1, 0)) + self.bias
            return high_precision_silu(y, out_dtype=torch.bfloat16)

    _compile_and_check(M(), (_input_a(),), atol=1.5, expect_fused=1, expect_kinds=["evt_col"])


# ─────────────────────────────────────────────────────────────────────────────
# SM90 D stride padding regression — exercises the fix where make_args() uses
# ea.ldd (= n_pad) instead of N for stride_D.  When N is not 128-byte aligned
# the runtime pads D to (M, n_pad) and passes the (M, N) slice; the TMA
# descriptor must use n_pad as the globalStride or every row after the first
# is written to the wrong offset.
# ─────────────────────────────────────────────────────────────────────────────


@_SM90_ONLY
def test_evt_sm90_d_stride_padding_silu():
    """SM90 D stride regression: N=1032 is not 128-byte aligned for bf16.

    Runtime pads D to n_pad=1088 (next 64-element boundary for bf16).
    Before the fix, stride_D was built from N instead of ldd,
    corrupting every row after the first.
    N must be a multiple of 8 so Inductor doesn't pad the weight.
    """
    K = 1024
    N = 1032

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(N, K))

        def forward(self, a):
            y = torch.mm(a, self.weight.permute(1, 0))
            return high_precision_silu(y, out_dtype=torch.bfloat16)

    a = torch.randn(_M, K, device="cuda", dtype=torch.bfloat16)
    _compile_and_check(M(), (a,), atol=0.5, expect_fused=1, expect_kinds=["evt_col"])


@_SM90_ONLY
def test_evt_sm90_d_stride_padding_swiglu():
    """SM90 D stride regression for swiglu: N=1040, n_out=520.

    520 bf16 elements = 1040 bytes, not 128-byte aligned.
    Runtime pads to n_pad=576 (next 64-element boundary).
    N must be a multiple of 8 so Inductor doesn't pad the weight.
    """
    K = 1024
    N = 1040

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(N, K))

        def forward(self, a):
            y = torch.mm(a, self.weight.permute(1, 0))
            return swiglu(y, out_dtype=torch.bfloat16)

    a = torch.randn(_M, K, device="cuda", dtype=torch.bfloat16)
    _compile_and_check(M(), (a,), atol=0.5, rtol=0.05, expect_fused=1, expect_kinds=["swiglu_dual"])


@_SM90_ONLY
def test_evt_sm90_d_stride_padding_add_scalar():
    """SM90 D stride regression: N=200 (not 128-byte aligned for bf16).

    200 bf16 elements = 400 bytes. Runtime pads to n_pad=256 (512 bytes).
    Exercises the stride mismatch (ldd=256 vs N=200) on a scalar-add chain.
    """
    K = 1024
    N = 200

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(N, K))

        def forward(self, a):
            return (torch.mm(a, self.weight.permute(1, 0)) + 0.5).to(torch.bfloat16)

    a = torch.randn(_M, K, device="cuda", dtype=torch.bfloat16)
    _compile_and_check(M(), (a,), atol=1.5, expect_fused=1, expect_kinds=["evt_col"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
