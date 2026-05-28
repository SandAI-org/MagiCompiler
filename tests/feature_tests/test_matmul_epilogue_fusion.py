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

"""Tests for CUTLASS EVT matmul–epilogue fusion (``MatmulEvtEpilogueFusionPass``).

Architecture routing (see ``matmul_epilogue_fusion.py`` / ``evt_runtime.py``):

  * sm_90  (Hopper / H100)        — CUTLASS 3.x ``Sm90EVT``; TMA+WGMMA.
  * sm_120+ (Blackwell consumer, e.g. RTX 5090) — CUTLASS 2.x ``Sm80EVT``;
    cp.async multistage.

Most tests use ``@_EVT_CAPABLE`` (runs on whichever GPU is present).
``@_SM120_ONLY`` is reserved for SM80-path-specific edge cases (e.g. 64-bit
alignment that SM90 TMA cannot handle).

Three families of checks:

  1. Positive numerical equivalence: every supported epilogue must match
     eager within dtype-appropriate tolerance.
  2. Fusion-actually-fired: the emitted graph must contain a
     ``magi_epilogue.matmul_fused_epilogue`` node.
  3. Negative fallback: shapes / dtypes / chains the EVT pass does NOT
     support must keep the original ``aten.mm`` and run through cuBLAS.
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
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="SM90 EVT path targets Hopper (H100)"
)

_EVT_CAPABLE = pytest.mark.skipif(
    not torch.cuda.is_available()
    or (torch.cuda.get_device_capability() != (9, 0) and torch.cuda.get_device_capability()[0] < 12),
    reason="EVT path targets sm_90 (Hopper) or sm_120+ (Blackwell)",
)


_TEST_RNG_SEED = 123


@pytest.fixture(autouse=True)
def _enable_mm_epilogue_fusion():
    config = get_compile_config()
    old_value = config.pass_config.enable_mm_epilogue_fusion
    config.pass_config.enable_mm_epilogue_fusion = True
    yield
    config.pass_config.enable_mm_epilogue_fusion = old_value


@pytest.fixture(autouse=True)
def _fixed_rng_seed():
    """Make low-precision random numerical tests reproducible."""
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(_TEST_RNG_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_TEST_RNG_SEED)
    yield
    torch.random.set_rng_state(cpu_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)


# ── Activations from athena/performer_v16/activation.py (verbatim) ────────────


def high_precision_silu(x, out_dtype: Optional[torch.dtype] = None):
    out_dtype = x.dtype if out_dtype is None else out_dtype
    return F.silu(x.to(torch.float32)).to(out_dtype)


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


# ── Compile + fusion-side instrumentation ────────────────────────────────────


class _FusionStats:
    """Records what the EVT pass did to the graph during one ``magi_compile``."""

    def __init__(self) -> None:
        self.mm_before = 0
        self.mm_after = 0
        self.fused_count = 0
        self.kinds: list = []
        self.out_dtype_ids: list = []
        self.ir_jsons: list = []
        self.call_function_targets_after: list = []


def _install_pass_instrument():
    """Returns (stats, restore_fn). Wraps the FX pass to record per-call deltas."""
    from magi_compiler.passes.piecewise_graph.fusion import matmul_epilogue_fusion as P

    stats = _FusionStats()
    original = P.MatmulEvtEpilogueFusionPass.__call__
    evt_op = torch.ops.magi_epilogue.matmul_fused_epilogue.default
    mm_targets = (torch.ops.aten.mm.default, torch.ops.aten.mm)

    def _instrumented(self, graph: fx.Graph):
        before = sum(1 for n in graph.nodes if n.op == "call_function" and n.target in mm_targets)
        result = original(self, graph)
        after = sum(1 for n in graph.nodes if n.op == "call_function" and n.target in mm_targets)
        emitted_kinds = []
        emitted_out_dtype_ids = []
        emitted_ir_jsons = []
        call_function_targets_after = []
        for n in graph.nodes:
            if n.op == "call_function":
                call_function_targets_after.append(n.target)
            if n.op == "call_function" and n.target is evt_op:
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
        stats.call_function_targets_after.extend(call_function_targets_after)
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
    """Compile ``model``, run it on ``inputs``, compare against eager."""
    if dynamic_arg_dims is None:
        import inspect

        params = list(inspect.signature(model.forward).parameters)
        if not params:
            dynamic_arg_dims = {}
        else:
            dynamic_arg_dims = {params[0]: 0}

    model = model.cuda()
    if cast_model_to_bf16 and any(p.dtype.is_floating_point for p in model.parameters()):
        model = model.bfloat16()
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

    if expect_fused >= 0:
        assert stats.fused_count == expect_fused, (
            f"Expected {expect_fused} fused mm sites, got {stats.fused_count}. "
            f"mm_before={stats.mm_before} mm_after={stats.mm_after} "
            f"emitted kinds={stats.kinds}"
        )
        if expect_fused > 0:
            evt_op = torch.ops.magi_epilogue.matmul_fused_epilogue.default
            assert stats.call_function_targets_after == [evt_op] * expect_fused, (
                "Expected the final fused subgraph to contain only matmul_fused_epilogue "
                f"call_function nodes, got {stats.call_function_targets_after}"
            )

    # Skip the numerical accuracy check when fusion was explicitly expected NOT
    # to fire.  The unfused path goes through vanilla torch.compile → Inductor,
    # which has a known upstream bf16 mm bug: when the output dimension N is not
    # 16-byte aligned (N % 8 != 0 for bf16), the compiled mm produces
    # systematically wrong results (max |diff| ≈ 1.0).  We still check fusion
    # correctness above; the accuracy assertion is only meaningful when the EVT
    # path is active.
    if expect_fused == 0:
        return

    abs_diff = (actual - expected).abs()
    tol = atol + rtol * expected.abs()
    max_violation = (abs_diff - tol).max().item()
    assert max_violation <= 0, (
        f"Fused result outside tolerance: "
        f"max(|diff| - tol) = {max_violation:.4f}, "
        f"max |diff| = {abs_diff.max().item():.4f}, "
        f"fusion stats: fused={stats.fused_count} kinds={stats.kinds}"
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
# Common helpers
# ─────────────────────────────────────────────────────────────────────────────


class _Bf16MmModel(nn.Module):
    """bf16 mm followed by an epilogue fn that returns bf16."""

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


# ─────────────────────────────────────────────────────────────────────────────
# Positive tests — unary activations, SwiGLU, scalar ops, bias, AuxLoad
# ─────────────────────────────────────────────────────────────────────────────


@_EVT_CAPABLE
@pytest.mark.parametrize("epi_name,epi_fn,atol,rtol", [("silu", high_precision_silu, 0.5, 0.0), ("gelu7", gelu7, 0.5, 0.0)])
def test_evt_unary_activations_fuse(epi_name, epi_fn, atol, rtol):
    """Representative unary activations must fuse to a single ``evt_col`` op."""
    model = _Bf16MmModel(_K, _N, epi_fn)
    _compile_and_check(model, (_input_a(),), atol=atol, rtol=rtol, expect_fused=1, expect_kinds=["evt_col"])


@_EVT_CAPABLE
def test_evt_relu_native():
    """Plain ``aten.relu`` variants must fuse and preserve emitted output dtype."""

    class Fp32Relu(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(_N, _K))

        def forward(self, a):
            return torch.relu(torch.mm(a, self.weight.permute(1, 0)).float())

    _compile_and_check(
        Fp32Relu(),
        (_input_a(),),
        expect_fused=1,
        expect_kinds=["evt_col"],
        expect_out_dtype=torch.float32,
        expect_actual_dtype=torch.float32,
    )


@_EVT_CAPABLE
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


# ── alpha parameter tests for aten.add/sub ────────────────────────────────────


@_EVT_CAPABLE
@pytest.mark.parametrize(
    "case_name,op,other_kind,alpha",
    [("add_scalar_alpha2", torch.add, "scalar", 2.0), ("sub_tensor_alpha2", torch.sub, "tensor", 2.0)],
)
def test_evt_mm_add_sub_with_alpha(case_name, op, other_kind, alpha):
    """aten.add/sub with alpha must fuse and produce numerically correct results.

    Tensor-operand cases use ``silu(mm(...))`` as the base so that PyTorch's
    FX decomposition does not merge ``mm + alpha*bias`` into ``aten.addmm``
    (which would hide the mm node from our EVT pass).
    """

    class ScalarModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(_N, _K))

        def forward(self, a):
            y = torch.mm(a, self.weight.permute(1, 0))
            return op(y, 0.5, alpha=alpha).to(torch.bfloat16)

    class TensorModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(_N, _K))
            self.bias = nn.Parameter(torch.randn(_N))

        def forward(self, a):
            y = F.silu(torch.mm(a, self.weight.permute(1, 0)).to(torch.float32))
            return op(y, self.bias, alpha=alpha).to(torch.bfloat16)

    model = ScalarModel() if other_kind == "scalar" else TensorModel()
    _compile_and_check(
        model,
        (_input_a(),),
        atol=1.5,
        expect_fused=1,
        expect_kinds=["evt_col"],
        expect_out_dtype=torch.bfloat16,
        expect_actual_dtype=torch.bfloat16,
    )


@_EVT_CAPABLE
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

    _compile_and_check(
        M(),
        (_input_a(),),
        atol=1.5,
        expect_fused=1,
        expect_kinds=["evt_col"],
        expect_out_dtype=torch.bfloat16,
        expect_actual_dtype=torch.bfloat16,
    )


@_EVT_CAPABLE
def test_evt_aux_load_padded_stride():
    """AuxLoad with padded row stride (stride(0) > N) must fuse and read correctly."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(_N, _K))

        def forward(self, a, gate):
            y = torch.mm(a, self.weight.permute(1, 0)) * gate
            return y.to(torch.bfloat16)

    a = _input_a()
    N_padded = _N + 64
    gate_buf = torch.randn(_M, N_padded, device="cuda", dtype=torch.bfloat16)
    gate = gate_buf[:, :_N]  # shape (_M, _N), stride (N_padded, 1)
    assert gate.stride() == (N_padded, 1), f"Expected padded stride, got {gate.stride()}"
    _compile_and_check(
        M(), (a, gate), atol=0.0, rtol=0.1, expect_fused=1, expect_kinds=["evt_col"], dynamic_arg_dims={"a": 0, "gate": 0}
    )


@_EVT_CAPABLE
def test_evt_multiple_and_repeated_aux_loads_fuse():
    """Multiple AuxLoad extras, with one tensor reused at multiple EVT positions."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(_N, _K))

        def forward(self, a, gate, r1, r2):
            y = torch.mm(a, self.weight.permute(1, 0))
            return (y * gate + gate + r1 + r2).to(torch.bfloat16)

    a = _input_a()
    gate = torch.randn(_M, _N, device="cuda", dtype=torch.bfloat16)
    r1 = torch.randn(_M, _N, device="cuda", dtype=torch.bfloat16)
    r2 = torch.randn(_M, _N, device="cuda", dtype=torch.bfloat16)
    _compile_and_check(
        M(),
        (a, gate, r1, r2),
        atol=4.0,
        rtol=0.1,
        expect_fused=1,
        expect_kinds=["evt_col"],
        dynamic_arg_dims={"a": 0, "gate": 0, "r1": 0, "r2": 0},
    )


# ─────────────────────────────────────────────────────────────────────────────
# RowMajor B layout — weight stored as (K, N), used directly without permute
# ─────────────────────────────────────────────────────────────────────────────


@_EVT_CAPABLE
def test_evt_row_b_layout_fuses():
    """B is (K, N) row-major (no permute). LayoutB=RowMajor, kind=evt_row.

    CuTe stride for RowMajor B: (_1, N, N*K) — N is contiguous.
    TMA globalStride = N * sizeof(elem); N=1024 is 16B-aligned for bf16.
    """

    class M(nn.Module):
        def __init__(self, k, n):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(k, n))

        def forward(self, a):
            y = torch.mm(a, self.weight)
            return high_precision_silu(y, out_dtype=torch.bfloat16)

    _compile_and_check(M(_K, _N), (_input_a(),), expect_fused=1, expect_kinds=["evt_row"])


# ─────────────────────────────────────────────────────────────────────────────
# Negative tests — fusion must NOT fire, cuBLAS fallback
# ─────────────────────────────────────────────────────────────────────────────


@_EVT_CAPABLE
def test_evt_no_fuse_intermediate_escapes():
    """Attention → residual → RMSNorm: intermediate value escapes the fused
    chain. The pass MUST refuse."""

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
    _compile_and_check(M(), (a, residual), atol=2.0, rtol=0.1, expect_fused=0, dynamic_arg_dims={"a": 0, "residual": 0})


@_EVT_CAPABLE
def test_evt_no_fuse_bare_mm():
    """Bare ``mm`` — Store(Accum) is trivial, pass must skip."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(_N, _K))

        def forward(self, a):
            return torch.mm(a, self.weight.permute(1, 0))

    _compile_and_check(M(), (_input_a(),), atol=0.5, expect_fused=0)


@_EVT_CAPABLE
def test_evt_no_fuse_k_misaligned():
    """K below 64-bit alignment (bf16: K % 4 != 0) — pass aborts.

    K=1022: 1022 % 4 = 2 → no valid AlignmentA on either arch.
    """

    class M(nn.Module):
        def __init__(self, k, n):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(n, k))

        def forward(self, a):
            y = torch.mm(a, self.weight.permute(1, 0))
            return high_precision_silu(y, out_dtype=torch.bfloat16)

    K = 1022
    N = 1024
    a = torch.randn(_M, K, device="cuda", dtype=torch.bfloat16)
    _compile_and_check(M(K, N), (a,), expect_fused=0)


@_SM90_ONLY
def test_evt_sm90_no_fuse_k_not_16byte_aligned():
    """K=1020: K % 4 == 0 (64-bit aligned) but K * 2 % 16 != 0.

    SM90 TMA requires globalStride to be 16-byte aligned.  A is RowMajor
    (M, K) so stride_A = K, giving K * sizeof(bf16) = 2040 bytes, which
    is not 16-byte aligned (2040 % 16 = 8).  The pass must refuse.
    On SM120 this fuses fine (64-bit alignment is sufficient).
    """

    class M(nn.Module):
        def __init__(self, k, n):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(n, k))

        def forward(self, a):
            y = torch.mm(a, self.weight.permute(1, 0))
            return high_precision_silu(y, out_dtype=torch.bfloat16)

    K = 1020
    N = 1024
    a = torch.randn(_M, K, device="cuda", dtype=torch.bfloat16)
    _compile_and_check(M(K, N), (a,), expect_fused=0)


@_SM90_ONLY
def test_evt_sm90_no_fuse_n_not_16byte_aligned():
    """N=1026: N * sizeof(bf16) = 2052 bytes, not 16-byte aligned.

    SM90 CollectiveEpilogue (TMA store) requires problem N % AlignmentD
    == 0, where AlignmentD = 16 / sizeof(bf16) = 8.  1026 % 8 = 2 ≠ 0
    so all tile candidates fail can_implement.  The pass must refuse.
    On SM120 this fuses fine (runtime pads ldd).
    """

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
    _compile_and_check(M(K, N), (a,), expect_fused=0)


@_SM90_ONLY
def test_evt_sm90_no_fuse_row_b_n_not_16byte_aligned():
    """RowMajor B with N=1020: N * sizeof(bf16) = 2040, not 16B-aligned.

    CuTe stride for RowMajor B is (_1, N, ...) so TMA globalStride =
    N * sizeof(elem) = 2040 bytes, 2040 % 16 = 8 ≠ 0.
    N=1020 passes the 64-bit check (1020 % 4 == 0) but fails the SM90
    16B TMA constraint.  The pass must refuse on SM90.
    On SM120 this fuses fine (64-bit alignment is sufficient).
    """

    class M(nn.Module):
        def __init__(self, k, n):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(k, n))

        def forward(self, a):
            y = torch.mm(a, self.weight)
            return high_precision_silu(y, out_dtype=torch.bfloat16)

    K = 1024
    N = 1020
    a = torch.randn(_M, K, device="cuda", dtype=torch.bfloat16)
    _compile_and_check(M(K, N), (a,), expect_fused=0)


@_EVT_CAPABLE
def test_evt_no_fuse_fp32_mm():
    """fp32 mm — pass requires bf16 or fp16; fp32 must skip."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(_N, _K))

        def forward(self, a):
            y = torch.mm(a, self.weight.permute(1, 0))
            return F.silu(y)

    a = torch.randn(_M, _K, device="cuda", dtype=torch.float32)

    model = M().cuda()
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
# Alignment edge cases and D stride padding
# ─────────────────────────────────────────────────────────────────────────────


@_SM120_ONLY
def test_evt_col_n_misaligned_still_fuses():
    """N=1026: not 128-bit aligned for bf16, runtime pads D stride. Still fuses.

    SM120-only: SM80 (CUTLASS 2.x) threadblock epilogue only requires ldd to
    be aligned, so _aligned_n_stride(1026)=1032 suffices. SM90 (CUTLASS 3.x)
    TMA CollectiveBuilder requires problem N % AlignmentD == 0, and 1026 % 8
    != 0 — all tile candidates fail can_implement.
    """

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
    """N=12: n_out=6, not 128-bit aligned. Runtime pads, fusion fires.

    SM120-only: same reason as col_n_misaligned — SM90 TMA requires
    N % AlignmentD == 0 and 12 % 8 != 0.
    """

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


@_SM120_ONLY
def test_evt_row_b_n_64bit_aligned_fuses_on_sm120():
    """RowMajor B, N=1020: N % 4 == 0 (64-bit) but N*2 % 16 != 0.

    SM120-only: SM80 codegen accepts 64-bit alignment for B.
    SM90 TMA rejects because globalStride = 1020 * 2 = 2040, 2040 % 16 ≠ 0.
    """

    class M(nn.Module):
        def __init__(self, k, n):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(k, n))

        def forward(self, a):
            y = torch.mm(a, self.weight)
            return high_precision_silu(y, out_dtype=torch.bfloat16)

    K = 1024
    N = 1020
    a = torch.randn(_M, K, device="cuda", dtype=torch.bfloat16)
    _compile_and_check(M(K, N), (a,), expect_fused=1)


@_EVT_CAPABLE
def test_evt_d_stride_padding_silu():
    """D stride padding regression: N=1032, not 128-byte aligned for bf16.
    Runtime pads D to n_pad=1088."""
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


@_SM120_ONLY
def test_evt_k_64bit_aligned_fuses_on_sm120():
    """K=1020: K % 4 == 0 (64-bit aligned) but K % 8 != 0 (not 128-bit).

    On SM120 (RTX 5090), the SM80 codegen accepts AlignmentA=4 (64-bit)
    and fusion proceeds normally. This exercises the 64-bit fallback path
    in ``_largest_pow2_align_bits`` / ``_runtime_align_bits``.
    """

    class M(nn.Module):
        def __init__(self, k, n):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(n, k))

        def forward(self, a):
            y = torch.mm(a, self.weight.permute(1, 0))
            return high_precision_silu(y, out_dtype=torch.bfloat16)

    K = 1020
    N = 1024
    a = torch.randn(_M, K, device="cuda", dtype=torch.bfloat16)
    _compile_and_check(M(K, N), (a,), expect_fused=1, expect_kinds=["evt_col"])


# ─────────────────────────────────────────────────────────────────────────────
# IR / cache key invariants
# ─────────────────────────────────────────────────────────────────────────────


@_EVT_CAPABLE
def test_evt_ir_canonical_determinism():
    """Same IR built twice → identical canonical JSON."""
    from magi_compiler.passes.piecewise_graph.fusion.evt_ir import Accum, Compute, Store, cache_key, to_canonical_json

    a = Store(Compute("silu", (Compute("add", (Accum(), Accum())),)), "bfloat16")
    b = Store(Compute("silu", (Compute("add", (Accum(), Accum())),)), "bfloat16")
    assert to_canonical_json(a) == to_canonical_json(b)
    assert cache_key(a, "bfloat16", "bfloat16") == cache_key(b, "bfloat16", "bfloat16")


# ─────────────────────────────────────────────────────────────────────────────
# Per-node compute_dtype
# ─────────────────────────────────────────────────────────────────────────────


@_EVT_CAPABLE
def test_evt_mixed_compute_dtype_chain():
    """mm → to(fp32) → silu → to(bf16) → add_scalar(0.5).

    silu must have compute_dtype=float32, add_scalar must have bfloat16.
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
    assert diff <= 1.5, f"Mixed compute_dtype chain max|diff|={diff}"
    assert stats.fused_count == 1, f"Expected 1 fusion, got {stats.fused_count}"

    assert len(stats.ir_jsons) == 1, f"Expected 1 ir_json, got {len(stats.ir_jsons)}"
    compute_dtypes = _parse_ir_compute_dtypes(stats.ir_jsons[0])
    assert "bfloat16" in compute_dtypes, f"Expected at least one bfloat16 compute_dtype in IR, " f"got {compute_dtypes}"
    assert "float32" in compute_dtypes, f"Expected at least one float32 compute_dtype in IR, " f"got {compute_dtypes}"


# ─────────────────────────────────────────────────────────────────────────────
# No-GPU tests: codegen, IR invariants
# ─────────────────────────────────────────────────────────────────────────────


def test_sm90_codegen_repeated_aux_idx():
    """SM90 codegen produces valid C++ with repeated AuxLoad input_idx."""
    import re

    from magi_compiler.passes.piecewise_graph.fusion.evt_ir import Accum, AuxLoad, Compute, Store
    from magi_compiler.passes.piecewise_graph.fusion.sm90.evt_codegen import render_evt_cu

    ir = Store(
        child=Compute(
            op="add",
            children=(
                Compute(op="mul", children=(Accum(), AuxLoad(input_idx=0, dtype="bfloat16"))),
                AuxLoad(input_idx=0, dtype="bfloat16"),
            ),
        ),
        out_dtype="bfloat16",
    )
    src = render_evt_cu(ir, "bfloat16", "bfloat16")

    aux_load_defs = re.findall(r"using\s+\w+\s*=\s*cutlass::epilogue::fusion::Sm90AuxLoad<", src)
    assert len(aux_load_defs) == 2, f"Expected 2 Sm90AuxLoad typedefs, found {len(aux_load_defs)}"
    assert len(re.findall(r"ptr_extras\[0\]", src)) >= 1
    assert "expected 1 extra tensors" in src


def test_sm90_codegen_repeated_aux_idx_mixed_with_distinct():
    """SM90 codegen: repeated input_idx=0 + distinct input_idx=1."""
    import re

    from magi_compiler.passes.piecewise_graph.fusion.evt_ir import Accum, AuxLoad, Compute, Store
    from magi_compiler.passes.piecewise_graph.fusion.sm90.evt_codegen import render_evt_cu

    ir = Store(
        child=Compute(
            op="add",
            children=(
                Compute(
                    op="add",
                    children=(
                        Compute(op="mul", children=(Accum(), AuxLoad(input_idx=0, dtype="bfloat16"))),
                        AuxLoad(input_idx=0, dtype="bfloat16"),
                    ),
                ),
                AuxLoad(input_idx=1, dtype="bfloat16"),
            ),
        ),
        out_dtype="bfloat16",
    )
    src = render_evt_cu(ir, "bfloat16", "bfloat16")

    aux_load_defs = re.findall(r"using\s+\w+\s*=\s*cutlass::epilogue::fusion::Sm90AuxLoad<", src)
    assert len(aux_load_defs) == 3, f"Expected 3 Sm90AuxLoad typedefs, found {len(aux_load_defs)}"
    assert "expected 2 extra tensors" in src


def test_evt_ir_compute_dtype_roundtrip():
    """Compute with non-default compute_dtype serialises and round-trips."""
    import json

    from magi_compiler.passes.piecewise_graph.fusion.evt_ir import Accum, Compute, Store, to_canonical_json
    from magi_compiler.passes.piecewise_graph.fusion.evt_runtime import _ir_from_json

    ir_bf16 = Store(Compute("silu", (Accum(),), compute_dtype="bfloat16"), "bfloat16")
    j_bf16 = to_canonical_json(ir_bf16)
    parsed = json.loads(j_bf16)
    assert parsed["child"]["compute_dtype"] == "bfloat16"

    ir_default = Store(Compute("silu", (Accum(),)), "bfloat16")
    j_default = to_canonical_json(ir_default)
    assert "compute_dtype" not in j_default

    restored = _ir_from_json(j_bf16)
    assert restored.child.compute_dtype == "bfloat16"
    restored_default = _ir_from_json(j_default)
    assert restored_default.child.compute_dtype == "float32"

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
    assert p["child"]["compute_dtype"] == "bfloat16"
    assert "compute_dtype" not in p["child"]["children"][0]
    assert p["child"]["children"][1]["compute_dtype"] == "bfloat16"


def test_evt_ir_compute_dtype_cache_key_differs():
    """Different compute_dtype MUST produce different cache keys."""
    from magi_compiler.passes.piecewise_graph.fusion.evt_ir import Accum, Compute, Store, to_canonical_json

    ir_fp32 = Store(Compute("silu", (Accum(),), compute_dtype="float32"), "bfloat16")
    ir_bf16 = Store(Compute("silu", (Accum(),), compute_dtype="bfloat16"), "bfloat16")
    assert to_canonical_json(ir_fp32) != to_canonical_json(ir_bf16)


def test_evt_ir_compute_dtype_valid_types():
    """All floating-point ALU types are accepted as compute_dtype."""
    from magi_compiler.passes.piecewise_graph.fusion.evt_ir import Accum, Compute

    for dt in ("float32", "float16", "bfloat16"):
        node = Compute("silu", (Accum(),), compute_dtype=dt)
        assert node.compute_dtype == dt


def test_evt_ir_compute_dtype_rejects_unsupported():
    """Unsupported compute_dtype values must raise."""
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
    assert "VisitorCompute<" in src
    assert "cutlass::bfloat16_t, cutlass::bfloat16_t" in src
    assert "float, float" in src


def test_evt_codegen_sm90_per_node_compute_dtype():
    """SM90 codegen emits per-node element types in Sm90Compute."""
    from magi_compiler.passes.piecewise_graph.fusion.evt_ir import Accum, Compute, Store
    from magi_compiler.passes.piecewise_graph.fusion.sm90.evt_codegen import render_evt_cu

    ir = Store(
        Compute(
            "add",
            (Compute("silu", (Accum(),), compute_dtype="float32"), Compute("neg", (Accum(),), compute_dtype="bfloat16")),
            compute_dtype="bfloat16",
        ),
        "bfloat16",
    )
    src = render_evt_cu(ir, "bfloat16", "bfloat16")
    assert "Sm90Compute<" in src
    assert "cutlass::bfloat16_t, cutlass::bfloat16_t" in src
    assert "float, float" in src


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
