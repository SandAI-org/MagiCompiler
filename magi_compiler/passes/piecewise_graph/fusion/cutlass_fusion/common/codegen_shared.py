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

"""Arch-agnostic codegen helpers shared by the SM80 and SM90 EVT codegens.

The two paths render structurally different .cu sources (CUTLASS 2.x Sm80EVT
vs CUTLASS 3.x Sm90EVT), but the dtype tables, built-in op table, custom
functor bodies, and helper functions are identical. Keep them here as the
single source of truth.
"""

from __future__ import annotations

import textwrap

# ── PyTorch dtype string → CUTLASS type ──────────────────────────────────────
_DTYPE_TO_CUTLASS = {"bfloat16": "cutlass::bfloat16_t", "float16": "cutlass::half_t", "float32": "float"}

# PyTorch dtype string → at::ScalarType used in TORCH_CHECK.
_DTYPE_TO_AT = {"bfloat16": "at::kBFloat16", "float16": "at::kHalf", "float32": "at::kFloat"}

# For data_ptr<T>() casts at the C++ layer.
_DTYPE_TO_AT_CPP = {"bfloat16": "at::BFloat16", "float16": "at::Half", "float32": "float"}


# ── Built-in CUTLASS op names for the visitor template-template parameter ────
# Maps IR op name → CUTLASS template name. Each value must be a
# ``template <class> class`` accepting a single type arg. These names exist in
# both CUTLASS 2.x (Sm80EVT) and CUTLASS 3.x (Sm90EVT) under the same
# namespaces, so the table is arch-agnostic.
_BUILTIN_FN_TEMPLATE = {
    # binary
    "add": "cutlass::plus",
    "sub": "cutlass::minus",
    "mul": "cutlass::multiplies",
    "div": "cutlass::divides",
    "max": "cutlass::maximum",
    "min": "cutlass::minimum",
    # unary
    "neg": "cutlass::negate",
    "sigmoid": "cutlass::epilogue::thread::Sigmoid",
    "silu": "cutlass::epilogue::thread::SiLu",
    "tanh": "cutlass::epilogue::thread::Tanh",
    "relu": "cutlass::epilogue::thread::ReLu",
    "abs": "cutlass::absolute_value_op",
}

# Unary ops that need a custom emitted functor (CUTLASS has no built-in).
# Each maps to a body template; the body uses ``T`` as the element type and
# operates on a single ``T`` value named ``x``.
_CUSTOM_UNARY_BODY = {
    "square": "return x * x;",
    "exp": "return cutlass::fast_exp(x);",
    "log": "return cutlass::fast_log(x);",
    "sqrt": "return cutlass::fast_sqrt(x);",
    "rsqrt": "return cutlass::fast_rsqrt(x);",
    "erf": "return T(erff(float(x)));",
    "gelu_erf": "return T(0.5f) * x * (T(1.0f) + T(erff(float(x) * 0.70710678118654752f)));",
    "gelu_tanh": (
        "float v = float(x);" " return T(0.5f * v * (1.0f + tanhf(" "0.7978845608028654f * (v + 0.044715f * v * v * v))));"
    ),
}

# Scalar-baked unary ops. The body template uses ``x`` and ``c`` (the baked
# constant, emitted as a ``T`` literal — never a runtime value).
_CUSTOM_SCALAR_BODY = {
    "add_scalar": "return x + c;",
    "sub_scalar": "return x - c;",
    "mul_scalar": "return x * c;",
    "div_scalar": "return x / c;",
    "rsub_scalar": "return c - x;",
    "clamp_min_c": "return x < c ? c : x;",
    "clamp_max_c": "return x < c ? x : c;",
    # scaled_silu_alpha(x, alpha) = x * sigmoid(alpha * x). Used by GELU7.
    "scaled_silu_alpha": (
        "T t = c * x;" " T one = T(1.0f);" " T sig = one / (one + cutlass::fast_exp(-t));" " return x * sig;"
    ),
    # pow_scalar(x, c) – emit as repeated multiplies for small int c.
    # Otherwise fall back to powf.
    "pow_scalar": "return T(powf(float(x), float(c)));",
}


# ── Greedy alignment selector — shared by FX-pass + runtime ─────────────────
_VALID_ALIGN_BITS = (128, 64)


def _scalar_literal_T(value: float) -> str:
    """Emit a constant as a ``T(...)`` cast that survives bf16 / fp16 / fp32."""
    # repr keeps round-trip precision; "f" suffix forces float in C++.
    return f"T({float(value)!r}f)"


def _emit_custom_functor(name: str, op: str, scalar=None) -> str:
    """Emit a unary CUTLASS-compatible functor (scalar + Array<T,N> spec).

    The same functor template body works on both Sm80EVT and Sm90EVT — both
    paths instantiate it as a ``template<class>``-shaped op. The
    ``cutlass::Array<T, N>`` specialisation lets the per-thread vector path
    apply the op element-wise to a packed array.
    """
    if op in _CUSTOM_UNARY_BODY:
        body = _CUSTOM_UNARY_BODY[op]
        scalar_decl = ""
    elif op in _CUSTOM_SCALAR_BODY:
        if scalar is None:
            raise ValueError(f"Scalar op {op!r} needs a baked constant")
        body = _CUSTOM_SCALAR_BODY[op]
        scalar_decl = f"        const T c = {_scalar_literal_T(scalar)};\n"
    else:
        raise ValueError(f"No custom functor body for op {op!r}")
    return textwrap.dedent(
        f"""\
        template <typename T>
        struct {name} {{
            static const bool kIsHeavy = true;
            CUTLASS_HOST_DEVICE
            T operator()(T const& x) const {{
        {scalar_decl}        {body}
            }}
        }};

        template <typename T, int N>
        struct {name}<cutlass::Array<T, N>> {{
            static const bool kIsHeavy = true;
            CUTLASS_HOST_DEVICE
            cutlass::Array<T, N> operator()(cutlass::Array<T, N> const& v) const {{
                {name}<T> op;
                cutlass::Array<T, N> out;
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < N; ++i) out[i] = op(v[i]);
                return out;
            }}
        }};
        """
    )
