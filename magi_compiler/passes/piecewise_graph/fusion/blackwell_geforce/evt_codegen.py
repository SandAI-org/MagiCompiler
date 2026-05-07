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

"""Render a CUTLASS .cu source from an EVT IR tree.

The output is a single self-contained file that:
  1. Declares any custom functor templates required by scalar-baked ops
     (ClampMaxC, ScaledSiLuAlpha, GeluErf, …) — each baked with its constant.
  2. Declares the bottom-up Sm80EVT typedef chain.
  3. Declares the GemmKernel + DeviceGemm + entry point.
  4. Exposes ``evt_matmul_out`` via PYBIND11.

We use CUTLASS 2.x ``Sm80EVT`` running backward-compat on sm_120; this matches
``$MAGI_CUTLASS_ROOT/examples/99_evt_demo/heavy_epi_torch_ext.cu`` (default
``/opt/cutlass/...``) which has been verified to deliver +5..+12 % vs the
Triton TMA path on RTX 5090 bf16.
"""

from __future__ import annotations

import textwrap
from typing import Dict, List, Tuple

from .evt_ir import Accum, AuxLoad, ColBroadcast, Compute, RowBroadcast, Store, walk_leaves

# ── PyTorch dtype string → CUTLASS type ──────────────────────────────────────
_DTYPE_TO_CUTLASS = {"bfloat16": "cutlass::bfloat16_t", "float16": "cutlass::half_t", "float32": "float"}

# PyTorch dtype string → at::ScalarType / pybind dtype string used in TORCH_CHECK.
_DTYPE_TO_AT = {"bfloat16": "at::kBFloat16", "float16": "at::kHalf", "float32": "at::kFloat"}


# ── Per-M-bucket tile candidate sets, hand-tuned for RTX 5090 (sm_120) ──────
# Hardware constraints driving these choices:
#   * 170 SMs — the optimal grid size is some multiple of 170; small tiles
#     keep more CTAs in flight when M is short.
#   * 100 KB SMEM / SM — per-stage SMEM = (BM + BN) * BK * 2 (bf16). With
#     stages=4 and (128,128,32) we land at 128 KB which exceeds budget; we
#     prefer stages=3 in that case. (128,128,32)*4 = 128KB, (128,256,32)*3=144KB,
#     (256,128,32)*3=144KB are still over budget but CUTLASS auto-shrinks
#     stages on Sm80 if SMEM doesn't fit. We rely on can_implement / init to
#     reject illegal combos at autotune time.
#   * Decode-style M (≤256) loses parallelism on big tiles — 1 wave covers
#     just a handful of N tiles. Need small BM.
#   * Prefill-style M (>2048) has plenty of parallelism — bigger tiles win
#     because they amortise loads better.
#
# Each tuple is (BM, BN, BK, WM, WN, WK, NumStages, label).
# WarpShape is conventionally TileShape / (2, 2) along (M, N), keeping 4 warps.
# We include WK == BK to match Sm80 TensorOp's default warp tiling.
_TILE_CANDIDATES_5090: dict = {
    # ── small (decode / single-token) ────────────────────────────────────────
    # M ≤ 256: low parallelism along M. Use small BM to launch more CTAs along N.
    # All candidates have BM*BN ≤ 16384 to keep occupancy high on 170 SMs.
    "small": [
        (64, 64, 32, 32, 32, 32, 4, "T<64,64,32>_S4"),
        (64, 64, 64, 32, 32, 64, 3, "T<64,64,64>_S3"),
        (64, 128, 32, 32, 64, 32, 3, "T<64,128,32>_S3"),
        (64, 128, 32, 32, 64, 32, 4, "T<64,128,32>_S4"),
        (64, 128, 64, 32, 64, 64, 3, "T<64,128,64>_S3"),
        (64, 256, 32, 32, 64, 32, 3, "T<64,256,32>_S3"),
        (128, 64, 32, 64, 32, 32, 3, "T<128,64,32>_S3"),
        (128, 64, 32, 64, 32, 32, 4, "T<128,64,32>_S4"),
    ],
    # ── medium (256 < M ≤ 2048) ──────────────────────────────────────────────
    # Standard CUTLASS bf16 sweet spot. Mix BM=128/256 with BN=64/128/256.
    "medium": [
        (128, 128, 32, 64, 64, 32, 3, "T<128,128,32>_S3"),
        (128, 128, 32, 64, 64, 32, 4, "T<128,128,32>_S4"),
        (128, 128, 64, 64, 64, 64, 3, "T<128,128,64>_S3"),
        (128, 256, 32, 64, 64, 32, 3, "T<128,256,32>_S3"),
        (256, 128, 32, 64, 64, 32, 3, "T<256,128,32>_S3"),
        (128, 64, 64, 64, 32, 64, 4, "T<128,64,64>_S4"),
        (64, 128, 64, 32, 64, 64, 4, "T<64,128,64>_S4"),
    ],
    # ── large (M > 2048) ─────────────────────────────────────────────────────
    # Plenty of parallelism — bigger tiles for better arith density. SMEM
    # budget on 5090 (100 KB) restricts (256,128) and (128,256) to stages=3.
    "large": [
        (128, 256, 32, 64, 64, 32, 3, "T<128,256,32>_S3"),
        (256, 128, 32, 64, 64, 32, 3, "T<256,128,32>_S3"),
        (128, 128, 32, 64, 64, 32, 4, "T<128,128,32>_S4"),
        (128, 128, 64, 64, 64, 64, 3, "T<128,128,64>_S3"),
        (256, 128, 64, 64, 64, 64, 3, "T<256,128,64>_S3"),
        (128, 256, 64, 64, 64, 64, 3, "T<128,256,64>_S3"),
    ],
}


def _emit_tile_candidates(m_bucket: str) -> str:
    """Emit C++ EVT_TILE_CANDIDATE(...) statements for a given M bucket."""
    candidates = _TILE_CANDIDATES_5090.get(m_bucket, _TILE_CANDIDATES_5090["medium"])
    lines = []
    for bm, bn, bk, wm, wn, wk, stages, label in candidates:
        lines.append(f'    EVT_TILE_CANDIDATE({bm}, {bn}, {bk}, {wm}, {wn}, {wk}, ' f'{stages}, "{label}");')
    return "\n".join(lines)


# For data_ptr<T>() casts at the C++ layer.
_DTYPE_TO_AT_CPP = {"bfloat16": "at::BFloat16", "float16": "at::Half", "float32": "float"}


# ── Built-in CUTLASS op names for the visitor template-template parameter ────
# Maps IR op name → (CUTLASS template name, is_class_template_with_T_only)
# Each value must be a `template <class> class` accepting a single type arg.
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


def _scalar_literal_T(value: float) -> str:
    """Emit a constant as a ``T(...)`` cast that survives bf16 / fp16 / fp32."""
    # repr keeps round-trip precision; "f" suffix forces float in C++.
    return f"T({float(value)!r}f)"


def _emit_custom_functor(name: str, op: str, scalar=None) -> str:
    """Emit a unary CUTLASS-compatible functor (scalar + Array<T,N> spec)."""
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


# ── EVT typedef + leaf args walker ────────────────────────────────────────────


class _EvtEmitter:
    """Bottom-up walker that emits typedef chains + leaf placeholders."""

    def __init__(self, root: Store):
        self.root = root
        self.typedef_lines: List[str] = []
        self.functor_decls: List[str] = []
        self._emitted_functors: Dict[Tuple[str, str], str] = {}
        self._tmp_counter = 0
        # Per-leaf metadata captured during walk: leaf identity (object id) →
        # (typedef_name, leaf_kind, input_idx_or_None, dtype_str)
        self.leaf_typedefs: List[Tuple[str, str, "int | None", str]] = []
        self.scalar_functor_counter = 0

    def _new_name(self, prefix: str) -> str:
        self._tmp_counter += 1
        return f"{prefix}_{self._tmp_counter}"

    def _functor_name_for(self, op: str, scalar) -> str:
        """Unique struct name for a custom functor, deduped by (op, scalar)."""
        key = (op, repr(scalar) if scalar is not None else "")
        if key in self._emitted_functors:
            return self._emitted_functors[key]
        # Strip dots from the scalar so the name stays a valid C++ identifier.
        scalar_tag = ""
        if scalar is not None:
            self.scalar_functor_counter += 1
            scalar_tag = f"_v{self.scalar_functor_counter}"
        name = f"Magi_{op}{scalar_tag}"
        self._emitted_functors[key] = name
        self.functor_decls.append(_emit_custom_functor(name, op, scalar))
        return name

    def _compute_op_template(self, node: Compute) -> str:
        """Return the C++ template-name passed as ComputeFn to VisitorCompute."""
        if node.op in _BUILTIN_FN_TEMPLATE and node.scalar is None:
            return _BUILTIN_FN_TEMPLATE[node.op]
        # Custom functor — either scalar-baked or unary-no-builtin (e.g. erf).
        return self._functor_name_for(node.op, node.scalar)

    def emit(self) -> str:
        """Walk the IR; return the typedef name of the root EVT type (EVT_D)."""
        # Recurse from Store.child first to build up subtrees.
        body_root = self._emit_node(self.root.child)
        # The store leaf itself is the StoreD typedef wrapping body_root.
        store_name = self._new_name("StoreD")
        self.typedef_lines.append(
            "using {name} = cutlass::epilogue::threadblock::VisitorAuxStore<\n"
            "    OutputTileThreadMap, ElementC,\n"
            "    cutlass::FloatRoundStyle::round_to_nearest,\n"
            "    cute::Stride<int64_t, _1, int64_t>>;".format(name=store_name)
        )
        evt_d = self._new_name("EVT_D")
        self.typedef_lines.append(
            f"using {evt_d} = cutlass::epilogue::threadblock::Sm80EVT<\n" f"    {store_name}, {body_root}>;"
        )
        # Track the StoreD leaf metadata so the launcher knows where to bind D.
        self.leaf_typedefs.append((store_name, "store", None, self.root.out_dtype))
        return evt_d

    def _emit_node(self, node) -> str:
        if isinstance(node, Accum):
            name = self._new_name("Accum")
            self.typedef_lines.append(f"using {name} = cutlass::epilogue::threadblock::VisitorAccFetch;")
            return name
        if isinstance(node, RowBroadcast):
            name = self._new_name("RowBcast")
            elem = _DTYPE_TO_CUTLASS[node.dtype]
            self.typedef_lines.append(
                f"using {name} = cutlass::epilogue::threadblock::VisitorRowBroadcast<\n"
                f"    OutputTileThreadMap, {elem},\n"
                f"    cute::Stride<_0, _1, int32_t>>;"
            )
            self.leaf_typedefs.append((name, "row_bcast", node.input_idx, node.dtype))
            return name
        if isinstance(node, ColBroadcast):
            name = self._new_name("ColBcast")
            elem = _DTYPE_TO_CUTLASS[node.dtype]
            self.typedef_lines.append(
                f"using {name} = cutlass::epilogue::threadblock::VisitorColBroadcast<\n"
                f"    OutputTileThreadMap, {elem},\n"
                f"    cute::Stride<_1, _0, int32_t>>;"
            )
            self.leaf_typedefs.append((name, "col_bcast", node.input_idx, node.dtype))
            return name
        if isinstance(node, AuxLoad):
            name = self._new_name("Aux")
            elem = _DTYPE_TO_CUTLASS[node.dtype]
            self.typedef_lines.append(
                f"using {name} = cutlass::epilogue::threadblock::VisitorAuxLoad<\n"
                f"    OutputTileThreadMap, {elem},\n"
                f"    cute::Stride<int64_t, _1, int64_t>>;"
            )
            self.leaf_typedefs.append((name, "aux_load", node.input_idx, node.dtype))
            return name
        if isinstance(node, Compute):
            child_names = [self._emit_node(c) for c in node.children]
            compute_name = self._new_name(f"Cmp_{node.op}")
            fn_template = self._compute_op_template(node)
            self.typedef_lines.append(
                f"using {compute_name} = cutlass::epilogue::threadblock::VisitorCompute<\n"
                f"    {fn_template}, ElementCompute, ElementCompute,\n"
                f"    cutlass::FloatRoundStyle::round_to_nearest>;"
            )
            evt_name = self._new_name(f"EVT_{node.op}")
            child_typedef_list = ", ".join(child_names)
            self.typedef_lines.append(
                f"using {evt_name} = cutlass::epilogue::threadblock::Sm80EVT<\n" f"    {compute_name}, {child_typedef_list}>;"
            )
            return evt_name
        raise TypeError(f"Unknown IR node type: {type(node).__name__}")


# ── Argument-tree emitter (matches EVT typedef tree) ──────────────────────────


def _emit_args_tree(node, leaf_args: Dict[int, str], indent: int = 4) -> str:
    """Emit the nested-brace runtime callback-args literal matching the IR.

    ``leaf_args[input_idx]`` for non-Accum leaves is a small C++ snippet like
    ``{ptrBias, ElementC(0), {_0{}, _1{}, int32_t(N)}}``. Accum / Compute /
    Store args are empty braces ``{}``. The Store arg is ``{ptrD, {N, _1{},
    MN}}`` and is handled by the caller — this emitter only renders the body
    inside StoreD.
    """
    pad = " " * indent
    if isinstance(node, Accum):
        return f"{pad}{{}}"
    if isinstance(node, (RowBroadcast, ColBroadcast, AuxLoad)):
        return f"{pad}{leaf_args[node.input_idx]}"
    if isinstance(node, Compute):
        children_str = ",\n".join(_emit_args_tree(c, leaf_args, indent + 2) for c in node.children)
        return f"{pad}{{\n" f"{children_str},\n" f"{pad}  {{}}\n" f"{pad}}}"
    raise TypeError(f"Unknown IR node type: {type(node).__name__}")


# ── Public API: render a complete .cu source string ──────────────────────────


_KERNEL_PREAMBLE = """\
// AUTO-GENERATED by magi_compiler/passes/piecewise_graph/fusion/evt_codegen.py
// Do not edit by hand. Regenerate by re-running the FX pass.
//
// IR cache key: {cache_key}

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <memory>
#include <unordered_map>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/functional.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm_universal.h"

#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

using cute::_0;
using cute::_1;

////////////////////////////////////////////////////////////////////////////////
// Custom functors (one per unique scalar-baked op or non-builtin unary).
////////////////////////////////////////////////////////////////////////////////
{functor_decls}

////////////////////////////////////////////////////////////////////////////////
// Data types and layouts
////////////////////////////////////////////////////////////////////////////////

using ElementA       = {a_elem};
using ElementB       = {b_elem};
using ElementC       = {c_elem};
using ElementAcc     = float;
using ElementCompute = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::{b_layout};
using LayoutC = cutlass::layout::RowMajor;

constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
// Uniform 128-bit alignment for A, B, and D. The host pads D's row stride
// (ldd) up to AlignmentC element boundaries when n_out doesn't naturally
// divide it; the runtime passes the padded stride via EvtArgs.ldd.
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ArchTag          = cutlass::arch::Sm80;
using OperatorClass    = cutlass::arch::OpClassTensorOp;
using InstructionShape = cutlass::gemm::GemmShape< 16,   8, 16>;
constexpr int EVTEpilogueStages = 1;

////////////////////////////////////////////////////////////////////////////////
// Per-tile-config GEMM type. The OutputTileThreadMap depends on
// ThreadblockShape/WarpShape, which forces every EVT typedef to be re-built
// per tile. We package the whole tree inside a template struct keyed on the
// tile/warp/stages parameters so each autotune candidate is a distinct type.
////////////////////////////////////////////////////////////////////////////////

template <class TbShape, class WarpShape, int NumStages>
struct EvtConfig {{
  using TheTbShape = TbShape;
  using TheWarpShape = WarpShape;

  using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
      TbShape, WarpShape, ElementC, AlignmentC, EVTEpilogueStages>;

  ////////////////////////////////////////////////////////////////////////////
  // EVT (Epilogue Visitor Tree) typedefs — generated from the IR tree.
  ////////////////////////////////////////////////////////////////////////////
{typedef_block}

  ////////////////////////////////////////////////////////////////////////////
  // GemmKernel / DeviceGemm
  ////////////////////////////////////////////////////////////////////////////
  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignmentA,
      ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignmentB,
      ElementC, LayoutC, AlignmentC,
      ElementAcc,
      ElementCompute,
      OperatorClass,
      ArchTag,
      TbShape,
      WarpShape,
      InstructionShape,
      {evt_root_name},
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      NumStages,
      cutlass::arch::OpMultiplyAdd,
      EVTEpilogueStages>::GemmKernel;

  using DeviceGemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}};

////////////////////////////////////////////////////////////////////////////////
// Autotune runner — one candidate per tile/warp/stages combination; first call
// at a new (M, N, K) tuple times every candidate and caches the winner.
////////////////////////////////////////////////////////////////////////////////

struct EvtArgs {{
  int M;
  int N;
  int K;
  void* ptr_A;
  void* ptr_B;
  void* ptr_D;
  // Row stride of D in elements. Equals N when D is contiguous; > N when
  // the host padded D up to AlignmentC. Threaded into LayoutC at runtime.
  int64_t ldd;
  // Extras pointers, in IR-leaf order.
  std::vector<void*> ptr_extras;
}};

class EvtConcept {{
 public:
  virtual ~EvtConcept() = default;
  virtual size_t get_workspace_size(const EvtArgs&) = 0;
  virtual cutlass::Status initialize(const EvtArgs&, void* ws, cudaStream_t s) = 0;
  virtual cutlass::Status run(cudaStream_t stream) = 0;
  virtual const char* name() const = 0;
}};

template <class Cfg>
class EvtImpl : public EvtConcept {{
 public:
  using GemmType = typename Cfg::DeviceGemm;
  using EvtRoot  = typename Cfg::{evt_root_name};

  explicit EvtImpl(const char* name) : name_(name) {{}}

  typename GemmType::Arguments make_args(const EvtArgs& a) {{
    auto ptrA = reinterpret_cast<ElementA*>(a.ptr_A);
    auto ptrB = reinterpret_cast<ElementB*>(a.ptr_B);
    auto ptrD = reinterpret_cast<ElementC*>(a.ptr_D);
    int const M = a.M;
    int const N = a.N;
    int const K = a.K;
    int64_t const MN = static_cast<int64_t>(M) * static_cast<int64_t>(N);
    // ldd = D's row stride in elements; padded by host to satisfy AlignmentC.
    int64_t const ldd = a.ldd;
    int64_t const stride_d_total = static_cast<int64_t>(M) * ldd;

    typename EvtRoot::Arguments callback_args{{
{args_tree}
        ,
        {{ptrD, {{ldd, _1{{}}, stride_d_total}}}}
    }};

    cutlass::gemm::GemmCoord problem{{M, N, K}};
    typename GemmType::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem,
        /*batch_count=*/1,
        callback_args,
        ptrA, ptrB,
        /*ptr_C=*/nullptr, /*ptr_D=*/nullptr,
        /*batch_stride_A=*/static_cast<int64_t>(M) * K,
        /*batch_stride_B=*/static_cast<int64_t>(N) * K,
        /*batch_stride_C=*/0, /*batch_stride_D=*/0,
        /*stride_a=*/static_cast<int64_t>(K),
        /*stride_b=*/static_cast<int64_t>({stride_b_expr}),
        /*stride_c=*/0, /*stride_d=*/0);
    return args;
  }}

  size_t get_workspace_size(const EvtArgs& a) override {{
    auto args = make_args(a);
    return GemmType::get_workspace_size(args);
  }}
  cutlass::Status initialize(const EvtArgs& a, void* ws, cudaStream_t s) override {{
    auto args = make_args(a);
    return gemm_.initialize(args, ws, s);
  }}
  cutlass::Status run(cudaStream_t stream) override {{
    return gemm_.run(stream);
  }}
  const char* name() const override {{ return name_; }}

 private:
  GemmType gemm_;
  const char* name_;
}};

////////////////////////////////////////////////////////////////////////////////
// Python-facing launcher
////////////////////////////////////////////////////////////////////////////////
"""


_LAUNCHER_TEMPLATE = """\
////////////////////////////////////////////////////////////////////////////////
// Tile candidate registration. Each AutoConfigBuilder invocation instantiates
// the full EVT typedef tree + GemmKernel for that (TileShape, WarpShape,
// NumStages) tuple. Compile time grows linearly with the candidate count, so
// keep the list small and shape-relevant.
////////////////////////////////////////////////////////////////////////////////

#define EVT_TILE_CANDIDATE(tb_m, tb_n, tb_k, wa_m, wa_n, wa_k, stages, label)        \\
  configs_.push_back(std::make_unique<EvtImpl<EvtConfig<                              \\
      cutlass::gemm::GemmShape<tb_m, tb_n, tb_k>,                                     \\
      cutlass::gemm::GemmShape<wa_m, wa_n, wa_k>,                                     \\
      stages>>>(label))

class EvtAutoTuneRunner {{
 public:
  EvtAutoTuneRunner() {{
{tile_candidate_block}
  }}

  void operator()(at::Tensor A, at::Tensor B,
                  std::vector<at::Tensor> extras, at::Tensor D) {{
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && D.is_cuda(),
                "evt_matmul_out: A/B/D must be CUDA tensors");
    TORCH_CHECK(A.scalar_type() == {a_at_dtype}, "A must be {a_dtype}");
    TORCH_CHECK(B.scalar_type() == {b_at_dtype}, "B must be {b_dtype}");
    TORCH_CHECK(D.scalar_type() == {c_at_dtype}, "D must be {c_dtype}");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && D.dim() == 2, "A, B, D must be 2D");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(),
                "A, B must be contiguous (row-major)");

    int const M = static_cast<int>(A.size(0));
    int const K = static_cast<int>(A.size(1));
    int const N = static_cast<int>({n_dim_expr});

    TORCH_CHECK(D.size(0) == M && D.size(1) == N,
                "D must be (M, N); got ", D.sizes());
    // D may be a strided view of a host-padded (M, n_padded) buffer: inner
    // stride must be 1, row stride (ldd) must be >= N.
    TORCH_CHECK(D.stride(1) == 1, "D innermost stride must be 1; got ", D.stride(1));
    TORCH_CHECK(D.stride(0) >= N,
                "D row stride must be >= N; got stride(0)=", D.stride(0), ", N=", N);
    TORCH_CHECK(extras.size() == {n_extras}, "expected {n_extras} extra tensors, got ", extras.size());

{extras_validation}

    EvtArgs ea;
    ea.M = M; ea.N = N; ea.K = K;
    ea.ptr_A = A.data_ptr<{a_at_cpp}>();
    ea.ptr_B = B.data_ptr<{b_at_cpp}>();
    ea.ptr_D = D.data_ptr<{c_at_cpp}>();
    ea.ldd = static_cast<int64_t>(D.stride(0));
    ea.ptr_extras.reserve({n_extras});
{extras_ptrs}

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.device().index()).stream();

    // Single autotune per module. The .cu is compiled per (IR, M-bucket,
    // b_layout, N, K) on the Python side — every distinct weight (N, K)
    // gets its own .cu, so this runner instance hosts exactly one (N, K)
    // and one bucket of M values. Autotune once on the first call; all
    // subsequent calls (any M inside the bucket) reuse `best_idx_`.
    if (best_idx_ < 0) {{
      best_idx_ = autotune(ea, stream);
    }}
    int idx = best_idx_;

    auto& gemm = configs_[idx];
    size_t ws_sz = gemm->get_workspace_size(ea);
    if (!ws_.defined() || ws_.numel() < (int64_t)ws_sz) {{
      ws_ = at::empty({{(int64_t)ws_sz + 1}},
          at::TensorOptions().dtype(at::kByte).device(A.device()));
    }}
    auto st = gemm->initialize(ea, ws_sz > 0 ? ws_.data_ptr() : nullptr, stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess,
                "CUTLASS init failed (", gemm->name(), "): ", cutlassGetStatusString(st));
    st = gemm->run(stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess,
                "CUTLASS run failed (", gemm->name(), "): ", cutlassGetStatusString(st));
  }}

  int num_configs() const {{ return (int)configs_.size(); }}

 private:
  int autotune(const EvtArgs& ea, cudaStream_t stream) {{
    int best_idx = -1;
    float best_time = 1e30f;
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    for (size_t i = 0; i < configs_.size(); ++i) {{
      auto& g = configs_[i];
      size_t ws_sz = 0;
      try {{ ws_sz = g->get_workspace_size(ea); }}
      catch (...) {{ continue; }}
      if (!ws_.defined() || ws_.numel() < (int64_t)ws_sz) {{
        ws_ = at::empty({{(int64_t)ws_sz + 1}},
            at::TensorOptions().dtype(at::kByte).device(at::kCUDA));
      }}
      void* ws_ptr = ws_sz > 0 ? ws_.data_ptr() : nullptr;
      if (g->initialize(ea, ws_ptr, stream) != cutlass::Status::kSuccess) {{
        continue;
      }}

      // Warmup — 10 iters so L2 / inst caches settle (3 was too few — first
      // timed iter saw a cold L2 and biased the choice towards smaller tiles).
      for (int w = 0; w < 10; ++w) g->run(stream);
      cudaStreamSynchronize(stream);

      // Time — 20 iters for ~1% timing noise, matching torch.compile defaults.
      cudaEventRecord(s, stream);
      int iters = 20;
      for (int p = 0; p < iters; ++p) g->run(stream);
      cudaEventRecord(e, stream);
      cudaEventSynchronize(e);
      float ms = 0;
      cudaEventElapsedTime(&ms, s, e);
      float avg = ms / iters;
      if (avg < best_time) {{ best_time = avg; best_idx = (int)i; }}
    }}
    cudaEventDestroy(s); cudaEventDestroy(e);
    TORCH_CHECK(best_idx >= 0,
                "EVT AutoTune: no candidate succeeded for (M,N,K)=(",
                ea.M, ",", ea.N, ",", ea.K, ")");
    return best_idx;
  }}

  std::vector<std::unique_ptr<EvtConcept>> configs_;
  int best_idx_ = -1;     // -1 = not yet autotuned; sticky after first call.
  at::Tensor ws_;
}};

static EvtAutoTuneRunner& runner() {{
  static EvtAutoTuneRunner R;
  return R;
}}

void evt_matmul_out(at::Tensor A, at::Tensor B,
                    std::vector<at::Tensor> extras,
                    at::Tensor D) {{
  runner()(std::move(A), std::move(B), std::move(extras), std::move(D));
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.doc() = "Magi compiler EVT-fused matmul (auto-generated, autotune)";
    m.def("evt_matmul_out", &evt_matmul_out,
          "Fused EVT matmul: D = epilogue(A @ B, extras...)",
          pybind11::arg("A"), pybind11::arg("B"),
          pybind11::arg("extras"), pybind11::arg("D"));
    m.def("num_configs", []() {{ return runner().num_configs(); }});
}}
"""


def render_evt_cu(
    ir: Store, a_dtype: str, b_dtype: str, cache_key_str: str = "", b_layout: str = "row", m_bucket: str = "medium"
) -> str:
    """Render a complete .cu source for the given EVT IR.

    Parameters
    ----------
    ir : Store
        Root of the EVT IR tree.
    a_dtype, b_dtype : str
        Element types for A and B (typically ``"bfloat16"``). Output dtype is
        taken from ``ir.out_dtype``.
    cache_key_str : str
        Optional hash echoed in a top-level comment, useful for debugging.
    b_layout : "row" | "col"
        ``"row"`` (default): B is contiguous (K, N) row-major; LayoutB =
        RowMajor; ldB = N. ``"col"``: B is the underlying (N, K) row-major
        weight (== column-major (K, N)); LayoutB = ColumnMajor; ldB = K. Use
        ``"col"`` when the FX graph passes ``permute([1,0])(weight)`` as B.
    m_bucket : "small" | "medium" | "large"
        Picks a tile-candidate set tuned for RTX 5090 (sm_120) at the given M
        regime. The runner inside the rendered .cu autotunes across all
        candidates in that bucket on the first call per (M, N, K) shape and
        caches the winner.
    """
    if b_layout not in ("row", "col"):
        raise ValueError(f"b_layout must be 'row' or 'col', got {b_layout!r}")
    if m_bucket not in _TILE_CANDIDATES_5090:
        raise ValueError(f"unknown m_bucket {m_bucket!r}; " f"expected one of {list(_TILE_CANDIDATES_5090)}")
    if not isinstance(ir, Store):
        raise TypeError("render_evt_cu expects a Store node as root")
    tile_candidate_block = _emit_tile_candidates(m_bucket)

    a_elem = _DTYPE_TO_CUTLASS[a_dtype]
    b_elem = _DTYPE_TO_CUTLASS[b_dtype]
    c_elem = _DTYPE_TO_CUTLASS[ir.out_dtype]

    emitter = _EvtEmitter(ir)
    evt_root = emitter.emit()

    # Build per-leaf runtime arg fragments. These get inlined into
    # ``EvtImpl::make_args`` (a method on a different class than the launcher
    # that fills ea.ptr_extras). The only shared state between the two scopes
    # is the EvtArgs struct ``a``, so we read pointers from a.ptr_extras[i]
    # and cast back to the leaf's element type.
    leaves = walk_leaves(ir)
    leaf_args: Dict[int, str] = {}
    for leaf in leaves:
        # Accum has no extras pointer / dtype — skip; it consumes the GEMM
        # accumulator directly via VisitorAccFetch.
        if not isinstance(leaf, (RowBroadcast, ColBroadcast, AuxLoad)):
            continue
        elem = _DTYPE_TO_CUTLASS[leaf.dtype]
        ptr_expr = f"reinterpret_cast<{elem}*>(a.ptr_extras[{leaf.input_idx}])"
        if isinstance(leaf, RowBroadcast):
            leaf_args[leaf.input_idx] = f"{{{ptr_expr}, {elem}(0), {{_0{{}}, _1{{}}, int32_t(N)}}}}"
        elif isinstance(leaf, ColBroadcast):
            leaf_args[leaf.input_idx] = f"{{{ptr_expr}, {elem}(0), {{_1{{}}, _0{{}}, int32_t(M)}}}}"
        else:  # AuxLoad
            leaf_args[leaf.input_idx] = f"{{{ptr_expr}, {elem}(0), {{int64_t(N), _1{{}}, MN}}}}"
        # Accum has no explicit args entry.

    args_tree = _emit_args_tree(ir.child, leaf_args, indent=8)

    # Extras-validation + pointer-extraction blocks. The same external tensor
    # (same input_idx) may appear at multiple leaves in the IR tree — e.g. an
    # ``add(mm, bias)`` value flowing into both ``sigmoid`` and ``mul`` creates
    # two RowBroadcast(0) leaves. We must declare ``ptr_extra_0`` exactly once
    # in the launcher; the runtime args tree still references the same ptr
    # name from each leaf-arg fragment so this dedup is purely a C++ scope fix.
    extras_validation_lines = []
    extras_ptr_lines = []
    seen_extras: set = set()
    extra_leaves = [n for n in leaves if not isinstance(n, Accum)]
    n_extras = max((leaf.input_idx for leaf in extra_leaves), default=-1) + 1
    for leaf in extra_leaves:
        i = leaf.input_idx
        if i in seen_extras:
            continue
        seen_extras.add(i)
        at_dtype = _DTYPE_TO_AT[leaf.dtype]
        at_cpp = _DTYPE_TO_AT_CPP[leaf.dtype]
        _DTYPE_TO_CUTLASS[leaf.dtype]
        if isinstance(leaf, RowBroadcast):
            extras_validation_lines.append(f'    TORCH_CHECK(extras[{i}].numel() == N, "extras[{i}] must have N elements");')
        elif isinstance(leaf, ColBroadcast):
            extras_validation_lines.append(f'    TORCH_CHECK(extras[{i}].numel() == M, "extras[{i}] must have M elements");')
        elif isinstance(leaf, AuxLoad):
            extras_validation_lines.append(
                f'    TORCH_CHECK(extras[{i}].size(0) == M && extras[{i}].size(1) == N,' f' "extras[{i}] must be (M,N)");'
            )
        extras_validation_lines.append(
            f'    TORCH_CHECK(extras[{i}].scalar_type() == {at_dtype},' f' "extras[{i}] must be {leaf.dtype}");'
        )
        extras_validation_lines.append(f'    TORCH_CHECK(extras[{i}].is_cuda(), "extras[{i}] must be CUDA");')
        # Push raw pointer into ea.ptr_extras for the make_args() side to
        # read (it lives in a different scope than this launcher fn).
        extras_ptr_lines.append(f"    ea.ptr_extras.push_back(static_cast<void*>(" f"extras[{i}].data_ptr<{at_cpp}>()));")

    extras_validation = "\n".join(extras_validation_lines) if extras_validation_lines else "    // no extras"
    extras_ptrs = "\n".join(extras_ptr_lines) if extras_ptr_lines else ""

    # Emit. The functor decls already end with a trailing newline each.
    functor_decls = "\n".join(emitter.functor_decls) if emitter.functor_decls else "// (no custom functors)"
    # typedef_block lives inside ``struct EvtConfig`` — indent each line by 2
    # spaces so member typedefs read consistently with the surrounding struct.
    typedef_block = "\n".join("  " + l if l.strip() else l for l in "\n".join(emitter.typedef_lines).split("\n"))

    cutlass_b_layout = "RowMajor" if b_layout == "row" else "ColumnMajor"
    if b_layout == "row":
        # B is (K, N) row-major contiguous: K from B.size(0), N from B.size(1), ldB = N.
        n_dim_expr = "B.size(1)"
        stride_b_expr = "N"
    else:
        # B is the underlying (N, K) row-major weight (we read the same
        # bytes via ColumnMajor (K, N)): N from B.size(0), K from B.size(1), ldB = K.
        n_dim_expr = "B.size(0)"
        stride_b_expr = "K"

    preamble = _KERNEL_PREAMBLE.format(
        cache_key=cache_key_str,
        functor_decls=functor_decls,
        a_elem=a_elem,
        b_elem=b_elem,
        c_elem=c_elem,
        typedef_block=typedef_block,
        evt_root_name=evt_root,
        b_layout=cutlass_b_layout,
        # EvtImpl::make_args uses args_tree + stride_b_expr; same values as
        # the launcher (per-IR / per-layout, not per-tile-config).
        args_tree=args_tree,
        stride_b_expr=stride_b_expr,
    )
    launcher = _LAUNCHER_TEMPLATE.format(
        evt_root_name=evt_root,
        args_tree=args_tree,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=ir.out_dtype,
        a_at_dtype=_DTYPE_TO_AT[a_dtype],
        b_at_dtype=_DTYPE_TO_AT[b_dtype],
        c_at_dtype=_DTYPE_TO_AT[ir.out_dtype],
        a_at_cpp=_DTYPE_TO_AT_CPP[a_dtype],
        b_at_cpp=_DTYPE_TO_AT_CPP[b_dtype],
        c_at_cpp=_DTYPE_TO_AT_CPP[ir.out_dtype],
        n_extras=n_extras,
        extras_validation=extras_validation,
        extras_ptrs=extras_ptrs,
        n_dim_expr=n_dim_expr,
        stride_b_expr=stride_b_expr,
        tile_candidate_block=tile_candidate_block,
    )
    return preamble + launcher
