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

"""Render a CUTLASS 2.x Sm80EVT .cu source from an EVT IR tree.

Used on sm_120 (RTX 5090) and all non-sm_90 arches. The H100 path is
``../sm90/evt_codegen.py``, selected by ``evt_runtime`` on sm_90 devices.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from ..common.codegen_shared import (
    _BUILTIN_FN_TEMPLATE,
    _DTYPE_TO_AT,
    _DTYPE_TO_AT_CPP,
    _DTYPE_TO_CUTLASS,
    _VALID_ALIGN_BITS,
    _emit_custom_functor,
)
from ..evt_ir import Accum, AuxLoad, ColBroadcast, Compute, RowBroadcast, Store, walk_leaves

# (BM, BN, BK, WM, WN, WK, NumStages, label).
# RTX 5090: 170 SMs, 100 KB SMEM / SM; tile×stages kept inside that envelope.
_TILE_CANDIDATES_SM120: dict = {
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
    "medium": [
        (128, 128, 32, 64, 64, 32, 3, "T<128,128,32>_S3"),
        (128, 128, 32, 64, 64, 32, 4, "T<128,128,32>_S4"),
        (128, 128, 64, 64, 64, 64, 3, "T<128,128,64>_S3"),
        (128, 256, 32, 64, 64, 32, 3, "T<128,256,32>_S3"),
        (256, 128, 32, 64, 64, 32, 3, "T<256,128,32>_S3"),
        (128, 64, 64, 64, 32, 64, 4, "T<128,64,64>_S4"),
        (64, 128, 64, 32, 64, 64, 4, "T<64,128,64>_S4"),
    ],
    "large": [
        (128, 256, 32, 64, 64, 32, 3, "T<128,256,32>_S3"),
        (256, 128, 32, 64, 64, 32, 3, "T<256,128,32>_S3"),
        (128, 128, 32, 64, 64, 32, 4, "T<128,128,32>_S4"),
        (128, 128, 64, 64, 64, 64, 3, "T<128,128,64>_S3"),
    ],
}

# Backward-compat alias: some external callers still reference this name.
_TILE_CANDIDATES_5090 = _TILE_CANDIDATES_SM120


def _emit_tile_candidates(m_bucket: str) -> str:
    """Emit C++ EVT_TILE_CANDIDATE(...) statements for the given M bucket."""
    candidates = _TILE_CANDIDATES_SM120.get(m_bucket, _TILE_CANDIDATES_SM120["medium"])
    lines = []
    for bm, bn, bk, wm, wn, wk, stages, label in candidates:
        lines.append(f'    EVT_TILE_CANDIDATE({bm}, {bn}, {bk}, {wm}, {wn}, {wk}, ' f'{stages}, "{label}");')
    return "\n".join(lines)


class _EvtEmitter:
    """Bottom-up walker that emits typedef chains + leaf placeholders."""

    def __init__(self, root: Store):
        self.root = root
        self.typedef_lines: List[str] = []
        self.functor_decls: List[str] = []
        self._emitted_functors: Dict[Tuple[str, str], str] = {}
        self._tmp_counter = 0
        self.leaf_typedefs: List[Tuple[str, str, "int | None", str]] = []
        self.scalar_functor_counter = 0

    def _new_name(self, prefix: str) -> str:
        self._tmp_counter += 1
        return f"{prefix}_{self._tmp_counter}"

    def _functor_name_for(self, op: str, scalar) -> str:
        key = (op, repr(scalar) if scalar is not None else "")
        if key in self._emitted_functors:
            return self._emitted_functors[key]
        scalar_tag = ""
        if scalar is not None:
            self.scalar_functor_counter += 1
            scalar_tag = f"_v{self.scalar_functor_counter}"
        name = f"Magi_{op}{scalar_tag}"
        self._emitted_functors[key] = name
        self.functor_decls.append(_emit_custom_functor(name, op, scalar))
        return name

    def _compute_op_template(self, node: Compute) -> str:
        if node.op in _BUILTIN_FN_TEMPLATE and node.scalar is None:
            return _BUILTIN_FN_TEMPLATE[node.op]
        # Custom functor — either scalar-baked or unary-no-builtin (e.g. erf).
        return self._functor_name_for(node.op, node.scalar)

    def emit(self) -> str:
        """Walk the IR; return the typedef name of the root EVT type."""
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
            elem_compute = _DTYPE_TO_CUTLASS[node.compute_dtype]
            self.typedef_lines.append(
                f"using {compute_name} = cutlass::epilogue::threadblock::VisitorCompute<\n"
                f"    {fn_template}, {elem_compute}, {elem_compute},\n"
                f"    cutlass::FloatRoundStyle::round_to_nearest>;"
            )
            evt_name = self._new_name(f"EVT_{node.op}")
            child_typedef_list = ", ".join(child_names)
            self.typedef_lines.append(
                f"using {evt_name} = cutlass::epilogue::threadblock::Sm80EVT<\n" f"    {compute_name}, {child_typedef_list}>;"
            )
            return evt_name
        raise TypeError(f"Unknown IR node type: {type(node).__name__}")


def _emit_args_tree(node, leaf_args: Dict[int, str], indent: int = 4) -> str:
    """Emit the nested-brace runtime args literal matching the EVT typedef tree."""
    pad = " " * indent
    if isinstance(node, Accum):
        return f"{pad}{{}}"
    if isinstance(node, (RowBroadcast, ColBroadcast, AuxLoad)):
        return f"{pad}{leaf_args[node.input_idx]}"
    if isinstance(node, Compute):
        children_str = ",\n".join(_emit_args_tree(c, leaf_args, indent + 2) for c in node.children)
        return f"{pad}{{\n" f"{children_str},\n" f"{pad}  {{}}\n" f"{pad}}}"
    raise TypeError(f"Unknown IR node type: {type(node).__name__}")


_KERNEL_PREAMBLE = """\
// AUTO-GENERATED by magi_compiler/passes/piecewise_graph/fusion/sm80/evt_codegen.py
// Do not edit by hand. Regenerate by re-running the FX pass.
//
// IR cache key: {cache_key}

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
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

// AlignmentA / AlignmentB / AlignmentC are baked from the (greedy) bit-width
// chosen at runtime to match the actual K, N, and ldd divisibility — 128
// bits when shapes allow vector loads, 64 bits as a fallback for shapes that
// only meet 8-byte alignment (e.g. K = 12 for bf16). For C the host already
// over-pads D's row stride to a full cache line (see ``_aligned_n_stride``
// in evt_runtime.py), so AlignmentC = 128 is almost always achievable —
// keeping it tunable lets a smaller-padding mode drop to 64 without a
// CUTLASS template rebuild from scratch.
constexpr int AlignmentA = {alignment_a_bits} / cutlass::sizeof_bits<ElementA>::value;
constexpr int AlignmentB = {alignment_b_bits} / cutlass::sizeof_bits<ElementB>::value;
constexpr int AlignmentC = {alignment_c_bits} / cutlass::sizeof_bits<ElementC>::value;

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
  int64_t lda;
  int64_t ldb;
  int64_t ldd;
  // Extras pointers, in IR-leaf order.
  std::vector<void*> ptr_extras;
  // Row strides for AuxLoad extras (stride(0) in elements). Indexed in
  // the same order as ptr_extras; RowBroadcast/ColBroadcast entries are
  // unused but still present so indices stay aligned.
  std::vector<int64_t> stride_extras;
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
    int64_t const lda = a.lda;
    int64_t const ldb = a.ldb;
    typename GemmType::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem,
        /*batch_count=*/1,
        callback_args,
        ptrA, ptrB,
        /*ptr_C=*/nullptr, /*ptr_D=*/nullptr,
        /*batch_stride_A=*/static_cast<int64_t>(M) * lda,
        /*batch_stride_B=*/static_cast<int64_t>(N) * ldb,
        /*batch_stride_C=*/0, /*batch_stride_D=*/0,
        /*stride_a=*/lda,
        /*stride_b=*/ldb,
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
    // A is always row-major (M, K), so its innermost (K) stride must be 1.
    // We don't require A.is_contiguous() because Inductor often hands us a
    // reinterpret_tensor that has the right strides but trips that check.
    TORCH_CHECK(A.stride(1) == 1, "A innermost stride must be 1; got ", A.stride(1));
    TORCH_CHECK(A.stride(0) >= A.size(1),
                "A row stride must be >= K; got stride(0)=", A.stride(0), ", K=", A.size(1));
    // B's stride contract depends on b_layout (substituted at codegen time):
    //   row: B is (K, N) row-major          → B.stride(1) == 1, B.stride(0) >= N
    //   col: B is the underlying (N, K)     → B.stride(1) == 1, B.stride(0) >= K
    //        row-major weight read as
    //        ColumnMajor (K, N) by CUTLASS
    {b_stride_check}

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
    // Real strides from the at::Tensor — handles Inductor reinterpret_tensor
    // cases where lda > K or ldb > size(1). Both stride(0) values are in
    // elements since stride(1) == 1 was just validated above.
    ea.lda = static_cast<int64_t>(A.stride(0));
    ea.ldb = static_cast<int64_t>(B.stride(0));
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

    // Drain any pre-existing CUDA error so we don't blame our first candidate
    // for an upstream failure.
    (void)cudaGetLastError();

    for (size_t i = 0; i < configs_.size(); ++i) {{
      auto& g = configs_[i];
      size_t ws_sz = 0;
      try {{ ws_sz = g->get_workspace_size(ea); }}
      catch (...) {{ (void)cudaGetLastError(); continue; }}
      if (!ws_.defined() || ws_.numel() < (int64_t)ws_sz) {{
        ws_ = at::empty({{(int64_t)ws_sz + 1}},
            at::TensorOptions().dtype(at::kByte).device(at::kCUDA));
      }}
      void* ws_ptr = ws_sz > 0 ? ws_.data_ptr() : nullptr;
      // initialize() can fail synchronously (e.g. cudaFuncSetAttribute returns
      // cudaErrorInvalidValue for tiles whose SharedStorage exceeds the
      // device opt-in cap). Clear the sticky CUDA error before moving on —
      // otherwise the next launch (or post-autotune user run) inherits it
      // and surfaces a misleading "Error Internal" against an unrelated tile.
      if (g->initialize(ea, ws_ptr, stream) != cutlass::Status::kSuccess) {{
        (void)cudaGetLastError();
        continue;
      }}

      // Warmup — 10 iters so L2 / inst caches settle (3 was too few — first
      // timed iter saw a cold L2 and biased the choice towards smaller tiles).
      // Capture run() status and sync return codes so an async launch failure
      // (e.g. invalid grid, latent SMEM issue) disqualifies the tile cleanly.
      bool tile_ok = true;
      for (int w = 0; w < 10; ++w) {{
        if (g->run(stream) != cutlass::Status::kSuccess) {{ tile_ok = false; break; }}
      }}
      if (tile_ok && cudaStreamSynchronize(stream) != cudaSuccess) {{
        tile_ok = false;
      }}
      if (!tile_ok) {{
        (void)cudaGetLastError();
        continue;
      }}

      // Time — 20 iters for ~1% timing noise, matching torch.compile defaults.
      cudaEventRecord(s, stream);
      int iters = 20;
      for (int p = 0; p < iters; ++p) {{
        if (g->run(stream) != cutlass::Status::kSuccess) {{ tile_ok = false; break; }}
      }}
      cudaEventRecord(e, stream);
      if (cudaEventSynchronize(e) != cudaSuccess) tile_ok = false;
      if (!tile_ok) {{
        (void)cudaGetLastError();
        continue;
      }}
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
    ir: Store,
    a_dtype: str,
    b_dtype: str,
    cache_key_str: str = "",
    b_layout: str = "row",
    m_bucket: str = "medium",
    alignment_a_bits: int = 128,
    alignment_b_bits: int = 128,
    alignment_c_bits: int = 128,
    arch: str = "sm120",
) -> str:
    """Render a complete .cu source for the given EVT IR.

    ``b_layout``: "row" = B is (K, N) RowMajor; "col" = underlying (N, K) weight
    read as ColumnMajor. ``m_bucket`` selects the tile-candidate set for autotune.
    ``alignment_*_bits``: greedy-picked 128 or 64 to match actual shape divisibility.
    ``arch`` accepted for signature parity with sm90 renderer; ignored here.
    """
    if b_layout not in ("row", "col"):
        raise ValueError(f"b_layout must be 'row' or 'col', got {b_layout!r}")
    if m_bucket not in _TILE_CANDIDATES_SM120:
        raise ValueError(f"unknown m_bucket {m_bucket!r}; " f"expected one of {list(_TILE_CANDIDATES_SM120)}")
    if (
        alignment_a_bits not in _VALID_ALIGN_BITS
        or alignment_b_bits not in _VALID_ALIGN_BITS
        or alignment_c_bits not in _VALID_ALIGN_BITS
    ):
        raise ValueError(
            f"alignment_*_bits must be one of {_VALID_ALIGN_BITS}; "
            f"got A={alignment_a_bits}, B={alignment_b_bits}, C={alignment_c_bits}"
        )
    if not isinstance(ir, Store):
        raise TypeError("render_evt_cu expects a Store node as root")
    del arch
    tile_candidate_block = _emit_tile_candidates(m_bucket)

    a_elem = _DTYPE_TO_CUTLASS[a_dtype]
    b_elem = _DTYPE_TO_CUTLASS[b_dtype]
    c_elem = _DTYPE_TO_CUTLASS[ir.out_dtype]

    emitter = _EvtEmitter(ir)
    evt_root = emitter.emit()

    leaves = walk_leaves(ir)
    leaf_args: Dict[int, str] = {}
    for leaf in leaves:
        if not isinstance(leaf, (RowBroadcast, ColBroadcast, AuxLoad)):
            continue
        elem = _DTYPE_TO_CUTLASS[leaf.dtype]
        ptr_expr = f"reinterpret_cast<{elem}*>(a.ptr_extras[{leaf.input_idx}])"
        if isinstance(leaf, RowBroadcast):
            leaf_args[leaf.input_idx] = f"{{{ptr_expr}, {elem}(0), {{_0{{}}, _1{{}}, int32_t(N)}}}}"
        elif isinstance(leaf, ColBroadcast):
            leaf_args[leaf.input_idx] = f"{{{ptr_expr}, {elem}(0), {{_1{{}}, _0{{}}, int32_t(M)}}}}"
        else:  # AuxLoad
            stride_expr = f"a.stride_extras[{leaf.input_idx}]"
            mn_expr = f"(static_cast<int64_t>(M) * {stride_expr})"
            leaf_args[leaf.input_idx] = f"{{{ptr_expr}, {elem}(0), {{{stride_expr}, _1{{}}, {mn_expr}}}}}"

    args_tree = _emit_args_tree(ir.child, leaf_args, indent=8)

    # Dedup by input_idx — same tensor may appear at multiple IR leaves.
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
        if isinstance(leaf, RowBroadcast):
            extras_validation_lines.append(f'    TORCH_CHECK(extras[{i}].numel() == N, "extras[{i}] must have N elements");')
        elif isinstance(leaf, ColBroadcast):
            extras_validation_lines.append(f'    TORCH_CHECK(extras[{i}].numel() == M, "extras[{i}] must have M elements");')
        elif isinstance(leaf, AuxLoad):
            extras_validation_lines.append(
                f'    TORCH_CHECK(extras[{i}].size(0) == M && extras[{i}].size(1) == N,' f' "extras[{i}] must be (M,N)");'
            )
            extras_validation_lines.append(
                f'    TORCH_CHECK(extras[{i}].stride(1) == 1 && extras[{i}].stride(0) >= N,'
                f' "extras[{i}] must be row-major with stride(1)==1 and stride(0)>=N");'
            )
        extras_validation_lines.append(
            f'    TORCH_CHECK(extras[{i}].scalar_type() == {at_dtype},' f' "extras[{i}] must be {leaf.dtype}");'
        )
        extras_validation_lines.append(f'    TORCH_CHECK(extras[{i}].is_cuda(), "extras[{i}] must be CUDA");')
        extras_ptr_lines.append(f"    ea.ptr_extras.push_back(static_cast<void*>(" f"extras[{i}].data_ptr<{at_cpp}>()));")
        extras_ptr_lines.append(f"    ea.stride_extras.push_back(static_cast<int64_t>(extras[{i}].stride(0)));")

    extras_validation = "\n".join(extras_validation_lines) if extras_validation_lines else "    // no extras"
    extras_ptrs = "\n".join(extras_ptr_lines) if extras_ptr_lines else ""

    functor_decls = "\n".join(emitter.functor_decls) if emitter.functor_decls else "// (no custom functors)"
    typedef_block = "\n".join("  " + l if l.strip() else l for l in "\n".join(emitter.typedef_lines).split("\n"))

    cutlass_b_layout = "RowMajor" if b_layout == "row" else "ColumnMajor"
    if b_layout == "row":
        n_dim_expr = "B.size(1)"
        stride_b_expr = "N"
        b_stride_check = (
            'TORCH_CHECK(B.stride(1) == 1, "B innermost stride must be 1; got ", B.stride(1));\n'
            '    TORCH_CHECK(B.stride(0) >= B.size(1),\n'
            '                "B row stride must be >= N; got stride(0)=", B.stride(0), ", N=", B.size(1));'
        )
    else:
        n_dim_expr = "B.size(0)"
        stride_b_expr = "K"
        b_stride_check = (
            'TORCH_CHECK(B.stride(1) == 1, "B innermost stride must be 1; got ", B.stride(1));\n'
            '    TORCH_CHECK(B.stride(0) >= B.size(1),\n'
            '                "B row stride must be >= K; got stride(0)=", B.stride(0), ", K=", B.size(1));'
        )

    preamble = _KERNEL_PREAMBLE.format(
        cache_key=cache_key_str,
        functor_decls=functor_decls,
        a_elem=a_elem,
        b_elem=b_elem,
        c_elem=c_elem,
        typedef_block=typedef_block,
        evt_root_name=evt_root,
        b_layout=cutlass_b_layout,
        alignment_a_bits=alignment_a_bits,
        alignment_b_bits=alignment_b_bits,
        alignment_c_bits=alignment_c_bits,
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
        b_stride_check=b_stride_check,
        tile_candidate_block=tile_candidate_block,
    )
    return preamble + launcher
