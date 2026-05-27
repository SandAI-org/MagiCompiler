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

"""Render a CUTLASS 3.x Sm90EVT .cu source from an EVT IR tree — H100 path.

Uses TMA + WGMMA via warp-specialized collective builders; ~1.6-2x faster
than the SM80 path on H100. Selected by ``evt_runtime`` when arch == sm_90.

All AuxLoad nodes use ``Sm90AuxLoad<0>`` (inline ld.global, no SMEM
staging). The C-operand TMA channel is left unused (ptr_C = nullptr).
The same ``AuxLoad.input_idx`` may appear at multiple positions in the
EVT tree (matching SM80 behaviour); the leaf-args dict produces
identical expressions for the same index so the overwrite is harmless.
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

# (TM, TN, TK, CM, CN, CK, schedule, label).
# Cluster_M=1 → Pingpong; Cluster_M>=2 → Cooperative. Mismatched combos
# fail at can_implement and are skipped by autotune.
# H100: 132 SMs, 228 KB SMEM / SM.

_TILE_CANDIDATES_SM90: dict = {
    "small": [
        (64, 128, 64, 1, 1, 1, "pingpong", "T<64,128,64>_Cl<1,1,1>_PP"),
        (64, 256, 64, 1, 1, 1, "pingpong", "T<64,256,64>_Cl<1,1,1>_PP"),
        (128, 128, 64, 1, 1, 1, "pingpong", "T<128,128,64>_Cl<1,1,1>_PP"),
        (128, 256, 64, 1, 1, 1, "pingpong", "T<128,256,64>_Cl<1,1,1>_PP"),
        (64, 128, 128, 1, 1, 1, "pingpong", "T<64,128,128>_Cl<1,1,1>_PP"),
        (64, 256, 128, 1, 1, 1, "pingpong", "T<64,256,128>_Cl<1,1,1>_PP"),
    ],
    "medium": [
        (128, 128, 64, 1, 1, 1, "pingpong", "T<128,128,64>_Cl<1,1,1>_PP"),
        (128, 256, 64, 1, 1, 1, "pingpong", "T<128,256,64>_Cl<1,1,1>_PP"),
        (128, 128, 64, 2, 1, 1, "cooperative", "T<128,128,64>_Cl<2,1,1>_CO"),
        (128, 256, 64, 2, 1, 1, "cooperative", "T<128,256,64>_Cl<2,1,1>_CO"),
        (256, 128, 64, 2, 1, 1, "cooperative", "T<256,128,64>_Cl<2,1,1>_CO"),
        (256, 256, 64, 2, 1, 1, "cooperative", "T<256,256,64>_Cl<2,1,1>_CO"),
    ],
    "large": [
        (128, 256, 64, 2, 1, 1, "cooperative", "T<128,256,64>_Cl<2,1,1>_CO"),
        (256, 128, 64, 2, 1, 1, "cooperative", "T<256,128,64>_Cl<2,1,1>_CO"),
        (256, 256, 64, 2, 1, 1, "cooperative", "T<256,256,64>_Cl<2,1,1>_CO"),
        (128, 256, 64, 2, 2, 1, "cooperative", "T<128,256,64>_Cl<2,2,1>_CO"),
        (256, 128, 64, 2, 2, 1, "cooperative", "T<256,128,64>_Cl<2,2,1>_CO"),
        (256, 256, 64, 2, 2, 1, "cooperative", "T<256,256,64>_Cl<2,2,1>_CO"),
    ],
}


_SCHEDULE_TYPES = {
    "pingpong": ("cutlass::gemm::KernelTmaWarpSpecializedPingpong", "cutlass::epilogue::TmaWarpSpecialized"),
    "cooperative": ("cutlass::gemm::KernelTmaWarpSpecializedCooperative", "cutlass::epilogue::TmaWarpSpecializedCooperative"),
}


def _emit_tile_candidates(m_bucket: str) -> str:
    """Emit C++ EVT_TILE_CANDIDATE(...) statements for the given M bucket."""
    candidates = _TILE_CANDIDATES_SM90.get(m_bucket, _TILE_CANDIDATES_SM90["medium"])
    lines = []
    for tm, tn, tk, cm, cn, ck, schedule, label in candidates:
        kernel_sched, epi_sched = _SCHEDULE_TYPES[schedule]
        lines.append(
            f"    EVT_TILE_CANDIDATE(" f"{tm}, {tn}, {tk}, {cm}, {cn}, {ck}, " f"{kernel_sched}, {epi_sched}, " f'"{label}");'
        )
    return "\n".join(lines)


class _Sm90EvtEmitter:
    """Bottom-up walker emitting Sm90EVT typedef chains.

    Unlike SM80, there is no Store wrapper — the CollectiveEpilogue owns
    the store; the EVT root is the topmost compute node.
    """

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
        return self._functor_name_for(node.op, node.scalar)

    def emit(self) -> str:
        """Walk the IR and return the typedef name of the EVT root."""
        return self._emit_node(self.root.child)

    def _emit_node(self, node) -> str:
        if isinstance(node, Accum):
            name = self._new_name("AccFetch")
            self.typedef_lines.append(f"using {name} = cutlass::epilogue::fusion::Sm90AccFetch;")
            return name
        if isinstance(node, RowBroadcast):
            name = self._new_name("RowBcast")
            elem = _DTYPE_TO_CUTLASS[node.dtype]
            self.typedef_lines.append(
                f"using {name} = cutlass::epilogue::fusion::Sm90RowBroadcast<\n"
                f"    /*Stages=*/0, TileShape, {elem}, ElementCompute>;"
            )
            self.leaf_typedefs.append((name, "row_bcast", node.input_idx, node.dtype))
            return name
        if isinstance(node, ColBroadcast):
            name = self._new_name("ColBcast")
            elem = _DTYPE_TO_CUTLASS[node.dtype]
            self.typedef_lines.append(
                f"using {name} = cutlass::epilogue::fusion::Sm90ColBroadcast<\n"
                f"    /*Stages=*/0, TileShape, {elem}, ElementCompute>;"
            )
            self.leaf_typedefs.append((name, "col_bcast", node.input_idx, node.dtype))
            return name
        if isinstance(node, AuxLoad):
            elem = _DTYPE_TO_CUTLASS[node.dtype]
            name = self._new_name("AuxLoad")
            self.typedef_lines.append(
                f"using {name} = cutlass::epilogue::fusion::Sm90AuxLoad<\n"
                f"    /*Stages=*/0, /*EpilogueTile=*/void, {elem},\n"
                f"    cutlass::layout::RowMajor, /*SmemLayoutAtom=*/void, /*CopyOpS2R=*/void>;"
            )
            self.leaf_typedefs.append((name, "aux_load_inline", node.input_idx, node.dtype))
            return name
        if isinstance(node, Compute):
            child_names = [self._emit_node(c) for c in node.children]
            compute_name = self._new_name(f"Cmp_{node.op}")
            fn_template = self._compute_op_template(node)
            elem_compute = _DTYPE_TO_CUTLASS[node.compute_dtype]
            self.typedef_lines.append(
                f"using {compute_name} = cutlass::epilogue::fusion::Sm90Compute<\n"
                f"    {fn_template}, {elem_compute}, {elem_compute},\n"
                f"    cutlass::FloatRoundStyle::round_to_nearest>;"
            )
            evt_name = self._new_name(f"EVT_{node.op}")
            child_typedef_list = ", ".join(child_names)
            self.typedef_lines.append(
                f"using {evt_name} = cutlass::epilogue::fusion::Sm90EVT<\n" f"    {compute_name}, {child_typedef_list}>;"
            )
            return evt_name
        raise TypeError(f"Unknown IR node type: {type(node).__name__}")


def _emit_args_tree(node, leaf_args: Dict[int, str], indent: int = 8) -> str:
    """Emit the nested-brace runtime args literal mirroring the Sm90EVT tree."""
    pad = " " * indent
    if isinstance(node, Accum):
        return f"{pad}{{}}"
    if isinstance(node, (AuxLoad, RowBroadcast, ColBroadcast)):
        return f"{pad}{leaf_args[node.input_idx]}"
    if isinstance(node, Compute):
        children_str = ",\n".join(_emit_args_tree(c, leaf_args, indent + 2) for c in node.children)
        return f"{pad}{{\n" f"{children_str},\n" f"{pad}  {{}}\n" f"{pad}}}"  # this Sm90Compute's op args (always empty)
    raise TypeError(f"Unknown IR node type: {type(node).__name__}")


_KERNEL_PREAMBLE_SM90 = """\
// AUTO-GENERATED by magi_compiler/passes/piecewise_graph/fusion/sm90/evt_codegen.py
// Do not edit by hand. Regenerate by re-running the FX pass.
//
// IR cache key: {cache_key}

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <memory>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/functional.h"
#include "cutlass/fast_math.h"

#include "cute/tensor.hpp"

#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cutlass/util/packed_stride.hpp"

using namespace cute;

////////////////////////////////////////////////////////////////////////////////
// Custom functors (one per unique scalar-baked op or non-builtin unary).
////////////////////////////////////////////////////////////////////////////////
{functor_decls}

////////////////////////////////////////////////////////////////////////////////
// Data types and layouts
////////////////////////////////////////////////////////////////////////////////

using ElementA       = {a_elem};
using ElementB       = {b_elem};
// C-operand TMA channel is unused (all AuxLoad nodes use Sm90AuxLoad<0>
// which loads via ld.global). ElementC = ElementD; ptr_C = nullptr.
using ElementC       = {c_elem};
using ElementD       = {d_elem};
using ElementAccumulator = float;
using ElementCompute     = float;

using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::{b_layout};
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;

constexpr int AlignmentA = {alignment_a_bits} / cutlass::sizeof_bits<ElementA>::value;
constexpr int AlignmentB = {alignment_b_bits} / cutlass::sizeof_bits<ElementB>::value;
constexpr int AlignmentC = {alignment_c_bits} / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = {alignment_c_bits} / cutlass::sizeof_bits<ElementD>::value;

using ArchTag       = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;

constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;

////////////////////////////////////////////////////////////////////////////////
// Per-tile-config GEMM type. The Sm90 EVT typedefs reference TileShape (each
// Sm90RowBroadcast / Sm90ColBroadcast bakes the tile dims into its on-the-fly
// loader), and CollectiveBuilder consumes (TileShape, ClusterShape, Schedule)
// — so every autotune candidate must re-instantiate the entire EVT chain +
// CollectiveEpilogue + CollectiveMainloop + GemmKernel. We package the whole
// tree inside a template struct keyed on the four tile/cluster/schedule
// parameters so each candidate is a distinct C++ type that can live side-by-
// side in ``configs_``.
////////////////////////////////////////////////////////////////////////////////

template <class TileShape_, class ClusterShape_, class KernelSchedule_, class EpilogueSchedule_>
struct EvtConfig {{
  using TileShape        = TileShape_;
  using ClusterShape     = ClusterShape_;
  using KernelSchedule   = KernelSchedule_;
  using EpilogueSchedule = EpilogueSchedule_;

  ////////////////////////////////////////////////////////////////////////////
  // EVT (Sm90 Epilogue Visitor Tree) typedefs — generated from the IR.
  // No outermost StoreD wrapper — the CollectiveEpilogue owns the store; the
  // EVT root is the topmost compute / leaf node.
  ////////////////////////////////////////////////////////////////////////////
{typedef_block}

  using FusionCallbacks = {evt_root_name};

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutCTag, AlignmentC,
      ElementD, LayoutDTag, AlignmentD,
      EpilogueSchedule,
      FusionCallbacks
    >::CollectiveOp;

  // AutoCarveout picks the max stages that fit in the actual epilogue's
  // SharedStorage footprint for the target arch. On H100 this lands on ~6-7
  // stages for typical TileShape<128,128,64>; bigger tiles automatically get
  // fewer stages. Aggressive choice is safe because this codegen is sm_90-
  // only (the runtime dispatcher routes other arches to sm80/evt_codegen.py).
  using StageCountType = cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutATag, AlignmentA,
      ElementB, LayoutBTag, AlignmentB,
      ElementAccumulator,
      TileShape, ClusterShape,
      StageCountType,
      KernelSchedule
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
}};

////////////////////////////////////////////////////////////////////////////////
// Autotune runner — one candidate per (TileShape, ClusterShape, Schedule)
// tuple; first call at a new (M, N, K) tuple times every candidate that
// can_implement accepts and caches the winner.
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
  // Extras pointers, in IR-leaf order. Each AuxLoad / RowBroadcast /
  // ColBroadcast looks up its pointer from this vector by its IR
  // input_idx baked into the launcher.
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
  virtual cutlass::Status can_implement(const EvtArgs&) = 0;
  virtual cutlass::Status initialize(const EvtArgs&, void* ws, cudaStream_t s) = 0;
  virtual cutlass::Status run(cudaStream_t stream) = 0;
  virtual const char* name() const = 0;
}};

template <class Cfg>
class EvtImpl : public EvtConcept {{
 public:
  using GemmType = typename Cfg::Gemm;
  using StrideA  = typename Cfg::StrideA;
  using StrideB  = typename Cfg::StrideB;
  using StrideC  = typename Cfg::StrideC;
  using StrideD  = typename Cfg::StrideD;

  explicit EvtImpl(const char* name) : name_(name) {{}}

  typename GemmType::Arguments make_args(const EvtArgs& a) {{
    auto ptrA = reinterpret_cast<ElementA const*>(a.ptr_A);
    auto ptrB = reinterpret_cast<ElementB const*>(a.ptr_B);
    auto ptrD = reinterpret_cast<ElementD*>      (a.ptr_D);
    int const M = a.M;
    int const N = a.N;
    int const K = a.K;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{{}}, cute::make_shape(M, K, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{{}}, cute::make_shape(N, K, 1));
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{{}}, cute::make_shape(M, N, 1));
    // D's row stride comes from the actual tensor (ea.ldd = D.stride(0)),
    // which may be larger than N when the runtime pads the output buffer to
    // a 16-byte boundary.  Using N here would give TMA a wrong
    // globalStride, corrupting every row after the first.
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{{}}, cute::make_shape(M, static_cast<int>(a.ldd), 1));
    // Per-AuxLoad strides — each extra may have a different row stride
    // (e.g. padded buffers where stride(0) > N). Emitted unconditionally;
    // nvcc -O3 drops unused variables.
{aux_stride_decls}

    // C-operand TMA channel unused — all AuxLoad nodes use Sm90AuxLoad<0>
    // (inline ld.global). ptr_C is nullptr; no node reports
    // is_C_load_needed()=true so CollectiveEpilogue skips the C TMA load.
    auto ptrC = {ptr_C_expr_in_make_args};

    typename GemmType::Arguments args{{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {{M, N, K, 1}},
      {{ ptrA, stride_A, ptrB, stride_B }},
      {{   // epilogue args = ( FusionCallbacks_args, ptr_C, stride_C, ptr_D, stride_D )
{args_tree},
        ptrC,  stride_C,
        ptrD,  stride_D
      }}
    }};
    return args;
  }}

  size_t get_workspace_size(const EvtArgs& a) override {{
    auto args = make_args(a);
    return GemmType::get_workspace_size(args);
  }}
  cutlass::Status can_implement(const EvtArgs& a) override {{
    auto args = make_args(a);
    return gemm_.can_implement(args);
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
// Python-facing launcher — same evt_matmul_out signature as the SM80 path
// so the dispatcher in evt_runtime.py picks up the same attribute name.
////////////////////////////////////////////////////////////////////////////////
"""


_LAUNCHER_TEMPLATE_SM90 = """\
////////////////////////////////////////////////////////////////////////////////
// Tile candidate registration. Each EVT_TILE_CANDIDATE invocation instantiates
// the full EvtConfig — EVT typedef tree + CollectiveEpilogue + CollectiveMain-
// loop + GemmKernel — for that (TileShape, ClusterShape, Schedule) tuple.
// Compile time grows linearly with the candidate count; bucket lists are kept
// at ~6 candidates each. Mismatched (schedule, cluster) combos compile fine
// but die at can_implement and are skipped silently by autotune().
////////////////////////////////////////////////////////////////////////////////

#define EVT_TILE_CANDIDATE(tm, tn, tk, cm, cn, ck, kernel_sched, epi_sched, label) \\
  configs_.push_back(std::make_unique<EvtImpl<EvtConfig<                            \\
      Shape<Int<tm>, Int<tn>, Int<tk>>,                                             \\
      Shape<Int<cm>, Int<cn>, Int<ck>>,                                             \\
      kernel_sched, epi_sched>>>(label))

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
    TORCH_CHECK(D.scalar_type() == {d_at_dtype}, "D must be {d_dtype}");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && D.dim() == 2, "A, B, D must be 2D");
    // Stride-based contiguity (Inductor's reinterpret_tensor often trips
    // .is_contiguous() with the "right" strides).
    TORCH_CHECK(A.stride(1) == 1, "A innermost stride must be 1; got ", A.stride(1));
    TORCH_CHECK(A.stride(0) >= A.size(1),
                "A row stride must be >= K; got stride(0)=", A.stride(0), ", K=", A.size(1));
    {b_stride_check}

    int const M = static_cast<int>(A.size(0));
    int const K = static_cast<int>(A.size(1));
    int const N = static_cast<int>({n_dim_expr});

    TORCH_CHECK(D.size(0) == M && D.size(1) == N,
                "D must be (M, N); got ", D.sizes());
    TORCH_CHECK(D.stride(1) == 1, "D innermost stride must be 1; got ", D.stride(1));
    TORCH_CHECK(D.stride(0) >= N,
                "D row stride must be >= N; got stride(0)=", D.stride(0), ", N=", N);
    TORCH_CHECK(extras.size() == {n_extras}, "expected {n_extras} extra tensors, got ", extras.size());

{extras_validation}

    const c10::cuda::CUDAGuard guard(A.device());
    auto stream = at::cuda::getCurrentCUDAStream(A.device().index()).stream();

    EvtArgs ea;
    ea.M = M; ea.N = N; ea.K = K;
    ea.ptr_A = A.data_ptr<{a_at_cpp}>();
    ea.ptr_B = B.data_ptr<{b_at_cpp}>();
    ea.ptr_D = D.data_ptr<{d_at_cpp}>();
    ea.lda = static_cast<int64_t>(A.stride(0));
    ea.ldb = static_cast<int64_t>(B.stride(0));
    ea.ldd = static_cast<int64_t>(D.stride(0));
    ea.ptr_extras.reserve({n_extras});
{extras_ptrs}

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
                "Sm90 EVT init failed (", gemm->name(), "): ", cutlassGetStatusString(st));
    st = gemm->run(stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess,
                "Sm90 EVT run failed (", gemm->name(), "): ", cutlassGetStatusString(st));
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
      // can_implement gates illegal (schedule, cluster) combos and shapes
      // that don't satisfy the kernel's M/N/K divisibility — these would
      // crash at initialize() otherwise.
      if (g->can_implement(ea) != cutlass::Status::kSuccess) continue;
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
                "Sm90 EVT AutoTune: no candidate succeeded for (M,N,K)=(",
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
                    std::vector<at::Tensor> extras, at::Tensor D) {{
  runner()(std::move(A), std::move(B), std::move(extras), std::move(D));
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
  m.doc() = "Magi compiler EVT-fused matmul (Sm90 TMA + WGMMA, autotune)";
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
    arch: str = "sm90",
) -> str:
    """Render the SM90 .cu source for ``ir``."""
    if b_layout not in ("row", "col"):
        raise ValueError(f"b_layout must be 'row' or 'col', got {b_layout!r}")
    if m_bucket not in _TILE_CANDIDATES_SM90:
        raise ValueError(f"unknown m_bucket {m_bucket!r}; " f"expected one of {list(_TILE_CANDIDATES_SM90)}")
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
        raise TypeError("render_evt_cu (sm90) expects a Store node as root")
    del arch

    a_elem = _DTYPE_TO_CUTLASS[a_dtype]
    b_elem = _DTYPE_TO_CUTLASS[b_dtype]
    d_elem = _DTYPE_TO_CUTLASS[ir.out_dtype]

    emitter = _Sm90EvtEmitter(ir)
    evt_root = emitter.emit()

    # No Sm90SrcFetch — the C-operand TMA channel is unused (ptr_C = nullptr).
    # ElementC must still be a concrete type for the CollectiveBuilder template.
    c_elem = d_elem

    leaves = walk_leaves(ir)
    leaf_args: Dict[int, str] = {}
    aux_stride_decl_lines: List[str] = []
    extras_validation_lines: List[str] = []
    extras_ptr_lines: List[str] = []
    seen_extras: set = set()
    extra_leaves = [n for n in leaves if not isinstance(n, Accum)]
    n_extras = max((leaf.input_idx for leaf in extra_leaves), default=-1) + 1
    for leaf in extra_leaves:
        i = leaf.input_idx
        elem = _DTYPE_TO_CUTLASS[leaf.dtype]
        if isinstance(leaf, RowBroadcast):
            ptr_expr = f"reinterpret_cast<{elem} const*>(a.ptr_extras[{i}])"
            leaf_args[i] = f"{{ {ptr_expr} }}"
        elif isinstance(leaf, ColBroadcast):
            ptr_expr = f"reinterpret_cast<{elem} const*>(a.ptr_extras[{i}])"
            leaf_args[i] = f"{{ {ptr_expr} }}"
        elif isinstance(leaf, AuxLoad):
            ptr_expr = f"reinterpret_cast<{elem} const*>(a.ptr_extras[{i}])"
            stride_var = f"stride_aux_{i}"
            leaf_args[i] = f"{{ {ptr_expr}, {elem}(0), {stride_var} }}"
            if i not in seen_extras:
                aux_stride_decl_lines.append(
                    f"    auto {stride_var} = cutlass::make_cute_packed_stride(\n"
                    f"        cute::Stride<int64_t, cute::_1, int64_t>{{}},\n"
                    f"        cute::make_shape(M, static_cast<int>(a.stride_extras[{i}]), 1));"
                )

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

    args_tree = _emit_args_tree(ir.child, leaf_args, indent=8)

    ptr_C_expr_in_make_args = "static_cast<ElementC const*>(nullptr)"

    extras_validation = "\n".join(extras_validation_lines) if extras_validation_lines else "    // no extras"
    extras_ptrs = "\n".join(extras_ptr_lines) if extras_ptr_lines else ""
    aux_stride_decls = "\n".join(aux_stride_decl_lines) if aux_stride_decl_lines else "    // (no AuxLoad strides)"

    functor_decls = "\n".join(emitter.functor_decls) if emitter.functor_decls else "// (no custom functors)"
    typedef_block = "\n".join("  " + l if l.strip() else l for l in "\n".join(emitter.typedef_lines).split("\n"))

    cutlass_b_layout = "RowMajor" if b_layout == "row" else "ColumnMajor"
    if b_layout == "row":
        n_dim_expr = "B.size(1)"
        b_stride_check = (
            'TORCH_CHECK(B.stride(1) == 1, "B innermost stride must be 1; got ", B.stride(1));\n'
            '    TORCH_CHECK(B.stride(0) >= B.size(1),\n'
            '                "B row stride must be >= N; got stride(0)=", B.stride(0), ", N=", B.size(1));'
        )
    else:
        n_dim_expr = "B.size(0)"
        b_stride_check = (
            'TORCH_CHECK(B.stride(1) == 1, "B innermost stride must be 1; got ", B.stride(1));\n'
            '    TORCH_CHECK(B.stride(0) >= B.size(1),\n'
            '                "B row stride must be >= K; got stride(0)=", B.stride(0), ", K=", B.size(1));'
        )

    tile_candidate_block = _emit_tile_candidates(m_bucket)

    preamble = _KERNEL_PREAMBLE_SM90.format(
        cache_key=cache_key_str,
        functor_decls=functor_decls,
        a_elem=a_elem,
        b_elem=b_elem,
        c_elem=c_elem,
        d_elem=d_elem,
        b_layout=cutlass_b_layout,
        alignment_a_bits=alignment_a_bits,
        alignment_b_bits=alignment_b_bits,
        alignment_c_bits=alignment_c_bits,
        typedef_block=typedef_block,
        evt_root_name=evt_root,
        ptr_C_expr_in_make_args=ptr_C_expr_in_make_args,
        args_tree=args_tree,
        aux_stride_decls=aux_stride_decls,
    )
    launcher = _LAUNCHER_TEMPLATE_SM90.format(
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        d_dtype=ir.out_dtype,
        a_at_dtype=_DTYPE_TO_AT[a_dtype],
        b_at_dtype=_DTYPE_TO_AT[b_dtype],
        d_at_dtype=_DTYPE_TO_AT[ir.out_dtype],
        a_at_cpp=_DTYPE_TO_AT_CPP[a_dtype],
        b_at_cpp=_DTYPE_TO_AT_CPP[b_dtype],
        d_at_cpp=_DTYPE_TO_AT_CPP[ir.out_dtype],
        n_dim_expr=n_dim_expr,
        b_stride_check=b_stride_check,
        n_extras=n_extras,
        extras_validation=extras_validation,
        extras_ptrs=extras_ptrs,
        tile_candidate_block=tile_candidate_block,
    )
    return preamble + launcher
