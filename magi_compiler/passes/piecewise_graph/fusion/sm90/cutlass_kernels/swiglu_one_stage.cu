// Copyright (c) 2026 SandAI. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Single-kernel fully-fused swiglu on Hopper (sm_90a) using the vendored
// Sm90 DualGemm (TMA + WGMMA, warp-specialized cooperative consumer
// warpgroups). User contract is byte-for-byte identical to the SM80
// sibling at ../../sm80/cutlass_kernels/swiglu_one_stage.cu — same Python
// signature, same B gate/linear interleaved layout (ldB = 2K col-major
// view), same SwArgs shape, same stride-based input checks.
//
//   D = swiglu(A @ B.T)
//
//   A : (M, K)   bf16 row-major
//   B : (N, K)   bf16 row-major   (torch.nn.Linear weight convention; N even)
//   D : (M, N/2) bf16 row-major   (strided view of (M, ldd) host-padded buffer)
//
// AUTOTUNE: at first call per (M, N, K) tuple the runner times every
// registered (TileShape, Stages) candidate and caches the fastest one. The
// candidate set targets H100's ~228 KiB dynamic-smem budget; per-stage smem
// for Sm90DualGemm = (BM + 2*BN) * BK * 2 (bf16) * stages.
//
// Built by magi_compiler/passes/piecewise_graph/fusion/cutlass_fusion/
// evt_runtime.py::_compile_swiglu_dual when the live device's compute
// capability is sm_90; everything else routes to the SM80 sibling.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <memory>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/scale_type.h"

// Vendored at cutlass_kernels/hopper_dual_gemm/. Resolved by adding
// cutlass_kernels/ itself to nvcc's extra_include_paths in evt_runtime.py.
#include "hopper_dual_gemm/device/sm90_dual_gemm.h"
#include "swiglu_combine.h"

////////////////////////////////////////////////////////////////////////////////
// Data types
////////////////////////////////////////////////////////////////////////////////

using ElementA       = cutlass::bfloat16_t;
using ElementB       = cutlass::bfloat16_t;
using ElementC       = cutlass::bfloat16_t;
using ElementAcc     = float;
using ElementCompute = float;

using LayoutA  = cutlass::layout::RowMajor;
using LayoutB0 = cutlass::layout::ColumnMajor;   // strided ldB = 2K view
using LayoutB1 = cutlass::layout::ColumnMajor;   // strided ldB = 2K view
using LayoutC  = cutlass::layout::RowMajor;

// Greedy-picked on the host side via -DMAGI_SWIGLU_ALIGN_*_BITS — same macro
// plumbing as the sm_80 path. Defaults give 128-bit (8 elem for bf16) loads /
// stores; the host can drop to 64-bit when a shape only meets 8B alignment.
#ifndef MAGI_SWIGLU_ALIGN_A_BITS
#define MAGI_SWIGLU_ALIGN_A_BITS 128
#endif
#ifndef MAGI_SWIGLU_ALIGN_B_BITS
#define MAGI_SWIGLU_ALIGN_B_BITS 128
#endif
#ifndef MAGI_SWIGLU_ALIGN_C_BITS
#define MAGI_SWIGLU_ALIGN_C_BITS 128
#endif
constexpr int AlignmentA = MAGI_SWIGLU_ALIGN_A_BITS / cutlass::sizeof_bits<ElementA>::value;
constexpr int AlignmentB = MAGI_SWIGLU_ALIGN_B_BITS / cutlass::sizeof_bits<ElementB>::value;
constexpr int EpilogueVecCount = MAGI_SWIGLU_ALIGN_C_BITS / cutlass::sizeof_bits<ElementC>::value;

constexpr auto kScaleType        = cutlass::epilogue::thread::ScaleType::Nothing;
constexpr bool kSplitKSerial     = false;
constexpr bool kStoreD0          = false;
constexpr bool kStoreD1          = false;

////////////////////////////////////////////////////////////////////////////////
// Per-tile Sm90DualGemm wrapper. Each autotune candidate instantiates the
// full kernel for its (TileShape, Stages) tuple. Compile time grows linearly
// with the candidate count — keep the set small and shape-relevant.
////////////////////////////////////////////////////////////////////////////////

template <class TileShape_, int Stages_>
struct DualGemmConfigSm90 {
  using TileShape = TileShape_;
  static constexpr int kStages = Stages_;

  using EpilogueOp0 = cutlass::epilogue::thread::LinearCombination<
      ElementC, EpilogueVecCount, ElementAcc, ElementCompute, kScaleType>;
  using EpilogueOp1 = cutlass::epilogue::thread::LinearCombination<
      ElementC, EpilogueVecCount, ElementAcc, ElementCompute, kScaleType>;
  using EpilogueOp2 = cutlass::epilogue::thread::SwigluCombine<
      ElementC, EpilogueVecCount, ElementAcc, ElementCompute>;

  using Gemm = cutlass::gemm::device::Sm90DualGemm<
      ElementA, LayoutA,
      ElementB, LayoutB0, LayoutB1,
      ElementC, LayoutC,
      ElementAcc,
      TileShape,
      EpilogueOp0, EpilogueOp1, EpilogueOp2,
      kStages,
      kStoreD0, kStoreD1, kSplitKSerial,
      AlignmentA, AlignmentB>;
};

////////////////////////////////////////////////////////////////////////////////
// Type-erased runner concept; one instance per autotune candidate.
// Same SwArgs layout as the sm_80 path — keeps the host wrapper identical.
////////////////////////////////////////////////////////////////////////////////

struct SwArgs {
  int M;          // activations rows
  int N_out;      // = N/2  (output cols)
  int K;
  void* ptr_A;
  void* ptr_B;    // (N, K) row-major weight; gate/linear interleaved
  void* ptr_D;    // (M, N_out) — strided view of an (M, ldd) padded buffer
  int64_t ldd;    // D's row stride in elements; >= N_out, multiple of EpilogueVecCount
  float alpha;    // silu_alpha scaling: x * sigmoid(alpha * x)
  float limit;    // clamp bound: clamp(gate, max=limit), clamp(linear, -limit, limit)
  float one;      // additive offset: (x_linear + one)
};

class SwSm90Concept {
 public:
  virtual ~SwSm90Concept() = default;
  virtual size_t get_workspace_size(const SwArgs&) = 0;
  virtual cutlass::Status initialize(const SwArgs&, void* ws, cudaStream_t) = 0;
  virtual cutlass::Status run(cudaStream_t stream) = 0;
  virtual const char* name() const = 0;
};

template <class Cfg>
class SwSm90Impl : public SwSm90Concept {
 public:
  using GemmType = typename Cfg::Gemm;
  using EpilogueOp0 = typename Cfg::EpilogueOp0;
  using EpilogueOp1 = typename Cfg::EpilogueOp1;
  using EpilogueOp2 = typename Cfg::EpilogueOp2;

  explicit SwSm90Impl(const char* name) : name_(name) {}

  typename GemmType::Arguments make_args(const SwArgs& a) {
    auto ptrA = reinterpret_cast<ElementA*>(a.ptr_A);
    auto ptrB = reinterpret_cast<ElementB*>(a.ptr_B);
    auto ptrD = reinterpret_cast<ElementC*>(a.ptr_D);
    int const M = a.M, N_out = a.N_out, K = a.K;

    int64_t const ldB_strided = static_cast<int64_t>(2) * K;
    LayoutB0 layoutB_gate(ldB_strided);
    LayoutB1 layoutB_linear(ldB_strided);
    // ldd carries the host-padded row stride; Sm90DualGemm reads it via
    // ref_D2.stride(0) at run() time, so a strided D view works without
    // touching the vendored device/kernel headers.
    LayoutC  layoutC(a.ldd);

    using TensorRefA  = cutlass::TensorRef<ElementA const, LayoutA>;
    using TensorRefB0 = cutlass::TensorRef<ElementB const, LayoutB0>;
    using TensorRefB1 = cutlass::TensorRef<ElementB const, LayoutB1>;
    using TensorRefCi = cutlass::TensorRef<ElementC const, LayoutC>;
    using TensorRefDo = cutlass::TensorRef<ElementC,       LayoutC>;

    TensorRefA  ref_A0(ptrA,        LayoutA(static_cast<int64_t>(K)));
    TensorRefB0 ref_B0(ptrB,        layoutB_gate);                 // W_gate (even rows)
    TensorRefCi ref_C0(nullptr,     LayoutC(0));
    TensorRefDo ref_D0(nullptr,     LayoutC(0));
    TensorRefB1 ref_B1(ptrB + K,    layoutB_linear);               // W_linear (odd rows)
    TensorRefCi ref_C1(nullptr,     LayoutC(0));
    TensorRefDo ref_D1(nullptr,     LayoutC(0));
    TensorRefDo ref_D2(ptrD,        layoutC);                      // output

    typename EpilogueOp0::Params epi0{ElementCompute(1.0f), ElementCompute(0.0f)};
    typename EpilogueOp1::Params epi1{ElementCompute(1.0f), ElementCompute(0.0f)};
    typename EpilogueOp2::Params epi2{
        ElementCompute(a.alpha), ElementCompute(a.limit), ElementCompute(a.one)};

    cutlass::gemm::GemmCoord problem{M, N_out, K};

    typename GemmType::Arguments args(
        cutlass::gemm::DualGemmMode::kGemm,
        problem,
        ref_A0,
        ref_B0, ref_C0, ref_D0,
        ref_B1, ref_C1, ref_D1,
        ref_D2,
        epi0, epi1, epi2,
        /*split_k_slices=*/1,
        /*batch_count=*/1);
    return args;
  }

  size_t get_workspace_size(const SwArgs& a) override {
    return GemmType::get_workspace_size(make_args(a));
  }
  cutlass::Status initialize(const SwArgs& a, void* ws, cudaStream_t s) override {
    return gemm_.initialize(make_args(a), ws, s);
  }
  cutlass::Status run(cudaStream_t stream) override {
    return gemm_.run(stream);
  }
  const char* name() const override { return name_; }

 private:
  GemmType gemm_;
  const char* name_;
};

////////////////////////////////////////////////////////////////////////////////
// AutoTune runner — first call per (M, N_out, K) shape times all candidates.
////////////////////////////////////////////////////////////////////////////////

#define SW_SM90_TILE(bm, bn, bk, stages, label)                                 \
  configs_.push_back(std::make_unique<                                            \
      SwSm90Impl<DualGemmConfigSm90<                                             \
          cute::Shape<cute::Int<bm>, cute::Int<bn>, cute::Int<bk>>,               \
          stages>>>(label))

class SwSm90AutoTuneRunner {
 public:
  SwSm90AutoTuneRunner() {
    // Tile candidates for H100 (sm_90a, ~228 KiB dynamic SMEM/SM, 132 SMs).
    //
    // SMEM cost = (BM + 2*BN) * BK * 2 (bf16) * stages. Stay under ~200 KiB
    // to leave room for barriers and TMA descriptors. Sm90DualGemm requires
    // BM >= 128 to enable cooperative dual consumer warpgroups (the perf
    // sweet spot); smaller BM falls back to a single-wg path.
    //
    // Candidates intentionally span small/medium/large M; the runner picks
    // the best one per (M, N_out, K) tuple at first call.

    // ── Reference / prefill sweet spot ───────────────────────────────────────
    SW_SM90_TILE(128, 128, 64, 4, "Sm90<128,128,64>_S4");   // 192 KiB
    SW_SM90_TILE(128, 128, 64, 3, "Sm90<128,128,64>_S3");   // 144 KiB

    // ── Decode-style small M ─────────────────────────────────────────────────
    SW_SM90_TILE(64,  128, 64, 4, "Sm90<64,128,64>_S4");    // 160 KiB
    SW_SM90_TILE(64,   64, 64, 4, "Sm90<64,64,64>_S4");     //  96 KiB

    // ── Alternate small-N ────────────────────────────────────────────────────
    SW_SM90_TILE(128,  64, 64, 4, "Sm90<128,64,64>_S4");    // 128 KiB

    // ── Large prefill ────────────────────────────────────────────────────────
    SW_SM90_TILE(256, 128, 64, 2, "Sm90<256,128,64>_S2");   // 128 KiB
  }

  void operator()(at::Tensor A, at::Tensor B, at::Tensor D,
                  float alpha, float limit, float one) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && D.is_cuda(),
                "all inputs must be CUDA tensors");
    TORCH_CHECK(A.scalar_type() == at::kBFloat16 && B.scalar_type() == at::kBFloat16
                    && D.scalar_type() == at::kBFloat16,
                "all inputs must be bf16");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && D.dim() == 2, "A, B, D must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "K mismatch (A.size(1) vs B.size(1))");
    // Stride-based contiguity check (mirrors sm_80 path) — Inductor's
    // reinterpret_tensor often hands us a tensor with the right strides but
    // tripped is_contiguous() (e.g. larger storage than sizes would imply).
    TORCH_CHECK(A.stride(1) == 1, "A innermost stride must be 1; got ", A.stride(1));
    TORCH_CHECK(A.stride(0) >= A.size(1),
                "A row stride must be >= K; got stride(0)=", A.stride(0), ", K=", A.size(1));
    TORCH_CHECK(B.stride(1) == 1, "B innermost stride must be 1; got ", B.stride(1));
    TORCH_CHECK(B.stride(0) >= B.size(1),
                "B row stride must be >= K; got stride(0)=", B.stride(0), ", K=", B.size(1));

    int const M = static_cast<int>(A.size(0));
    int const K = static_cast<int>(A.size(1));
    int const N = static_cast<int>(B.size(0));
    TORCH_CHECK((N % 2) == 0, "N must be even, got ", N);
    // Sm90DualGemm uses TMA for A/B loads; TMA requires the innermost stride
    // **in bytes** to be a multiple of 16 (cudaTensorMapEncodeTiled's hard
    // constraint, also enforced by sm90_dual_gemm.h's can_implement via
    //   constexpr int min_k_align = 128 / sizeof_bits<ElementA>;
    //   if (problem_size.k() % min_k_align != 0) return kErrorInvalidProblem;
    // ). Express in bytes so a future fp8 / fp32 swiglu path inherits the
    // gate without a one-line dtype change. For bf16 (sizeof = 2) this
    // reduces to K % 8 == 0; for fp32 (sizeof = 4) → K % 4; for fp8 → K % 16.
    constexpr int kMinKAlignBytes = 16;
    constexpr int kElemBytes      = sizeof(ElementA);
    constexpr int kMinKAlignElems = kMinKAlignBytes / kElemBytes;
    TORCH_CHECK((K % kMinKAlignElems) == 0,
                "Sm90 swiglu requires K * sizeof(elem) % 16 == 0 (TMA's 128-bit "
                "alignment in bytes); got K=", K, ", elem_bytes=", kElemBytes,
                ", required K % ", kMinKAlignElems,
                " == 0. This shape is fusion-eligible only on the sm_80/sm_120 path.");
    int const N_out = N / 2;
    TORCH_CHECK(D.size(0) == M && D.size(1) == N_out,
                "D must be (M, N/2) = (", M, ",", N_out, ")");
    TORCH_CHECK(D.stride(1) == 1, "D innermost stride must be 1; got ", D.stride(1));
    TORCH_CHECK(D.stride(0) >= N_out,
                "D row stride must be >= N_out; got stride(0)=", D.stride(0), ", N_out=", N_out);

    SwArgs ea;
    ea.M = M; ea.N_out = N_out; ea.K = K;
    ea.ptr_A = A.data_ptr<at::BFloat16>();
    ea.ptr_B = B.data_ptr<at::BFloat16>();
    ea.ptr_D = D.data_ptr<at::BFloat16>();
    ea.ldd = static_cast<int64_t>(D.stride(0));
    ea.alpha = alpha; ea.limit = limit; ea.one = one;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.device().index()).stream();

    // Single autotune per module. The .cu is compiled per (m_bucket, N, K,
    // alignA, alignB, alignC) on the Python side — every distinct shape
    // bucket gets its own runner instance with isolated `best_idx_`.
    if (best_idx_ < 0) {
      best_idx_ = autotune(ea, stream);
    }
    int idx = best_idx_;

    auto& gemm = configs_[idx];
    size_t ws_sz = gemm->get_workspace_size(ea);
    if (!ws_.defined() || ws_.numel() < (int64_t)ws_sz) {
      ws_ = at::empty({(int64_t)ws_sz + 1},
          at::TensorOptions().dtype(at::kByte).device(A.device()));
    }
    auto st = gemm->initialize(ea, ws_sz > 0 ? ws_.data_ptr() : nullptr, stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess,
                "Sm90DualGemm init failed (", gemm->name(), "): ",
                cutlassGetStatusString(st));
    st = gemm->run(stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess,
                "Sm90DualGemm run failed (", gemm->name(), "): ",
                cutlassGetStatusString(st));
  }

  int num_configs() const { return (int)configs_.size(); }

 private:
  int autotune(const SwArgs& ea, cudaStream_t stream) {
    int best_idx = -1;
    float best_time = 1e30f;
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    for (size_t i = 0; i < configs_.size(); ++i) {
      auto& g = configs_[i];
      size_t ws_sz = 0;
      try { ws_sz = g->get_workspace_size(ea); }
      catch (...) { continue; }
      if (!ws_.defined() || ws_.numel() < (int64_t)ws_sz) {
        ws_ = at::empty({(int64_t)ws_sz + 1},
            at::TensorOptions().dtype(at::kByte).device(at::kCUDA));
      }
      void* ws_ptr = ws_sz > 0 ? ws_.data_ptr() : nullptr;
      if (g->initialize(ea, ws_ptr, stream) != cutlass::Status::kSuccess) {
        continue;
      }

      // Warmup — 10 iters so the L2 / instruction cache settle.
      for (int w = 0; w < 10; ++w) g->run(stream);
      cudaStreamSynchronize(stream);

      // Time — 50 iters keeps timing noise to <1%.
      cudaEventRecord(s, stream);
      int iters = 50;
      for (int p = 0; p < iters; ++p) g->run(stream);
      cudaEventRecord(e, stream);
      cudaEventSynchronize(e);
      float ms = 0;
      cudaEventElapsedTime(&ms, s, e);
      float avg = ms / iters;
      if (avg < best_time) { best_time = avg; best_idx = (int)i; }
    }
    cudaEventDestroy(s); cudaEventDestroy(e);
    TORCH_CHECK(best_idx >= 0,
                "Sm90DualGemm AutoTune: no candidate succeeded for (M,N_out,K)=(",
                ea.M, ",", ea.N_out, ",", ea.K, ")");
    return best_idx;
  }

  std::vector<std::unique_ptr<SwSm90Concept>> configs_;
  int best_idx_ = -1;     // -1 = not yet autotuned; sticky after first call.
  at::Tensor ws_;
};

static SwSm90AutoTuneRunner& runner() {
  static SwSm90AutoTuneRunner R;
  return R;
}

void swiglu_dual_matmul_out(at::Tensor A, at::Tensor B, at::Tensor D,
                             float alpha, float limit, float one) {
  runner()(std::move(A), std::move(B), std::move(D), alpha, limit, one);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "CUTLASS Sm90 DualGemm fully-fused swiglu (bf16) on sm_90a — autotune";
  m.def("swiglu_dual_matmul_out",
        &swiglu_dual_matmul_out,
        "D = swiglu(A @ B.T) in a single fused Sm90 (TMA+WGMMA) kernel; "
        "A:(M,K) bf16, B:(N,K) bf16 (N even), D:(M,N/2) bf16 (strided ok)",
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("D"),
        pybind11::arg("alpha") = 1.702f,
        pybind11::arg("limit") = 7.0f,
        pybind11::arg("one") = 1.0f);
  m.def("num_configs", []() { return runner().num_configs(); });
}
