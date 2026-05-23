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

// Single-kernel fully-fused swiglu — SM80 multistage path.
//
// Routes from sm_80 / sm_86 / sm_89 / sm_120 (Blackwell GeForce). The
// Hopper (sm_90) native TMA + WGMMA implementation lives at
// ../../sm90/cutlass_kernels/swiglu_one_stage.cu and is selected by
// _compile_swiglu_dual in evt_runtime.py per device compute capability.
//
//   D = swiglu(A @ B.T)
//
//   A : (M, K)   bf16 row-major
//   B : (N, K)   bf16 row-major   (torch.nn.Linear weight convention; N even)
//   D : (M, N/2) bf16 row-major   (strided view of (M, ldd) host-padded buffer)
//
// Implementation uses cutlass::gemm::device::DualGemm — the two GEMMs
// A @ W_gate.T and A @ W_linear.T run in the same threadblock sharing A's
// smem stages; their accumulators stay in registers and a custom
// SwigluCombine epilogue functor combines them and writes only D.
//
// AUTOTUNE: at first call per (M, N, K) tuple the runner times every
// registered (TileShape, WarpShape, Stages) candidate and caches the
// fastest one. Candidate set is sized to the sm_120 / Ada SMEM budget
// (~96 KB per CTA); see SwAutoTuneRunner for SMEM math.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <memory>
#include <unordered_map>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/util/host_tensor.h"

#include "45_dual_gemm/device/dual_gemm.h"
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

// AlignmentA / AlignmentB / AlignmentC are picked greedily on the Python side
// (128 → 64 bits) and passed in via -D at JIT time, so weights/activations
// whose K only divides 64 bits (e.g. K = 12 for bf16) still fuse onto this
// kernel instead of falling back to cuBLAS. AlignmentC normally stays at 128
// because the host pads D's row stride to a full cache line, but exposing it
// keeps the parity with A/B and lets a smaller-pad mode drop to 64 without
// editing this file. Defaults preserve the prior 128-bit behaviour for
// callers that don't override.
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
// Output vector store width = ldd's alignment expressed in elements. Host-side
// padding (see _aligned_n_stride in evt_runtime.py) normally guarantees 128
// bits / 8 elements for bf16 — kept tunable here for parity with A/B.
constexpr int EpilogueVecCount = MAGI_SWIGLU_ALIGN_C_BITS / cutlass::sizeof_bits<ElementC>::value;

using ArchTag          = cutlass::arch::Sm80;
using OperatorClass    = cutlass::arch::OpClassTensorOp;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

constexpr auto kScaleType        = cutlass::epilogue::thread::ScaleType::Nothing;
constexpr bool kSplitKSerial     = false;
constexpr bool kStoreD0          = false;
constexpr bool kStoreD1          = false;

////////////////////////////////////////////////////////////////////////////////
// Per-tile DualGemm wrapper. The DualGemm device type is templated on
// (TileShape, WarpShape, Stages) — every autotune candidate instantiates the
// full kernel for its tuple. Compile time grows linearly with candidate count
// but DualGemm Sm80 is much cheaper to compile than the EVT path (no visitor
// tree), so we can afford 8–10 candidates.
////////////////////////////////////////////////////////////////////////////////

template <class TbShape, class WaShape, int Stages>
struct DualGemmConfig {
  using EpilogueOp0 = cutlass::epilogue::thread::LinearCombination<
      ElementC, EpilogueVecCount, ElementAcc, ElementCompute, kScaleType>;
  using EpilogueOp1 = cutlass::epilogue::thread::LinearCombination<
      ElementC, EpilogueVecCount, ElementAcc, ElementCompute, kScaleType>;
  using EpilogueOp2 = cutlass::epilogue::thread::SwigluCombine<
      ElementC, EpilogueVecCount, ElementAcc, ElementCompute>;

  using Gemm = cutlass::gemm::device::DualGemm<
      ElementA, LayoutA,
      ElementB, LayoutB0, LayoutB1,
      ElementC, LayoutC,
      ElementAcc,
      OperatorClass, ArchTag,
      TbShape, WaShape, InstructionShape,
      EpilogueOp0, EpilogueOp1, EpilogueOp2,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      Stages,
      kStoreD0, kStoreD1, kSplitKSerial,
      AlignmentA, AlignmentB>;
};

////////////////////////////////////////////////////////////////////////////////
// Type-erased runner concept; one instance per autotune candidate.
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

class SwConcept {
 public:
  virtual ~SwConcept() = default;
  virtual size_t get_workspace_size(const SwArgs&) = 0;
  virtual cutlass::Status initialize(const SwArgs&, void* ws, cudaStream_t) = 0;
  virtual cutlass::Status run(cudaStream_t stream) = 0;
  virtual const char* name() const = 0;
};

template <class Cfg>
class SwImpl : public SwConcept {
 public:
  using GemmType = typename Cfg::Gemm;
  using EpilogueOp0 = typename Cfg::EpilogueOp0;
  using EpilogueOp1 = typename Cfg::EpilogueOp1;
  using EpilogueOp2 = typename Cfg::EpilogueOp2;

  explicit SwImpl(const char* name) : name_(name) {}

  typename GemmType::Arguments make_args(const SwArgs& a) {
    auto ptrA = reinterpret_cast<ElementA*>(a.ptr_A);
    auto ptrB = reinterpret_cast<ElementB*>(a.ptr_B);
    auto ptrD = reinterpret_cast<ElementC*>(a.ptr_D);
    int const M = a.M, N_out = a.N_out, K = a.K;

    int64_t const ldB_strided = static_cast<int64_t>(2) * K;
    LayoutB0 layoutB_gate(ldB_strided);
    LayoutB1 layoutB_linear(ldB_strided);
    LayoutC  layoutC(a.ldd);

    using TensorRefA  = cutlass::TensorRef<ElementA const, LayoutA>;
    using TensorRefB0 = cutlass::TensorRef<ElementB const, LayoutB0>;
    using TensorRefB1 = cutlass::TensorRef<ElementB const, LayoutB1>;
    using TensorRefCi = cutlass::TensorRef<ElementC const, LayoutC>;
    using TensorRefDo = cutlass::TensorRef<ElementC,       LayoutC>;

    TensorRefA  ref_A0(ptrA,        LayoutA(static_cast<int64_t>(K)));
    TensorRefB0 ref_B0(ptrB,        layoutB_gate);
    TensorRefCi ref_C0(nullptr,     LayoutC(0));
    TensorRefDo ref_D0(nullptr,     LayoutC(0));
    TensorRefB1 ref_B1(ptrB + K,    layoutB_linear);
    TensorRefCi ref_C1(nullptr,     LayoutC(0));
    TensorRefDo ref_D1(nullptr,     LayoutC(0));
    TensorRefDo ref_D2(ptrD,        layoutC);

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
        /*batch_count=*/1,
        /*batch_stride_A=*/0,
        /*batch_stride_B0=*/0,
        /*batch_stride_B1=*/0,
        /*batch_stride_C=*/0,
        /*batch_stride_D=*/0);
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

#define SW_TILE(tb_m, tb_n, tb_k, wa_m, wa_n, wa_k, stages, label)            \
  configs_.push_back(std::make_unique<                                          \
      SwImpl<DualGemmConfig<                                                   \
          cutlass::gemm::GemmShape<tb_m, tb_n, tb_k>,                           \
          cutlass::gemm::GemmShape<wa_m, wa_n, wa_k>,                           \
          stages>>>(label))

class SwAutoTuneRunner {
 public:
  SwAutoTuneRunner() {
    // SMEM cost for DualGemm = (BM + 2*BN) * BK * 2B * stages because both
    // B operands live in smem simultaneously. Budget cap ~96 KB matches
    // sm_120's per-SM SMEM (also fits sm_80 / sm_86 / sm_89).
    //
    // Bucket of M doesn't drive a separate .cu here — DualGemm compiles
    // fast enough that one runner with all candidates handles every M, and
    // the per-shape cache picks the best for whatever M it sees.

    // Small / decode-friendly tiles
    SW_TILE(64,  64, 32, 32, 32, 32, 4, "T<64,64,32>_S4");      // 36 KB
    SW_TILE(64,  64, 64, 32, 32, 64, 3, "T<64,64,64>_S3");      // 72 KB
    SW_TILE(64, 128, 32, 32, 64, 32, 3, "T<64,128,32>_S3");     // 60 KB
    SW_TILE(64, 128, 32, 32, 64, 32, 4, "T<64,128,32>_S4");     // 80 KB

    // Medium tiles (CUTLASS bf16 reference defaults)
    SW_TILE(128,  64, 32, 64, 32, 32, 3, "T<128,64,32>_S3");    // 48 KB
    SW_TILE(128,  64, 32, 64, 32, 32, 4, "T<128,64,32>_S4");    // 64 KB
    SW_TILE(128,  64, 64, 64, 32, 64, 3, "T<128,64,64>_S3");    // 96 KB
    SW_TILE(128, 128, 32, 64, 64, 32, 3, "T<128,128,32>_S3");   // 72 KB
    SW_TILE(128, 128, 32, 64, 64, 32, 4, "T<128,128,32>_S4");   // 96 KB

    // Large prefill tiles
    SW_TILE(256,  64, 32, 64, 32, 32, 3, "T<256,64,32>_S3");    // 72 KB
    // (256, 128, 32)*3 = 96 KB exact-budget, prone to SMEM alloc fail; omitted.
    // (128, 256, 32)*3 = 120 KB > 96 — omitted.
    // (64,  256, 32)*3 = 108 KB > 96 — omitted.
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
    // Stride-based contiguity instead of A.is_contiguous() / B.is_contiguous():
    // Inductor's reinterpret_tensor often hands us a tensor with the right
    // strides but tripped is_contiguous() (e.g. bigger storage than sizes
    // would imply). The kernel only cares that A's innermost is K-stride 1
    // and B's innermost is K-stride 1 (B is the (N, K) row-major weight,
    // CUTLASS reads it via ColumnMajor + ldB=2K).
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
    int const N_out = N / 2;
    TORCH_CHECK(D.size(0) == M && D.size(1) == N_out,
                "D must be (M, N/2) = (", M, ",", N_out, ")");
    // D may be a strided view of a host-padded (M, ldd) buffer.
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

    // Single autotune per module. The .cu is compiled per (M-bucket, N, K)
    // on the Python side — every distinct weight (N, K) gets its own .cu,
    // so this runner instance hosts exactly one (N, K) and one bucket. The
    // first call autotunes; all subsequent calls (any M in the bucket)
    // reuse `best_idx_`.
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
                "DualGemm init failed (", gemm->name(), "): ",
                cutlassGetStatusString(st));
    st = gemm->run(stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess,
                "DualGemm run failed (", gemm->name(), "): ",
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

      // Warmup — 10 iters so the L2 / instruction cache settle. With only
      // 3 warmups (the original count) the first timed iter sees a cold L2
      // and inflates the average, sometimes flipping the best-config choice.
      for (int w = 0; w < 10; ++w) g->run(stream);
      cudaStreamSynchronize(stream);

      // Time — 50 iters keeps timing noise to <1% so 2–3 % perf gaps
      // between candidates are distinguishable.
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
                "swiglu AutoTune: no candidate succeeded for (M,N_out,K)=(",
                ea.M, ",", ea.N_out, ",", ea.K, ")");
    return best_idx;
  }

  std::vector<std::unique_ptr<SwConcept>> configs_;
  int best_idx_ = -1;     // -1 = not yet autotuned; sticky after first call.
  at::Tensor ws_;
};

static SwAutoTuneRunner& runner() {
  static SwAutoTuneRunner R;
  return R;
}

void swiglu_dual_matmul_out(at::Tensor A, at::Tensor B, at::Tensor D,
                             float alpha, float limit, float one) {
  runner()(std::move(A), std::move(B), std::move(D), alpha, limit, one);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "CUTLASS DualGemm fully-fused swiglu (bf16) on sm_120 — autotune";
  m.def("swiglu_dual_matmul_out",
        &swiglu_dual_matmul_out,
        "D = swiglu(A @ B.T) in a single fused kernel; "
        "A:(M,K) bf16, B:(N,K) bf16 (N even), D:(M,N/2) bf16",
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("D"),
        pybind11::arg("alpha") = 1.702f,
        pybind11::arg("limit") = 7.0f,
        pybind11::arg("one") = 1.0f);
  m.def("num_configs", []() { return runner().num_configs(); });
}
