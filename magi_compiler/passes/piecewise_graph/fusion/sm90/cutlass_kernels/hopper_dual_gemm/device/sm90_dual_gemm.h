// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// VENDORED from upstream CUTLASS examples on 2026-05-09:
//   examples/49_hopper_dual_gemm/device/sm90_dual_gemm.h
// To resync, copy the upstream file verbatim over this one. Don't edit
// in-tree — the swiglu path on top of it is in
// magi_compiler/passes/piecewise_graph/fusion/cutlass_fusion/sm90/
// cutlass_kernels/swiglu_one_stage.cu and works around any contract quirks
// at the host side, leaving this file as a drop-in upstream copy.
//
// Sm90 DualGemm — device-level wrapper.
//
// Public API mirrors examples/45_dual_gemm/device/dual_gemm.h as closely as
// the SM90 idiom permits, so existing call sites that build on
// `cutlass::gemm::device::DualGemm<...>` migrate to
// `cutlass::gemm::device::Sm90DualGemm<...>` with only the template-parameter
// list changing (TileShape/ClusterShape replace ThreadblockShape/WarpShape/
// InstructionShape; ArchTag is implicit).
//
// Functional contract:
//
//   D2 = epilogue2( A @ B0,  A @ B1 )
//
// Both matmuls accumulate in fp32 (or whatever ElementAccumulator the user
// picks), the binary `epilogue2` (e.g. cutlass::epilogue::thread::SwigluCombine)
// fuses them into a single ElementC output. D0 / D1 are not stored — the
// only currently supported mode is StoreD0 = StoreD1 = false (the same mode
// used by the Sm80 swiglu one-stage example).
//
// Hardware: requires sm_90a (Hopper WGMMA + TMA). The kernel uses a single
// 128-thread warpgroup per CTA, no cluster, non-persistent grid.

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/cluster_launch.hpp"

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/arch/mma_sm90_gmma.hpp"
#include "cute/atom/mma_traits_sm90_gmma.hpp"

#include "../kernel/sm90_dual_gemm_kernel.hpp"
// VENDORED CHANGE: upstream points at "../../45_dual_gemm/dual_gemm_common.h"
// (examples-relative). We co-located the file under our 49_hopper_dual_gemm/
// to make the vendored tree self-contained. Resync: leave this `#include` as
// `"../dual_gemm_common.h"` even if upstream changes its path.
#include "../dual_gemm_common.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace device {

namespace sm90_dual_gemm_detail {

using namespace cute;

// ---------------------------------------------------------------------------
// CUTLASS 2.x layout tag → cute Major / stride for SM90 GMMA.
//
// CUTLASS 2.x convention (operand-aware):
//   A (M×K):       RowMajor → K contig (TN-A)        ColMajor → M contig (NT-A)
//   B (K×N):       RowMajor → N contig (NT-B)        ColMajor → K contig (TN-B)
//   C/D (M×N):     RowMajor → N contig               ColMajor → M contig
//
// The SM90 kernel views B as cute shape (N, K) (CUTLASS 3.x convention),
// so for operand B the relationship between the layout tag and which mode
// is contiguous is *flipped* relative to A and C.
//
// The Tag below selects between operand semantics; for each operand we
// derive a uniform cute Stride pair (int64_t, _1) (or (_1, int64_t)) plus
// the corresponding GMMA::Major.
// ---------------------------------------------------------------------------

enum class Operand { A, B, C };

// Which mode of the (mode0, mode1) cute tensor is contiguous?
//   K_contig=true  → cute stride = (int64_t, _1)   (K contiguous, GMMA::Major::K)
//   K_contig=false → cute stride = (_1, int64_t)   (MN contiguous, GMMA::Major::MN)
//
// For A (M, K):  RowMajor=K_contig=true,  ColMajor=K_contig=false
// For B (N, K):  RowMajor=K_contig=false, ColMajor=K_contig=true   (flipped — see above)
// For C (M, N):  treat the K-contig flag as N-contig → RowMajor=true, ColMajor=false
template <Operand Op, class LayoutTag>
struct LayoutTraits;

// ---- A operand
template <>
struct LayoutTraits<Operand::A, cutlass::layout::RowMajor> {
  using Stride = cute::Stride<int64_t, cute::_1>;
  static constexpr cute::GMMA::Major Major = cute::GMMA::Major::K;
  template <class T> CUTE_HOST_DEVICE static
  Stride make(T ld) { return cute::make_stride(int64_t(ld), cute::_1{}); }
};
template <>
struct LayoutTraits<Operand::A, cutlass::layout::ColumnMajor> {
  using Stride = cute::Stride<cute::_1, int64_t>;
  static constexpr cute::GMMA::Major Major = cute::GMMA::Major::MN;
  template <class T> CUTE_HOST_DEVICE static
  Stride make(T ld) { return cute::make_stride(cute::_1{}, int64_t(ld)); }
};

// ---- B operand (note: layout-tag sense is flipped vs A because
//                cute view is (N, K) but CUTLASS-2.x tag is "B as K×N")
template <>
struct LayoutTraits<Operand::B, cutlass::layout::RowMajor> {
  // CUTLASS 2.x "RowMajor B" = N contig in (K, N) = MN-contig in our (N, K)
  using Stride = cute::Stride<cute::_1, int64_t>;
  static constexpr cute::GMMA::Major Major = cute::GMMA::Major::MN;
  template <class T> CUTE_HOST_DEVICE static
  Stride make(T ld) { return cute::make_stride(cute::_1{}, int64_t(ld)); }
};
template <>
struct LayoutTraits<Operand::B, cutlass::layout::ColumnMajor> {
  // CUTLASS 2.x "ColumnMajor B" = K contig in (K, N) = K-contig in our (N, K)
  using Stride = cute::Stride<int64_t, cute::_1>;
  static constexpr cute::GMMA::Major Major = cute::GMMA::Major::K;
  template <class T> CUTE_HOST_DEVICE static
  Stride make(T ld) { return cute::make_stride(int64_t(ld), cute::_1{}); }
};

// ---- C/D operand (M, N): same mapping as A but interpreting "K-contig" as N-contig.
template <>
struct LayoutTraits<Operand::C, cutlass::layout::RowMajor> {
  using Stride = cute::Stride<int64_t, cute::_1>;
  static constexpr cute::GMMA::Major Major = cute::GMMA::Major::K;  // unused for C
  template <class T> CUTE_HOST_DEVICE static
  Stride make(T ld) { return cute::make_stride(int64_t(ld), cute::_1{}); }
};
template <>
struct LayoutTraits<Operand::C, cutlass::layout::ColumnMajor> {
  using Stride = cute::Stride<cute::_1, int64_t>;
  static constexpr cute::GMMA::Major Major = cute::GMMA::Major::MN;  // unused for C
  template <class T> CUTE_HOST_DEVICE static
  Stride make(T ld) { return cute::make_stride(cute::_1{}, int64_t(ld)); }
};

} // namespace sm90_dual_gemm_detail

////////////////////////////////////////////////////////////////////////////////
// Sm90DualGemm — public template
////////////////////////////////////////////////////////////////////////////////

template <
    typename ElementA_,
    typename LayoutA_,
    typename ElementB_,
    typename LayoutB0_,
    typename LayoutB1_,
    typename ElementC_,
    typename LayoutC_,
    typename ElementAccumulator_,
    /// CTA tile shape:    cute::Shape<_M, _N, _K>          (e.g. <_128,_128,_64>)
    typename TileShape_,
    /// Per-GEMM linear-combination ops (only used when StoreD0/D1 are true).
    typename EpilogueOutputOp0_,
    typename EpilogueOutputOp1_,
    /// Binary combine functor (e.g. cutlass::epilogue::thread::SwigluCombine).
    typename EpilogueOutputOp2_,
    /// Pipeline stages.  Defaults to 3 — bumping higher needs more dyn-smem.
    int Stages = 3,
    /// Reserved for parity with the Sm80 DualGemm — must be false today.
    bool StoreD0 = false,
    bool StoreD1 = false,
    /// Reserved for parity with the Sm80 DualGemm — must be false today.
    bool SplitKSerial = false,
    int AlignmentA = 8,
    int AlignmentB = 8>
class Sm90DualGemm {
 public:

  using ElementA            = ElementA_;
  using LayoutA             = LayoutA_;
  using ElementB            = ElementB_;
  using LayoutB0            = LayoutB0_;
  using LayoutB1            = LayoutB1_;
  using ElementC            = ElementC_;
  using LayoutC             = LayoutC_;
  using ElementAccumulator  = ElementAccumulator_;
  using TileShape           = TileShape_;
  using EpilogueOutputOp0   = EpilogueOutputOp0_;
  using EpilogueOutputOp1   = EpilogueOutputOp1_;
  using EpilogueOutputOp2   = EpilogueOutputOp2_;

  static constexpr int kStages     = Stages;
  static constexpr bool kStoreD0   = StoreD0;
  static constexpr bool kStoreD1   = StoreD1;
  static constexpr bool kSplitKSerial = SplitKSerial;
  static constexpr int kAlignmentA = AlignmentA;
  static constexpr int kAlignmentB = AlignmentB;

  static_assert(!StoreD0, "Sm90DualGemm: StoreD0=true is not yet implemented (D0 is consumed in registers).");
  static_assert(!StoreD1, "Sm90DualGemm: StoreD1=true is not yet implemented (D1 is consumed in registers).");
  static_assert(!SplitKSerial, "Sm90DualGemm: split-K is not yet implemented.");

  // Same TensorRef typedefs as the Sm80 DualGemm wrapper for API parity.
  using TensorRefA  = TensorRef<ElementA const, LayoutA>;
  using TensorRefB0 = TensorRef<ElementB const, LayoutB0>;
  using TensorRefB1 = TensorRef<ElementB const, LayoutB1>;
  using TensorRefC  = TensorRef<ElementC const, LayoutC>;
  using TensorRefD  = TensorRef<ElementC,       LayoutC>;

  static_assert(cute::is_static<TileShape>::value, "TileShape must be a static cute::Shape.");
  static constexpr int kBlockM = cute::size<0>(TileShape{});
  static constexpr int kBlockN = cute::size<1>(TileShape{});
  static constexpr int kBlockK = cute::size<2>(TileShape{});

  static_assert(kBlockM % 64 == 0, "BLK_M must be a multiple of 64 (WGMMA constraint).");

  // ---------------------- cute-side type setup ----------------------
 private:

  using TraitsA  = sm90_dual_gemm_detail::LayoutTraits<sm90_dual_gemm_detail::Operand::A, LayoutA >;
  using TraitsB0 = sm90_dual_gemm_detail::LayoutTraits<sm90_dual_gemm_detail::Operand::B, LayoutB0>;
  using TraitsB1 = sm90_dual_gemm_detail::LayoutTraits<sm90_dual_gemm_detail::Operand::B, LayoutB1>;
  using TraitsC  = sm90_dual_gemm_detail::LayoutTraits<sm90_dual_gemm_detail::Operand::C, LayoutC >;

  static constexpr cute::GMMA::Major kMajorA  = TraitsA::Major;
  static constexpr cute::GMMA::Major kMajorB0 = TraitsB0::Major;
  static constexpr cute::GMMA::Major kMajorB1 = TraitsB1::Major;
  static_assert(kMajorB0 == kMajorB1,
      "B0 and B1 must share the same Major (= same K-major / MN-major orientation).");

  using StrideA  = typename TraitsA::Stride;
  using StrideB  = typename TraitsB0::Stride;
  using StrideD  = typename TraitsC::Stride;

  // Cooperative warpgroup count. Splits the BLK_M dim of each CTA tile across
  // this many consumer warpgroups (each runs 128 threads), so a 128x128 tile
  // with 2 wgs has each wg owning 64x128 of the accumulator. This caps the
  // dual-acc per-thread register pressure regardless of BLK_M.
  static constexpr int kNumConsumerWgs =
      (kBlockM >= 128) ? 2 : 1;     // M ≥ 128 ⇒ cooperative (64 M per wg)

  // The cute SS atom selector picks the WGMMA atom for the *single-wg view*
  // of the tile: it expects size<0>(TileShape) == kBlockM / kNumConsumerWgs
  // (the per-wg M sub-tile). We construct a synthetic per-wg tile shape for
  // the selector, then re-tile across wgs via the TiledMma layout below.
  using PerWgTileShape = cute::Shape<
      cute::Int<kBlockM / kNumConsumerWgs>, cute::Int<kBlockN>, cute::Int<kBlockK>>;
  using GmmaAtom = decltype(cute::SM90::GMMA::ss_op_selector<
      ElementA, ElementB, ElementAccumulator, PerWgTileShape, kMajorA, kMajorB0>());
  // Cooperative TiledMma: replicate the atom kNumConsumerWgs× along M.
  using TiledMma = decltype(cute::make_tiled_mma(
      GmmaAtom{},
      cute::Layout<cute::Shape<cute::Int<kNumConsumerWgs>, cute::_1, cute::_1>>{}));

  // Smem layout atoms — per-Major canonical SW128 atoms.
  using SmemLayoutAtomA = cute::conditional_t<
      kMajorA == cute::GMMA::Major::K,
      cute::GMMA::Layout_K_SW128_Atom<ElementA>,
      cute::GMMA::Layout_MN_SW128_Atom<ElementA>>;
  using SmemLayoutAtomB = cute::conditional_t<
      kMajorB0 == cute::GMMA::Major::K,
      cute::GMMA::Layout_K_SW128_Atom<ElementB>,
      cute::GMMA::Layout_MN_SW128_Atom<ElementB>>;

  using PipeStages_ = cute::Int<kStages>;
  using SmemLayoutA = decltype(cute::tile_to_shape(
      SmemLayoutAtomA{},
      cute::make_shape(cute::Int<kBlockM>{}, cute::Int<kBlockK>{}, PipeStages_{})));
  using SmemLayoutB = decltype(cute::tile_to_shape(
      SmemLayoutAtomB{},
      cute::make_shape(cute::Int<kBlockN>{}, cute::Int<kBlockK>{}, PipeStages_{})));

  // TMA atom decltypes — the actual TMA atoms have to be constructed on host
  // (they bake the gmem tensor's runtime shape into a copy descriptor), so
  // we only use these for the `decltype(...) const` kernel-template parameter.
  using TmaA  = decltype(cute::make_tma_atom(
      cute::SM90_TMA_LOAD{},
      cute::make_tensor(static_cast<ElementA const*>(nullptr),
                        cute::make_shape(int(0), int(0)),
                        StrideA{}),
      SmemLayoutA{}(cute::_, cute::_, 0),
      cute::make_shape(cute::Int<kBlockM>{}, cute::Int<kBlockK>{})));
  using TmaB  = decltype(cute::make_tma_atom(
      cute::SM90_TMA_LOAD{},
      cute::make_tensor(static_cast<ElementB const*>(nullptr),
                        cute::make_shape(int(0), int(0)),
                        StrideB{}),
      SmemLayoutB{}(cute::_, cute::_, 0),
      cute::make_shape(cute::Int<kBlockN>{}, cute::Int<kBlockK>{})));

  using SharedStorage = kernel::sm90_dual_gemm_detail::DualGemmSharedStorage<
      ElementA, ElementB, SmemLayoutA, SmemLayoutB>;

  static constexpr int kSmemBytes = static_cast<int>(sizeof(SharedStorage));

 public:

  // -------------------------- Arguments --------------------------
  struct Arguments {
    DualGemmMode mode;
    GemmCoord    problem_size;

    TensorRefA   ref_A0;
    TensorRefB0  ref_B0;
    TensorRefC   ref_C0;
    TensorRefD   ref_D0;
    TensorRefB1  ref_B1;
    TensorRefC   ref_C1;
    TensorRefD   ref_D1;
    TensorRefD   ref_D2;

    typename EpilogueOutputOp0::Params epilogue0;
    typename EpilogueOutputOp1::Params epilogue1;
    typename EpilogueOutputOp2::Params epilogue2;

    int     split_k_slices = 1;
    int     batch_count    = 1;
    int64_t batch_stride_A = 0;
    int64_t batch_stride_B0 = 0;
    int64_t batch_stride_B1 = 0;
    int64_t batch_stride_C  = 0;
    int64_t batch_stride_D  = 0;

    CUTLASS_HOST_DEVICE Arguments() : problem_size(0, 0, 0) {}

    CUTLASS_HOST_DEVICE Arguments(
        DualGemmMode mode_,
        GemmCoord problem_size_,
        TensorRefA  ref_A0_,
        TensorRefB0 ref_B0_,
        TensorRefC  ref_C0_,
        TensorRefD  ref_D0_,
        TensorRefB1 ref_B1_,
        TensorRefC  ref_C1_,
        TensorRefD  ref_D1_,
        TensorRefD  ref_D2_,
        typename EpilogueOutputOp0::Params epilogue0_ = typename EpilogueOutputOp0::Params(),
        typename EpilogueOutputOp1::Params epilogue1_ = typename EpilogueOutputOp1::Params(),
        typename EpilogueOutputOp2::Params epilogue2_ = typename EpilogueOutputOp2::Params(),
        int split_k_slices_ = 1,
        int batch_count_    = 1,
        int64_t batch_stride_A_  = 0,
        int64_t batch_stride_B0_ = 0,
        int64_t batch_stride_B1_ = 0,
        int64_t batch_stride_C_  = 0,
        int64_t batch_stride_D_  = 0)
      : mode(mode_), problem_size(problem_size_),
        ref_A0(ref_A0_), ref_B0(ref_B0_), ref_C0(ref_C0_), ref_D0(ref_D0_),
        ref_B1(ref_B1_), ref_C1(ref_C1_), ref_D1(ref_D1_), ref_D2(ref_D2_),
        epilogue0(epilogue0_), epilogue1(epilogue1_), epilogue2(epilogue2_),
        split_k_slices(split_k_slices_),
        batch_count(batch_count_),
        batch_stride_A(batch_stride_A_),
        batch_stride_B0(batch_stride_B0_),
        batch_stride_B1(batch_stride_B1_),
        batch_stride_C(batch_stride_C_),
        batch_stride_D(batch_stride_D_) {}
  };

 private:
  // Captured inside `initialize` for `run` to use later.
  Arguments args_{};
  bool      initialized_ = false;

 public:

  Sm90DualGemm() = default;

  static Status can_implement(Arguments const& args) {
    if (args.mode != DualGemmMode::kGemm) {
      return Status::kErrorInvalidProblem;
    }
    if (args.split_k_slices > 1) {
      return Status::kErrorInvalidProblem;
    }
    if (args.batch_count != 1) {
      return Status::kErrorInvalidProblem;
    }
    if (args.problem_size.m() <= 0 || args.problem_size.n() <= 0 || args.problem_size.k() <= 0) {
      return Status::kErrorInvalidProblem;
    }
    if (args.ref_D2.data() == nullptr) {
      return Status::kErrorInvalidProblem;
    }
    // D0/D1 must be null when StoreD0/D1 is false (matches Sm80 DualGemm contract).
    if ((kStoreD0 != (args.ref_D0.data() != nullptr)) ||
        (kStoreD1 != (args.ref_D1.data() != nullptr))) {
      return Status::kErrorInvalidProblem;
    }
    // K alignment: must be a multiple of TMA's 128-bit minimum (= 8 bf16 elts).
    constexpr int min_k_align = 128 / cutlass::sizeof_bits<ElementA>::value;
    if (args.problem_size.k() % min_k_align != 0) {
      return Status::kErrorInvalidProblem;
    }
    return Status::kSuccess;
  }

  static size_t get_workspace_size(Arguments const& /*args*/) {
    return 0;
  }

  Status initialize(Arguments const& args, void* /*workspace*/ = nullptr,
                    cudaStream_t /*stream*/ = nullptr) {
    Status s = can_implement(args);
    if (s != Status::kSuccess) return s;
    args_ = args;
    initialized_ = true;
    return Status::kSuccess;
  }

  Status update(Arguments const& args, void* /*workspace*/ = nullptr) {
    Status s = can_implement(args);
    if (s != Status::kSuccess) return s;
    args_ = args;
    return Status::kSuccess;
  }

  Status run(cudaStream_t stream = nullptr) {
    if (!initialized_) return Status::kErrorInternal;

    int const M = args_.problem_size.m();
    int const N = args_.problem_size.n();
    int const K = args_.problem_size.k();

    // Stride conversion: TensorRef<...,LayoutX>::layout().stride() carries the
    // leading dim, which is what cute needs.
    auto dA  = TraitsA ::make(args_.ref_A0.stride(0));
    auto dB0 = TraitsB0::make(args_.ref_B0.stride(0));
    auto dB1 = TraitsB1::make(args_.ref_B1.stride(0));
    auto dD2 = TraitsC ::make(args_.ref_D2.stride(0));

    auto* ptrA  = args_.ref_A0.data();
    auto* ptrB0 = args_.ref_B0.data();
    auto* ptrB1 = args_.ref_B1.data();
    auto* ptrD2 = args_.ref_D2.data();

    // Build TMA atoms host-side (they capture the full gmem-shape descriptor).
    auto mA  = cute::make_tensor(ptrA,  cute::make_shape(M, K), dA );
    auto mB0 = cute::make_tensor(ptrB0, cute::make_shape(N, K), dB0);
    auto mB1 = cute::make_tensor(ptrB1, cute::make_shape(N, K), dB1);

    auto tmaA  = cute::make_tma_atom(cute::SM90_TMA_LOAD{}, mA,
                                     SmemLayoutA{}(cute::_, cute::_, 0),
                                     cute::make_shape(cute::Int<kBlockM>{}, cute::Int<kBlockK>{}));
    auto tmaB0 = cute::make_tma_atom(cute::SM90_TMA_LOAD{}, mB0,
                                     SmemLayoutB{}(cute::_, cute::_, 0),
                                     cute::make_shape(cute::Int<kBlockN>{}, cute::Int<kBlockK>{}));
    auto tmaB1 = cute::make_tma_atom(cute::SM90_TMA_LOAD{}, mB1,
                                     SmemLayoutB{}(cute::_, cute::_, 0),
                                     cute::make_shape(cute::Int<kBlockN>{}, cute::Int<kBlockK>{}));

    typename EpilogueOutputOp2::Params op2_params = args_.epilogue2;
    EpilogueOutputOp2 combine_op(op2_params);

    auto cta_tiler  = TileShape{};
    auto prob_shape = cute::make_shape(M, N, K);

    auto* kernel_ptr = &kernel::sm90_dual_gemm_detail::sm90_dual_gemm_device<
        decltype(prob_shape), TileShape,
        ElementA, SmemLayoutA, decltype(tmaA),
        ElementB, SmemLayoutB, decltype(tmaB0),
        ElementC, decltype(dD2),
        TiledMma, EpilogueOutputOp2>;

    cudaError_t err = cudaFuncSetAttribute(
        reinterpret_cast<void const*>(kernel_ptr),
        cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
    if (err != cudaSuccess) return Status::kErrorInternal;

    dim3 grid(static_cast<unsigned>((M + kBlockM - 1) / kBlockM),
              static_cast<unsigned>((N + kBlockN - 1) / kBlockN),
              1);
    // 1 producer warpgroup (128 threads, only lane 0 of warp 0 is live)
    // + kNumConsumerWgs consumer warpgroups (128 threads each).
    dim3 block(static_cast<unsigned>(128 * (kNumConsumerWgs + 1)), 1, 1);
    dim3 cluster(1, 1, 1);

    cutlass::ClusterLaunchParams launch_params{grid, block, cluster, kSmemBytes, stream};
    cutlass::Status st = cutlass::launch_kernel_on_cluster(
        launch_params,
        reinterpret_cast<void const*>(kernel_ptr),
        prob_shape, cta_tiler,
        ptrA,  tmaA,
        ptrB0, tmaB0,
        ptrB1, tmaB1,
        ptrD2, dD2,
        TiledMma{},
        combine_op);
    return st;
  }

  Status operator()(cudaStream_t stream = nullptr) { return run(stream); }

  Status operator()(Arguments const& args, void* workspace = nullptr,
                    cudaStream_t stream = nullptr) {
    Status s = initialize(args, workspace, stream);
    if (s == Status::kSuccess) s = run(stream);
    return s;
  }
};

} // namespace device
} // namespace gemm
} // namespace cutlass
