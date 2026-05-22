// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// VENDORED from upstream CUTLASS examples on 2026-05-09:
//   examples/49_hopper_dual_gemm/kernel/sm90_dual_gemm_kernel.hpp
// To resync, copy the upstream file verbatim over this one.
//
// Sm90 DualGemm kernel — fused dual-WGMMA producer/consumer pipeline,
// warp-specialized.
//
// Computes (in a single kernel launch):
//
//   Acc0 = A @ B0
//   Acc1 = A @ B1
//   D2   = combine(Acc0, Acc1)
//
// A is loaded once and consumed by both WGMMA chains in the same K-stage,
// so the gate / linear matmuls share A's smem traffic — the whole point
// of DualGemm. Neither D0 nor D1 ever spills to HBM.
//
// Architecture
// ------------
// Three warpgroups per CTA (1 producer + N consumer), no clusters,
// non-persistent grid:
//
//   * Producer warpgroup (warps 0-3, threads 0-127): only lane 0 of warp 0
//     is "live"; the rest call setmaxnreg.dec<40> and exit. The live thread
//     issues TMA loads for A + B0 + B1 of the next K-stage and arrives on
//     a per-stage producer barrier. Reg-deallocated to <=40 to free SM
//     registers for the consumers.
//
//   * Consumer warpgroups (warps 4..N+3, threads 128..128*(N+1)-1): each
//     wg does setmaxnreg.inc<240> and runs two WGMMA chains that share
//     the same A smem buffer (the TiledMma's _N_-warpgroup M-tiling splits
//     A's M dim between them). Each wg owns its own accumulator pair
//     (acc0, acc1) and emits its M-sub-tile of D2 via predicated STG.
//
// The number of consumer warpgroups is determined by the TiledMma's
// thread-count: `NumConsumerWgs = size(TiledMma{}) / 128`. The user
// configures this on the host side via the cooperative make_tiled_mma
// (e.g. `Layout<_2,_1,_1>` doubles M-side compute per CTA).
//
// K-pipeline
// ----------
// Two barriers per stage:
//
//   producer_mbar[s] : ClusterTransactionBarrier
//                      Producer arrives once after `cp.async.bulk` issue
//                      (3 TMAs share one barrier, transaction-bytes count
//                       all three). Consumer waits before issuing WGMMA.
//
//   consumer_mbar[s] : ClusterBarrier
//                      Consumer arrives 128× after `warpgroup_wait` releases
//                      the stage. Producer waits before issuing the next
//                      TMA into the same stage.
//
// Pipelining is across K-tiles: the consumer issues a new WGMMA batch
// then immediately calls `warpgroup_wait<K_PIPE_MMAS>()` which keeps
// K_PIPE_MMAS batches in flight. With K_PIPE_MMAS=1 the loop-carried
// chain is kept full and the next-stage barrier wait + next WGMMA can
// overlap with the trailing WGMMA's tensor-core latency.
//
// Bounds
// ------
// M and N can be arbitrary. TMA naturally zero-fills out-of-bound loads
// (so accumulators stay correct), and stores are predicated per (m, n)
// coordinate.

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/device_kernel.h"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include "cutlass/cluster_launch.hpp"
#include "cutlass/arch/mma_sm90.h"

#include "cute/tensor.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/algorithm/functional.hpp"

namespace cutlass {
namespace gemm {
namespace kernel {

namespace sm90_dual_gemm_detail {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////
// SharedStorage for one Sm90 dual-GEMM CTA.
//
// Three pipelined smem buffers (A, B0, B1), one producer barrier per stage
// (TMA-arrival), one consumer barrier per stage (MMA-completion-release).
////////////////////////////////////////////////////////////////////////////////

template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct DualGemmSharedStorage {
  static constexpr int K_PIPE_MAX = size<2>(SmemLayoutA{});

  alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> sA;
  alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> sB0;
  alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> sB1;

  alignas(16) uint64_t producer_mbar[K_PIPE_MAX];
  alignas(16) uint64_t consumer_mbar[K_PIPE_MAX];
};

////////////////////////////////////////////////////////////////////////////////
// Kernel
//
// Threading: 256 threads / CTA = 2 warpgroups
//   - wg 0 (threads   0-127): producer (only lane 0 of warp 0 is live)
//   - wg 1 (threads 128-255): consumer (full WGMMA + epilogue)
////////////////////////////////////////////////////////////////////////////////

template <
    class ProblemShape,
    class CtaTiler,
    class ElementA, class SmemLayoutA, class TmaA,
    class ElementB, class SmemLayoutB, class TmaB,
    class ElementC, class CStride, class TiledMma,
    class CombineOp>
__global__
__launch_bounds__(/*MaxThreads=*/(decltype(size(TiledMma{}))::value + 128), 1)
void
sm90_dual_gemm_device(
    ProblemShape shape_MNK,
    CtaTiler cta_tiler,
    ElementA const* /*ptr_A — only here so TMA atom can be constructed host-side*/,
    CUTLASS_GRID_CONSTANT TmaA  const tma_a,
    ElementB const* /*ptr_B0*/,
    CUTLASS_GRID_CONSTANT TmaB  const tma_b0,
    ElementB const* /*ptr_B1*/,
    CUTLASS_GRID_CONSTANT TmaB  const tma_b1,
    ElementC*       ptr_D2,     CStride dD2,
    TiledMma        mma,
    CombineOp       combine_op)
{
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  using namespace cute;

  // ---------- preconditions ----------
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});
  static_assert(is_static<SmemLayoutA>::value);
  static_assert(is_static<SmemLayoutB>::value);
  static_assert(decltype(size(TiledMma{}))::value % 128 == 0,
                "Sm90 dual gemm: TiledMma thread-count must be a multiple of "
                "128 (one consumer warpgroup per 128 threads).");

  constexpr int kNumConsumerWgs   = decltype(size(TiledMma{}))::value / 128;
  constexpr int kConsumerThreads  = 128 * kNumConsumerWgs;
  constexpr int kProducerThreads  = 128;
  constexpr int kBarrierArvCount  = kConsumerThreads;

  // ---------- gmem tensors ----------
  auto [M, N, K] = shape_MNK;
  Tensor mA  = tma_a .get_tma_tensor(make_shape(M, K));
  Tensor mB0 = tma_b0.get_tma_tensor(make_shape(N, K));
  Tensor mB1 = tma_b1.get_tma_tensor(make_shape(N, K));
  Tensor mD2 = make_tensor(make_gmem_ptr(ptr_D2), make_shape(M, N), dD2);

  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA  = local_tile(mA,  cta_tiler, cta_coord, Step<_1, X,_1>{});
  Tensor gB0 = local_tile(mB0, cta_tiler, cta_coord, Step< X,_1,_1>{});
  Tensor gB1 = local_tile(mB1, cta_tiler, cta_coord, Step< X,_1,_1>{});
  Tensor gD2 = local_tile(mD2, cta_tiler, cta_coord, Step<_1,_1, X>{});

  // ---------- smem tensors ----------
  extern __shared__ char smem_buf[];
  using Storage = DualGemmSharedStorage<ElementA, ElementB, SmemLayoutA, SmemLayoutB>;
  Storage& storage = *reinterpret_cast<Storage*>(smem_buf);

  Tensor sA  = make_tensor(make_smem_ptr(storage.sA .begin()), SmemLayoutA{});
  Tensor sB0 = make_tensor(make_smem_ptr(storage.sB0.begin()), SmemLayoutB{});
  Tensor sB1 = make_tensor(make_smem_ptr(storage.sB1.begin()), SmemLayoutB{});

  // ---------- TMA partitioning ----------
  auto [tAgA,  tAsA ] = tma_partition(tma_a , Int<0>{}, Layout<_1>{},
                                      group_modes<0,2>(sA ), group_modes<0,2>(gA ));
  auto [tBgB0, tBsB0] = tma_partition(tma_b0, Int<0>{}, Layout<_1>{},
                                      group_modes<0,2>(sB0), group_modes<0,2>(gB0));
  auto [tBgB1, tBsB1] = tma_partition(tma_b1, Int<0>{}, Layout<_1>{},
                                      group_modes<0,2>(sB1), group_modes<0,2>(gB1));

  constexpr uint32_t tma_transaction_bytes =
      static_cast<uint32_t>(sizeof(make_tensor_like(tensor<0>(tAsA))) +
                            sizeof(make_tensor_like(tensor<0>(tBsB0))) +
                            sizeof(make_tensor_like(tensor<0>(tBsB1))));

  constexpr int K_PIPE_MAX  = Storage::K_PIPE_MAX;
  constexpr int K_PIPE_MMAS = 1;

  int  k_tile_count = size<1>(tAgA);

  // ---------- warpgroup role ----------
  int thr_idx        = threadIdx.x;
  int warp_idx       = cutlass::canonical_warp_idx_sync();
  // wg_idx == 0          → producer warpgroup
  // wg_idx == 1..N       → consumer warpgroup #(wg_idx-1) of the cooperative pair/triple/...
  int wg_idx         = thr_idx / 128;
  int cons_thr_idx   = thr_idx - 128;                // [0, kConsumerThreads) for consumer wgs

  using ProducerBar = cutlass::arch::ClusterTransactionBarrier;
  using ConsumerBar = cutlass::arch::ClusterBarrier;

  // ---------- barrier init (one thread total) ----------
  if (warp_idx == 0 && cute::elect_one_sync()) {
    CUTLASS_PRAGMA_UNROLL
    for (int p = 0; p < K_PIPE_MAX; ++p) {
      ProducerBar::init(&storage.producer_mbar[p], 1);
      ConsumerBar::init(&storage.consumer_mbar[p], kBarrierArvCount);
    }
  }
  // Make barrier inits visible to all threads in the CTA before they start
  // consuming them.
  __syncthreads();

  // ============================================================================
  // Producer warpgroup
  // ============================================================================
  if (wg_idx == 0) {
    cutlass::arch::warpgroup_reg_dealloc<40>();

    // Inactive lanes / warps in the producer wg exit early after reg-dealloc.
    // Only lane 0 of warp 0 issues TMAs.
    if (warp_idx != 0) return;
    if (!cute::elect_one_sync()) return;

    // Prefetch up to K_PIPE_MAX stages without waiting — those are the
    // initial fills that the consumer hasn't yet reached. State advance is
    // done implicitly by issuing into stages 0..prefetch_count-1.
    int const prefetch_count =
        (k_tile_count < K_PIPE_MAX) ? k_tile_count : K_PIPE_MAX;
    CUTLASS_PRAGMA_UNROLL
    for (int p = 0; p < prefetch_count; ++p) {
      ProducerBar::arrive_and_expect_tx(&storage.producer_mbar[p],
                                        tma_transaction_bytes);
      copy(tma_a .with(storage.producer_mbar[p]),
           tAgA (_, p), tAsA (_, p));
      copy(tma_b0.with(storage.producer_mbar[p]),
           tBgB0(_, p), tBsB0(_, p));
      copy(tma_b1.with(storage.producer_mbar[p]),
           tBgB1(_, p), tBsB1(_, p));
    }

    // Steady-state main loop. Each iteration: wait for consumer to release
    // the next stage, then re-arm the producer barrier and issue a fresh
    // TMA into it.  write_phase starts at 0 (matching the initial parity
    // of consumer_mbar) and flips on every wrap of write_pipe.
    int      write_pipe  = 0;
    uint32_t write_phase = 0;
    CUTLASS_PRAGMA_NO_UNROLL
    for (int k = K_PIPE_MAX; k < k_tile_count; ++k) {
      ConsumerBar::wait(&storage.consumer_mbar[write_pipe], write_phase);

      ProducerBar::arrive_and_expect_tx(&storage.producer_mbar[write_pipe],
                                        tma_transaction_bytes);
      copy(tma_a .with(storage.producer_mbar[write_pipe]),
           tAgA (_, k), tAsA (_, write_pipe));
      copy(tma_b0.with(storage.producer_mbar[write_pipe]),
           tBgB0(_, k), tBsB0(_, write_pipe));
      copy(tma_b1.with(storage.producer_mbar[write_pipe]),
           tBgB1(_, k), tBsB1(_, write_pipe));

      ++write_pipe;
      if (write_pipe == K_PIPE_MAX) {
        write_pipe = 0;
        write_phase ^= 1;
      }
    }
    return;
  }

  // ============================================================================
  // Consumer warpgroup(s) — cooperative when kNumConsumerWgs > 1
  // ============================================================================
  // Register budget: SM has 64K regs total. 1 producer wg × 40 + N consumer
  // wgs × R must satisfy 40 + N·R ≤ 65536 / 128 = 512.
  //   N=1 ⇒ R ≤ 472, pick 240 (matches CUTLASS pingpong)
  //   N=2 ⇒ R ≤ 236, pick 232 (cooperative; matches CUTLASS cooperative)
  if constexpr (kNumConsumerWgs == 1) {
    cutlass::arch::warpgroup_reg_alloc<240>();
  } else {
    cutlass::arch::warpgroup_reg_alloc<232>();
  }

  // For a cooperative TiledMma whose layout spans multiple warpgroups, the
  // thread slice must be queried with the *flattened* index across the math
  // wgs (0 .. kConsumerThreads-1). Each math wg's threads naturally cover
  // its sub-tile of the (BLK_M, BLK_N) accumulator.
  ThrMMA thr_mma = mma.get_thread_slice(cons_thr_idx);
  Tensor tCsA  = thr_mma.partition_A(sA );
  Tensor tCsB0 = thr_mma.partition_B(sB0);
  Tensor tCsB1 = thr_mma.partition_B(sB1);

  Tensor tCgC  = thr_mma.partition_C(gD2);
  Tensor tCrC0 = thr_mma.make_fragment_C(tCgC);
  Tensor tCrC1 = thr_mma.make_fragment_C(tCgC);
  clear(tCrC0);
  clear(tCrC1);

  Tensor tCrA  = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB0 = thr_mma.make_fragment_B(tCsB0);
  Tensor tCrB1 = thr_mma.make_fragment_B(tCsB1);

  int      read_pipe     = 0;
  uint32_t read_phase    = 0;
  int      release_pipe  = 0;
  uint32_t release_phase = 0;

  // ---------- Prologue: queue K_PIPE_MMAS WGMMA batches without releasing ----
  int prologue_count = (k_tile_count < K_PIPE_MMAS) ? k_tile_count : K_PIPE_MMAS;
  CUTLASS_PRAGMA_UNROLL
  for (int p = 0; p < prologue_count; ++p) {
    ProducerBar::wait(&storage.producer_mbar[read_pipe], read_phase);

    cute::warpgroup_arrive();
    cute::gemm(mma, tCrA(_,_,_,read_pipe), tCrB0(_,_,_,read_pipe), tCrC0);
    cute::gemm(mma, tCrA(_,_,_,read_pipe), tCrB1(_,_,_,read_pipe), tCrC1);
    cute::warpgroup_commit_batch();

    ++read_pipe;
    if (read_pipe == K_PIPE_MAX) { read_pipe = 0; read_phase ^= 1; }
  }

  // ---------- Mainloop: issue, wait for K_PIPE_MMAS-old batch, release --------
  int mainloop_count = k_tile_count - prologue_count;
  CUTLASS_PRAGMA_NO_UNROLL
  for (int k = 0; k < mainloop_count; ++k) {
    ProducerBar::wait(&storage.producer_mbar[read_pipe], read_phase);

    cute::warpgroup_arrive();
    cute::gemm(mma, tCrA(_,_,_,read_pipe), tCrB0(_,_,_,read_pipe), tCrC0);
    cute::gemm(mma, tCrA(_,_,_,read_pipe), tCrB1(_,_,_,read_pipe), tCrC1);
    cute::warpgroup_commit_batch();

    cute::warpgroup_wait<K_PIPE_MMAS>();

    ConsumerBar::arrive(&storage.consumer_mbar[release_pipe]);

    ++read_pipe;
    if (read_pipe == K_PIPE_MAX) { read_pipe = 0; read_phase ^= 1; }
    ++release_pipe;
    if (release_pipe == K_PIPE_MAX) { release_pipe = 0; release_phase ^= 1; }
  }

  // ---------- Drain remaining in-flight WGMMAs and release their stages ------
  cute::warpgroup_wait<0>();
  CUTLASS_PRAGMA_UNROLL
  for (int p = 0; p < prologue_count; ++p) {
    ConsumerBar::arrive(&storage.consumer_mbar[release_pipe]);
    ++release_pipe;
    if (release_pipe == K_PIPE_MAX) { release_pipe = 0; release_phase ^= 1; }
  }

  // ---------- Epilogue: combine (acc0, acc1) and predicate-store --------------
  Tensor cD2  = make_identity_tensor(make_shape(size<0>(gD2), size<1>(gD2)));
  Tensor tCcD = thr_mma.partition_C(cD2);

  int const m_offset = blockIdx.x * size<0>(gD2);
  int const n_offset = blockIdx.y * size<1>(gD2);

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < size(tCrC0); ++i) {
    auto coord = tCcD(i);
    int  m_g   = m_offset + get<0>(coord);
    int  n_g   = n_offset + get<1>(coord);
    if (m_g < M && n_g < N) {
      ElementC c0 = static_cast<ElementC>(tCrC0(i));
      ElementC c1 = static_cast<ElementC>(tCrC1(i));
      tCgC(i) = combine_op(c0, c1);
    }
  }
#endif // CUTLASS_ARCH_MMA_SM90_SUPPORTED
}

} // namespace sm90_dual_gemm_detail

} // namespace kernel
} // namespace gemm
} // namespace cutlass
