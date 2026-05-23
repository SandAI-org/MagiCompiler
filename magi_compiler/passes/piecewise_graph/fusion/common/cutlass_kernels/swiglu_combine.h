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

// Binary epilogue combine functor for the swiglu DualGemm fusion.
//
//   D = silu_alpha( clamp(lhs, max=limit) ) * ( clamp(rhs, -limit, limit) + 1 )
//
//   silu_alpha(x) = x * sigmoid(alpha * x)        default: alpha = 1.702, limit = 7.0
//
// `lhs` is the gate-path output fragment (Op0 applied to A @ W_gate.T),
// `rhs` is the linear-path output fragment (Op1 applied to A @ W_linear.T).
// Both arrive as ElementOutput (bf16) fragments — this is dictated by the
// dual-epilogue call site (examples/45_dual_gemm/threadblock/dual_epilogue.h:413
// passes `output_frag_ptr[0][i]` and `[1][i]`, which are post-conversion
// output-type fragments, not raw accumulator fragments). The combine upcasts
// to ElementCompute (fp32) internally, evaluates the swiglu expression, and
// converts back to bf16.
//
// Note on precision: the gate/linear matmuls accumulate in fp32 inside the
// MMAs. Op0/Op1 (LinearCombination, ScaleType::Nothing) downcast those fp32
// accumulators to bf16 before this combine runs. The swiglu math itself
// stays in fp32 here, so the only extra precision loss vs the two-stage EVT
// version is the single fp32→bf16 round-trip on each accumulator at the
// epilogue boundary. Empirically this is well within the bf16 noise floor.
//
// Modelled on cutlass/examples/45_dual_gemm/thread/left_silu_and_mul.h —
// same interface contract: ElementOutput / ElementAccumulator / ElementCompute
// typedefs, kCount fragment width, empty Params, two operator() overloads
// (fragment + scalar), is_source_needed() returning true.

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/epilogue/thread/activation.h"

namespace cutlass {
namespace epilogue {
namespace thread {

template <
    typename ElementOutput_,
    int Count,
    typename ElementAccumulator_ = ElementOutput_,
    typename ElementCompute_     = ElementOutput_,
    FloatRoundStyle Round        = FloatRoundStyle::round_to_nearest>
class SwigluCombine {
public:

    using ElementOutput      = ElementOutput_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementCompute     = ElementCompute_;

    static int const kCount = Count;
    using FragmentOutput      = Array<ElementOutput,      kCount>;
    using FragmentAccumulator = Array<ElementAccumulator, kCount>;
    using ComputeFragment     = Array<ElementCompute,     kCount>;

    static FloatRoundStyle const kRound = Round;

    struct Params {
        ElementCompute alpha;
        ElementCompute limit;
        ElementCompute one;

        CUTLASS_HOST_DEVICE
        Params() : alpha(ElementCompute(1.702f)),
                   limit(ElementCompute(7.0f)),
                   one(ElementCompute(1.0f)) {}

        CUTLASS_HOST_DEVICE
        Params(ElementCompute alpha_, ElementCompute limit_, ElementCompute one_)
            : alpha(alpha_), limit(limit_), one(one_) {}
    };

public:

    CUTLASS_HOST_DEVICE
    SwigluCombine(Params const& p) : alpha_(p.alpha), limit_(p.limit), one_(p.one) {}

    CUTLASS_HOST_DEVICE
    bool is_source_needed() const { return true; }

    CUTLASS_HOST_DEVICE
    void set_k_partition(int /*k_partition*/, int /*k_partition_count*/) {
        // swiglu cannot be split-K-reduced (non-linear epilogue).
        assert(false);
    }

    // Fragment-level. lhs = gate output fragment (bf16, post Op0),
    //                  rhs = linear output fragment (bf16, post Op1).
    CUTLASS_HOST_DEVICE
    FragmentOutput operator()(FragmentOutput const& lhs,
                              FragmentOutput const& rhs) const {
        NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round> in2c;
        NumericArrayConverter<ElementOutput,  ElementCompute, kCount, Round> c2o;

        ComputeFragment gate = in2c(lhs);
        ComputeFragment lin  = in2c(rhs);
        ComputeFragment out;

        Sigmoid<ElementCompute> sig;
        ElementCompute const nlimit = -limit_;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kCount; ++i) {
            ElementCompute g = gate[i] < limit_  ? gate[i] : limit_;
            ElementCompute r = lin[i]  < nlimit ? nlimit
                              : (lin[i] > limit_ ? limit_ : lin[i]);
            ElementCompute silu_g = g * sig(alpha_ * g);
            out[i] = silu_g * (r + one_);
        }
        return c2o(out);
    }

    // Scalar overload — required by the DualGemm epilogue boilerplate.
    CUTLASS_HOST_DEVICE
    ElementOutput operator()(ElementOutput const& lhs,
                             ElementOutput const& rhs) const {
        ElementCompute g(lhs), r(rhs);
        ElementCompute const nlimit = -limit_;

        Sigmoid<ElementCompute> sig;

        g = g < limit_  ? g : limit_;
        r = r < nlimit ? nlimit : (r > limit_ ? limit_ : r);
        ElementCompute silu_g = g * sig(alpha_ * g);
        return ElementOutput(silu_g * (r + one_));
    }

private:
    ElementCompute alpha_;
    ElementCompute limit_;
    ElementCompute one_;
};

} // namespace thread
} // namespace epilogue
} // namespace cutlass
