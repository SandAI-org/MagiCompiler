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

"""Operator-level determinism matrix for the ``TORCH_DETERMINISTIC`` switch.

Determinism is compositional: if every primitive operator is bitwise-reproducible
for identical inputs, so is any graph composed from them. We therefore verify the
operator layer directly, rather than per model. The cases cover the operators
most relevant to LLM and video/diffusion models drawn from PyTorch's documented
nondeterministic list (atomicAdd-based scatter/index, embedding/cross-entropy,
interpolate/grid_sample backward, SDPA), plus deterministic references.

For each operator we run two checks:

  - Contract (asserted), under ``use_deterministic_algorithms(True)`` + cuDNN
    deterministic + cuBLAS workspace + fixed seed: the result over N runs on the
    same input must be either (a) bitwise identical, or (b) a RuntimeError when no
    deterministic implementation exists -- a refusal is acceptable, silent drift
    is not. For case (b) we additionally re-run under ``warn_only=True`` and
    report whether the fallback kernel is reproducible (so it is measured, not
    skipped; e.g. grid_sample backward both raises and is non-reproducible).

  - Probe (report-only, hardware/build dependent): the same operator with the
    flag off, reporting which operators the switch actually rescues.

Scope. This matrix verifies eager operators only. ``scaled_dot_product_attention``
dispatches to backend kernels whose backward atomicAdd is not a listed aten op,
so it is included as an explicit SDPA case. Nondeterminism introduced by the
compiled path (fused Inductor/Triton kernels) is outside this operator layer.

Usage:
    CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m pytest \\
        tests/feature_tests/test_torch_op_determinism_matrix.py -v -s
"""

import os

# Best-effort: cuBLAS determinism requires this BEFORE the first cuBLAS handle.
# Prefer exporting it in CI; setdefault here helps when this module is imported early.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from contextlib import contextmanager  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

DEVICE = "cuda"
N_RUNS = 5


# =============================================================================
# Harness
# =============================================================================
@contextmanager
def torch_deterministic(enabled: bool, seed: int = 0, warn_only: bool = False):
    """Pin the full TORCH_DETERMINISTIC context.

    ``warn_only=False`` (strict, default): ops with no deterministic
    implementation RAISE -- the loud refusal is the safe contract.
    ``warn_only=True``: those ops only warn and fall back to their
    nondeterministic kernel, so we can still MEASURE their run-to-run
    reproducibility instead of skipping them.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    prev = torch.are_deterministic_algorithms_enabled()
    prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_d = torch.backends.cudnn.deterministic
    cudnn_b = torch.backends.cudnn.benchmark
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        yield
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn)
        torch.backends.cudnn.deterministic = cudnn_d
        torch.backends.cudnn.benchmark = cudnn_b


def _gen(seed: int = 0):
    return torch.Generator(device=DEVICE).manual_seed(seed)


def run_collect(builder, n: int = N_RUNS):
    """Call ``builder()`` ``n`` times on identical inputs; clone each result."""
    outs = []
    for _ in range(n):
        outs.append(builder().detach().clone())
    torch.cuda.synchronize()
    return outs


def all_bitwise_identical(outs):
    ref = outs[0]
    return all(torch.equal(ref, o) for o in outs[1:])


# =============================================================================
# Operator cases
#   Each builder returns ONE tensor to compare (forward output for forward-
#   atomic ops; input/weight grad for backward-atomic ops). Inputs are rebuilt
#   from a FIXED seed every call, so any run-to-run difference is the kernel's.
# =============================================================================
# --- forward atomicAdd via duplicate-index accumulation (HIGH collision) ---
_NOUT, _NSRC, _D = 16, 200_000, 32


def _src():
    return torch.randn(_NSRC, _D, generator=_gen(0), device=DEVICE)


def _idx_rows():
    return torch.randint(0, _NOUT, (_NSRC,), generator=_gen(1), device=DEVICE)


def case_index_add():
    return torch.zeros(_NOUT, _D, device=DEVICE).index_add(0, _idx_rows(), _src())


def case_scatter_add():
    idx = _idx_rows().unsqueeze(1).expand(-1, _D)
    return torch.zeros(_NOUT, _D, device=DEVICE).scatter_add(0, idx, _src())


def case_scatter_reduce_sum():
    idx = _idx_rows().unsqueeze(1).expand(-1, _D)
    return torch.zeros(_NOUT, _D, device=DEVICE).scatter_reduce(0, idx, _src(), reduce="sum", include_self=True)


def case_index_put_accumulate():
    base = torch.zeros(_NOUT, _D, device=DEVICE)
    base.index_put_((_idx_rows(),), _src(), accumulate=True)
    return base


# --- backward atomicAdd (gradient accumulation) -- the real risk regime ---
def case_embedding_backward():
    # token embedding lookup -- every LLM uses it; its dense backward scatters
    # grads of repeated token ids into the weight rows via atomicAdd.
    w = torch.randn(_NOUT, _D, generator=_gen(0), device=DEVICE, requires_grad=True)
    idx = torch.randint(0, _NOUT, (_NSRC,), generator=_gen(1), device=DEVICE)
    F.embedding(idx, w).sum().backward()
    return w.grad


def case_cross_entropy_backward():
    # the LLM training loss (logits over vocab). Backward path touches the same
    # nll/log_softmax kernels that PyTorch lists as nondeterministic on CUDA.
    logits = torch.randn(4096, 128, generator=_gen(0), device=DEVICE, requires_grad=True)
    target = torch.randint(0, 128, (4096,), generator=_gen(1), device=DEVICE)
    F.cross_entropy(logits, target).backward()
    return logits.grad


def case_interpolate_bilinear_backward():
    # spatial upsampling in diffusion/video U-Net & VAE decoders.
    x = torch.randn(8, 8, 64, 64, generator=_gen(0), device=DEVICE, requires_grad=True)
    F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False).sum().backward()
    return x.grad


def case_grid_sample_backward():
    # warping / optical-flow resampling used by video models. PyTorch has NO
    # deterministic implementation for its CUDA backward -> strict mode raises.
    x = torch.randn(8, 8, 64, 64, generator=_gen(0), device=DEVICE, requires_grad=True)
    grid = torch.rand(8, 128, 128, 2, generator=_gen(1), device=DEVICE) * 2 - 1
    F.grid_sample(x, grid, align_corners=False).sum().backward()
    return x.grad


# --- attention: scaled_dot_product_attention DISPATCHES to flash/mem-efficient/
#     cudnn/math backends. The forward is deterministic; the flash/mem-efficient
#     BACKWARD uses atomicAdd over dQ/dK/dV -- the real determinism risk, and it
#     is NOT a single aten op on the nondeterministic list. fp16 is used so the
#     flash/mem-efficient backend (the risky one) is eligible. ---
def _qkv(requires_grad: bool):
    q = torch.randn(8, 16, 512, 64, generator=_gen(0), device=DEVICE, dtype=torch.float16, requires_grad=requires_grad)
    k = torch.randn(8, 16, 512, 64, generator=_gen(1), device=DEVICE, dtype=torch.float16, requires_grad=requires_grad)
    v = torch.randn(8, 16, 512, 64, generator=_gen(2), device=DEVICE, dtype=torch.float16, requires_grad=requires_grad)
    return q, k, v


def case_sdpa_forward():
    q, k, v = _qkv(requires_grad=False)
    return F.scaled_dot_product_attention(q, k, v)


def case_sdpa_backward():
    q, k, v = _qkv(requires_grad=True)
    F.scaled_dot_product_attention(q, k, v).sum().backward()
    return q.grad


# --- "deterministic-anyway" references (no atomics) ---
def case_matmul():
    a = torch.randn(1024, 1024, generator=_gen(0), device=DEVICE)
    b = torch.randn(1024, 1024, generator=_gen(1), device=DEVICE)
    return a @ b


def case_cumsum():
    x = torch.randn(4096, 512, generator=_gen(0), device=DEVICE)
    return torch.cumsum(x, dim=0)


def case_sort_indices():
    x = torch.randint(0, 8, (2048, 512), generator=_gen(0), device=DEVICE).float()
    return torch.sort(x, dim=-1).indices


def case_topk_indices():
    x = torch.randint(0, 8, (2048, 512), generator=_gen(0), device=DEVICE).float()
    return torch.topk(x, 16, dim=-1).indices


# Ordered registry: (name, category, builder)
CASES = [
    # MoE expert combine / token scatter -- the classic high-collision atomicAdd.
    ("index_add", "fwd-atomic", case_index_add),
    ("scatter_add", "fwd-atomic", case_scatter_add),
    ("scatter_reduce_sum", "fwd-atomic", case_scatter_reduce_sum),
    ("index_put_accumulate", "fwd-atomic", case_index_put_accumulate),
    # token embedding (LLM) + cross-entropy loss (LLM training).
    ("embedding_backward", "bwd-atomic", case_embedding_backward),
    ("cross_entropy_backward", "bwd-atomic", case_cross_entropy_backward),
    # diffusion / video: upsampling + warp/resampling.
    ("interpolate_bilinear_backward", "bwd-atomic", case_interpolate_bilinear_backward),
    ("grid_sample_backward", "bwd-atomic", case_grid_sample_backward),
    # attention (every transformer): SDPA fwd + bwd (flash/mem-efficient atomicAdd).
    ("sdpa_forward", "attention", case_sdpa_forward),
    ("sdpa_backward", "attention", case_sdpa_backward),
    # references: matmul / cumsum (RoPE & cumulative ops) / MoE-routing & sampling topk-sort.
    ("matmul", "reference", case_matmul),
    ("cumsum", "reference", case_cumsum),
    ("sort_indices", "reference", case_sort_indices),
    ("topk_indices", "reference", case_topk_indices),
]
_IDS = [c[0] for c in CASES]


# =============================================================================
# Tests
# =============================================================================
class TestTorchDeterministicContract:
    """Under TORCH_DETERMINISTIC, every op is bitwise-identical OR loudly refuses
    (RuntimeError). It must NEVER silently drift."""

    @pytest.mark.parametrize("name,category,builder", CASES, ids=_IDS)
    def test_flag_on_is_deterministic_or_refuses(self, name, category, builder):
        # Strict mode: the op either runs (and must be bitwise identical) or has
        # no deterministic implementation and RAISES.
        try:
            with torch_deterministic(True):
                outs = run_collect(builder)
        except RuntimeError as e:
            # No deterministic implementation: the raise is the safe contract.
            # Re-run in warn_only mode to measure run-to-run reproducibility of
            # the fallback kernel (report-only: it may legitimately drift).
            with torch_deterministic(True, warn_only=True):
                fb = run_collect(builder)
            reproducible = all_bitwise_identical(fb)
            print(
                f"\n[{category:11s}] {name:32s} flag ON -> NO DET IMPL (raises); "
                f"warn_only fallback is {'REPRODUCIBLE' if reproducible else 'NON-REPRODUCIBLE'} "
                f"-- {str(e).splitlines()[0]}"
            )
            return

        identical = all_bitwise_identical(outs)
        print(f"\n[{category:11s}] {name:32s} flag ON -> {'IDENTICAL' if identical else 'DRIFT!!!'}")
        assert identical, (
            f"{name}: NOT bitwise identical even under TORCH_DETERMINISTIC -- "
            "this op silently drifts with the flag on, a determinism hole"
        )


class TestFlagOffProbe:
    """Report-only: which ops actually NEED the flag (drift when it is off).
    Hardware/PyTorch-build dependent -> no hard assertions."""

    @pytest.mark.parametrize("name,category,builder", CASES, ids=_IDS)
    def test_probe_flag_off_drift(self, name, category, builder):
        with torch_deterministic(False):
            try:
                outs = run_collect(builder)
            except RuntimeError as e:  # pragma: no cover - unexpected with flag off
                print(f"\n[probe][{category:11s}] {name:32s} flag OFF -> RAISES: {str(e).splitlines()[0]}")
                return
        verdict = "SAME (det anyway)" if all_bitwise_identical(outs) else "DRIFT (flag needed)"
        print(f"\n[probe][{category:11s}] {name:32s} flag OFF -> {verdict}")
