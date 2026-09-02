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

"""
Tests for symbolic over-unification in torch.split with dynamic modality sizes.

Problem: When a modality has 0 tokens during initial tracing, Dynamo unifies
symbolic variables (e.g. total_tokens == video_tokens), causing AssertionError
on subsequent calls with different shapes when guard_filter_fn bypasses
recompilation.  The order matters: only the first compilation's shape
distribution determines whether symbols get unified.

Fix: Use a CPU "size carrier" tensor with mark_unbacked dimensions so each
modality size becomes an independent unbacked SymInt (u0, u1, u2).  When
the dispatcher is created inside @torch.compile (two-level compile
architecture), tolist() triggers a graph break, so mark_unbacked runs in
eager.  The is_compiling() guard is a safety net for this case.

Part A/B: Raw bug and carrier-tensor fix in isolation (single-level compile).
Part C: CP4-like cache reuse with carrier-tensor fix (single-level compile).
Part D: Two-level compile (@torch.compile outer + @magi_compile inner)
        with carrier + guard.
"""

import os

import pytest
import torch
import torch.nn as nn

from magi_compiler import magi_compile
from magi_compiler.utils.envs import IS_PT_212

HIDDEN = 64
NUM_MODALITIES = 3


# ─── Model definitions ────────────────────────────────────────────────


class MultiModalSplitNorm(nn.Module):
    """Minimal model reproducing the symbolic unification issue.

    Uses torch.split with list[int] group_sizes → Dynamo sees concrete ints,
    unifies symbols when some are 0.  Mirrors the real ModalityDispatcher +
    MultiModalityRMSNorm pattern where each chunk goes through a per-modality
    linear layer.
    """

    def __init__(self, hidden_size: int, num_modalities: int):
        super().__init__()
        self.num_modalities = num_modalities
        self.experts = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias=False, dtype=torch.bfloat16) for _ in range(num_modalities)]
        )

    def forward(self, x: torch.Tensor, group_sizes: list[int]) -> torch.Tensor:
        chunks = list(torch.split(x, group_sizes, dim=0))
        for i in range(self.num_modalities):
            chunks[i] = self.experts[i](chunks[i].to(torch.bfloat16)).float()
        return torch.cat(chunks, dim=0)


class MultiModalSplitNormFixed(nn.Module):
    """Fixed version using carrier tensor with mark_unbacked dimensions.

    group_sizes come from carrier.shape[i] which are unbacked SymInts,
    preventing symbolic unification. Mirrors the real ModalityDispatcher
    which unconditionally splits and processes all chunks.
    """

    def __init__(self, hidden_size: int, num_modalities: int):
        super().__init__()
        self.num_modalities = num_modalities
        self.experts = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias=False, dtype=torch.bfloat16) for _ in range(num_modalities)]
        )

    def forward(self, x: torch.Tensor, size_carrier: torch.Tensor) -> torch.Tensor:
        group_sizes = [size_carrier.shape[i] for i in range(self.num_modalities)]
        chunks = list(torch.split(x, group_sizes, dim=0))
        for i in range(self.num_modalities):
            chunks[i] = self.experts[i](chunks[i].to(torch.bfloat16)).float()
        return torch.cat(chunks, dim=0)


# ─── Helpers ───────────────────────────────────────────────────────────


def _make_input(sizes: list[int], hidden: int = HIDDEN, device="cuda"):
    total = sum(sizes)
    return torch.randn(total, hidden, device=device, dtype=torch.float32)


@torch.compiler.disable()
def _mark_carrier_unbacked(carrier: torch.Tensor, num_modalities: int) -> torch.Tensor:
    """Mark each dim of carrier as unbacked, in a Dynamo-safe wrapper.

    @torch.compiler.disable() is required for PT 2.12 two-level compile
    (outer @torch.compile + inner @magi_compile): the is_compiling() guard
    alone is insufficient — magi_compile's guard bypass causes Dynamo to
    still unify symbols when some modalities have 0 tokens.
    """
    for i in range(num_modalities):
        torch._dynamo.decorators.mark_unbacked(carrier, i)
    return carrier


def _make_carrier(sizes: list[int]):
    """Create a CPU carrier tensor and mark each dim as unbacked."""
    return _mark_carrier_unbacked(torch.empty(*sizes), len(sizes))


def _bypass_all_guards(guards):
    return [False for _ in guards]


# ─── Test scenarios ────────────────────────────────────────────────────
# First call: only one modality has tokens (audio=0, text=0) so Dynamo
# deduces total_tokens == video_tokens, unifying symbols.
# Second call: all modalities > 0 with a different total → triggers
# AssertionError because the unified symbolic expression is wrong.

SCENARIO_FIRST = [128, 0, 0]  # only video, audio=text=0
SCENARIO_SECOND = [60, 20, 20]  # all > 0, different total


# =======================================================================
# Part A: Reproduce the symbolic over-unification problem
# =======================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_split_symbolic_unification_torch_compile():
    """torch.compile + guard bypass: list[int] group_sizes cause symbolic
    unification when one modality is 0, leading to AssertionError on reuse."""
    torch._dynamo.reset()
    model = MultiModalSplitNorm(HIDDEN, NUM_MODALITIES).cuda().eval()

    compiled = torch.compile(
        model, fullgraph=True, dynamic=True, backend="inductor", options={"guard_filter_fn": _bypass_all_guards}
    )

    x1 = _make_input(SCENARIO_FIRST)
    with torch.no_grad():
        out1 = compiled(x1, SCENARIO_FIRST)
    assert out1.shape == (sum(SCENARIO_FIRST), HIDDEN)

    x2 = _make_input(SCENARIO_SECOND)
    with pytest.raises((RuntimeError, AssertionError)):
        with torch.no_grad():
            compiled(x2, SCENARIO_SECOND)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_split_symbolic_unification_magi_compile():
    """magi_compile path: same issue with guard bypass built into magi."""
    torch._dynamo.reset()

    @magi_compile(dynamic_arg_dims={"x": 0})
    class BuggyModel(MultiModalSplitNorm):
        pass

    model = BuggyModel(HIDDEN, NUM_MODALITIES).cuda().eval()

    x1 = _make_input(SCENARIO_FIRST)
    with torch.no_grad():
        out1 = model(x1, SCENARIO_FIRST)
    assert out1.shape == (sum(SCENARIO_FIRST), HIDDEN)

    x2 = _make_input(SCENARIO_SECOND)
    with pytest.raises((RuntimeError, AssertionError)):
        with torch.no_grad():
            model(x2, SCENARIO_SECOND)


# =======================================================================
# Part B: Verify the mark_unbacked carrier tensor fix
# =======================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mark_unbacked_carrier_torch_compile():
    """Fixed model with carrier tensor + mark_unbacked: torch.compile path."""
    torch._dynamo.reset()
    model = MultiModalSplitNormFixed(HIDDEN, NUM_MODALITIES).cuda().eval()

    compiled = torch.compile(
        model, fullgraph=True, dynamic=True, backend="inductor", options={"guard_filter_fn": _bypass_all_guards}
    )

    scenarios = [[128, 0, 64], [100, 40, 52], [0, 200, 0], [50, 50, 50], [256, 0, 0]]

    with torch.no_grad():
        for sizes in scenarios:
            x = _make_input(sizes)
            carrier = _make_carrier(sizes)
            out = compiled(x, carrier)
            assert out.shape == (sum(sizes), HIDDEN), f"Failed for sizes={sizes}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(
    IS_PT_212,
    reason=(
        "PT 2.12: accumulated CUDA state from CUTLASS EVT kernel compilation "
        "causes illegal-memory-access when autotune_at_compile_time=True. "
        "Resolved by 8-shard CI (pass/ and feature-runtime run in separate processes)."
    ),
)
def test_mark_unbacked_carrier_autotune_compat():
    """Verify carrier tensor + mark_unbacked works with autotune_at_compile_time
    both enabled and disabled. standalone_compile hardcodes autotune=True, so
    MagiCompiler needs to override it to False for unbacked SymInt compat."""
    from torch._inductor import config as inductor_config

    scenarios = [[128, 0, 0], [60, 20, 20], [0, 100, 0], [50, 50, 50]]

    for autotune_val in [False, True]:
        torch._dynamo.reset()
        model = MultiModalSplitNormFixed(HIDDEN, NUM_MODALITIES).cuda().eval()

        with inductor_config.patch("triton.autotune_at_compile_time", autotune_val):
            compiled = torch.compile(
                model, fullgraph=True, dynamic=True, backend="inductor", options={"guard_filter_fn": _bypass_all_guards}
            )

            with torch.no_grad():
                for sizes in scenarios:
                    x = _make_input(sizes)
                    carrier = _make_carrier(sizes)
                    out = compiled(x, carrier)
                    assert out.shape == (sum(sizes), HIDDEN), f"Failed for sizes={sizes}, autotune={autotune_val}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mark_unbacked_carrier_magi_compile():
    """Fixed model with carrier tensor: magi_compile path."""
    torch._dynamo.reset()

    @magi_compile(dynamic_arg_dims={"x": 0})
    class FixedModel(MultiModalSplitNormFixed):
        pass

    model = FixedModel(HIDDEN, NUM_MODALITIES).cuda().eval()

    scenarios = [[128, 0, 64], [100, 40, 52], [0, 200, 0], [50, 50, 50], [256, 0, 0]]

    with torch.no_grad():
        for sizes in scenarios:
            x = _make_input(sizes)
            carrier = _make_carrier(sizes)
            out = model(x, carrier)
            assert out.shape == (sum(sizes), HIDDEN), f"Failed for sizes={sizes}"


# =======================================================================
# Part C: CP4-like cache reuse — single-level @magi_compile with
#         mark_unbacked carrier tensor
# =======================================================================
# Tests the carrier-tensor fix in a CP4-like setting (single-level compile).
# The dispatcher uses mark_unbacked to prevent symbolic over-unification
# when some ranks get 0 tokens for certain modalities.
#
# This tests the single-level case.  See Part D for the two-level
# compile pattern (@torch.compile outer + @magi_compile inner).


class ModalityDispatcherMock:
    """Mock dispatcher for single-level @magi_compile tests.

    Uses a carrier tensor with mark_unbacked to prevent symbolic
    over-unification.  This is needed in single-level compile because
    Dynamo traces group_size_cpu directly and would unify symbols when
    some modalities have 0 tokens.

    Note: In two-level compile (@torch.compile outer + @magi_compile inner)
    the is_compiling() guard is also needed.  See ModalityDispatcherMockV2.
    """

    def __init__(self, modality_mapping: torch.Tensor, num_modalities: int):
        self.num_modalities = num_modalities
        self.permute_mapping = torch.argsort(modality_mapping)
        permuted = modality_mapping[self.permute_mapping]
        group_sizes = torch.bincount(permuted, minlength=num_modalities).tolist()

        self._size_carrier = torch.empty(*[int(s) for s in group_sizes])
        for i in range(num_modalities):
            torch._dynamo.decorators.mark_unbacked(self._size_carrier, i)

    @property
    def group_size_cpu(self) -> list[int]:
        return [self._size_carrier.shape[i] for i in range(self.num_modalities)]

    def dispatch(self, x: torch.Tensor) -> list[torch.Tensor]:
        return list(torch.split(x, self.group_size_cpu, dim=0))

    def undispatch(self, *groups: torch.Tensor) -> torch.Tensor:
        return torch.cat(groups, dim=0)

    def permute(self, x: torch.Tensor) -> torch.Tensor:
        return x[self.permute_mapping]


class TransformerBlockMock(nn.Module):
    """Simplified transformer block using ModalityDispatcherMock."""

    def __init__(self, hidden_size: int, num_modalities: int):
        super().__init__()
        self.num_modalities = num_modalities
        self.experts = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias=False, dtype=torch.bfloat16) for _ in range(num_modalities)]
        )
        self.norm_weights = nn.ParameterList([nn.Parameter(torch.ones(hidden_size)) for _ in range(num_modalities)])

    def forward(self, x: torch.Tensor, dispatcher: ModalityDispatcherMock) -> torch.Tensor:
        chunks = dispatcher.dispatch(x)
        for i in range(self.num_modalities):
            normed = chunks[i].float() * (self.norm_weights[i] + 1)
            chunks[i] = self.experts[i](normed.to(torch.bfloat16)).float()
        return dispatcher.undispatch(*chunks)


def _cp_split(total_seq: int, cp_size: int) -> list[int]:
    """Mimic ulysses_scheduler._dispatch split logic: divide seq as evenly
    as possible, remainder tokens go to the first ranks."""
    base = total_seq // cp_size
    remainder = total_seq % cp_size
    return [base + 1] * remainder + [base] * (cp_size - remainder)


def _build_global_modality_mapping(video_tokens: int, audio_tokens: int, text_tokens: int, device="cuda") -> torch.Tensor:
    """Build a global modality_mapping like the real pipeline does:
    [0]*video ++ [1]*audio ++ [2]*text."""
    parts = []
    if video_tokens > 0:
        parts.append(torch.zeros(video_tokens, dtype=torch.long, device=device))
    if audio_tokens > 0:
        parts.append(torch.ones(audio_tokens, dtype=torch.long, device=device))
    if text_tokens > 0:
        parts.append(torch.full((text_tokens,), 2, dtype=torch.long, device=device))
    if not parts:
        return torch.zeros(0, dtype=torch.long, device=device)
    return torch.cat(parts)


# Each sample is (description, video_tokens, audio_tokens, text_tokens).
# The CP split is computed automatically — some ranks will naturally get
# 0 tokens for certain modalities, just like production.
CP4_SAMPLES = [
    ("duration=1s (video-heavy, short audio/text)", 128, 16, 16),
    ("duration=2s (video-heavy, more audio/text)", 256, 32, 32),
    ("duration=1s (balanced)", 64, 48, 48),
    ("duration=2s (video-only)", 160, 0, 0),
    ("duration=1s (all modalities, odd total)", 97, 31, 19),
]
CP_SIZE = 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cp4_cache_reuse_magi_compile():
    """Verify magi_compile cache reuse across CP4 rank distributions.

    For each sample, builds a global modality_mapping and splits it across
    CP_SIZE=4 ranks (mirroring real ulysses_scheduler._dispatch).  A single
    compiled model handles all resulting per-rank shapes — including those
    where some modalities have 0 tokens — without recompilation.
    """
    torch._dynamo.reset()

    @magi_compile(dynamic_arg_dims={"x": 0})
    class CompiledBlock(TransformerBlockMock):
        pass

    model = CompiledBlock(HIDDEN, NUM_MODALITIES).cuda().eval()

    with torch.no_grad():
        for desc, n_video, n_audio, n_text in CP4_SAMPLES:
            total_seq = n_video + n_audio + n_text
            if total_seq == 0:
                continue

            global_mapping = _build_global_modality_mapping(n_video, n_audio, n_text)
            split_sizes = _cp_split(total_seq, CP_SIZE)

            rank_mappings = torch.split(global_mapping, split_sizes, dim=0)

            for rank, rank_mapping in enumerate(rank_mappings):
                rank_seq = rank_mapping.shape[0]
                if rank_seq == 0:
                    continue

                dispatcher = ModalityDispatcherMock(rank_mapping, NUM_MODALITIES)
                x = torch.randn(rank_seq, HIDDEN, device="cuda", dtype=torch.float32)
                x_permuted = dispatcher.permute(x)

                out = model(x_permuted, dispatcher)

                assert out.shape == (rank_seq, HIDDEN), (
                    f"Shape mismatch for {desc} rank{rank}: " f"expected ({rank_seq}, {HIDDEN}), got {out.shape}"
                )


# =======================================================================
# Part D: Two-level compile — dispatcher created inside @torch.compile
# =======================================================================
# Simulates a two-level compile architecture:
#   - Outer model: @torch.compile(dynamic=True, fullgraph=False)
#   - Inner block: @magi_compile
# ModalityDispatcher is created inside Transformer.forward (the outer
# @torch.compile region).  tolist() triggers a graph break inside
# __init__, so the remaining code (carrier tensor + mark_unbacked)
# runs in eager.  The is_compiling() guard is a safety net in case
# graph-break behavior changes in future PyTorch versions.


class ModalityDispatcherMockV2:
    """Mock dispatcher matching the real ModalityDispatcher — carrier
    tensor with mark_unbacked behind is_compiling() guard."""

    def __init__(self, modality_mapping: torch.Tensor, num_modalities: int):
        self.num_modalities = num_modalities
        self.permute_mapping = torch.argsort(modality_mapping)
        permuted = modality_mapping[self.permute_mapping]
        group_sizes = torch.bincount(permuted, minlength=num_modalities).tolist()

        self._size_carrier = torch.empty(*[int(s) for s in group_sizes])
        if not torch.compiler.is_compiling():
            for i in range(num_modalities):
                torch._dynamo.decorators.mark_unbacked(self._size_carrier, i)

    @property
    def group_size_cpu(self) -> list[int]:
        return [self._size_carrier.shape[i] for i in range(self.num_modalities)]

    def dispatch(self, x: torch.Tensor) -> list[torch.Tensor]:
        return list(torch.split(x, self.group_size_cpu, dim=0))

    def undispatch(self, *groups: torch.Tensor) -> torch.Tensor:
        return torch.cat(groups, dim=0)

    def permute(self, x: torch.Tensor) -> torch.Tensor:
        return x[self.permute_mapping]


class ModalityDispatcherMockV2Fixed:
    """Fixed dispatcher using @torch.compiler.disable() wrapper.

    On PT 2.12, the is_compiling() guard in ModalityDispatcherMockV2 is
    insufficient for two-level compile (outer @torch.compile + inner
    @magi_compile) with bad-order inputs (first call has zero-token
    modalities).  magi_compile's guard bypass causes Dynamo to still
    unify symbols, leading to shape-mismatch assertions at runtime.

    The fix: wrap mark_unbacked in @torch.compiler.disable() so Dynamo
    unconditionally skips the function body during tracing.
    """

    def __init__(self, modality_mapping: torch.Tensor, num_modalities: int):
        self.num_modalities = num_modalities
        self.permute_mapping = torch.argsort(modality_mapping)
        permuted = modality_mapping[self.permute_mapping]
        group_sizes = torch.bincount(permuted, minlength=num_modalities).tolist()

        self._size_carrier = _mark_carrier_unbacked(torch.empty(*[int(s) for s in group_sizes]), num_modalities)

    @property
    def group_size_cpu(self) -> list[int]:
        return [self._size_carrier.shape[i] for i in range(self.num_modalities)]

    def dispatch(self, x: torch.Tensor) -> list[torch.Tensor]:
        return list(torch.split(x, self.group_size_cpu, dim=0))

    def undispatch(self, *groups: torch.Tensor) -> torch.Tensor:
        return torch.cat(groups, dim=0)

    def permute(self, x: torch.Tensor) -> torch.Tensor:
        return x[self.permute_mapping]


class OuterModel(nn.Module):
    """Uses ModalityDispatcherMockV2 (is_compiling guard).

    Works for good-order inputs on both PT versions, but fails for
    bad-order inputs on PT 2.12 due to symbolic unification.
    """

    def __init__(self, inner_block: nn.Module):
        super().__init__()
        self.block = inner_block

    @torch.compile(dynamic=True, fullgraph=False)
    def forward(self, x: torch.Tensor, modality_mapping: torch.Tensor):
        dispatcher = ModalityDispatcherMockV2(modality_mapping, NUM_MODALITIES)
        x_perm = dispatcher.permute(x)
        out = self.block(x_perm, dispatcher)
        return out


class OuterModelFixed(nn.Module):
    """Uses ModalityDispatcherMockV2Fixed (@torch.compiler.disable).

    Works for both good-order and bad-order inputs on both PT versions.
    """

    def __init__(self, inner_block: nn.Module):
        super().__init__()
        self.block = inner_block

    @torch.compile(dynamic=True, fullgraph=False)
    def forward(self, x: torch.Tensor, modality_mapping: torch.Tensor):
        dispatcher = ModalityDispatcherMockV2Fixed(modality_mapping, NUM_MODALITIES)
        x_perm = dispatcher.permute(x)
        out = self.block(x_perm, dispatcher)
        return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_two_level_compile_cache_reuse_good_order():
    """Two-level compile: first call has all modalities > 0.

    When the initial compilation sees all modalities with tokens, Dynamo
    assigns independent symbols.  Subsequent calls with zero-token
    modalities reuse the cache safely.  This is the easy case — even
    without mark_unbacked it would work, but we test the full production
    pattern (carrier + guard) for consistency.
    """
    torch._dynamo.reset()

    @magi_compile(dynamic_arg_dims={"x": 0})
    class InnerBlock(TransformerBlockMock):
        pass

    inner = InnerBlock(HIDDEN, NUM_MODALITIES).cuda().eval()
    model = OuterModel(inner).cuda().eval()

    shapes = [
        (32, 16, 16),  # all > 0 → first compile gets independent symbols
        (20, 0, 12),  # audio = 0
        (10, 8, 6),  # all > 0
        (0, 20, 12),  # video = 0
        (64, 0, 0),  # only video
    ]

    with torch.no_grad():
        for v, a, t in shapes:
            total = v + a + t
            if total == 0:
                continue
            x = torch.randn(total, HIDDEN, device="cuda", dtype=torch.float32)
            mm = _build_global_modality_mapping(v, a, t)
            out = model(x, mm)
            assert out.shape == (total, HIDDEN), (
                f"Shape mismatch for ({v},{a},{t}): " f"expected ({total}, {HIDDEN}), got {out.shape}"
            )


def _find_inductor_cache_dir() -> str:
    """Locate the inductor cache directory used by the current environment.

    magi_compiler may redirect the cache via MAGI_COMPILE_CACHE_ROOT_DIR;
    otherwise fall back to PyTorch's default (/tmp/torchinductor_<user>).
    """
    from magi_compiler.config import get_compile_config

    magi_dir = os.path.join(get_compile_config().cache_root_dir, "inductor_cache")
    if os.path.isdir(magi_dir):
        return magi_dir
    default_dir = os.path.join("/tmp", f"torchinductor_{os.environ.get('USER', 'root')}")
    if os.path.isdir(default_dir):
        return default_dir
    raise FileNotFoundError(f"Inductor cache not found at {magi_dir} or {default_dir}")


def _collect_py_files(cache_dir: str) -> list[str]:
    py_files = []
    for root, _dirs, files in os.walk(cache_dir):
        for f in files:
            if f.endswith(".py"):
                py_files.append(os.path.join(root, f))
    assert py_files, f"No .py files found in {cache_dir}"
    return py_files


def _check_inductor_cache_has_independent_symbols_u():
    """PT 2.9: verify generated kernels use independent unbacked SymInts
    (u0, u1, u2) rather than a single backed symbol for all modalities."""
    py_files = _collect_py_files(_find_inductor_cache_dir())

    found_independent = False
    for path in py_files:
        with open(path) as fh:
            code = fh.read()
        if "u0" in code and "u1" in code and "u2" in code and "(u0 + u1 + u2)" in code:
            found_independent = True
            break

    assert found_independent, (
        "Inductor cache does not contain independent unbacked SymInts " "(u0, u1, u2).  mark_unbacked may not be working."
    )


_BAD_ORDER_SHAPES = [
    (64, 0, 0),  # only video → first compile, audio=text=0
    (32, 16, 16),  # all > 0 → reuse cache
    (0, 20, 12),  # video = 0 → reuse cache
    (10, 8, 6),  # all > 0
    (20, 0, 12),  # audio = 0
]


def _run_bad_order_test(outer_model_cls):
    """Shared logic for bad-order two-level compile test."""
    torch._dynamo.reset()

    @magi_compile(dynamic_arg_dims={"x": 0})
    class InnerBlock(TransformerBlockMock):
        pass

    inner = InnerBlock(HIDDEN, NUM_MODALITIES).cuda().eval()
    model = outer_model_cls(inner).cuda().eval()

    with torch.no_grad():
        for v, a, t in _BAD_ORDER_SHAPES:
            total = v + a + t
            if total == 0:
                continue
            x = torch.randn(total, HIDDEN, device="cuda", dtype=torch.float32)
            mm = _build_global_modality_mapping(v, a, t)
            out = model(x, mm)
            assert out.shape == (total, HIDDEN), (
                f"Shape mismatch for ({v},{a},{t}): " f"expected ({total}, {HIDDEN}), got {out.shape}"
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(IS_PT_212, reason="PT 2.12 uses s-prefixed symbols; see _pt212 variant")
def test_two_level_compile_cache_reuse_bad_order_pt29():
    """Two-level compile: first call has zero-token modalities (PT 2.9).

    This is the critical case — initial compilation with some modalities = 0
    would cause symbolic over-unification without mark_unbacked.  The carrier
    tensor + is_compiling() guard ensures mark_unbacked runs in eager (after
    tolist() graph break) even when __init__ is called inside @torch.compile.

    After the first compile we also inspect the generated Inductor cache to
    confirm that three independent unbacked SymInts (u0, u1, u2) are used
    instead of a single backed symbol.
    """
    _run_bad_order_test(OuterModel)
    _check_inductor_cache_has_independent_symbols_u()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not IS_PT_212, reason="PT 2.9 uses u-prefixed symbols; see _pt29 variant")
def test_two_level_compile_cache_reuse_bad_order_pt212():
    """Two-level compile: first call has zero-token modalities (PT 2.12).

    Same critical scenario, but requires @torch.compiler.disable() wrapper
    instead of is_compiling() guard.  On PT 2.12, magi_compile's guard
    bypass causes the is_compiling() approach to still unify symbols when
    some modalities have 0 tokens — see test_is_compiling_guard_insufficient_pt212.

    PT 2.12 names unbacked SymInts with high-numbered s-prefixed ids
    (e.g. s66, s69, s98) instead of u0/u1/u2.
    """
    # Runtime correctness: all 5 shape combos pass without AssertionError.
    # Unlike PT 2.9 (where u0/u1/u2 are visible in the inductor cache),
    # magi_compile concretizes the 3 modality sizes before Inductor
    # generates kernels, so cache symbol inspection is not applicable.
    _run_bad_order_test(OuterModelFixed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not IS_PT_212, reason="is_compiling() guard works on PT 2.9; see _pt29 variant")
def test_is_compiling_guard_insufficient_pt212():
    """PT 2.12: is_compiling() guard causes symbolic unification in bad-order.

    With magi_compile's guard bypass + two-level compile, the is_compiling()
    guard approach (ModalityDispatcherMockV2) fails to prevent symbolic
    unification when the first call has zero-token modalities.  The compiled
    graph reuses a shape that was unified (e.g. total == video), causing an
    assert_size_stride failure on subsequent calls with different shapes.

    This demonstrates why @torch.compiler.disable() (ModalityDispatcherMockV2Fixed)
    is necessary on PT 2.12.
    """
    torch._dynamo.reset()

    @magi_compile(dynamic_arg_dims={"x": 0})
    class InnerBlock(TransformerBlockMock):
        pass

    inner = InnerBlock(HIDDEN, NUM_MODALITIES).cuda().eval()
    model = OuterModel(inner).cuda().eval()

    with pytest.raises((AssertionError, RuntimeError)):
        with torch.no_grad():
            for v, a, t in _BAD_ORDER_SHAPES:
                total = v + a + t
                if total == 0:
                    continue
                x = torch.randn(total, HIDDEN, device="cuda", dtype=torch.float32)
                mm = _build_global_modality_mapping(v, a, t)
                model(x, mm)


class ModalityDispatcherMockNoGuard:
    """Dispatcher with mark_unbacked but WITHOUT is_compiling() guard.

    Demonstrates that removing the guard causes 'forbidden callable' error
    when __init__ is called inside @torch.compile."""

    def __init__(self, modality_mapping: torch.Tensor, num_modalities: int):
        self.num_modalities = num_modalities
        self.permute_mapping = torch.argsort(modality_mapping)
        permuted = modality_mapping[self.permute_mapping]
        group_sizes = torch.bincount(permuted, minlength=num_modalities).tolist()

        self._size_carrier = torch.empty(*[int(s) for s in group_sizes])
        for i in range(num_modalities):
            torch._dynamo.decorators.mark_unbacked(self._size_carrier, i)

    @property
    def group_size_cpu(self) -> list[int]:
        return [self._size_carrier.shape[i] for i in range(self.num_modalities)]

    def dispatch(self, x: torch.Tensor) -> list[torch.Tensor]:
        return list(torch.split(x, self.group_size_cpu, dim=0))

    def undispatch(self, *groups: torch.Tensor) -> torch.Tensor:
        return torch.cat(groups, dim=0)

    def permute(self, x: torch.Tensor) -> torch.Tensor:
        return x[self.permute_mapping]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_two_level_compile_no_guard_raises():
    """Without is_compiling() guard, mark_unbacked triggers 'forbidden callable'.

    Dynamo's resume frame after tolist() graph break still traces the code;
    mark_unbacked is registered via _disallow_in_graph, so even referencing
    it during tracing is an error.  The is_compiling() guard is required.
    """
    torch._dynamo.reset()

    @magi_compile(dynamic_arg_dims={"x": 0})
    class InnerBlock(TransformerBlockMock):
        pass

    class OuterModelNoGuard(nn.Module):
        def __init__(self, inner_block: nn.Module):
            super().__init__()
            self.block = inner_block

        @torch.compile(dynamic=True, fullgraph=False)
        def forward(self, x: torch.Tensor, modality_mapping: torch.Tensor):
            dispatcher = ModalityDispatcherMockNoGuard(modality_mapping, NUM_MODALITIES)
            x_perm = dispatcher.permute(x)
            return self.block(x_perm, dispatcher)

    inner = InnerBlock(HIDDEN, NUM_MODALITIES).cuda().eval()
    model = OuterModelNoGuard(inner).cuda().eval()

    x = torch.randn(64, HIDDEN, device="cuda", dtype=torch.float32)
    mm = _build_global_modality_mapping(32, 16, 16)

    with pytest.raises(AssertionError, match="forbidden callable"):
        with torch.no_grad():
            model(x, mm)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
