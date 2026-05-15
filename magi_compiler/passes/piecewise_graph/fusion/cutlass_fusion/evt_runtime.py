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

"""Runtime side of the EVT fusion: torch.library op + JIT loader + dispatch.

This file owns:
  * The ``magi_epilogue::matmul_custom_evt`` torch.library op + fake impl.
  * A process-level cache mapping IR JSON → compiled cpp_extension module.
  * Dispatch to one of two backends:
      - ``kind == "evt"``         → JIT-compiled CUTLASS Sm80EVT kernel.
      - ``kind == "swiglu7_dual"`` → vendored DualGemm one-stage kernel.
        Routes to the SM80 cp.async multistage path on sm_120 (RTX 5090) and
        to the SM90 TMA + WGMMA path on sm_90 (H100). Both expose the same
        ``swiglu7_dual_matmul_out(A, B, D)`` PYBIND callable, so the
        dispatcher is arch-agnostic.

The kernel build directory uses the IR cache key + arch tag as its name so
re-runs and multi-process Inductor compile workers all hit the same on-disk
cache, and so a binary built for one arch never gets reused on another.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Optional

import torch

from magi_compiler.config import get_compile_config

from .evt_ir import Accum, AuxLoad, ColBroadcast, Compute, RowBroadcast, Store
from .sm80.evt_codegen import render_evt_cu as _render_evt_cu_sm80
from .sm90.evt_codegen import render_evt_cu as _render_evt_cu_sm90

# ── torch.library op definition ───────────────────────────────────────────────
# Reuse the existing ``magi_epilogue`` library so all our custom matmul ops
# live under one namespace. Defining a fresh op here is harmless even if
# ``matmul_epilogue_fusion.py`` has already initialised the library.
_LIB = torch.library.Library("magi_epilogue", "FRAGMENT")
_LIB.define(
    "matmul_custom_evt(Tensor A, Tensor B, Tensor[] extras, str ir_json," " str kind, int n_out, int out_dtype_id) -> Tensor"
)


# ── Output-dtype encoding (must round-trip through torch.library int args) ────
_OUT_DTYPE_ID = {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2}
_ID_TO_DTYPE = {v: k for k, v in _OUT_DTYPE_ID.items()}
_DTYPE_TO_STR = {torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32"}


def out_dtype_id(dt: torch.dtype) -> int:
    """Encode a torch.dtype as a small int for inclusion in op args."""
    if dt not in _OUT_DTYPE_ID:
        raise ValueError(f"Unsupported EVT output dtype {dt}")
    return _OUT_DTYPE_ID[dt]


def out_dtype_from_id(i: int) -> torch.dtype:
    return _ID_TO_DTYPE[i]


# ── Greedy AlignmentA / AlignmentB picker (matches FX-side gate) ────────────
# CUTLASS only requires the leading dim divides AlignmentX. We pick the
# largest power-of-2 in (128, 64) bits that fits the actual K (or N), giving
# us 128-bit vector loads when shapes allow but admitting 64-bit-aligned
# shapes (e.g. K = 12 for bf16 → 4 elems, 64 bits) that the strict 128-bit
# gate previously rejected. The FX pass admits the fusion any time at least
# 64 bits fits; the runtime then picks the actual width per call (cache-keyed
# on (N, K) so each shape gets its own compiled kernel).
_GREEDY_ALIGN_BITS_RT = (128, 64)


def _runtime_align_bits(dim: int, dtype: torch.dtype) -> int:
    n_int = int(dim)
    for bits in _GREEDY_ALIGN_BITS_RT:
        align_elems = max(1, bits // (dtype.itemsize * 8))
        if n_int % align_elems == 0:
            return bits
    raise ValueError(f"dim={n_int} not even {_GREEDY_ALIGN_BITS_RT[-1]}-bit-aligned for dtype={dtype}")


def _aligned_n_stride(n_out: int, dtype: torch.dtype) -> int:
    """Round n_out up to a 128-byte (one L2 cache line) element count.

    The CUTLASS-side requirement is only ``ldd % AlignmentC == 0`` where
    ``AlignmentC = 128 / sizeof_bits<ElementC>`` (= 8 elements for bf16),
    i.e. a 16-byte boundary. We over-align here to 128 bytes — a full L2
    cache line — for two reasons:

      1. Every row starts on a cache-line boundary, so the contiguous block
         of cp.async / ld.global issued by the next op (typically a cuBLAS
         GEMM that consumes our strided D) sees clean cache-line packing.
      2. cuBLAS's GEMM heuristic picks a different (and on RTX 5090 measurably
         slower) kernel for "awkward" lda values that are not 128-byte
         multiples. Bumping the pad from one vector store (16 B) to one
         cache line (128 B) costs at most 63 extra elements per row — under
         a hundred KB even at large M — and recovers the cuBLAS kernel
         heuristic's first-class path.

    Bytes-based formula keeps this dtype-agnostic:
      bf16 / fp16 → 64 element pad boundary
      fp32        → 32 element pad boundary
      fp8         → 128 element pad boundary
    """
    align_bytes = 128
    align = max(1, align_bytes // dtype.itemsize)
    n = int(n_out)
    return ((n + align - 1) // align) * align


# ── Compile cache + per-key build lock ────────────────────────────────────────
_MODULE_CACHE: dict = {}  # cache_key (sha256 str) → loaded cpp_extension module
# Hot-path fast cache — avoids ``json.dumps + sha256`` (~10–30 μs/call) when
# the module has already been compiled. Keyed by the 4-tuple of (Python-)
# hashable inputs that uniquely determine the rendered .cu, since equality on
# the tuple is sufficient (no need to canonicalise twice). Populated on the
# slow path inside ``_compile_evt_module``.
_MODULE_FAST_CACHE: dict = {}  # (ir_json, a_dtype, b_dtype, b_layout) → module
_MODULE_LOCKS: dict = {}  # cache_key → threading.Lock
_MODULE_LOCKS_GLOBAL = threading.Lock()
_SWIGLU7_LOCK = threading.Lock()  # serialises insertions into _SWIGLU7_FAST_CACHE


# ── D output-buffer cache ────────────────────────────────────────────────────
# Single-entry greedy cache, keyed by (M, n_pad, dtype, device_idx). The hot
# path in ``_matmul_custom_evt_cuda`` reads/writes this dict directly (the
# resolver was inlined for ~1 μs/call savings), so this module only owns the
# storage and a disable switch.
#
# Every D allocation is sized ``(M, n_pad)`` where
# ``n_pad = _aligned_n_stride(n_out, dtype)`` rounds n_out up to a full L2
# cache line (128 B) — over-aligned vs. CUTLASS's vector-store requirement
# of one 16 B boundary, so that downstream cuBLAS GEMMs that consume our
# strided D land on the heuristic's first-class kernel. The op returns the
# strided view ``D_pad[:, :n_out]`` (stride(0) == n_pad, stride(1) == 1) so
# downstream Inductor sees a (M, n_out) tensor whose row stride is the
# padded one. Two distinct n_out values that round to the same n_pad share
# the same buffer.
#
# To opt out (e.g. when bench-scripting with overlapping streams), set the
# env var ``MAGI_EVT_DISABLE_D_CACHE=1``.
_D_BUF_CACHE: dict = {}
_D_CACHE_DISABLED: bool = os.environ.get("MAGI_EVT_DISABLE_D_CACHE", "0") not in ("0", "", "false", "False")


def _cutlass_root() -> str:
    # Default install location is /opt/cutlass (Dockerfile clones the source
    # tree there). Override with MAGI_CUTLASS_ROOT for ad-hoc dev checkouts.
    return os.environ.get("MAGI_CUTLASS_ROOT", "/opt/cutlass")


def _device_gencode_flags() -> list[str]:
    """Return nvcc -gencode flags matching the current CUDA device.

    Hardcoding ``sm_120`` (Blackwell GeForce) breaks any other arch — the
    nvcc output has no compatible SASS, kernel launch returns
    ``cudaErrorInvalidDeviceFunction``, and CUTLASS surfaces it as
    ``Status::kErrorInternal``. Detect the live device's compute capability
    and emit a matching gencode plus a forward-compat PTX so future arches
    can JIT.

    Special case: sm_90 must use the ``a`` (architecture-specific) feature
    variant because all WGMMA / TMA kernels in CUTLASS 3.x are gated on it.
    Plain ``sm_90`` exists in the toolchain but lacks WGMMA support, so any
    Hopper-native kernel we ship would fail to compile against it.

    Override with ``MAGI_EVT_GENCODE`` (semicolon-separated nvcc args) for
    ad-hoc multi-arch builds.
    """
    override = os.environ.get("MAGI_EVT_GENCODE")
    if override:
        return [a for a in override.split(";") if a]
    cap = torch.cuda.get_device_capability()
    arch = f"{cap[0]}{cap[1]}"  # "90" for H100, "120" for RTX 5090, "80" for A100
    # Use the wgmma-enabled "a" variant on Hopper; all other arches stay plain.
    arch_for_code = f"{arch}a" if arch == "90" else arch
    return [
        f"-gencode=arch=compute_{arch_for_code},code=sm_{arch_for_code}",
        # Embed PTX of the same arch so a slightly newer driver / different
        # minor revision JITs cleanly without rebuilding.
        f"-gencode=arch=compute_{arch_for_code},code=compute_{arch_for_code}",
    ]


def _device_arch_tag() -> str:
    """Short tag for the live device's compute capability (e.g. ``sm90``).

    Folded into build_dir / module name so binaries compiled for a different
    arch (e.g. running the same source tree on an H100 after using it on a
    Blackwell box) don't get reused.
    """
    cap = torch.cuda.get_device_capability()
    return f"sm{cap[0]}{cap[1]}"


def _evt_build_dir(key: str) -> str:
    cache_root = get_compile_config().cache_root_dir
    return os.path.join(cache_root, "evt_kernels", _device_arch_tag(), key)


def _per_key_lock(key: str) -> threading.Lock:
    """Return the per-key build lock; coalesces concurrent compile requests."""
    with _MODULE_LOCKS_GLOBAL:
        lock = _MODULE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _MODULE_LOCKS[key] = lock
        return lock


def _compile_evt_module(
    ir_json: str,
    a_dtype: torch.dtype,
    b_dtype: torch.dtype,
    b_layout: str = "row",
    m_bucket: str = "medium",
    N: int = 0,
    K: int = 0,
    alignment_a_bits: int = 128,
    alignment_b_bits: int = 128,
    alignment_c_bits: int = 128,
):
    """Render + JIT-compile the EVT kernel for ``ir_json``. Process-level cached.

    Cache key: (IR, A dtype, B dtype, b_layout, m_bucket, N, K, alignA, alignB,
    alignC, arch). Each distinct weight (N, K) lowers to its own .cu — even
    though the .cu source is identical (N/K stay runtime variables), splitting
    the modules gives every (N, K) its own runner instance with isolated
    `best_idx_`. ``alignment_*_bits`` are derived from runtime K (A), N or K
    (B), and ldd (C) via greedy 128 → 64 bit selection and baked into the
    rendered .cu via constexpr; including them in the key keeps two shapes
    that pick different alignments from sharing a .so.
    """
    # arch determines which per-bucket tile candidate set the codegen inlines.
    # Different arches must lower to different .cu files, so it goes into both
    # the fast key and the SHA key.
    arch = _device_arch_tag()

    # Hot-path fast cache: skip ``json.dumps + sha256`` (~10–30 μs each) on
    # subsequent calls with the same inputs.
    fast_key = (
        ir_json,
        a_dtype,
        b_dtype,
        b_layout,
        m_bucket,
        N,
        K,
        alignment_a_bits,
        alignment_b_bits,
        alignment_c_bits,
        arch,
    )
    cached = _MODULE_FAST_CACHE.get(fast_key)
    if cached is not None:
        return cached

    if b_layout not in ("row", "col"):
        raise ValueError(f"b_layout must be 'row' or 'col', got {b_layout!r}")
    a_str = _DTYPE_TO_STR[a_dtype]
    b_str = _DTYPE_TO_STR[b_dtype]
    extended = json.dumps(
        {
            "ir": ir_json,
            "a": a_str,
            "b": b_str,
            "b_layout": b_layout,
            "m_bucket": m_bucket,
            "N": int(N),
            "K": int(K),
            "alignA_bits": int(alignment_a_bits),
            "alignB_bits": int(alignment_b_bits),
            "alignC_bits": int(alignment_c_bits),
            "arch": arch,
            "version": 6,
        },
        sort_keys=True,
    ).encode("utf-8")
    key = hashlib.sha256(extended).hexdigest()

    cached = _MODULE_CACHE.get(key)
    if cached is not None:
        _MODULE_FAST_CACHE[fast_key] = cached
        return cached

    lock = _per_key_lock(key)
    with lock:
        cached = _MODULE_CACHE.get(key)
        if cached is not None:
            _MODULE_FAST_CACHE[fast_key] = cached
            return cached

        # Re-hydrate the IR tree from JSON for codegen. Pick renderer per arch:
        # sm_90 → CUTLASS 3.x Sm90EVT (TMA + WGMMA, ~1.6-2× faster on H100);
        # everything else → CUTLASS 2.x Sm80EVT (cp.async, runs on sm_80 / Ada
        # / Blackwell GeForce). Both renderers expose the same `evt_matmul_out`
        # PYBIND function so the dispatcher attribute lookup is uniform.
        ir = _ir_from_json(ir_json)
        render_fn = _render_evt_cu_sm90 if arch == "sm90" else _render_evt_cu_sm80
        src = render_fn(
            ir,
            a_str,
            b_str,
            cache_key_str=key,
            b_layout=b_layout,
            m_bucket=m_bucket,
            alignment_a_bits=alignment_a_bits,
            alignment_b_bits=alignment_b_bits,
            alignment_c_bits=alignment_c_bits,
            arch=arch,
        )

        build_dir = _evt_build_dir(key)
        os.makedirs(build_dir, exist_ok=True)
        src_path = os.path.join(build_dir, "evt.cu")
        # Write atomically (tmp + rename) so concurrent processes don't see a
        # half-written file. Use a process-specific tmp name to avoid races
        # across multiple rank processes generating the same kernel.
        tmp_path = f"{src_path}.{os.getpid()}.tmp"
        with open(tmp_path, "w") as f:
            f.write(src)
        os.replace(tmp_path, src_path)

        cutlass_root = _cutlass_root()
        from torch.utils.cpp_extension import load

        # SM90 EVT (CUTLASS 3.x) needs extra cflags for warp-specialized
        # collectives + extended MMA shape selection. SM80 EVT doesn't need
        # them and accepting them on sm_80 / sm_120 / sm_120 builds is also
        # harmless, but we only pass them on sm_90 to keep the build minimal.
        sm90_specific_cflags = (
            ["--expt-extended-lambda", "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED=1"] if arch == "sm90" else []
        )

        # cpp_extension.load uses its own file lock under build_directory, so
        # multi-process races resolve to a single nvcc invocation.
        module = load(
            name=f"magi_evt_{key[:12]}",
            sources=[src_path],
            extra_include_paths=[
                os.path.join(cutlass_root, "include"),
                os.path.join(cutlass_root, "tools", "util", "include"),
            ],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=(
                ["-std=c++17", "-O3", "--expt-relaxed-constexpr"] + sm90_specific_cflags + _device_gencode_flags()
            ),
            build_directory=build_dir,
            verbose=False,
        )
        _MODULE_CACHE[key] = module
        _MODULE_FAST_CACHE[fast_key] = module
        return module


# ── IR (de)serialisation ─────────────────────────────────────────────────────


def to_ir_json(node) -> str:
    from .evt_ir import to_canonical_json

    return to_canonical_json(node)


def _ir_from_json(s: str):
    """Inverse of ``to_canonical_json``. Used only to drive codegen at compile
    time — the FX pass holds the original Python objects and never round-trips
    its own IR through JSON in a hot loop."""
    d = json.loads(s)
    return _node_from_dict(d)


def _node_from_dict(d):
    kind = d["kind"]
    if kind == "accum":
        return Accum()
    if kind == "row_bcast":
        return RowBroadcast(input_idx=d["input_idx"], dtype=d["dtype"])
    if kind == "col_bcast":
        return ColBroadcast(input_idx=d["input_idx"], dtype=d["dtype"])
    if kind == "aux_load":
        return AuxLoad(input_idx=d["input_idx"], dtype=d["dtype"])
    if kind == "compute":
        scalar = d.get("scalar")
        scalar_val: Optional[float] = float(scalar) if scalar is not None else None
        return Compute(op=d["op"], children=tuple(_node_from_dict(c) for c in d["children"]), scalar=scalar_val)
    if kind == "store":
        return Store(child=_node_from_dict(d["child"]), out_dtype=d["out_dtype"])
    raise ValueError(f"Unknown IR kind {kind!r}")


# ── swiglu7 dual-gemm extension loader ────────────────────────────────────────
# Per-(m_bucket, N, K) cache. The .cu source is identical across keys (N/K stay
# runtime variables); we still build separate modules so each runner instance
# hosts exactly one (N, K), giving every weight shape its own isolated
# best_idx_. Two distinct (N, K) × two M-buckets ⇒ 4 modules.
_SWIGLU7_FAST_CACHE: dict = {}  # (m_bucket, N, K) → loaded module
_SWIGLU7_BUILD_LOCKS: dict = {}  # (m_bucket, N, K) → threading.Lock


def _compile_swiglu7_dual(
    m_bucket: str, N: int, K: int, alignment_a_bits: int = 128, alignment_b_bits: int = 128, alignment_c_bits: int = 128
):
    """Lazy-load a per-(bucket, N, K, align) instance of the vendored DualGemm kernel.

    Parameters
    ----------
    m_bucket : "small" | "medium" | "large"
        Bucket of the activation M dim — included in the cache key so e.g.
        small-M (decode) can autotune to a different best tile than large-M
        (prefill) for the same (N, K).
    N, K : int
        Static weight shape from B (the underlying (N, K) row-major tensor).
        Distinct (N, K) get distinct modules so their autotune state is
        independent.
    alignment_a_bits, alignment_b_bits, alignment_c_bits : int
        Alignment width baked into the .cu via -DMAGI_SWIGLU7_ALIGN_*_BITS at
        nvcc time. Greedy-picked from the actual K (A/B) and ldd (C):
        128 → 64 bits. K-aligned shapes get vectorised loads, K = 12-style
        shapes still fuse at 64. ``alignment_c_bits`` gates the epilogue
        store width (``EpilogueVecCount``); host padding normally satisfies
        128 but the parameter is exposed for parity with A/B.
        Distinct widths get distinct .so files since the change is at
        constexpr level and recompilation is the only way to thread it
        through the DualGemm template.
    """
    fast_key = (m_bucket, int(N), int(K), int(alignment_a_bits), int(alignment_b_bits), int(alignment_c_bits))
    cached = _SWIGLU7_FAST_CACHE.get(fast_key)
    if cached is not None:
        return cached

    with _SWIGLU7_LOCK:
        lock = _SWIGLU7_BUILD_LOCKS.get(fast_key)
        if lock is None:
            lock = threading.Lock()
            _SWIGLU7_BUILD_LOCKS[fast_key] = lock
    with lock:
        cached = _SWIGLU7_FAST_CACHE.get(fast_key)
        if cached is not None:
            return cached

        cutlass_root = _cutlass_root()
        here = os.path.dirname(os.path.abspath(__file__))
        # Pick the .cu source per device arch. sm_90 (Hopper / H100) gets the
        # native TMA + WGMMA implementation built on the vendored Sm90DualGemm
        # under sm90/cutlass_kernels/49_hopper_dual_gemm/. Everything else
        # (sm_120 Blackwell GeForce, Ada, Ampere…) falls back to the SM80
        # multistage path under sm80/cutlass_kernels/.
        arch_tag = _device_arch_tag()
        arch_subdir = "sm90" if arch_tag == "sm90" else "sm80"
        src = os.path.join(here, arch_subdir, "cutlass_kernels", "swiglu7_one_stage.cu")
        if not os.path.exists(src):
            raise FileNotFoundError(f"vendored swiglu7 source not found: {src}")
        cache_root = get_compile_config().cache_root_dir
        # Build dir embeds (arch, bucket, N, K, align) so distinct keys get
        # their own build artefacts. cpp_extension uses the dir as the cache
        # identity, and a stale binary from a different arch must NOT be
        # reused (CUDA driver would refuse to load and CUTLASS surfaces it
        # as Status::kErrorInternal).
        build_tag = f"{m_bucket}_N{N}_K{K}" f"_aA{alignment_a_bits}_aB{alignment_b_bits}_aC{alignment_c_bits}"
        build_dir = os.path.join(cache_root, "evt_kernels", arch_tag, f"swiglu7_dual_{build_tag}")
        os.makedirs(build_dir, exist_ok=True)
        from torch.utils.cpp_extension import load

        # SM90 path needs CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED for the WGMMA
        # tile selector and --expt-extended-lambda for the warp-specialized
        # collective. Other arches don't need (or accept) these, so they're
        # only added on the Hopper build.
        sm90_specific_cflags = (
            ["--expt-extended-lambda", "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED=1"] if arch_tag == "sm90" else []
        )

        # Both .cu files do `#include "swiglu7_combine.h"` (arch-agnostic
        # math). Lives under common/cutlass_kernels/ so a single -I covers
        # both arch builds. The sm_90 .cu additionally does
        # `#include "49_hopper_dual_gemm/device/sm90_dual_gemm.h"`, resolved
        # by sm90/cutlass_kernels/.
        sm90_include_paths = [os.path.join(here, "sm90", "cutlass_kernels")] if arch_tag == "sm90" else []

        module = load(
            name=f"magi_swiglu7_dual_{build_tag}",
            sources=[src],
            extra_include_paths=[
                os.path.join(cutlass_root, "include"),
                os.path.join(cutlass_root, "tools", "util", "include"),
                os.path.join(cutlass_root, "examples"),
                os.path.join(here, "common", "cutlass_kernels"),
                *sm90_include_paths,
            ],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=[
                "-std=c++17",
                "-O3",
                "--expt-relaxed-constexpr",
                *sm90_specific_cflags,
                *_device_gencode_flags(),
                f"-DMAGI_SWIGLU7_ALIGN_A_BITS={int(alignment_a_bits)}",
                f"-DMAGI_SWIGLU7_ALIGN_B_BITS={int(alignment_b_bits)}",
                f"-DMAGI_SWIGLU7_ALIGN_C_BITS={int(alignment_c_bits)}",
            ],
            build_directory=build_dir,
            verbose=False,
        )
        _SWIGLU7_FAST_CACHE[fast_key] = module
        return module


# ── torch.library backend impls ───────────────────────────────────────────────


# ── Dispatch fast-cache ──────────────────────────────────────────────────────
# Hot-path bottleneck reduction: collapse the four-step
#   out_dtype_from_id → _m_bucket → _compile_* → mod.attr-lookup
# chain into a single dict.get() returning a pre-bound callable plus the
# small amount of immutable metadata the kernel-launch site needs.
#
# Key shape: (kind, ir_json, A.dtype, B.dtype, N, K, m_bucket, out_dtype).
# Most of these are static per FX-emit site (kind / ir_json / dtypes / N / K)
# — only m_bucket varies with M. So the cache reaches steady state after the
# first time each (site, bucket) is seen.
#
# Each entry holds:
#   * kernel_call : pre-bound mod.evt_matmul_out / swiglu7_dual_matmul_out
#   * is_evt      : True for evt_row/evt_col (need extras list), False for swiglu7
#   * out_dtype   : torch.dtype to pass to D allocation
class _DispatchEntry:
    __slots__ = ("kernel_call", "is_evt", "out_dtype")

    def __init__(self, kernel_call, is_evt, out_dtype):
        self.kernel_call = kernel_call
        self.is_evt = is_evt
        self.out_dtype = out_dtype


_DISPATCH_CACHE: dict = {}


def _resolve_dispatch(kind, ir_json, a_dtype, b_dtype, N_w, K_w, m_bucket, out_dtype):
    """Slow-path resolver — compiles the .cu module (cache miss) and binds
    the kernel callable. Cached by (kind, ir_json, A_dt, B_dt, N, K, bucket,
    out_dtype) so each FX site × bucket only pays this once.

    AlignmentC is derived from the host-padded ldd that the runtime will pass
    to CUTLASS. Under the current ``_aligned_n_stride`` (128-byte / cache-line
    pad), n_pad is always a multiple of 8 bf16 elements ⇒ 128-bit AlignmentC
    is always picked. The greedy fallback to 64 is wired for parity with A/B
    so a future smaller-pad mode can drop without a code change here.
    """
    # n_out used by CUTLASS LayoutC = the kernel's logical output cols.
    # evt_row / evt_col output shape is (M, N); swiglu7 outputs (M, N/2).
    n_out_for_c = (N_w // 2) if kind == "swiglu7_dual" else N_w
    ldd = _aligned_n_stride(n_out_for_c, out_dtype)
    alignment_c_bits = _runtime_align_bits(ldd, out_dtype)

    if kind == "swiglu7_dual":
        # swiglu7 reads A's K and B's strided ldB = 2K. Both leading dims are
        # multiples of K, so the alignment that fits K also fits 2K — deriving
        # from K alone is sufficient. dtype is bf16 on both sides (FX gate).
        align_bits = _runtime_align_bits(K_w, a_dtype)
        mod = _compile_swiglu7_dual(
            m_bucket, N_w, K_w, alignment_a_bits=align_bits, alignment_b_bits=align_bits, alignment_c_bits=alignment_c_bits
        )
        return _DispatchEntry(mod.swiglu7_dual_matmul_out, False, out_dtype)
    if kind == "evt_row" or kind == "evt":
        b_layout = "row"
    elif kind == "evt_col":
        b_layout = "col"
    else:
        raise ValueError(f"Unknown EVT kind {kind!r}")
    # Greedy-pick AlignmentA / AlignmentB from actual K and the layout-relevant
    # B leading dim (N for row, K for col). Falls back from 128 → 64 bits when
    # 128-bit isn't divisible. The FX gate has already proven at least 64 bits
    # fits, so this can't raise here in practice.
    alignment_a_bits = _runtime_align_bits(K_w, a_dtype)
    b_lead_dim = N_w if b_layout == "row" else K_w
    alignment_b_bits = _runtime_align_bits(b_lead_dim, b_dtype)
    mod = _compile_evt_module(
        ir_json,
        a_dtype,
        b_dtype,
        b_layout=b_layout,
        m_bucket=m_bucket,
        N=N_w,
        K=K_w,
        alignment_a_bits=alignment_a_bits,
        alignment_b_bits=alignment_b_bits,
        alignment_c_bits=alignment_c_bits,
    )
    return _DispatchEntry(mod.evt_matmul_out, True, out_dtype)


@torch.library.impl(_LIB, "matmul_custom_evt", "CUDA")
def _matmul_custom_evt_cuda(A, B, extras, ir_json, kind, n_out, out_dtype_id_):
    """Runtime entry point for the EVT-fused matmul op.

    Hot path is heavily inlined to keep per-call Python overhead under ~2 μs:
    one dict.get() resolves the kernel callable + metadata, then we allocate D
    (with a single-entry greedy cache) and call straight into the C++ kernel.

    Layout contract — the FX pass owns this; do not rewrite operands here:
      * ``kind == "evt_row"`` : B is contiguous (K, N) row-major.
      * ``kind == "evt_col"`` : B is the underlying (N, K) row-major weight; the
        kernel was rendered with ``LayoutB = ColumnMajor`` so it reads (K, N)
        from the same bytes via stride (1, K).
      * ``kind == "swiglu7_dual"`` : B is the underlying (N, K) row-major weight
        (the FX pass already replaced the ``permute([1,0])`` view with its
        operand). The DualGemm kernel reads it as ColumnMajor + ldB=2K.

    Calling ``.contiguous()`` on B here would silently break the col / swiglu7
    paths by materialising a (K, N) row-major copy that no longer matches the
    LayoutB the kernel was compiled with — every B value would be wrong.
    """
    # ── Step 1: resolve dispatch entry (one dict lookup on the fast path) ──
    # B.size(0)/size(1) are slightly faster than .shape[0]/[1] (avoid Python
    # tuple construction). For all 3 kinds B's leading dim ≠ K — the launcher
    # / runner derives N internally from b_layout, but for the dispatch cache
    # key we just need a stable per-site discriminator, so passing the raw
    # B.size pair is enough.
    B_size0 = B.size(0)
    B_size1 = B.size(1)
    M = A.size(0)
    # Inline _m_bucket: avoid the ~300 ns function call.
    if M <= 256:
        m_bucket = "small"
    elif M <= 2048:
        m_bucket = "medium"
    else:
        m_bucket = "large"
    # Inline out_dtype_from_id: skip the function call frame.
    out_dtype = _ID_TO_DTYPE[out_dtype_id_]
    # B's (N, K) interpretation depends on kind. For evt_row B is (K, N),
    # for evt_col / swiglu7_dual B is the underlying (N, K). Either way we
    # only need (B_size0, B_size1) to disambiguate distinct weights — the
    # resolver re-computes N/K correctly for compilation.
    a_dtype = A.dtype
    b_dtype_ = B.dtype
    fast_key = (kind, ir_json, a_dtype, b_dtype_, B_size0, B_size1, m_bucket, out_dtype)
    entry = _DISPATCH_CACHE.get(fast_key)
    if entry is None:
        # Map B sizes to (N_w, K_w) in the layout the compile path expects.
        if kind == "evt_row":
            K_w, N_w = B_size0, B_size1
        else:
            # evt_col / swiglu7_dual: B is (N, K) underlying weight.
            N_w, K_w = B_size0, B_size1
        entry = _resolve_dispatch(kind, ir_json, a_dtype, b_dtype_, N_w, K_w, m_bucket, out_dtype)
        _DISPATCH_CACHE[fast_key] = entry

    # ── Step 2: alloc / fetch padded D (greedy single-entry cache, inlined) ──
    # Allocate D padded to AlignmentC element boundaries on the row stride.
    # The CUTLASS kernel only writes the first n_out columns; the rest of
    # each padded row is left untouched. The slice D_pad[:, :n_out] is what
    # we hand to the kernel and what we return — a strided view whose
    # stride(0) == n_pad. Cache key is on n_pad (not n_out) since that's the
    # actual buffer size; two n_out values that pad to the same n_pad share.
    n_pad = _aligned_n_stride(n_out, out_dtype)
    if _D_CACHE_DISABLED:
        D_pad = torch.empty((M, n_pad), device=A.device, dtype=out_dtype)
    else:
        dev_idx = A.device.index or 0
        d_key = (M, n_pad, out_dtype, dev_idx)
        D_pad = _D_BUF_CACHE.get(d_key)
        if D_pad is None:
            D_pad = torch.empty((M, n_pad), device=A.device, dtype=out_dtype)
            _D_BUF_CACHE.clear()
            _D_BUF_CACHE[d_key] = D_pad
    D = D_pad[:, :n_out] if n_pad != n_out else D_pad

    # ── Step 3: dispatch — pre-bound callable, single C++ trampoline ──
    kernel_call = entry.kernel_call
    if entry.is_evt:
        kernel_call(A, B, extras, D)
    else:
        # swiglu7_dual: extras is always [] here (FX pass guarantees).
        kernel_call(A, B, D)
    return D


@torch.library.register_fake("magi_epilogue::matmul_custom_evt")
def _matmul_custom_evt_fake(A, B, extras, ir_json, kind, n_out, out_dtype_id_):
    out_dtype = out_dtype_from_id(out_dtype_id_)
    # Strided (M, n_out) view of an (M, n_pad) buffer — must match the
    # stride layout the CUDA impl actually returns, otherwise Inductor's
    # downstream view metadata desyncs from the real tensor.
    n_pad = _aligned_n_stride(n_out, out_dtype)
    return A.new_empty_strided((A.shape[0], n_out), (n_pad, 1), dtype=out_dtype)
