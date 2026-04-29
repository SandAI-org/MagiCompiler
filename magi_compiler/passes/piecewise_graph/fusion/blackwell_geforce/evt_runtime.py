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

The kernel build directory uses the IR cache key as its name so re-runs and
multi-process Inductor compile workers all hit the same on-disk cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Optional

import torch

from magi_compiler.config import get_compile_config

from .evt_codegen import render_evt_cu
from .evt_ir import Accum, AuxLoad, ColBroadcast, Compute, RowBroadcast, Store

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
# Single-entry greedy cache, keyed by (M, n_out, dtype, device_idx). The hot
# path in ``_matmul_custom_evt_cuda`` reads/writes this dict directly (the
# resolver was inlined for ~1 μs/call savings), so this module only owns the
# storage and a disable switch.
#
# FX-pass guards (K % 8 == 0; generic N % 4 == 0; swiglu7 N % 8 == 0) ensure
# n_out is always a multiple of CUTLASS's AlignmentC = 4 elements, so D is
# always allocated as a true-contiguous ``torch.empty((M, n_out), dtype)`` —
# no padded stride / scratch buffer route exists. Anything that violates the
# guards is rejected upstream and falls back to torch.compile's default mm.
#
# To opt out (e.g. when bench-scripting with overlapping streams), set the
# env var ``MAGI_EVT_DISABLE_D_CACHE=1``.
_D_BUF_CACHE: dict = {}
_D_CACHE_DISABLED: bool = os.environ.get("MAGI_EVT_DISABLE_D_CACHE", "0") not in ("0", "", "false", "False")


def _cutlass_root() -> str:
    # Default install location is /opt/cutlass (Dockerfile clones the source
    # tree there). Override with MAGI_CUTLASS_ROOT for ad-hoc dev checkouts.
    return os.environ.get("MAGI_CUTLASS_ROOT", "/opt/cutlass")


def _evt_build_dir(key: str) -> str:
    cache_root = get_compile_config().cache_root_dir
    return os.path.join(cache_root, "evt_kernels", key)


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
):
    """Render + JIT-compile the EVT kernel for ``ir_json``. Process-level cached.

    Cache key: (IR, A dtype, B dtype, b_layout, m_bucket, N, K). Each distinct
    weight (N, K) lowers to its own .cu — even though the .cu source is
    identical (N/K stay runtime variables), splitting the modules gives every
    (N, K) its own runner instance with isolated `best_idx_`. This avoids
    cross-(N, K) autotune contamination and matches the user's per-(N, K)
    cache layout: e.g. two distinct (N, K) × two M-buckets ⇒ 4 .cu modules.
    """
    # Hot-path fast cache: skip ``json.dumps + sha256`` (~10–30 μs each) on
    # subsequent calls with the same inputs.
    fast_key = (ir_json, a_dtype, b_dtype, b_layout, m_bucket, N, K)
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
            "version": 3,
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

        # Re-hydrate the IR tree from JSON for codegen.
        ir = _ir_from_json(ir_json)
        src = render_evt_cu(ir, a_str, b_str, cache_key_str=key, b_layout=b_layout, m_bucket=m_bucket)

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
            extra_cuda_cflags=["-std=c++17", "-O3", "--expt-relaxed-constexpr", "-gencode=arch=compute_120,code=sm_120"],
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


def _compile_swiglu7_dual(m_bucket: str, N: int, K: int):
    """Lazy-load a per-(bucket, N, K) instance of the vendored DualGemm kernel.

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
    """
    fast_key = (m_bucket, int(N), int(K))
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
        src = os.path.join(here, "cutlass_kernels", "swiglu7_epi_one_stage.cu")
        if not os.path.exists(src):
            raise FileNotFoundError(f"vendored swiglu7 source not found: {src}")
        cache_root = get_compile_config().cache_root_dir
        # Build dir embeds (bucket, N, K) so distinct keys get their own
        # build artefacts. cpp_extension uses the dir as the cache identity.
        build_tag = f"{m_bucket}_N{N}_K{K}"
        build_dir = os.path.join(cache_root, "evt_kernels", f"swiglu7_dual_{build_tag}")
        os.makedirs(build_dir, exist_ok=True)
        from torch.utils.cpp_extension import load

        module = load(
            name=f"magi_swiglu7_dual_{build_tag}",
            sources=[src],
            extra_include_paths=[
                os.path.join(cutlass_root, "include"),
                os.path.join(cutlass_root, "tools", "util", "include"),
                os.path.join(cutlass_root, "examples"),
                os.path.join(here, "cutlass_kernels"),
            ],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-std=c++17", "-O3", "--expt-relaxed-constexpr", "-gencode=arch=compute_120,code=sm_120"],
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
    out_dtype) so each FX site × bucket only pays this once."""
    if kind == "swiglu7_dual":
        mod = _compile_swiglu7_dual(m_bucket, N_w, K_w)
        return _DispatchEntry(mod.swiglu7_dual_matmul_out, False, out_dtype)
    if kind == "evt_row" or kind == "evt":
        b_layout = "row"
    elif kind == "evt_col":
        b_layout = "col"
    else:
        raise ValueError(f"Unknown EVT kind {kind!r}")
    mod = _compile_evt_module(ir_json, a_dtype, b_dtype, b_layout=b_layout, m_bucket=m_bucket, N=N_w, K=K_w)
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

    # ── Step 2: alloc / fetch D (greedy single-entry cache, inlined) ──
    # FX pass guards (K % 8 == 0; generic N % 4 == 0; swiglu7 N % 8 == 0)
    # ensure n_out is a multiple of CUTLASS AlignmentC = 4 for every dtype,
    # so a plain ``torch.empty((M, n_out), dtype)`` is already CUTLASS-
    # contiguous — no padded stride / scratch buffer route is required.
    # Anything that violates the guards is rejected upstream and falls back
    # to torch.compile's default mm.
    if _D_CACHE_DISABLED:
        D = torch.empty((M, n_out), device=A.device, dtype=out_dtype)
    else:
        dev_idx = A.device.index or 0
        d_key = (M, n_out, out_dtype, dev_idx)
        D = _D_BUF_CACHE.get(d_key)
        if D is None:
            D = torch.empty((M, n_out), device=A.device, dtype=out_dtype)
            _D_BUF_CACHE.clear()
            _D_BUF_CACHE[d_key] = D

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
    # Contiguous (M, n_out) — see _D_BUF_CACHE comment for why padding is
    # never needed under the FX-pass alignment guards.
    return A.new_empty((A.shape[0], n_out), dtype=out_dtype)
