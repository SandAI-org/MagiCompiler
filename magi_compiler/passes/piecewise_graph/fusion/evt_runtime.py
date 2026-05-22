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


# Greedy alignment: 128 bits when divisible, 64-bit fallback.
GREEDY_ALIGN_BITS = (128, 64)


def _runtime_align_bits(dim: int, dtype: torch.dtype) -> int:
    n_int = int(dim)
    for bits in GREEDY_ALIGN_BITS:
        align_elems = max(1, bits // (dtype.itemsize * 8))
        if n_int % align_elems == 0:
            return bits
    raise ValueError(f"dim={n_int} not even {GREEDY_ALIGN_BITS[-1]}-bit-aligned for dtype={dtype}")


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
_MODULE_CACHE: dict = {}
# Fast cache keyed by hashable tuple — skips json.dumps + sha256 on hot path.
_MODULE_FAST_CACHE: dict = {}
_MODULE_LOCKS: dict = {}
_MODULE_LOCKS_GLOBAL = threading.Lock()
_SWIGLU7_LOCK = threading.Lock()


# Single-entry greedy D-buffer cache. Opt out with MAGI_EVT_DISABLE_D_CACHE=1.
_D_BUF_CACHE: dict = {}
_D_CACHE_DISABLED: bool = os.environ.get("MAGI_EVT_DISABLE_D_CACHE", "0") not in ("0", "", "false", "False")


def _device_gencode_flags() -> list[str]:
    """Return nvcc -gencode flags for the live device.

    sm_90 needs the ``a`` variant for WGMMA/TMA support.
    Override with MAGI_EVT_GENCODE (semicolon-separated).
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
    """Short tag for the live device (e.g. ``sm90``), folded into build_dir."""
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


# ``cpp_extension.load`` uses a ``FileBaton`` (torch/utils/file_baton.py) to
# serialise multi-process compile requests for the same extension: the holding
# process creates ``<build_dir>/lock`` and removes it inside a ``finally``.
# If the holder is SIGKILL'd mid-build (Ctrl-C → timeout escalation, OOM,
# container restart) ``release()`` never runs, the file stays on disk, and
# every subsequent ``load()`` poll-waits on ``os.path.exists(lock)`` forever.
#
# We harden against this in two ways:
#   1. **Skip the lock entirely when the .so is already built.** The hot path
#      after a warm on-disk cache should never touch FileBaton — we just
#      dlopen the .so directly.
#   2. **Probe the existing lock with fcntl.flock(LOCK_EX|LOCK_NB).** The
#      kernel releases flock'd advisory locks when the holding process dies
#      (graceful or SIGKILL), so flock gives us a *correct* liveness check
#      independent of mtime. If we can grab the flock, the previous owner is
#      gone and the file is stale regardless of how recently it was created.
#      mtime is only used as a coarse extra guard when flock isn't available.


def _try_dlopen_prebuilt(build_dir: str, mod_name: str):
    """Fast path: if the .so for this build_dir already exists, import it
    directly without going through cpp_extension.load (which would try to
    acquire FileBaton and could hang on a stale lock).

    Returns the loaded module on success, None if the .so isn't there yet.
    """
    so_path = os.path.join(build_dir, f"{mod_name}.so")
    if not os.path.isfile(so_path):
        return None
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(mod_name, so_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:
        # Anything goes wrong (corrupt .so, ABI mismatch, …) — let the slow
        # path through cpp_extension.load() handle it properly.
        return None


def _evict_stale_lock(build_dir: str) -> None:
    """Reclaim ``<build_dir>/lock`` if its owner is gone.

    Strategy: open the lock file and try to ``flock(LOCK_EX|LOCK_NB)``. The
    OS releases advisory locks when the holding process exits, including on
    SIGKILL, so a successful non-blocking acquisition proves the previous
    holder is dead. We then unlink the file and release our own flock so the
    next FileBaton.try_acquire() succeeds.

    If flock is unavailable (non-Unix) or the file is currently held by a
    live process, we leave it alone — letting cpp_extension.load() block as
    designed for genuine concurrent compiles.
    """
    lock_path = os.path.join(build_dir, "lock")
    if not os.path.exists(lock_path):
        return
    try:
        import fcntl
    except ImportError:
        # Windows / no fcntl — fall back to no-op; user must remove stale
        # locks manually. We do NOT use mtime-only eviction here because the
        # "rapid kill within N seconds" workflow can defeat any mtime cutoff.
        return
    try:
        fd = os.open(lock_path, os.O_RDWR)
    except FileNotFoundError:
        return
    except OSError:
        return
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError):
            # Someone is alive and holding the lock — leave it.
            return
        # We hold flock now; the previous owner is dead. Unlink and let our
        # flock be released by closing the fd.
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass
        except OSError:
            pass
    finally:
        os.close(fd)


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

    Each distinct (N, K) gets its own module so autotune state is isolated.
    """
    arch = _device_arch_tag()
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
            "version": 10,
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

        # sm_90 → Sm90EVT (TMA+WGMMA); else → Sm80EVT (cp.async).
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
        mod_name = f"magi_evt_{key[:12]}"

        # Warm-cache fast path: if a previous run already produced the .so for
        # this exact key, dlopen it directly and skip FileBaton entirely.
        # Avoids hanging on a stale lock when the .so is already usable, and
        # makes repeated kill+restart cycles converge as soon as one run
        # produced the binary.
        prebuilt = _try_dlopen_prebuilt(build_dir, mod_name)
        if prebuilt is not None:
            _MODULE_CACHE[key] = prebuilt
            _MODULE_FAST_CACHE[fast_key] = prebuilt
            return prebuilt

        src_path = os.path.join(build_dir, "evt.cu")
        # Atomic write: tmp + rename to avoid half-written files across ranks.
        tmp_path = f"{src_path}.{os.getpid()}.tmp"
        with open(tmp_path, "w") as f:
            f.write(src)
        os.replace(tmp_path, src_path)

        # Reap any FileBaton lock left by a previous SIGKILL'd build (flock
        # liveness check, mtime-independent). Must run inside the per-key
        # Python lock so concurrent threads in this process cannot race.
        _evict_stale_lock(build_dir)

        cutlass_root = get_compile_config().cutlass_root
        from torch.utils.cpp_extension import load

        # SM90 needs extra cflags for warp-specialized collectives + extended MMA.
        sm90_specific_cflags = (
            ["--expt-extended-lambda", "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED=1"] if arch == "sm90" else []
        )

        # -fvisibility=hidden gives each .so its own copy of CUTLASS template
        # static members like GemmUniversalBase<GemmKernel>::device_ordinal_.
        # Without this, two .so files that instantiate the same GemmKernel
        # type (e.g. medium and large m-bucket modules sharing the same EVT
        # chain + tile shape) collide on the static symbol — the first .so
        # to call init_device_props() poisons the cache for all later .so
        # files: their kernels never get cudaFuncSetAttribute called, so any
        # launch above the default 48 KB dynamic SMEM fails with cudaError-
        # InvalidValue ("invalid argument").
        module = load(
            name=mod_name,
            sources=[src_path],
            extra_include_paths=[
                os.path.join(cutlass_root, "include"),
                os.path.join(cutlass_root, "tools", "util", "include"),
            ],
            extra_cflags=["-O3", "-std=c++17", "-fvisibility=hidden", "-fvisibility-inlines-hidden"],
            extra_cuda_cflags=(
                [
                    "-std=c++17",
                    "-O3",
                    "--expt-relaxed-constexpr",
                    "-Xcompiler=-fvisibility=hidden",
                    "-Xcompiler=-fvisibility-inlines-hidden",
                ]
                + sm90_specific_cflags
                + _device_gencode_flags()
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
    """Inverse of ``to_canonical_json``. Used only at codegen time."""
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
        compute_dtype = d.get("compute_dtype", "float32")
        return Compute(
            op=d["op"],
            children=tuple(_node_from_dict(c) for c in d["children"]),
            scalar=scalar_val,
            compute_dtype=compute_dtype,
        )
    if kind == "store":
        return Store(child=_node_from_dict(d["child"]), out_dtype=d["out_dtype"])
    raise ValueError(f"Unknown IR kind {kind!r}")


# Per-(m_bucket, N, K, align) cache — separate modules so each runner has its
# own autotune state (best_idx_).
_SWIGLU7_FAST_CACHE: dict = {}
_SWIGLU7_BUILD_LOCKS: dict = {}


def _compile_swiglu7_dual(
    m_bucket: str, N: int, K: int, alignment_a_bits: int = 128, alignment_b_bits: int = 128, alignment_c_bits: int = 128
):
    """Lazy-load a per-(bucket, N, K, align) DualGemm kernel module."""
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

        cutlass_root = get_compile_config().cutlass_root
        here = os.path.dirname(os.path.abspath(__file__))
        # sm_90 → TMA+WGMMA DualGemm; else → SM80 multistage path.
        arch_tag = _device_arch_tag()
        arch_subdir = "sm90" if arch_tag == "sm90" else "sm80"
        src = os.path.join(here, arch_subdir, "cutlass_kernels", "swiglu7_one_stage.cu")
        if not os.path.exists(src):
            raise FileNotFoundError(f"vendored swiglu7 source not found: {src}")
        cache_root = get_compile_config().cache_root_dir
        # Build dir embeds (arch, bucket, N, K, align) — stale cross-arch
        # binaries cause cudaErrorInvalidDeviceFunction.
        build_tag = f"{m_bucket}_N{N}_K{K}" f"_aA{alignment_a_bits}_aB{alignment_b_bits}_aC{alignment_c_bits}"
        build_dir = os.path.join(cache_root, "evt_kernels", arch_tag, f"swiglu7_dual_{build_tag}")
        os.makedirs(build_dir, exist_ok=True)
        mod_name = f"magi_swiglu7_dual_{build_tag}"

        # Warm-cache fast path — see _compile_evt_module for rationale.
        prebuilt = _try_dlopen_prebuilt(build_dir, mod_name)
        if prebuilt is not None:
            _SWIGLU7_FAST_CACHE[fast_key] = prebuilt
            return prebuilt

        # See _evict_stale_lock — reap a SIGKILL-orphaned cpp_extension lock
        # before cpp_extension.load tries to acquire it.
        _evict_stale_lock(build_dir)

        from torch.utils.cpp_extension import load

        # SM90 needs extra cflags for WGMMA + warp-specialized collective.
        sm90_specific_cflags = (
            ["--expt-extended-lambda", "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED=1"] if arch_tag == "sm90" else []
        )

        sm90_include_paths = [os.path.join(here, "sm90", "cutlass_kernels")] if arch_tag == "sm90" else []

        # -fvisibility=hidden — see _compile_evt_module above for rationale.
        module = load(
            name=mod_name,
            sources=[src],
            extra_include_paths=[
                os.path.join(cutlass_root, "include"),
                os.path.join(cutlass_root, "tools", "util", "include"),
                os.path.join(cutlass_root, "examples"),
                os.path.join(here, "common", "cutlass_kernels"),
                *sm90_include_paths,
            ],
            extra_cflags=["-O3", "-std=c++17", "-fvisibility=hidden", "-fvisibility-inlines-hidden"],
            extra_cuda_cflags=[
                "-std=c++17",
                "-O3",
                "--expt-relaxed-constexpr",
                "-Xcompiler=-fvisibility=hidden",
                "-Xcompiler=-fvisibility-inlines-hidden",
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


# ── Dispatch fast-cache ──────────────────────────────────────────────────────
# Collapses out_dtype_from_id → _m_bucket → _compile_* → mod.attr-lookup
# into a single dict.get(). Keyed by (kind, ir_json, dtypes, N, K, m_bucket,
# out_dtype); reaches steady state after the first call per (site, bucket).
class _DispatchEntry:
    __slots__ = ("kernel_call", "is_evt", "out_dtype")

    def __init__(self, kernel_call, is_evt, out_dtype):
        self.kernel_call = kernel_call
        self.is_evt = is_evt
        self.out_dtype = out_dtype


_DISPATCH_CACHE: dict = {}


def _resolve_dispatch(kind, ir_json, a_dtype, b_dtype, N_w, K_w, m_bucket, out_dtype):
    """Slow-path resolver — compiles the .cu module and binds the kernel callable."""
    n_out_for_c = (N_w // 2) if kind == "swiglu7_dual" else N_w
    ldd = _aligned_n_stride(n_out_for_c, out_dtype)
    alignment_c_bits = _runtime_align_bits(ldd, out_dtype)

    if kind == "swiglu7_dual":
        # K alignment also covers ldB=2K.
        align_bits = _runtime_align_bits(K_w, a_dtype)
        mod = _compile_swiglu7_dual(
            m_bucket, N_w, K_w, alignment_a_bits=align_bits, alignment_b_bits=align_bits, alignment_c_bits=alignment_c_bits
        )
        sw7 = json.loads(ir_json) if ir_json else {}
        sw7_alpha = float(sw7.get("alpha", 1.702))
        sw7_limit = float(sw7.get("limit", 7.0))
        sw7_one = float(sw7.get("one", 1.0))
        kernel_fn = mod.swiglu7_dual_matmul_out

        def _sw7_call(A, B, D, _fn=kernel_fn, _a=sw7_alpha, _l=sw7_limit, _o=sw7_one):
            return _fn(A, B, D, _a, _l, _o)

        return _DispatchEntry(_sw7_call, False, out_dtype)
    if kind == "evt_row" or kind == "evt":
        b_layout = "row"
    elif kind == "evt_col":
        b_layout = "col"
    else:
        raise ValueError(f"Unknown EVT kind {kind!r}")
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
    """Runtime entry point. Do NOT call .contiguous() on B — the FX pass
    controls the layout (evt_row=RowMajor, evt_col/swiglu7=ColumnMajor)."""
    # B.size(0)/size(1) avoids the Python tuple construction of .shape.
    B_size0 = B.size(0)
    B_size1 = B.size(1)
    M = A.size(0)
    if M <= 256:
        m_bucket = "small"
    elif M <= 2048:
        m_bucket = "medium"
    else:
        m_bucket = "large"
    out_dtype = _ID_TO_DTYPE[out_dtype_id_]
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
