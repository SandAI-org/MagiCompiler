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

import json
import math
import os

import torch
import triton
import triton.language as tl

from magi_compiler.config import get_compile_config

# ── Python-level kernel caches ─────────────────────────────────────────────────
# (num_extras, epilogue_code, reduce_n_by_2) → kernel object
_KERNEL_CACHE: dict = {}
_KERNEL_TMA_CACHE: dict = {}

# ── Persistent autotune result caches (survive process restart) ────────────────
_cache_root = get_compile_config().cache_root_dir
_AUTOTUNE_FILE = os.path.join(_cache_root, "magi_epilogue_autotune.json")
_AUTOTUNE_FILE_TMA = os.path.join(_cache_root, "magi_epilogue_autotune_tma.json")
_AUTOTUNE_PERSIST: dict = {}
_AUTOTUNE_PERSIST_TMA: dict = {}


def _load_autotune_cache() -> None:
    global _AUTOTUNE_PERSIST
    try:
        with open(_AUTOTUNE_FILE) as f:
            _AUTOTUNE_PERSIST = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        _AUTOTUNE_PERSIST = {}


def _save_autotune_cache() -> None:
    os.makedirs(os.path.dirname(_AUTOTUNE_FILE), exist_ok=True)
    with open(_AUTOTUNE_FILE, "w") as f:
        json.dump(_AUTOTUNE_PERSIST, f)


def _load_autotune_cache_tma() -> None:
    global _AUTOTUNE_PERSIST_TMA
    try:
        with open(_AUTOTUNE_FILE_TMA) as f:
            _AUTOTUNE_PERSIST_TMA = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        _AUTOTUNE_PERSIST_TMA = {}


def _save_autotune_cache_tma() -> None:
    os.makedirs(os.path.dirname(_AUTOTUNE_FILE_TMA), exist_ok=True)
    with open(_AUTOTUNE_FILE_TMA, "w") as f:
        json.dump(_AUTOTUNE_PERSIST_TMA, f)


_load_autotune_cache()


def _check_tma() -> bool:
    """Return True when SM90+ TMA with device-side descriptors is available."""
    try:
        return (
            torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9 and hasattr(tl, "make_tensor_descriptor")
        )
    except Exception:
        return False


_TMA_AVAILABLE: bool = _check_tma()
_TMA_ALLOCATOR_SET: bool = False

if _TMA_AVAILABLE:
    _load_autotune_cache_tma()


def _ensure_tma_allocator() -> None:
    """Set a Triton global-memory allocator once; required by device-side TMA descriptors."""
    global _TMA_ALLOCATOR_SET
    if _TMA_ALLOCATOR_SET:
        return

    def _alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(_alloc_fn)
    _TMA_ALLOCATOR_SET = True


def _parse_static_dims(epilogue_code: str) -> dict:
    """Parse the ``# @static:{...}`` header injected by the fusion pass.

    Returns a dict like ``{"M": 2048, "K": 4096, "N": 8192}`` (only the keys
    that are actually static).  Missing keys mean the dimension is dynamic.
    """
    for line in epilogue_code.splitlines():
        if line.startswith("# @static:"):
            try:
                return json.loads(line[len("# @static:") :])
            except Exception:
                pass
    return {}


def _bucket_m(M: int) -> int:
    """Round M up to the nearest power-of-2 bucket.

    This drastically reduces the number of distinct (M, N, K) triples
    that trigger autotune: e.g. M=1000 and M=1023 both map to 1024,
    reusing the same benchmark result instead of each triggering 27 × 125
    device kernel launches.
    """
    return 1 << math.ceil(math.log2(max(M, 1)))


# ── Autotune config list ───────────────────────────────────────────────────────
# Shapes that prune_configs removes:
#   • BLOCK_M > M_bucket  → waste SM occupancy on empty rows
#   • BLOCK_K > K         → single-iteration k-loop, large overhead
#   • BLOCK_N > N         → waste on empty columns


def _prune_configs(configs, named_args, **kwargs):
    M = named_args["M"]
    N = named_args["N"]
    K = named_args["K"]
    pruned = []
    for cfg in configs:
        bm = cfg.kwargs["BLOCK_M"]
        bn = cfg.kwargs["BLOCK_N"]
        bk = cfg.kwargs["BLOCK_K"]
        # Keep configs whose tiles are no larger than 4× the dimension
        # (leaving room for the autotuner to still test large tiles that
        # can handle moderate-size matrices efficiently).
        if bm > 4 * M or bn > 4 * N or bk > K:
            continue
        pruned.append(cfg)
    # Always keep at least one fallback
    return pruned if pruned else [configs[0]]


# ── Shared autotune config list (embedded as a string in both templates) ───────
_AUTOTUNE_CONFIGS_BODY = """
    # ── Large-tile: high-throughput for large M/N (training) ──────────────────
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64,  "GROUP_M": 8}, num_stages=3, num_warps=8),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8}, num_stages=3, num_warps=8),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_stages=3, num_warps=8),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64,  "BLOCK_K": 128, "GROUP_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_stages=4, num_warps=4),
    # ── Medium-tile: balanced for mixed shapes ─────────────────────────────────
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64,  "GROUP_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 64,  "GROUP_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64,  "GROUP_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 128, "GROUP_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 32,  "GROUP_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32,  "GROUP_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64,  "GROUP_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64,  "GROUP_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32,  "GROUP_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32,  "GROUP_M": 8}, num_stages=4, num_warps=4),
    # ── Small-tile: high occupancy for small-M or tail dimensions ─────────────
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32,  "BLOCK_K": 128, "GROUP_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32,  "BLOCK_K": 64,  "GROUP_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 32,  "BLOCK_K": 64,  "GROUP_M": 8}, num_stages=5, num_warps=2),
    triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64,  "BLOCK_K": 64,  "GROUP_M": 8}, num_stages=5, num_warps=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32,  "BLOCK_K": 32,  "GROUP_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 32,  "BLOCK_K": 32,  "GROUP_M": 8}, num_stages=5, num_warps=2),
    triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64,  "BLOCK_K": 32,  "GROUP_M": 8}, num_stages=5, num_warps=2),
    triton.Config({"BLOCK_M": 32,  "BLOCK_N": 32,  "BLOCK_K": 64,  "GROUP_M": 8}, num_stages=5, num_warps=2),
    triton.Config({"BLOCK_M": 16,  "BLOCK_N": 32,  "BLOCK_K": 64,  "GROUP_M": 8}, num_stages=6, num_warps=2),
    triton.Config({"BLOCK_M": 32,  "BLOCK_N": 16,  "BLOCK_K": 64,  "GROUP_M": 8}, num_stages=6, num_warps=2),
"""


# ─────────────────────────────────────────────────────────────────────────────
# Non-persistent kernel template (all CUDA GPUs)
# Uses tl.where + tl.max_contiguous + tl.multiple_of for vectorised loads.
# ─────────────────────────────────────────────────────────────────────────────
KERNEL_TEMPLATE = """
import triton
import triton.language as tl

_AUTOTUNE_CONFIGS = [
{autotune_configs}
]

@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["M_BUCKET", "N", "K"],
    prune_configs_by={{"early_config_prune": {prune_fn_name}}},
    warmup=10,
    rep=30,
)
@triton.jit
def dynamic_matmul_epilogue_kernel(
    A_ptr, B_ptr, D_ptr,
    {extra_ptrs_args}
    M{M_annot}, N{N_annot}, K{K_annot},
    M_BUCKET,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_dm, stride_dn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id         = pid // num_pid_in_group
    first_pid_m      = group_id * GROUP_M
    group_size_m     = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N

    offs_am = start_m + tl.arange(0, BLOCK_M)
    offs_bn = start_n + tl.arange(0, BLOCK_N)
{offs_am_guard}{offs_bn_guard}    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_M), BLOCK_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_N), BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    A_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(A_ptrs{k_mask_a})
        b = tl.load(B_ptrs{k_mask_b})
        acc = tl.dot(a, b, acc)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    offs_dm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_dn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = {out_mask_expr}

{epilogue_code}

{store_code}
"""


# ─────────────────────────────────────────────────────────────────────────────
# TMA persistent kernel template (SM90+: H100 / Hopper and newer)
#
# Key advantages over the non-persistent path:
#   1. Device-side tl.make_tensor_descriptor — no host→device descriptor copy.
#   2. Persistent CTA loop — each SM processes multiple tiles, amortising
#      kernel-launch and L2-warmup overhead.
#   3. Hardware-managed OOB fill — TMA zero-fills out-of-bounds tile edges,
#      so the k-loop needs no software mask.
#   4. B read as [K, N] (no pre-transpose required).
#
# {epilogue_code} and {store_code} are injected at 8-space indent so they
# land inside the `for tile_id` persistent loop body.
# ─────────────────────────────────────────────────────────────────────────────
KERNEL_TEMPLATE_TMA_PERSISTENT = """
import triton
import triton.language as tl

_AUTOTUNE_CONFIGS_TMA = [
{autotune_configs}
]

@triton.autotune(
    configs=_AUTOTUNE_CONFIGS_TMA,
    key=["M_BUCKET", "N", "K"],
    prune_configs_by={{"early_config_prune": {prune_fn_name}}},
    warmup=10,
    rep=30,
)
@triton.jit
def dynamic_matmul_epilogue_kernel_tma(
    A_ptr, B_ptr, D_ptr,
    {extra_ptrs_args}
    M{M_annot}, N{N_annot}, K{K_annot},
    M_BUCKET,
    stride_dm, stride_dn,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Device-side TMA descriptor creation — eliminates host→device copy latency.
    # A is [M, K] row-major; B is [K, N] row-major (no pre-transpose needed).
    # TMA hardware zero-fills tiles that extend past the tensor boundary.
    a_desc = tl.make_tensor_descriptor(
        A_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K],
    )
    b_desc = tl.make_tensor_descriptor(
        B_ptr, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_K, BLOCK_N],
    )

    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_M * num_pid_n

    # Each CTA iterates over multiple tiles, stepping NUM_SMS at a time.
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        group_id     = tile_id // num_pid_in_group
        first_pid_m  = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (tile_id % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        offs_am = pid_m * BLOCK_M
        offs_bn = pid_n * BLOCK_N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            offs_k = k * BLOCK_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_k, offs_bn])
            acc = tl.dot(a, b, acc)

        offs_dm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_dn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = {out_mask_expr}

{epilogue_code}

{store_code}
"""


def _build_kernel_via_exec(
    template: str, kernel_name: str, num_extras: int, epilogue_code: str, reduce_n_by_2: bool, indent: int, persist_cache: dict
) -> object:
    """Compile *template* with exec() and return the resulting Triton kernel."""
    extra_ptrs_args = "".join([f"Extra_{i}_ptr, " for i in range(num_extras)])

    # ── Derive tl.constexpr annotations and static mask/guard expressions ────
    # The fusion pass prepends a "# @static:{...}" comment to epilogue_code
    # whenever it can prove (from FakeTensor meta) that a dimension is a plain
    # Python int rather than a SymInt.
    static_dims = _parse_static_dims(epilogue_code)
    M_static = static_dims.get("M")
    N_static = static_dims.get("N")
    K_static = static_dims.get("K")

    # tl.constexpr annotation: Triton JIT-compiles one kernel variant per
    # unique value, making all constexpr-dependent expressions compile-time
    # constants (loop bounds, tile counts, mask predicates, etc.).
    M_annot = ": tl.constexpr" if M_static is not None else ""
    N_annot = ": tl.constexpr" if N_static is not None else ""
    K_annot = ": tl.constexpr" if K_static is not None else ""

    # ── k-loop load masks ─────────────────────────────────────────────────────
    # Our BLOCK_K configs are {32, 64, 128}; the mask in the k-loop is needed
    # only when K is not a multiple of the chosen BLOCK_K.  If K % 128 == 0,
    # then K is a multiple of every BLOCK_K in the config set, so the mask
    # predicate is always all-true and we can emit bare (unmasked) tl.load
    # calls — the hottest path in the kernel.
    if K_static is not None and K_static % 128 == 0:
        k_mask_a = ""
        k_mask_b = ""
    else:
        k_mask_a = ", mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0"
        k_mask_b = ", mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0"

    # ── A / B index boundary guards ───────────────────────────────────────────
    # tl.where(offs < dim, offs, 0) prevents out-of-bounds pointer arithmetic
    # when a tile straddles the last row/column.  If dim is a multiple of the
    # largest BLOCK size (256 covers all configs {16,32,64,128,256}), every
    # tile is a full tile and the guard is dead code — remove it.
    m_tile_aligned = M_static is not None and M_static % 256 == 0
    n_tile_aligned = N_static is not None and N_static % 256 == 0

    offs_am_guard = "" if m_tile_aligned else "    offs_am = tl.where(offs_am < M, offs_am, 0)\n"
    offs_bn_guard = "" if n_tile_aligned else "    offs_bn = tl.where(offs_bn < N, offs_bn, 0)\n"

    # ── Output (and epilogue) mask ────────────────────────────────────────────
    # The mask tensor is referenced by both the output store and extra-tensor
    # loads inside epilogue_code.  When a dimension is tile-aligned we drop
    # its component from the predicate; both dropped → constant True mask (the
    # compiler will eliminate it entirely from the PTX).
    if m_tile_aligned and n_tile_aligned:
        out_mask_expr = "tl.full([BLOCK_M, BLOCK_N], True, dtype=tl.int1)"
    elif m_tile_aligned:
        out_mask_expr = "offs_dn[None, :] < N"
    elif n_tile_aligned:
        out_mask_expr = "offs_dm[:, None] < M"
    else:
        out_mask_expr = "(offs_dm[:, None] < M) & (offs_dn[None, :] < N)"

    pad = " " * indent
    indented_epilogue = "\n".join([f"{pad}{line}" for line in epilogue_code.strip().split("\n") if line])

    if reduce_n_by_2:
        # For SwiGLU the output N is N//2; output BLOCK size is BLOCK_N//2
        # whose maximum across configs is 128.  Tile-alignment condition:
        # (N_static // 2) % 128 == 0  ↔  N_static % 256 == 0  (same as n_tile_aligned).
        if m_tile_aligned and n_tile_aligned:
            mask_out_expr = "tl.full([BLOCK_M, BLOCK_N // 2], True, dtype=tl.int1)"
        elif m_tile_aligned:
            mask_out_expr = "offs_dn_out[None, :] < N // 2"
        elif n_tile_aligned:
            mask_out_expr = "offs_dm[:, None] < M"
        else:
            mask_out_expr = "(offs_dm[:, None] < M) & (offs_dn_out[None, :] < N // 2)"
        store_code = (
            f"{pad}offs_dn_out = pid_n * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)\n"
            f"{pad}mask_out = {mask_out_expr}\n"
            f"{pad}D_ptrs = D_ptr + stride_dm * offs_dm[:, None] + stride_dn * offs_dn_out[None, :]\n"
            f"{pad}tl.store(D_ptrs, acc.to(D_ptr.dtype.element_ty), mask=mask_out)"
        )
    else:
        store_code = (
            f"{pad}D_ptrs = D_ptr + stride_dm * offs_dm[:, None] + stride_dn * offs_dn[None, :]\n"
            f"{pad}tl.store(D_ptrs, acc.to(D_ptr.dtype.element_ty), mask=mask)"
        )

    code = template.format(
        autotune_configs=_AUTOTUNE_CONFIGS_BODY,
        extra_ptrs_args=extra_ptrs_args,
        epilogue_code=indented_epilogue,
        store_code=store_code,
        prune_fn_name="_prune_configs",
        M_annot=M_annot,
        N_annot=N_annot,
        K_annot=K_annot,
        offs_am_guard=offs_am_guard,
        offs_bn_guard=offs_bn_guard,
        k_mask_a=k_mask_a,
        k_mask_b=k_mask_b,
        out_mask_expr=out_mask_expr,
    )

    import linecache
    import uuid

    filename = f"<dynamic_kernel_{uuid.uuid4().hex}>"
    linecache.cache[filename] = (len(code), None, [line + "\n" for line in code.splitlines()], filename)
    compiled = compile(code, filename, "exec")

    namespace: dict = {}
    exec(compiled, {"triton": triton, "tl": tl, "_prune_configs": _prune_configs}, namespace)
    kernel = namespace[kernel_name]

    # Warm the in-process autotune cache from the persisted JSON so that
    # known shapes skip the benchmark entirely on restart.
    key_str = str((num_extras, epilogue_code, reduce_n_by_2))
    for cache_key, best_cfg in persist_cache.items():
        if cache_key.startswith(key_str + "|"):
            suffix = cache_key[len(key_str) + 1 :]
            try:
                m_bucket, n, k = (int(x) for x in suffix.split(","))
            except ValueError:
                continue
            triton_key = (m_bucket, n, k)
            cfg = triton.Config(
                {k2: v for k2, v in best_cfg["kwargs"].items()},
                num_stages=best_cfg["num_stages"],
                num_warps=best_cfg["num_warps"],
            )
            kernel.cache[triton_key] = cfg

    return kernel


def get_dynamic_kernel(num_extras: int, epilogue_code: str, reduce_n_by_2: bool):
    key = (num_extras, epilogue_code, reduce_n_by_2)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    kernel = _build_kernel_via_exec(
        KERNEL_TEMPLATE,
        "dynamic_matmul_epilogue_kernel",
        num_extras,
        epilogue_code,
        reduce_n_by_2,
        indent=4,
        persist_cache=_AUTOTUNE_PERSIST,
    )
    _KERNEL_CACHE[key] = kernel
    return kernel


def get_dynamic_kernel_tma(num_extras: int, epilogue_code: str, reduce_n_by_2: bool):
    """Build the TMA-persistent variant via exec()."""
    key = (num_extras, epilogue_code, reduce_n_by_2)
    if key in _KERNEL_TMA_CACHE:
        return _KERNEL_TMA_CACHE[key]
    kernel = _build_kernel_via_exec(
        KERNEL_TEMPLATE_TMA_PERSISTENT,
        "dynamic_matmul_epilogue_kernel_tma",
        num_extras,
        epilogue_code,
        reduce_n_by_2,
        indent=8,  # epilogue/store are inside the persistent for-loop
        persist_cache=_AUTOTUNE_PERSIST_TMA,
    )
    _KERNEL_TMA_CACHE[key] = kernel
    return kernel


def _record_best_config(kernel, epilogue_key: str, M_bucket: int, N: int, K: int, persist: dict, save_fn) -> None:
    """Persist the winning autotune config to disk after it is chosen."""
    triton_key = (M_bucket, N, K)
    cfg = kernel.cache.get(triton_key)
    if cfg is None:
        return
    cache_key = f"{epilogue_key}|{M_bucket},{N},{K}"
    persist[cache_key] = {"kwargs": dict(cfg.kwargs), "num_stages": cfg.num_stages, "num_warps": cfg.num_warps}
    save_fn()


def matmul_custom_epilogue(
    A: torch.Tensor, B: torch.Tensor, extras: list[torch.Tensor], epilogue_code: str, reduce_n_by_2: bool
) -> torch.Tensor:
    M, K = A.shape
    _, N = B.shape
    M_bucket = _bucket_m(M)

    N_out = N // 2 if reduce_n_by_2 else N

    # Align the row stride to 128 bytes so a subsequent cuBLAS mm can read
    # this buffer as its A operand without Inductor inserting a row-padding copy.
    elem_size = A.element_size()
    align_elems = 128 // elem_size
    N_stride = (N_out + align_elems - 1) // align_elems * align_elems
    D = torch.empty((M, N_stride), device=A.device, dtype=A.dtype)[:, :N_out]

    epilogue_key = str((len(extras), epilogue_code, reduce_n_by_2))
    triton_key = (M_bucket, N, K)

    use_tma = _TMA_AVAILABLE and A.is_contiguous() and B.is_contiguous()

    if use_tma:
        # ── TMA persistent path (SM90+) ───────────────────────────────────────
        # Device-side descriptors + persistent CTA loop over NUM_SMS SMs.
        # B is read as [K, N] row-major; no pre-transpose required.
        _ensure_tma_allocator()
        NUM_SMS = torch.cuda.get_device_properties(A.device).multi_processor_count
        kernel = get_dynamic_kernel_tma(len(extras), epilogue_code, reduce_n_by_2)
        needs_persist = triton_key not in kernel.cache

        grid = lambda meta: (min(NUM_SMS, triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"])),)

        args = [A, B, D]
        args.extend(extras)
        args.extend([M, N, K, M_bucket, D.stride(0), D.stride(1), NUM_SMS])

        kernel[grid](*args)

        if needs_persist:
            _record_best_config(kernel, epilogue_key, M_bucket, N, K, _AUTOTUNE_PERSIST_TMA, _save_autotune_cache_tma)

    else:
        # ── Non-persistent pointer-arithmetic path (all CUDA GPUs) ───────────
        kernel = get_dynamic_kernel(len(extras), epilogue_code, reduce_n_by_2)
        needs_persist = triton_key not in kernel.cache

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

        args = [A, B, D]
        args.extend(extras)
        args.extend([M, N, K, M_bucket, A.stride(0), A.stride(1), B.stride(0), B.stride(1), D.stride(0), D.stride(1)])

        kernel[grid](*args)

        if needs_persist:
            _record_best_config(kernel, epilogue_key, M_bucket, N, K, _AUTOTUNE_PERSIST, _save_autotune_cache)

    return D
