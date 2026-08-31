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
Memory-peak tests for shared-memory weight materialization.

Background
----------
``_materialize_shm_weights`` copies ALL parameters into an mmap file, then
calls ``load_state_dict(assign=True)`` to replace them.  During the copy the
original params (RssAnon) and the new mmap pages (RssFile) coexist, pushing
the process to ~2× model size.

The streaming alternative replaces each parameter immediately after copying,
so only one parameter's worth of duplication exists at any moment.

Measurement
-----------
Each test runs in a **subprocess** (clean VmHWM baseline).
VmHWM (RSS high-water mark from ``/proc/self/status``) tracks the growth
contributed by the mmap copy:

- **batch**:    VmHWM growth ≈ model_size  (old params + mmap ≈ 2×)
- **streaming**: VmHWM growth ≈ model_size / num_params  (≪ 0.3×)

No distributed / CUDA required.
"""

import gc
import multiprocessing as mp
import os
import tempfile

import pytest
import torch
import torch.nn as nn

_IS_LINUX = os.path.exists("/proc/self/status")
_skip_no_procfs = pytest.mark.skipif(not _IS_LINUX, reason="requires /proc/self/status for VmHWM")

PARAM_MB = 512
NUM_PARAMS = 4


def _read_vm(key: str = "VmHWM") -> float:
    """Read a VmXxx field from /proc/self/status (MB)."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1]) / 1024
    raise RuntimeError(f"{key} not found")


class HeavyModule(nn.Module):
    def __init__(self, numel_per_param: int, num_params: int = NUM_PARAMS, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        for i in range(num_params):
            self.register_parameter(f"w{i}", nn.Parameter(torch.randn(numel_per_param, dtype=dtype)))

    def forward(self, x):
        return x


# ── batch (current code pattern) ────────────────────────────


def _batch_materialize(module: nn.Module, shm_dir: str) -> None:
    full_state_dict = module.state_dict()
    grouped: dict[torch.dtype, list[tuple[str, torch.Tensor]]] = {}
    for name, tensor in full_state_dict.items():
        grouped.setdefault(tensor.dtype, []).append((name, tensor))

    shared_state: dict[str, torch.Tensor] = {}
    buffers: list[torch.Tensor] = []

    for dtype, param_list in grouped.items():
        total_numel = sum(t.numel() for _, t in param_list)
        elem_size = torch.empty(0, dtype=dtype).element_size()
        path = os.path.join(shm_dir, f"batch_{dtype}.bin")
        with open(path, "wb") as f:
            f.truncate(total_numel * elem_size)
        giant = torch.from_file(path, shared=True, size=total_numel, dtype=dtype, device="cpu")
        offset = 0
        for _, tensor in param_list:
            n = tensor.numel()
            giant[offset : offset + n].copy_(tensor.view(-1))
            offset += n
        offset = 0
        for name, orig in param_list:
            n = orig.numel()
            view = giant[offset : offset + n].view(orig.shape)
            if orig.requires_grad:
                view.requires_grad_(True)
            shared_state[name] = view
            offset += n
        buffers.append(giant)
        if os.path.exists(path):
            os.remove(path)

    module.load_state_dict(shared_state, assign=True)
    module._buffers_ref = buffers


# ── streaming (fix) ─────────────────────────────────────────


def _assign_param(module: nn.Module, dotted_name: str, new_tensor: torch.Tensor) -> None:
    parts = dotted_name.rsplit(".", 1)
    parent = module.get_submodule(parts[0]) if len(parts) == 2 else module
    attr = parts[-1]
    old = getattr(parent, attr)
    if isinstance(old, nn.Parameter):
        parent.register_parameter(attr, nn.Parameter(new_tensor, requires_grad=new_tensor.requires_grad))
    else:
        setattr(parent, attr, new_tensor)


def _streaming_materialize(module: nn.Module, shm_dir: str) -> None:
    full_state_dict = module.state_dict()
    grouped: dict[torch.dtype, list[tuple[str, torch.Tensor]]] = {}
    for name, tensor in full_state_dict.items():
        grouped.setdefault(tensor.dtype, []).append((name, tensor))
    del full_state_dict
    gc.collect()

    buffers: list[torch.Tensor] = []
    for dtype, param_list in grouped.items():
        total_numel = sum(t.numel() for _, t in param_list)
        elem_size = torch.empty(0, dtype=dtype).element_size()
        path = os.path.join(shm_dir, f"stream_{dtype}.bin")
        with open(path, "wb") as f:
            f.truncate(total_numel * elem_size)
        giant = torch.from_file(path, shared=True, size=total_numel, dtype=dtype, device="cpu")
        offset = 0
        for i, (name, tensor) in enumerate(param_list):
            n = tensor.numel()
            giant[offset : offset + n].copy_(tensor.view(-1))
            view = giant[offset : offset + n].view(tensor.shape)
            if tensor.requires_grad:
                view.requires_grad_(True)
            _assign_param(module, name, view)
            param_list[i] = (name, view)
            offset += n
        buffers.append(giant)
        if os.path.exists(path):
            os.remove(path)

    module._buffers_ref = buffers
    gc.collect()


# ── subprocess workers ──────────────────────────────────────


def _worker(result_dict, param_mb, materialize_fn):
    elem_bytes = 2  # bf16
    numel_per = param_mb * 1024 * 1024 // (elem_bytes * NUM_PARAMS)
    model = HeavyModule(numel_per, NUM_PARAMS)
    gc.collect()

    hwm_before = _read_vm("VmHWM")
    with tempfile.TemporaryDirectory() as d:
        materialize_fn(model, d)
    gc.collect()
    hwm_after = _read_vm("VmHWM")

    result_dict["hwm_before"] = hwm_before
    result_dict["hwm_after"] = hwm_after
    result_dict["param_mb"] = param_mb


def _run_in_subprocess(materialize_fn, param_mb):
    ctx = mp.get_context("fork")
    mgr = ctx.Manager()
    result = mgr.dict()
    p = ctx.Process(target=_worker, args=(result, param_mb, materialize_fn))
    p.start()
    p.join(timeout=120)
    assert p.exitcode == 0, f"subprocess exited with code {p.exitcode}"
    return dict(result)


# ── tests ───────────────────────────────────────────────────


@_skip_no_procfs
def test_batch_materialize_has_high_peak():
    """BUG REPRO: batch materialize adds ~1× model size as mmap overhead.

    At peak: old params (RssAnon) + mmap copy (RssFile) ≈ 2× model size.
    VmHWM growth measures the mmap portion, expected > 0.5× model_size.
    """
    r = _run_in_subprocess(_batch_materialize, PARAM_MB)
    growth = r["hwm_after"] - r["hwm_before"]
    pm = r["param_mb"]

    print(
        f"\n[batch] hwm_before={r['hwm_before']:.0f} MB, "
        f"hwm_after={r['hwm_after']:.0f} MB, "
        f"growth={growth:.0f} MB, model_size={pm} MB, "
        f"ratio={growth / pm:.2f}x"
    )

    assert growth > pm * 0.5, (
        f"Expected mmap overhead > {pm * 0.5:.0f} MB (0.5× model) "
        f"but got {growth:.0f} MB ({growth / pm:.2f}×). "
        f"The 2× peak may have been optimized away."
    )


@_skip_no_procfs
def test_streaming_materialize_low_peak():
    """FIX VERIFIED: streaming avoids the mmap overhead peak.

    By replacing each param immediately, only ~1/N of the model is ever
    duplicated.  VmHWM growth should be well under 0.3× model size.
    """
    r = _run_in_subprocess(_streaming_materialize, PARAM_MB)
    growth = r["hwm_after"] - r["hwm_before"]
    pm = r["param_mb"]

    print(
        f"\n[streaming] hwm_before={r['hwm_before']:.0f} MB, "
        f"hwm_after={r['hwm_after']:.0f} MB, "
        f"growth={growth:.0f} MB, model_size={pm} MB, "
        f"ratio={growth / pm:.2f}x"
    )

    assert growth < pm * 0.3, (
        f"Expected mmap overhead < {pm * 0.3:.0f} MB (0.3× model) "
        f"but got {growth:.0f} MB ({growth / pm:.2f}×). "
        f"Streaming fix did not reduce peak."
    )


@_skip_no_procfs
def test_streaming_preserves_weights():
    """Streaming must produce identical weights to batch."""
    numel = 1024
    torch.manual_seed(42)
    model_a = HeavyModule(numel, NUM_PARAMS)
    torch.manual_seed(42)
    model_b = HeavyModule(numel, NUM_PARAMS)

    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        _batch_materialize(model_a, d1)
        _streaming_materialize(model_b, d2)

    for name in model_a.state_dict():
        assert torch.equal(model_a.state_dict()[name], model_b.state_dict()[name]), f"Mismatch on '{name}'"


# ── speed ───────────────────────────────────────────────────

import time


def _speed_worker(result_dict, param_mb, num_params, materialize_fn, repeats):
    elem_bytes = 2
    numel_per = param_mb * 1024 * 1024 // (elem_bytes * num_params)
    times = []
    for _ in range(repeats):
        torch.manual_seed(42)
        model = HeavyModule(numel_per, num_params)
        gc.collect()
        with tempfile.TemporaryDirectory() as d:
            t0 = time.perf_counter()
            materialize_fn(model, d)
            times.append(time.perf_counter() - t0)
        del model
        gc.collect()
    result_dict["times"] = times
    result_dict["avg"] = sum(times) / len(times)


def _run_speed_subprocess(materialize_fn, param_mb, num_params=NUM_PARAMS, repeats=3):
    ctx = mp.get_context("fork")
    mgr = ctx.Manager()
    result = mgr.dict()
    p = ctx.Process(target=_speed_worker, args=(result, param_mb, num_params, materialize_fn, repeats))
    p.start()
    p.join(timeout=300)
    assert p.exitcode == 0, f"subprocess exited with code {p.exitcode}"
    return dict(result)


@_skip_no_procfs
def test_streaming_not_slower_than_batch():
    """Streaming must not be significantly slower than batch.

    Allows up to 1.20x slowdown to account for per-param register_parameter
    overhead.  In practice streaming is often faster on large models because
    it avoids the final load_state_dict bulk copy.
    """
    mb = PARAM_MB
    max_slowdown = 1.20

    r_batch = _run_speed_subprocess(_batch_materialize, mb)
    r_stream = _run_speed_subprocess(_streaming_materialize, mb)

    ratio = r_stream["avg"] / r_batch["avg"]
    print(
        f"\n[speed] model={mb} MB, num_params={NUM_PARAMS}"
        f"  batch={r_batch['avg']:.3f}s"
        f"  streaming={r_stream['avg']:.3f}s"
        f"  ratio={ratio:.2f}x"
    )

    assert ratio < max_slowdown, (
        f"Streaming is {ratio:.2f}x slower than batch "
        f"(limit {max_slowdown}x). "
        f"batch={r_batch['times']}, stream={r_stream['times']}"
    )
