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
The *original* production code in ``_patch_cpu_offload_apply`` allocated an
intermediate ``flat_buffer = torch.zeros(total_numel)`` to pack all parameters,
wrote it to disk via ``.numpy().tofile()``, then deleted the buffer and mmap'd
the file back.  At peak, both the model parameters **and** the flat_buffer
coexist in anonymous memory.

The streaming alternative (``_stream_copy_and_replace``) writes each parameter
directly into an mmap file and replaces the module parameter immediately, so
only one parameter's worth of duplication exists at any moment.

Measurement
-----------
Each test runs in a **subprocess** (clean memory baseline) via
``subprocess.run`` to avoid fork+threads deadlocks in CI Docker.

Memory is measured via ``/proc/self/smaps_rollup`` ``Anonymous`` field, which
performs an accurate page-table walk.  This is preferred over ``RssAnon`` from
``/proc/self/status``, which uses per-CPU batched counters and systematically
under-reports by ~40% on multi-core machines.

Peak is captured **deterministically** at the exact code point where the
flat_buffer coexists with model parameters (no polling thread needed).

- **batch** (original code):  Anonymous growth >= 0.8x model size
- **streaming** (fix):        Anonymous growth < 0.1x model size

No distributed / CUDA required.
"""

import gc
import json
import os
import subprocess
import sys
import tempfile

import torch
import torch.nn as nn

from magi_compiler._api import _create_empty_shm, _stream_copy_and_replace

PARAM_MB = 64
NUM_PARAMS = 4


class HeavyModule(nn.Module):
    def __init__(self, numel_per_param: int, num_params: int = NUM_PARAMS, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        for i in range(num_params):
            self.register_parameter(f"w{i}", nn.Parameter(torch.randn(numel_per_param, dtype=dtype)))

    def forward(self, x):
        return x


# ── batch (original production code, faithfully replicated) ──


def _group_params(module: nn.Module) -> dict[torch.dtype, list[tuple[str, torch.Tensor]]]:
    """Group module params by dtype (shared by both batch and streaming paths)."""
    grouped: dict[torch.dtype, list[tuple[str, torch.Tensor]]] = {}
    for name, tensor in module.state_dict().items():
        grouped.setdefault(tensor.dtype, []).append((name, tensor))
    return grouped


def _batch_materialize(module: nn.Module, shm_dir: str) -> None:
    """Original production code: flat_buffer + tofile + from_file + load_state_dict.

    Faithfully replicates the ORIGINAL _patch_cpu_offload_apply logic that
    caused ~2x peak memory.  The intermediate ``flat_buffer`` is the root
    cause -- it coexists with the model parameters in Anonymous memory.

    NOT imported from production because this code path no longer exists
    (replaced by streaming).  We keep it here as the buggy baseline.
    """
    full_state_dict = module.state_dict()
    grouped: dict[torch.dtype, list[tuple[str, torch.Tensor]]] = {}
    for name, tensor in full_state_dict.items():
        grouped.setdefault(tensor.dtype, []).append((name, tensor))

    shared_state_dict: dict[str, torch.Tensor] = {}
    giant_buffers: list[torch.Tensor] = []

    for dtype, param_list in grouped.items():
        total_numel = sum(t.numel() for _, t in param_list)
        shared_bin_path = os.path.join(shm_dir, f"magi_model_shared_{dtype}.bin")

        flat_buffer = torch.zeros(total_numel, dtype=dtype)
        offset = 0
        for _, tensor in param_list:
            numel = tensor.numel()
            flat_buffer[offset : offset + numel].copy_(tensor.view(-1))
            offset += numel

        if dtype == torch.bfloat16:
            flat_buffer.view(torch.int16).numpy().tofile(shared_bin_path)
        elif dtype.itemsize == 1 and dtype.is_floating_point:
            flat_buffer.view(torch.uint8).numpy().tofile(shared_bin_path)
        else:
            flat_buffer.numpy().tofile(shared_bin_path)

        del flat_buffer
        gc.collect()

        giant_shared_tensor = torch.from_file(shared_bin_path, shared=True, size=total_numel, dtype=dtype, device="cpu")
        giant_buffers.append(giant_shared_tensor)

        offset = 0
        for name, original_tensor in param_list:
            numel = original_tensor.numel()
            shared_param = giant_shared_tensor[offset : offset + numel].view(original_tensor.shape)
            if original_tensor.requires_grad:
                shared_param.requires_grad_(True)
            shared_state_dict[name] = shared_param
            offset += numel

        if os.path.exists(shared_bin_path):
            os.remove(shared_bin_path)

    module.load_state_dict(shared_state_dict, assign=True)
    module._magi_giant_buffers = giant_buffers
    gc.collect()


# ── streaming (fix) ─────────────────────────────────────────


def _streaming_materialize(module: nn.Module, shm_dir: str) -> None:
    """Uses the REAL production functions to test actual behavior."""
    grouped = _group_params(module)
    buffers: list[torch.Tensor] = []

    for dtype, param_list in grouped.items():
        total_numel = sum(t.numel() for _, t in param_list)
        path = os.path.join(shm_dir, f"stream_{dtype}.bin")
        giant = _create_empty_shm(path, total_numel, dtype)
        _stream_copy_and_replace(module, giant, param_list)
        buffers.append(giant)
        if os.path.exists(path):
            os.remove(path)

    module._buffers_ref = buffers
    gc.collect()


# ── subprocess runner (deterministic peak via smaps_rollup Anonymous) ────


_WORKER_TEMPLATE = """
import gc, json, os, sys, tempfile
import torch, torch.nn as nn
from magi_compiler._api import _create_empty_shm, _stream_copy_and_replace

PARAM_MB = {param_mb}
FN_NAME = "{fn_name}"
RESULT_PATH = "{result_path}"
NUM_PARAMS = {num_params}

def _read_smaps_anon():
    with open("/proc/self/smaps_rollup") as fh:
        for line in fh:
            if line.startswith("Anonymous:"):
                return int(line.split()[1]) / 1024
    return 0.0

class HeavyModule(nn.Module):
    def __init__(self, numel, n=NUM_PARAMS, dt=torch.bfloat16):
        super().__init__()
        for i in range(n):
            self.register_parameter("w" + str(i), nn.Parameter(torch.randn(numel, dtype=dt)))
    def forward(self, x): return x

def _batch_with_peak(module, shm_dir):
    full_sd = module.state_dict()
    grouped = {{}}
    for name, tensor in full_sd.items():
        grouped.setdefault(tensor.dtype, []).append((name, tensor))
    shared_sd = {{}}
    bufs = []
    peak = 0.0
    for dtype, plist in grouped.items():
        total = sum(t.numel() for _, t in plist)
        path = os.path.join(shm_dir, "batch.bin")
        flat = torch.zeros(total, dtype=dtype)
        off = 0
        for _, t in plist:
            n = t.numel()
            flat[off:off+n].copy_(t.view(-1))
            off += n
        peak = max(peak, _read_smaps_anon())
        if dtype == torch.bfloat16:
            flat.view(torch.int16).numpy().tofile(path)
        else:
            flat.numpy().tofile(path)
        del flat
        gc.collect()
        giant = torch.from_file(path, shared=True, size=total, dtype=dtype, device="cpu")
        bufs.append(giant)
        off = 0
        for name, orig in plist:
            n = orig.numel()
            v = giant[off:off+n].view(orig.shape)
            if orig.requires_grad: v.requires_grad_(True)
            shared_sd[name] = v
            off += n
        if os.path.exists(path): os.remove(path)
    module.load_state_dict(shared_sd, assign=True)
    module._bufs = bufs
    gc.collect()
    return peak

def _streaming_with_peak(module, shm_dir):
    sd = module.state_dict()
    grouped = {{}}
    for name, tensor in sd.items():
        grouped.setdefault(tensor.dtype, []).append((name, tensor))
    bufs = []
    peak = _read_smaps_anon()
    for dtype, plist in grouped.items():
        total = sum(t.numel() for _, t in plist)
        path = os.path.join(shm_dir, "stream.bin")
        giant = _create_empty_shm(path, total, dtype)
        _stream_copy_and_replace(module, giant, plist)
        peak = max(peak, _read_smaps_anon())
        bufs.append(giant)
        if os.path.exists(path): os.remove(path)
    module._bufs = bufs
    gc.collect()
    return peak

fn = {{"batch": _batch_with_peak, "streaming": _streaming_with_peak}}[FN_NAME]
numel_per = PARAM_MB * 1024 * 1024 // (2 * NUM_PARAMS)
model = HeavyModule(numel_per)
gc.collect()
anon_baseline = _read_smaps_anon()
with tempfile.TemporaryDirectory() as d:
    peak_anon = fn(model, d)
gc.collect()
with open(RESULT_PATH, "w") as f:
    json.dump({{"anon_baseline": anon_baseline, "peak_anon": peak_anon, "param_mb": PARAM_MB}}, f)
"""


def _run_in_subprocess(fn_name: str, param_mb: int) -> dict:
    """Run materialize in a clean subprocess (no fork, no threads)."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as rf:
        result_path = rf.name
    script_content = _WORKER_TEMPLATE.format(
        param_mb=param_mb, fn_name=fn_name, result_path=result_path, num_params=NUM_PARAMS
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as sf:
        sf.write(script_content)
        script_path = sf.name
    try:
        r = subprocess.run([sys.executable, script_path], capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, f"Worker failed (rc={r.returncode}):\nstderr: {r.stderr}\nstdout: {r.stdout}"
        with open(result_path) as f:
            return json.load(f)
    finally:
        for p in (result_path, script_path):
            if os.path.exists(p):
                os.remove(p)


# ── speed subprocess runner ─────────────────────────────────

_SPEED_TEMPLATE = """
import gc, json, os, sys, tempfile, time
import torch, torch.nn as nn
from magi_compiler._api import _create_empty_shm, _stream_copy_and_replace

PARAM_MB = {param_mb}
FN_NAME = "{fn_name}"
RESULT_PATH = "{result_path}"
REPEATS = {repeats}
NUM_PARAMS = {num_params}

class HeavyModule(nn.Module):
    def __init__(self, numel, n=NUM_PARAMS, dt=torch.bfloat16):
        super().__init__()
        for i in range(n):
            self.register_parameter("w" + str(i), nn.Parameter(torch.randn(numel, dtype=dt)))
    def forward(self, x): return x

def _batch_materialize(module, shm_dir):
    sd = module.state_dict()
    grouped = {{}}
    for name, tensor in sd.items():
        grouped.setdefault(tensor.dtype, []).append((name, tensor))
    shared_sd = {{}}
    bufs = []
    for dtype, plist in grouped.items():
        total = sum(t.numel() for _, t in plist)
        path = os.path.join(shm_dir, "batch.bin")
        flat = torch.zeros(total, dtype=dtype)
        off = 0
        for _, t in plist:
            n = t.numel()
            flat[off:off+n].copy_(t.view(-1))
            off += n
        if dtype == torch.bfloat16:
            flat.view(torch.int16).numpy().tofile(path)
        else:
            flat.numpy().tofile(path)
        del flat; gc.collect()
        giant = torch.from_file(path, shared=True, size=total, dtype=dtype, device="cpu")
        bufs.append(giant)
        off = 0
        for name, orig in plist:
            n = orig.numel()
            v = giant[off:off+n].view(orig.shape)
            if orig.requires_grad: v.requires_grad_(True)
            shared_sd[name] = v
            off += n
        if os.path.exists(path): os.remove(path)
    module.load_state_dict(shared_sd, assign=True)
    module._bufs = bufs; gc.collect()

def _streaming_materialize(module, shm_dir):
    sd = module.state_dict()
    grouped = {{}}
    for name, tensor in sd.items():
        grouped.setdefault(tensor.dtype, []).append((name, tensor))
    bufs = []
    for dtype, plist in grouped.items():
        total = sum(t.numel() for _, t in plist)
        path = os.path.join(shm_dir, "stream.bin")
        giant = _create_empty_shm(path, total, dtype)
        _stream_copy_and_replace(module, giant, plist)
        bufs.append(giant)
        if os.path.exists(path): os.remove(path)
    module._bufs = bufs; gc.collect()

fn = {{"batch": _batch_materialize, "streaming": _streaming_materialize}}[FN_NAME]
numel_per = PARAM_MB * 1024 * 1024 // (2 * NUM_PARAMS)
times = []
for _ in range(REPEATS):
    torch.manual_seed(42)
    model = HeavyModule(numel_per)
    gc.collect()
    with tempfile.TemporaryDirectory() as d:
        t0 = time.perf_counter()
        fn(model, d)
        times.append(time.perf_counter() - t0)
    del model; gc.collect()
with open(RESULT_PATH, "w") as f:
    json.dump({{"times": times, "avg": sum(times)/len(times)}}, f)
"""


def _run_speed_subprocess(fn_name: str, param_mb: int, repeats: int = 2) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as rf:
        result_path = rf.name
    script_content = _SPEED_TEMPLATE.format(
        param_mb=param_mb, fn_name=fn_name, result_path=result_path, repeats=repeats, num_params=NUM_PARAMS
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as sf:
        sf.write(script_content)
        script_path = sf.name
    try:
        r = subprocess.run([sys.executable, script_path], capture_output=True, text=True, timeout=180)
        assert r.returncode == 0, f"Speed worker failed (rc={r.returncode}):\nstderr: {r.stderr}"
        with open(result_path) as f:
            return json.load(f)
    finally:
        for p in (result_path, script_path):
            if os.path.exists(p):
                os.remove(p)


# ── tests ───────────────────────────────────────────────────


def test_batch_materialize_has_high_peak():
    """BUG REPRO: original code's flat_buffer causes measurable memory overhead.

    The flat_buffer = torch.zeros(total_numel) in the original production code
    coexists with model parameters, causing ~1x extra Anonymous memory at peak.
    Measured via smaps_rollup Anonymous (accurate page-table walk).
    """
    r = _run_in_subprocess("batch", PARAM_MB)
    growth = r["peak_anon"] - r["anon_baseline"]
    pm = r["param_mb"]

    print(
        f"\n[batch] anon_baseline={r['anon_baseline']:.0f} MB, "
        f"peak_anon={r['peak_anon']:.0f} MB, "
        f"growth={growth:.0f} MB, model_size={pm} MB, "
        f"ratio={growth / pm:.2f}x"
    )

    assert growth > pm * 0.8, (
        f"Expected flat_buffer Anonymous overhead > {pm * 0.8:.0f} MB (0.8x model) "
        f"but got {growth:.0f} MB ({growth / pm:.2f}x). "
        f"The flat_buffer peak may have been optimized away."
    )


def test_streaming_materialize_low_peak():
    """FIX VERIFIED: streaming avoids the flat_buffer overhead peak.

    By writing directly into mmap and replacing each param immediately,
    no intermediate flat_buffer is needed.  Anonymous growth should be
    near zero (old params freed as they are replaced by mmap-backed views).
    """
    r = _run_in_subprocess("streaming", PARAM_MB)
    growth = r["peak_anon"] - r["anon_baseline"]
    pm = r["param_mb"]

    print(
        f"\n[streaming] anon_baseline={r['anon_baseline']:.0f} MB, "
        f"peak_anon={r['peak_anon']:.0f} MB, "
        f"growth={growth:.0f} MB, model_size={pm} MB, "
        f"ratio={growth / pm:.2f}x"
    )

    assert growth < pm * 0.1, (
        f"Expected no flat_buffer Anonymous overhead < {pm * 0.1:.0f} MB (0.1x model) "
        f"but got {growth:.0f} MB ({growth / pm:.2f}x). "
        f"Streaming should not increase Anonymous memory."
    )


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


def test_streaming_not_slower_than_batch():
    """Streaming must not be significantly slower than batch.

    Allows up to 1.50x slowdown to account for per-param register_parameter
    overhead.  In practice streaming is often faster on large models because
    it avoids the final load_state_dict bulk copy.
    """
    mb = PARAM_MB
    max_slowdown = 1.50

    r_batch = _run_speed_subprocess("batch", mb)
    r_stream = _run_speed_subprocess("streaming", mb)

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
