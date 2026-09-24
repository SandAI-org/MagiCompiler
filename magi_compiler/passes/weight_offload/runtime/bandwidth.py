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

"""Probe of the pinned host-to-device bandwidth the load placement is priced at.

The result is cached on the process's ``HostPool`` (see
``HostPool.h2d_bandwidth_bytes_per_ns``); this module only runs the probe.
"""

from __future__ import annotations

import torch

from magi_compiler.utils import magi_logger

PROBE_BYTES = 64 << 20

FALLBACK_BYTES_PER_NS = 20.0
"""~20 GB/s: a conservative PCIe Gen4 x16 pinned transfer, used when the probe
cannot run.  Under-estimating bandwidth over-estimates the load, which makes the
reorder pass hoist further than needed -- slower but never incorrect."""


def probe_world() -> int:
    import torch.distributed as dist

    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _probe_barrier() -> None:
    """Line every rank up on the probe, so it measures the contended bandwidth.

    Best-effort: a failure here costs accuracy, not correctness, and must never
    be the thing that hangs a compile.
    """
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        except Exception as exc:  # noqa: BLE001
            magi_logger.warning("host offload: bandwidth probe barrier failed (%s); measuring unsynchronized", exc)


def measure_h2d_bandwidth() -> float:
    """Pinned H2D bandwidth in bytes per nanosecond, measured with every rank probing at once."""
    if not torch.cuda.is_available():
        return FALLBACK_BYTES_PER_NS
    try:
        src = torch.empty(PROBE_BYTES, dtype=torch.uint8, device="cpu", pin_memory=True)
        dst = torch.empty(PROBE_BYTES, dtype=torch.uint8, device="cuda")
        stream = torch.cuda.current_stream()
        for _ in range(2):  # warm the driver's mapping before timing
            dst.copy_(src, non_blocking=True)
        stream.synchronize()

        # Enough iterations that the ranks stay overlapped for the whole window
        # rather than straggling apart after the barrier.
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        iters = 20
        _probe_barrier()
        start.record(stream)
        for _ in range(iters):
            dst.copy_(src, non_blocking=True)
        end.record(stream)
        stream.synchronize()
        _probe_barrier()
        elapsed_ns = start.elapsed_time(end) * 1e6
        if elapsed_ns <= 0:
            return FALLBACK_BYTES_PER_NS
        return PROBE_BYTES * iters / elapsed_ns
    except RuntimeError as exc:  # noqa: BLE001
        magi_logger.warning("host offload: H2D bandwidth probe failed (%s); assuming %.1f GB/s", exc, FALLBACK_BYTES_PER_NS)
        return FALLBACK_BYTES_PER_NS
