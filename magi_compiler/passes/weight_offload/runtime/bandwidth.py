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

The result is cached on the process's ``HostWeightStore`` (see
``HostWeightStore.h2d_bandwidth_bytes_per_ns``); this module only runs the probe.
The probe is rank-local and issues no collective: whether a rank compiles any
load, and whether its cache is warm, differ across ranks.
"""

from __future__ import annotations

import torch

from magi_compiler.utils import magi_logger

PROBE_BYTES = 64 << 20

FALLBACK_BYTES_PER_NS = 20.0
"""~20 GB/s: a conservative PCIe Gen4 x16 pinned transfer, used when the probe
cannot run.  Under-estimating bandwidth over-estimates the load, which makes the
reorder pass hoist further than needed -- slower but never incorrect."""


PROBE_ITERS = 20


def _probe_buffers() -> tuple[torch.Tensor, torch.Tensor] | None:
    """Pinned source and device destination with the driver's mapping warmed, or None if either fails."""
    try:
        src = torch.empty(PROBE_BYTES, dtype=torch.uint8, device="cpu", pin_memory=True)
        dst = torch.empty(PROBE_BYTES, dtype=torch.uint8, device="cuda")
        for _ in range(2):
            dst.copy_(src, non_blocking=True)
        torch.cuda.current_stream().synchronize()
    except RuntimeError as exc:
        magi_logger.warning(
            "host offload: H2D bandwidth probe setup failed (%s); assuming %.1f GB/s", exc, FALLBACK_BYTES_PER_NS
        )
        return None
    return src, dst


def _timed_copies_ns(src: torch.Tensor, dst: torch.Tensor) -> float:
    """Nanoseconds for ``PROBE_ITERS`` back-to-back copies, or 0.0 if a copy fails."""
    stream = torch.cuda.current_stream()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    try:
        start.record(stream)
        for _ in range(PROBE_ITERS):
            dst.copy_(src, non_blocking=True)
        end.record(stream)
        stream.synchronize()
    except RuntimeError as exc:
        magi_logger.warning("host offload: H2D bandwidth probe failed (%s); assuming %.1f GB/s", exc, FALLBACK_BYTES_PER_NS)
        return 0.0
    return start.elapsed_time(end) * 1e6


def measure_h2d_bandwidth() -> float:
    """Pinned H2D bandwidth in bytes per nanosecond, as this rank sees it alone.

    Ranks sharing a host link are not lined up, so the result can exceed the
    contended bandwidth; set ``offload_h2d_bandwidth_gbps`` to pin it."""
    if not torch.cuda.is_available():
        return FALLBACK_BYTES_PER_NS
    buffers = _probe_buffers()
    elapsed_ns = _timed_copies_ns(*buffers) if buffers is not None else 0.0
    if elapsed_ns <= 0:
        return FALLBACK_BYTES_PER_NS
    return PROBE_BYTES * PROBE_ITERS / elapsed_ns
