# Copyright (c) 2026 SandAI. All Rights Reserved.
"""Distributed utilities for MagiCompiler."""

from __future__ import annotations

import torch.distributed as dist

from magi_compiler.utils.logger import magi_logger

_CPU_GLOO_GROUP = "uninit"


def get_cpu_gloo_group() -> dist.ProcessGroup | None:
    """Return a gloo process group for CPU-tensor collectives.

    The default process group is typically NCCL, which only supports CUDA
    tensors.  This helper lazily creates a gloo group so that CPU tensors
    can participate in collectives like ``all_gather`` and ``all_reduce``.

    Returns ``None`` if gloo is unavailable (the caller should fall back
    to the default group or skip the collective).
    """
    global _CPU_GLOO_GROUP
    if _CPU_GLOO_GROUP != "uninit":
        return _CPU_GLOO_GROUP
    try:
        _CPU_GLOO_GROUP = dist.new_group(backend="gloo")
    except Exception as exc:  # noqa: BLE001
        magi_logger.warning("get_cpu_gloo_group: gloo unavailable (%s); returning None", exc)
        _CPU_GLOO_GROUP = None
    return _CPU_GLOO_GROUP
