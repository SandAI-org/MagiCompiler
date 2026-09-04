"""
FSDP Hook Patching System.

MagiCompiler patches PyTorch's FSDP (Fully Sharded Data Parallel)
communication collectives (all-gather, reduce-scatter, all-reduce) to
intercept them and record them as first-class nodes in the computation
graph.

The patching works at the ``torch.distributed`` level: every call to
``dist.all_gather``, ``dist.reduce_scatter``, or ``dist.all_reduce``
inside an ``FSDPHookContext`` is recorded into a global buffer that the
graph capture engine reads via ``get_and_clear_records()``.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Iterator

import torch.nn as nn

from magicompiler.utils.patches import (
    patch_fsdp_all_gather,
    patch_fsdp_all_reduce,
    patch_fsdp_reduce_scatter,
    unpatch_fsdp_all_gather,
    unpatch_fsdp_all_reduce,
    unpatch_fsdp_reduce_scatter,
)

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def FSDPHookContext(model: nn.Module) -> Iterator[None]:
    """Context manager that temporarily patches ``torch.distributed``
    collectives (all-gather, reduce-scatter, all-reduce) for graph capture.

    The patched functions record each collective call into a global buffer
    that the graph capture engine reads via :func:`get_and_clear_records`.

    Usage::

        with FSDPHookContext(model):
            output = model(input_tensor)
            records = get_and_clear_records()
    """
    patch_fsdp()
    try:
        yield
    finally:
        unpatch_fsdp()


# ── Public API ───────────────────────────────────────────────────────


def patch_fsdp() -> None:
    """Apply all FSDP patches for MagiCompiler graph capture.

    Patches ``torch.distributed.all_gather``, ``reduce_scatter``,
    and ``all_reduce`` to record calls into a global buffer.
    """
    patch_fsdp_all_gather()
    patch_fsdp_reduce_scatter()
    patch_fsdp_all_reduce()
    logger.info("FSDP patches applied.")


def unpatch_fsdp() -> None:
    """Restore all original ``torch.distributed`` collective operations."""
    unpatch_fsdp_all_gather()
    unpatch_fsdp_reduce_scatter()
    unpatch_fsdp_all_reduce()
    logger.info("FSDP patches removed.")
