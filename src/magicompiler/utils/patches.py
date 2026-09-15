"""
Utility patches for monkey-patching PyTorch FSDP communication collectives.

MagiCompiler intercepts FSDP's all-gather, reduce-scatter, and all-reduce
operations to record them as first-class nodes in the computation graph,
enabling cross-boundary fusion and optimization.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# Track original implementations so we can restore them
_ORIGINAL_FUNCTIONS: Dict[str, Callable] = {}

# Collectives recorded by the graph capture in the current pass
_collective_records: List[dict] = []


def _record_collective(kind: str, *args: Any, **kwargs: Any) -> dict:
    """Record a collective operation for graph capture."""
    record = {
        "kind": kind,
        "tensor_shape": tuple(args[0].shape) if args else None,
        "tensor_dtype": args[0].dtype if args else None,
        "group": kwargs.get("group", None),
        "async_op": kwargs.get("async_op", False),
    }
    _collective_records.append(record)
    return record


def get_and_clear_records() -> List[dict]:
    """Retrieve and clear the collective records buffer."""
    global _collective_records
    records = list(_collective_records)
    _collective_records.clear()
    return records


# ── Patched collectives ──────────────────────────────────────────────


def _patched_all_gather(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    group: Any = None,
    async_op: bool = False,
    **kwargs: Any,
) -> Optional[torch.distributed.Work]:
    """Patched all_gather that records the collective for graph capture."""
    _record_collective("all_gather", output_tensor, input_tensor,
                        group=group, async_op=async_op)

    orig = _ORIGINAL_FUNCTIONS.get("all_gather")
    if orig is None:
        raise RuntimeError("Original all_gather not saved. Call patch_fsdp() first.")
    return orig(output_tensor, input_tensor, group=group, async_op=async_op, **kwargs)


def _patched_reduce_scatter(
    output: torch.Tensor,
    input_list: List[torch.Tensor],
    group: Any = None,
    async_op: bool = False,
    **kwargs: Any,
) -> Optional[torch.distributed.Work]:
    """Patched reduce_scatter that records the collective for graph capture."""
    _record_collective("reduce_scatter", output, input_list,
                        group=group, async_op=async_op)

    orig = _ORIGINAL_FUNCTIONS.get("reduce_scatter")
    if orig is None:
        raise RuntimeError("Original reduce_scatter not saved. Call patch_fsdp() first.")
    return orig(output, input_list, group=group, async_op=async_op, **kwargs)


def _patched_all_reduce(
    tensor: torch.Tensor,
    group: Any = None,
    async_op: bool = False,
    **kwargs: Any,
) -> Optional[torch.distributed.Work]:
    """Patched all_reduce that records the collective for graph capture."""
    _record_collective("all_reduce", tensor, group=group, async_op=async_op)

    orig = _ORIGINAL_FUNCTIONS.get("all_reduce")
    if orig is None:
        raise RuntimeError("Original all_reduce not saved. Call patch_fsdp() first.")
    return orig(tensor, group=group, async_op=async_op, **kwargs)


# ── Patch management ─────────────────────────────────────────────────



# Patch functions — debug logs removed from hot-path to avoid overhead
# on every collective call.


def patch_fsdp_all_gather() -> None:
    """Patch ``torch.distributed.all_gather`` for graph capture."""
    if "all_gather" not in _ORIGINAL_FUNCTIONS:
        _ORIGINAL_FUNCTIONS["all_gather"] = dist.all_gather
        dist.all_gather = _patched_all_gather  # type: ignore[assignment]


def patch_fsdp_reduce_scatter() -> None:
    """Patch ``torch.distributed.reduce_scatter`` for graph capture."""
    if "reduce_scatter" not in _ORIGINAL_FUNCTIONS:
        _ORIGINAL_FUNCTIONS["reduce_scatter"] = dist.reduce_scatter
        dist.reduce_scatter = _patched_reduce_scatter  # type: ignore[assignment]


def patch_fsdp_all_reduce() -> None:
    """Patch ``torch.distributed.all_reduce`` for graph capture."""
    if "all_reduce" not in _ORIGINAL_FUNCTIONS:
        _ORIGINAL_FUNCTIONS["all_reduce"] = dist.all_reduce
        dist.all_reduce = _patched_all_reduce  # type: ignore[assignment]


def unpatch_fsdp_all_gather() -> None:
    """Restore original ``torch.distributed.all_gather``."""
    if "all_gather" in _ORIGINAL_FUNCTIONS:
        dist.all_gather = _ORIGINAL_FUNCTIONS.pop("all_gather")


def unpatch_fsdp_reduce_scatter() -> None:
    """Restore original ``torch.distributed.reduce_scatter``."""
    if "reduce_scatter" in _ORIGINAL_FUNCTIONS:
        dist.reduce_scatter = _ORIGINAL_FUNCTIONS.pop("reduce_scatter")


def unpatch_fsdp_all_reduce() -> None:
    """Restore original ``torch.distributed.all_reduce``."""
    if "all_reduce" in _ORIGINAL_FUNCTIONS:
        dist.all_reduce = _ORIGINAL_FUNCTIONS.pop("all_reduce")
