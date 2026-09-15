"""Tests for MagiCompiler patch utilities."""

import pytest
import torch
import torch.distributed as dist

from magicompiler.utils.patches import (
    patch_fsdp_all_gather,
    patch_fsdp_reduce_scatter,
    patch_fsdp_all_reduce,
    unpatch_fsdp_all_gather,
    unpatch_fsdp_reduce_scatter,
    unpatch_fsdp_all_reduce,
    get_and_clear_records,
)


# We don't initialize dist in unit tests, but we can still
# test that patches are applied and reverted correctly.


class TestPatchesAPI:
    """Test that patch APIs work correctly without distributed init."""

    def test_patch_unpatch_all_gather(self) -> None:
        """Patches should be applied and then removed cleanly."""
        original_fn = dist.all_gather
        patch_fsdp_all_gather()
        assert dist.all_gather is not original_fn
        unpatch_fsdp_all_gather()
        assert dist.all_gather is original_fn

    def test_patch_unpatch_reduce_scatter(self) -> None:
        original_fn = dist.reduce_scatter
        patch_fsdp_reduce_scatter()
        assert dist.reduce_scatter is not original_fn
        unpatch_fsdp_reduce_scatter()
        assert dist.reduce_scatter is original_fn

    def test_patch_unpatch_all_reduce(self) -> None:
        original_fn = dist.all_reduce
        patch_fsdp_all_reduce()
        assert dist.all_reduce is not original_fn
        unpatch_fsdp_all_reduce()
        assert dist.all_reduce is original_fn

    def test_idempotent_patch(self) -> None:
        """Patching twice should not break anything."""
        patch_fsdp_all_gather()
        patch_fsdp_all_gather()  # second patch should be a no-op
        unpatch_fsdp_all_gather()
        unpatch_fsdp_all_gather()  # second unpatch should be a no-op

    def test_get_and_clear_records(self) -> None:
        """Records should be cleared after retrieval."""
        records = get_and_clear_records()
        assert isinstance(records, list)
        # After clear, should be empty again
        records2 = get_and_clear_records()
        assert len(records2) == 0


class TestCollectiveRecords:
    """Test that collective recordings work."""

    def test_record_structure(self) -> None:
        """Manually created records should have the expected structure."""
        from magicompiler.utils.patches import _collective_records

        _collective_records.clear()
        _collective_records.append({
            "kind": "all_gather",
            "tensor_shape": (32, 64),
            "tensor_dtype": torch.float32,
            "group": None,
        })

        records = get_and_clear_records()
        assert len(records) == 1
        assert records[0]["kind"] == "all_gather"
        assert records[0]["tensor_shape"] == (32, 64)
