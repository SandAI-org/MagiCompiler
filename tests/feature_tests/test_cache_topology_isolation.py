# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Verify that different parallel topologies produce isolated cache directories,
preventing EP/CP cross-contamination of compiled artifacts."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import torch

from magi_compiler.config import _get_parallel_topology, magi_cache_dump_path, model_rank_dir_name


def _patch_dist(is_init=False, rank=0, world_size=1):
    """Context manager to mock torch.distributed state."""
    return (
        patch.object(torch.distributed, "is_initialized", return_value=is_init),
        patch.object(torch.distributed, "get_rank", return_value=rank),
        patch.object(torch.distributed, "get_world_size", return_value=world_size),
    )


class TestTopologyCacheIsolation:
    """Different topologies must map to different cache directory paths."""

    def test_model_rank_dir_includes_topology(self):
        p1, p2, p3 = _patch_dist(is_init=False)
        with p1, p2, p3:
            with patch("magi_compiler.config._get_parallel_topology", return_value="ep6_cp1"):
                name_a = model_rank_dir_name(0, None)
            with patch("magi_compiler.config._get_parallel_topology", return_value="ep8_cp1"):
                name_b = model_rank_dir_name(0, None)
        assert name_a != name_b
        assert "ep6_cp1" in name_a
        assert "ep8_cp1" in name_b

    def test_model_rank_dir_with_tag(self):
        p1, p2, p3 = _patch_dist(is_init=False)
        with p1, p2, p3:
            with patch("magi_compiler.config._get_parallel_topology", return_value="ws4"):
                name = model_rank_dir_name(1, "sr")
        assert "model_1_sr_rank_" in name
        assert "ws4" in name

    def test_magi_cache_dump_path_varies_with_topology(self, tmp_path: Path):
        p1, p2, p3 = _patch_dist(is_init=False)
        with p1, p2, p3:
            with patch("magi_compiler.config._get_parallel_topology", return_value="ep6_cp1"):
                path_a = magi_cache_dump_path(str(tmp_path), 0)
            with patch("magi_compiler.config._get_parallel_topology", return_value="ep8_cp1"):
                path_b = magi_cache_dump_path(str(tmp_path), 0)
        assert path_a != path_b
        assert path_a.parent == path_b.parent
        assert "ep6_cp1" in str(path_a)
        assert "ep8_cp1" in str(path_b)

    def test_no_dist_defaults_to_ws1(self):
        p1, p2, p3 = _patch_dist(is_init=False)
        with p1, p2, p3:
            topo = _get_parallel_topology()
        assert topo == "ws1"

    def test_dist_without_env_uses_world_size(self):
        """With dist but no MAGI_COMPILE_TOPOLOGY_KEY, falls back to ws{world_size}."""
        p1, p2, p3 = _patch_dist(is_init=True, rank=0, world_size=8)
        with p1, p2, p3, patch.dict("os.environ", {}, clear=False):
            # Ensure env var is NOT set
            import os

            os.environ.pop("MAGI_COMPILE_TOPOLOGY_KEY", None)
            topo = _get_parallel_topology()
            name = model_rank_dir_name(0, None)
        assert topo == "ws8"
        assert "ws8" in name

    def test_env_var_takes_priority(self):
        """MAGI_COMPILE_TOPOLOGY_KEY env var overrides all other resolution."""
        p1, p2, p3 = _patch_dist(is_init=True, rank=0, world_size=8)
        with p1, p2, p3, patch.dict("os.environ", {"MAGI_COMPILE_TOPOLOGY_KEY": "ep6_cp2_dp1"}):
            topo = _get_parallel_topology()
            name = model_rank_dir_name(0, None)
        assert topo == "ep6_cp2_dp1"
        assert "ep6_cp2_dp1" in name

    def test_same_topology_same_path(self):
        p1, p2, p3 = _patch_dist(is_init=False)
        with p1, p2, p3:
            with patch("magi_compiler.config._get_parallel_topology", return_value="ep6_cp2"):
                name1 = model_rank_dir_name(0, None)
                name2 = model_rank_dir_name(0, None)
        assert name1 == name2
