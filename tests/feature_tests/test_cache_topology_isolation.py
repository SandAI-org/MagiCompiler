# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Verify that different parallel topologies stay isolated, on disk and in memory.

Compiled artifacts have the ProcessGroup they traced with baked in, so a runtime that
changes CP/DP between calls -- adaptive DP reacting to queue depth -- must not reuse one
topology's artifacts under another. Two things have to be keyed by topology for that:
the cache directory, and the compile state held on the instance.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from magi_compiler._api import get_attr_name_for_state
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


class TestCompileStateIsolation:
    """The compile state on an instance is the in-memory counterpart of the cache dir.

    It owns the bytecode and AOT artifacts that the fast paths in _run_orchestration replay
    directly, without going through dynamo's guards -- so if two topologies share one state,
    the second silently runs the first one's graph, on the first one's process group.
    """

    def test_a_shared_state_is_what_lets_one_graph_serve_two_topologies(self):
        """The failure mode: with the key pinned, both topologies land on one attribute."""
        with patch.dict(os.environ, {"MAGI_COMPILE_TOPOLOGY_KEY": "pinned"}):
            under_cp8 = get_attr_name_for_state("forward")
            under_cp4 = get_attr_name_for_state("forward")
        assert under_cp8 == under_cp4

    def test_each_topology_gets_its_own_state(self):
        with patch.dict(os.environ, {"MAGI_COMPILE_TOPOLOGY_KEY": "cp8_dp1"}):
            under_cp8 = get_attr_name_for_state("forward")
        with patch.dict(os.environ, {"MAGI_COMPILE_TOPOLOGY_KEY": "cp4_dp2"}):
            under_cp4 = get_attr_name_for_state("forward")
        assert under_cp8 != under_cp4
        assert "cp8_dp1" in under_cp8 and "cp4_dp2" in under_cp4

    def test_a_single_fixed_topology_keeps_the_plain_name(self):
        """No key means no adaptive runtime, and the attribute must not change shape."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MAGI_COMPILE_TOPOLOGY_KEY", None)
            assert get_attr_name_for_state("forward") == "_magi_state_for_forward"

    def test_the_topology_at_call_time_decides_not_the_one_at_wrap_time(self):
        """The regression this guards.

        The attribute used to be resolved once while wrapping the instance, so a model built
        under cp=8 kept reaching for the cp=8 state no matter what the runtime switched to
        afterwards -- which made keying by topology inert.
        """
        from magi_compiler._api import _magi_compile_bound_method

        reached = []

        def fake_init(holder, target, dims, conf, tag, method, state_attr):
            reached.append(state_attr)
            setattr(
                holder,
                state_attr,
                SimpleNamespace(
                    compile_config=SimpleNamespace(offload_config=SimpleNamespace(model_cpu_offload=False)),
                    jit_compiled_code=None,
                ),
            )

        class Probe(torch.nn.Module):
            def forward(self, x):
                return x

        probe = Probe()
        with patch.dict(os.environ, {"MAGI_COMPILE_TOPOLOGY_KEY": "cp8_dp1"}):
            _magi_compile_bound_method(probe, {"x": 0}, SimpleNamespace(), "probe", method_name="forward")

        with patch.dict(os.environ, {"MAGI_COMPILE_TOPOLOGY_KEY": "cp4_dp2"}):
            with patch("magi_compiler._api._lazy_init_magi_state", side_effect=fake_init):
                with patch("magi_compiler._api._run_orchestration", return_value=None):
                    probe.forward(torch.zeros(2))

        assert reached == ["_magi_state_for_forward__cp4_dp2"], reached
