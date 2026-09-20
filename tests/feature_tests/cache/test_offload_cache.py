# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Compile-cache reuse when ``graph_weight_offload`` is on.

The artifact bakes process-local host-pool slots.  A sidecar records which
weight each integer referred to; a later process remaps those integers onto
the slots it minted.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.fx as fx
import torch.nn as nn

from magi_compiler.config import CompileConfig
from magi_compiler.magi_backend.magi_backend import CompilerManager
from magi_compiler.passes.weight_offload.cache_slots import (
    collect_host_slot_table,
    match_host_slot_tables,
    refresh_resident_flags,
    slot_identity,
)
from magi_compiler.passes.weight_offload.node_meta import mark_host_slot

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture(autouse=True)
def clean_pool():
    from magi_compiler.passes.weight_offload import host_pool

    host_pool.reset()
    yield
    host_pool.reset()


def _info(name, shape, dtype="torch.bfloat16", nbytes=32, resident=False):
    return {"name": name, "shape": shape, "dtype": dtype, "nbytes": nbytes, "resident": resident}


def test_match_host_slot_tables_remaps_when_identities_agree():
    sidecar = {7: _info("layers.0.w", (8, 4)), 8: _info("layers.1.w", (4, 4))}
    current = {0: _info("layers.0.w", (8, 4)), 1: _info("layers.1.w", (4, 4))}
    assert match_host_slot_tables(sidecar, current) == {7: 0, 8: 1}


def test_match_host_slot_tables_rejects_a_name_set_mismatch():
    sidecar = {0: _info("layers.0.w", (8, 4))}
    current = {0: _info("layers.0.w", (8, 4)), 1: _info("layers.1.w", (4, 4))}
    assert match_host_slot_tables(sidecar, current) is None


def test_match_host_slot_tables_rejects_duplicate_identities():
    sidecar = {0: _info("tied", (2, 2)), 1: _info("tied", (2, 2))}
    current = {0: _info("tied", (2, 2))}
    assert match_host_slot_tables(sidecar, current) is None


def test_slot_identity_uses_name_and_layout():
    assert slot_identity(_info("w", (2, 2))) == ("w", (2, 2), "torch.bfloat16")
    assert slot_identity(_info("w", (2, 2))) != slot_identity(_info("w", (2, 3)))


@requires_cuda
def test_collect_and_refresh_read_the_live_pool():
    from magi_compiler.passes.weight_offload import host_pool

    local = torch.randn(16, 8, device="cuda", dtype=torch.bfloat16)
    host = host_pool.reserve(tuple(local.shape), local.dtype, name="layers.0.w")
    host.copy_(local.detach())
    local.untyped_storage().resize_(0)
    slot = host_pool.adopt(host, local, name="layers.0.w")

    g = fx.Graph()
    node = g.placeholder("L_self_modules_layers_0_parameters_weight_")
    mark_host_slot(node, slot)
    g.output((node,))
    gm = fx.GraphModule(nn.Module(), g)

    table = collect_host_slot_table(gm)
    assert set(table) == {slot}
    assert table[slot]["name"] == "layers.0.w"
    assert table[slot]["shape"] == (16, 8)
    assert table[slot]["resident"] is False

    host_pool.make_resident(slot)
    refreshed = refresh_resident_flags(table)
    assert refreshed[slot]["resident"] is True


def _offload_manager(tmp_path: Path) -> CompilerManager:
    conf = CompileConfig()
    conf.offload_config.graph_weight_offload = True
    conf.offload_config.host_first_materialize = True
    mgr = CompilerManager(conf)
    mgr.initialize_cache(tmp_path)
    return mgr


def _tagged_graph(slot: int) -> fx.GraphModule:
    g = fx.Graph()
    node = g.placeholder("w")
    mark_host_slot(node, slot)
    g.output((node,))
    return fx.GraphModule(nn.Module(), g)


@requires_cuda
def test_bind_offload_cache_matches_sidecar_and_replays_resident(tmp_path: Path):
    from magi_compiler.passes.weight_offload import host_pool

    local = torch.randn(8, 8, device="cuda", dtype=torch.bfloat16)
    host = host_pool.reserve(tuple(local.shape), local.dtype, name="w")
    host.copy_(local.detach())
    local.untyped_storage().resize_(0)
    slot = host_pool.adopt(host, local, name="w")

    sidecar = {
        99: {"name": "w", "shape": (8, 8), "dtype": str(local.dtype), "nbytes": host_pool.slot_bytes(slot), "resident": True}
    }
    mgr = _offload_manager(tmp_path)
    mgr.host_slots_path.write_text(repr(sidecar))
    mgr.bind_offload_cache(_tagged_graph(slot))

    assert mgr._offload_cache_ready
    assert mgr._offload_remap == {99: slot}
    assert host_pool.is_resident(slot)


@requires_cuda
def test_bind_offload_cache_misses_without_sidecar(tmp_path: Path):
    from magi_compiler.passes.weight_offload import host_pool

    local = torch.randn(8, 8, device="cuda", dtype=torch.bfloat16)
    host = host_pool.reserve(tuple(local.shape), local.dtype, name="w")
    host.copy_(local.detach())
    local.untyped_storage().resize_(0)
    slot = host_pool.adopt(host, local, name="w")

    mgr = _offload_manager(tmp_path)
    mgr.bind_offload_cache(_tagged_graph(slot))
    assert not mgr._offload_cache_ready
    assert mgr._offload_remap is None


@requires_cuda
def test_save_to_file_writes_host_slots_sidecar(tmp_path: Path):
    from magi_compiler.passes.weight_offload import host_pool

    local = torch.randn(8, 8, device="cuda", dtype=torch.bfloat16)
    host = host_pool.reserve(tuple(local.shape), local.dtype, name="w")
    host.copy_(local.detach())
    local.untyped_storage().resize_(0)
    slot = host_pool.adopt(host, local, name="w")

    mgr = _offload_manager(tmp_path)
    mgr.bind_offload_cache(_tagged_graph(slot))
    assert not mgr._offload_cache_ready
    mgr.save_to_file()

    raw = ast.literal_eval(mgr.host_slots_path.read_text())
    assert raw[slot]["name"] == "w"
    assert tuple(raw[slot]["shape"]) == (8, 8)
    assert raw[slot]["resident"] is False


@requires_cuda
def test_replay_does_not_store_a_mixed_slot_artifact(tmp_path: Path):
    from magi_compiler.magi_backend._cache_data_cls import CacheEntry, CacheHandle
    from magi_compiler.passes.weight_offload import host_pool

    local = torch.randn(8, 8, device="cuda", dtype=torch.bfloat16)
    host = host_pool.reserve(tuple(local.shape), local.dtype, name="w")
    host.copy_(local.detach())
    local.untyped_storage().resize_(0)
    slot = host_pool.adopt(host, local, name="w")

    sidecar = {
        3: {"name": "w", "shape": (8, 8), "dtype": str(local.dtype), "nbytes": host_pool.slot_bytes(slot), "resident": False}
    }
    mgr = _offload_manager(tmp_path)
    mgr.host_slots_path.write_text(repr(sidecar))
    mgr.bind_offload_cache(_tagged_graph(slot))
    assert mgr._offload_remap == {3: slot}

    stored = mgr._maybe_store_cache_entry(
        CacheEntry(None, 0, "inductor_standalone"), CacheHandle("k", str(tmp_path), 0), None, "k"
    )
    assert stored is False
    assert mgr.cache == {}


@requires_cuda
def test_offload_cache_reuse_across_processes(tmp_path: Path):
    """Bake in one process, replay in another under assert_cache_hit."""
    helper_path = Path(__file__).parent / "cache_reuse_helper" / "offload_cache_reuse_helper.py"
    cache_root = tmp_path / "cache"
    out_bake = tmp_path / "bake.json"
    out_verify = tmp_path / "verify.json"

    env = os.environ.copy()
    env["MAGI_LOGGING_LEVEL"] = "info"
    env["MAGI_COMPILE_CACHE_ROOT_DIR"] = str(cache_root)

    p_bake = subprocess.run(
        [sys.executable, str(helper_path), "--output", str(out_bake)], env=env, capture_output=True, text=True
    )
    assert p_bake.returncode == 0, f"bake failed\nstdout:\n{p_bake.stdout}\nstderr:\n{p_bake.stderr}"

    sidecars = list(cache_root.rglob("host_slots.py"))
    assert sidecars, "bake must persist host_slots.py next to subgraph_indices.py"
    sidecar = ast.literal_eval(sidecars[0].read_text())
    assert sidecar, "bake must record at least one offloaded weight"

    verify_env = {**env, "MAGI_COMPILE_ASSERT_CACHE_HIT": "1"}
    p_verify = subprocess.run(
        [sys.executable, str(helper_path), "--output", str(out_verify)], env=verify_env, capture_output=True, text=True
    )
    assert p_verify.returncode == 0, (
        f"verify (assert_cache_hit) failed under graph_weight_offload.\n" f"stderr:\n{p_verify.stderr}"
    )
    assert "compile cache reusable" in p_verify.stderr or "Directly load" in p_verify.stderr, (
        "verify should have remapped the sidecar and loaded the artifact\n" f"stderr:\n{p_verify.stderr}"
    )

    bake = json.loads(out_bake.read_text())
    verify = json.loads(out_verify.read_text())
    assert bake["num_bound"] >= 1
    assert verify["num_bound"] == bake["num_bound"]
    assert verify["shape"] == bake["shape"]
    assert abs(bake["sum"] - verify["sum"]) < 1e-2
