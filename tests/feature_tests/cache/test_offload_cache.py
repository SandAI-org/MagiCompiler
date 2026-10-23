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
from magi_compiler.passes.weight_offload.cache.offload_cache import OffloadCache
from magi_compiler.passes.weight_offload.cache.slot_table import (
    collect_host_slot_table,
    match_host_slot_tables,
    refresh_resident_flags,
    slot_identity,
)
from magi_compiler.passes.weight_offload.node_meta import mark_host_slot

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture(autouse=True)
def reset_host_weight_store():
    from magi_compiler.passes.weight_offload import get_global_host_weight_store

    get_global_host_weight_store().reset()
    yield
    get_global_host_weight_store().reset()


def _slot_metadata(name, shape, dtype="torch.bfloat16", nbytes=32, resident=False):
    return {"name": name, "shape": shape, "dtype": dtype, "nbytes": nbytes, "resident": resident}


def test_match_host_slot_tables_remaps_when_identities_agree():
    sidecar = {7: _slot_metadata("layers.0.w", (8, 4)), 8: _slot_metadata("layers.1.w", (4, 4))}
    current = {0: _slot_metadata("layers.0.w", (8, 4)), 1: _slot_metadata("layers.1.w", (4, 4))}
    assert match_host_slot_tables(sidecar, current) == {7: 0, 8: 1}


def test_match_host_slot_tables_rejects_a_name_set_mismatch():
    sidecar = {0: _slot_metadata("layers.0.w", (8, 4))}
    current = {0: _slot_metadata("layers.0.w", (8, 4)), 1: _slot_metadata("layers.1.w", (4, 4))}
    assert match_host_slot_tables(sidecar, current) is None


def test_match_host_slot_tables_rejects_duplicate_identities():
    sidecar = {0: _slot_metadata("tied", (2, 2)), 1: _slot_metadata("tied", (2, 2))}
    current = {0: _slot_metadata("tied", (2, 2))}
    assert match_host_slot_tables(sidecar, current) is None


def test_slot_identity_uses_name_and_layout():
    assert slot_identity(_slot_metadata("w", (2, 2))) == ("w", (2, 2), "torch.bfloat16")
    assert slot_identity(_slot_metadata("w", (2, 2))) != slot_identity(_slot_metadata("w", (2, 3)))


@requires_cuda
def test_collect_and_refresh_read_the_live_pool():
    from magi_compiler.passes.weight_offload import get_global_host_weight_store

    device_standin = torch.randn(16, 8, device="cuda", dtype=torch.bfloat16)
    host_buffer = get_global_host_weight_store().reserve_host_buffer(
        tuple(device_standin.shape), device_standin.dtype, parameter_name="layers.0.w"
    )
    host_buffer.copy_(device_standin.detach())
    device_standin.untyped_storage().resize_(0)
    slot = get_global_host_weight_store().bind_host_buffer(host_buffer, device_standin, parameter_name="layers.0.w")

    g = fx.Graph()
    node = g.placeholder("L_self_modules_layers_0_parameters_weight_")
    mark_host_slot(node, slot)
    g.output((node,))
    gm = fx.GraphModule(nn.Module(), g)

    root = nn.Module()
    root.graph = object()  # An unrelated attribute must not hide the nested FX graph.
    root.add_module("compiled", gm)
    table = collect_host_slot_table(root)
    assert table == collect_host_slot_table(gm)
    assert set(table) == {slot}
    assert table[slot]["name"] == "layers.0.w"
    assert table[slot]["shape"] == (16, 8)
    assert table[slot]["resident"] is False

    get_global_host_weight_store().make_weight_resident(slot)
    refreshed = refresh_resident_flags(table)
    assert refreshed[slot]["resident"] is True


def test_disabled_offload_cache_is_a_noop(tmp_path: Path):
    cache = OffloadCache(enabled=False)
    cache.initialize(tmp_path)

    assert cache.can_replay_artifact()
    assert cache.can_store_artifact()
    assert cache._cached_to_current_slots is None
    cache.save_slot_sidecar()
    assert cache.host_slots_path is not None
    assert not cache.host_slots_path.exists()


def _make_offload_cache(tmp_path: Path) -> OffloadCache:
    cache = OffloadCache(enabled=True)
    cache.initialize(tmp_path)
    return cache


def _make_offload_manager(tmp_path: Path) -> CompilerManager:
    conf = CompileConfig()
    conf.offload_config.graph_weight_offload = True
    compiler_manager = CompilerManager(conf)
    compiler_manager.initialize_cache(tmp_path)
    return compiler_manager


def _make_slot_tagged_graph(slot: int) -> fx.GraphModule:
    g = fx.Graph()
    node = g.placeholder("w")
    mark_host_slot(node, slot)
    g.output((node,))
    return fx.GraphModule(nn.Module(), g)


def _park_test_weight(name: str = "w", shape=(8, 8)):
    from magi_compiler.passes.weight_offload import get_global_host_weight_store

    device_standin = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    host_buffer = get_global_host_weight_store().reserve_host_buffer(
        tuple(device_standin.shape), device_standin.dtype, parameter_name=name
    )
    host_buffer.copy_(device_standin.detach())
    device_standin.untyped_storage().resize_(0)
    slot = get_global_host_weight_store().bind_host_buffer(host_buffer, device_standin, parameter_name=name)
    return slot, device_standin


@requires_cuda
def test_bind_matches_sidecar_and_replays_resident(tmp_path: Path):
    from magi_compiler.passes.weight_offload import get_global_host_weight_store

    slot, device_standin = _park_test_weight()
    sidecar = {
        99: {
            "name": "w",
            "shape": (8, 8),
            "dtype": str(device_standin.dtype),
            "nbytes": get_global_host_weight_store().slot_bytes(slot),
            "resident": True,
        }
    }
    cache = _make_offload_cache(tmp_path)
    cache.host_slots_path.write_text(repr(sidecar))

    assert cache.prepare_graph_replay(_make_slot_tagged_graph(slot)) is None
    assert cache.can_replay_artifact()
    assert not cache.can_store_artifact()
    assert cache._cached_to_current_slots == {99: slot}
    assert get_global_host_weight_store().is_resident(slot)


@requires_cuda
def test_bind_misses_without_sidecar(tmp_path: Path):
    slot, _ = _park_test_weight()
    cache = _make_offload_cache(tmp_path)

    assert "sidecar missing" in cache.prepare_graph_replay(_make_slot_tagged_graph(slot))
    assert not cache.can_replay_artifact()
    assert cache.can_store_artifact()
    assert cache._cached_to_current_slots is None


@requires_cuda
def test_save_writes_host_slots_sidecar(tmp_path: Path):
    slot, _ = _park_test_weight()
    cache = _make_offload_cache(tmp_path)
    assert cache.prepare_graph_replay(_make_slot_tagged_graph(slot)) is not None
    cache.save_slot_sidecar()

    raw = ast.literal_eval(cache.host_slots_path.read_text())
    assert raw[slot]["name"] == "w"
    assert tuple(raw[slot]["shape"]) == (8, 8)
    assert raw[slot]["resident"] is False


@requires_cuda
def test_replay_does_not_rewrite_sidecar(tmp_path: Path):
    from magi_compiler.passes.weight_offload import get_global_host_weight_store

    slot, device_standin = _park_test_weight()
    sidecar = {
        3: {
            "name": "w",
            "shape": (8, 8),
            "dtype": str(device_standin.dtype),
            "nbytes": get_global_host_weight_store().slot_bytes(slot),
            "resident": False,
        }
    }
    cache = _make_offload_cache(tmp_path)
    cache.host_slots_path.write_text(repr(sidecar))
    assert cache.prepare_graph_replay(_make_slot_tagged_graph(slot)) is None
    cache.save_slot_sidecar()
    assert ast.literal_eval(cache.host_slots_path.read_text()) == sidecar


@requires_cuda
def test_bind_miss_drops_compile_indices(tmp_path: Path):
    from magi_compiler.magi_backend._cache_data_cls import CacheEntry, CacheHandle

    slot, _ = _park_test_weight()
    compiler_manager = _make_offload_manager(tmp_path)
    compiler_manager.cache[CacheEntry(None, 0, "inductor_standalone")] = CacheHandle("k", str(tmp_path), 0)
    compiler_manager.bind_offload_cache(_make_slot_tagged_graph(slot))

    assert compiler_manager.cache == {}
    assert not compiler_manager.offload_cache.can_replay_artifact()
    assert compiler_manager.offload_cache._cached_to_current_slots is None


@requires_cuda
def test_bind_hit_keeps_compile_indices(tmp_path: Path):
    from magi_compiler.magi_backend._cache_data_cls import CacheEntry, CacheHandle
    from magi_compiler.passes.weight_offload import get_global_host_weight_store

    slot, device_standin = _park_test_weight()
    sidecar = {
        3: {
            "name": "w",
            "shape": (8, 8),
            "dtype": str(device_standin.dtype),
            "nbytes": get_global_host_weight_store().slot_bytes(slot),
            "resident": False,
        }
    }
    compiler_manager = _make_offload_manager(tmp_path)
    entry = CacheEntry(None, 0, "inductor_standalone")
    compiler_manager.cache[entry] = CacheHandle("k", str(tmp_path), 0)
    compiler_manager.offload_cache.host_slots_path.write_text(repr(sidecar))
    compiler_manager.bind_offload_cache(_make_slot_tagged_graph(slot))

    assert entry in compiler_manager.cache
    assert compiler_manager.offload_cache.can_replay_artifact()


@requires_cuda
def test_replay_does_not_store_a_mixed_slot_artifact(tmp_path: Path):
    from magi_compiler.magi_backend._cache_data_cls import CacheEntry, CacheHandle
    from magi_compiler.passes.weight_offload import get_global_host_weight_store

    slot, device_standin = _park_test_weight()
    sidecar = {
        3: {
            "name": "w",
            "shape": (8, 8),
            "dtype": str(device_standin.dtype),
            "nbytes": get_global_host_weight_store().slot_bytes(slot),
            "resident": False,
        }
    }
    compiler_manager = _make_offload_manager(tmp_path)
    compiler_manager.offload_cache.host_slots_path.write_text(repr(sidecar))
    compiler_manager.bind_offload_cache(_make_slot_tagged_graph(slot))
    assert compiler_manager.offload_cache._cached_to_current_slots == {3: slot}
    assert not compiler_manager.offload_cache.can_store_artifact()

    stored = compiler_manager._maybe_store_cache_entry(
        CacheEntry(None, 0, "inductor_standalone"), CacheHandle("k", str(tmp_path), 0), None, "k"
    )
    assert stored is False
    assert compiler_manager.cache == {}


@requires_cuda
def test_wrap_loaded_applies_slot_remap(tmp_path: Path):
    from magi_compiler.passes.weight_offload import get_global_host_weight_store
    from magi_compiler.passes.weight_offload.runtime import slot_remap

    slot, device_standin = _park_test_weight()
    sidecar = {
        99: {
            "name": "w",
            "shape": (8, 8),
            "dtype": str(device_standin.dtype),
            "nbytes": get_global_host_weight_store().slot_bytes(slot),
            "resident": False,
        }
    }
    cache = _make_offload_cache(tmp_path)
    cache.host_slots_path.write_text(repr(sidecar))
    cache.prepare_graph_replay(_make_slot_tagged_graph(slot))

    seen: list[int] = []

    def compiled():
        seen.append(slot_remap.resolve_runtime_host_slot(99))
        return "ok"

    assert cache.wrap_with_slot_remap(compiled)() == "ok"
    assert seen == [slot]


@requires_cuda
def test_save_to_file_writes_host_slots_sidecar(tmp_path: Path):
    slot, _ = _park_test_weight()
    compiler_manager = _make_offload_manager(tmp_path)
    compiler_manager.bind_offload_cache(_make_slot_tagged_graph(slot))
    compiler_manager.save_to_file()

    raw = ast.literal_eval(compiler_manager.offload_cache.host_slots_path.read_text())
    assert raw[slot]["name"] == "w"
    assert tuple(raw[slot]["shape"]) == (8, 8)
    assert raw[slot]["resident"] is False


@requires_cuda
def test_bind_miss_raises_when_assert_cache_hit(tmp_path: Path):
    slot, _ = _park_test_weight()
    conf = CompileConfig()
    conf.offload_config.graph_weight_offload = True
    conf.assert_cache_hit = True
    compiler_manager = CompilerManager(conf)
    compiler_manager.initialize_cache(tmp_path)
    with pytest.raises(RuntimeError, match="host_slots sidecar"):
        compiler_manager.bind_offload_cache(_make_slot_tagged_graph(slot))


@requires_cuda
def test_offload_cache_reuse_across_processes(tmp_path: Path):
    """Bake in one process, replay in another under assert_cache_hit."""
    helper_path = Path(__file__).parent / "cache_reuse_helper" / "offload_cache_reuse_helper.py"
    cache_root = tmp_path / "cache"
    cold_run_report_path = tmp_path / "bake.json"
    replay_report_path = tmp_path / "verify.json"

    env = os.environ.copy()
    env["MAGI_LOGGING_LEVEL"] = "info"
    env["MAGI_COMPILE_CACHE_ROOT_DIR"] = str(cache_root)

    cold_run_result = subprocess.run(
        [sys.executable, str(helper_path), "--output", str(cold_run_report_path)], env=env, capture_output=True, text=True
    )
    assert (
        cold_run_result.returncode == 0
    ), f"bake failed\nstdout:\n{cold_run_result.stdout}\nstderr:\n{cold_run_result.stderr}"

    sidecars = list(cache_root.rglob("host_slots.py"))
    assert sidecars, "bake must persist host_slots.py next to subgraph_indices.py"
    sidecar = ast.literal_eval(sidecars[0].read_text())
    assert sidecar, "bake must record at least one offloaded weight"

    verify_env = {**env, "MAGI_COMPILE_ASSERT_CACHE_HIT": "1"}
    replay_result = subprocess.run(
        [sys.executable, str(helper_path), "--output", str(replay_report_path)], env=verify_env, capture_output=True, text=True
    )
    assert replay_result.returncode == 0, (
        f"verify (assert_cache_hit) failed under graph_weight_offload.\n" f"stderr:\n{replay_result.stderr}"
    )
    assert "compile cache reusable" in replay_result.stderr or "Directly load" in replay_result.stderr, (
        "verify should have remapped the sidecar and loaded the artifact\n" f"stderr:\n{replay_result.stderr}"
    )

    cold_run_report = json.loads(cold_run_report_path.read_text())
    replay_report = json.loads(replay_report_path.read_text())
    assert cold_run_report["weight_count"] >= 1
    assert replay_report["weight_count"] == cold_run_report["weight_count"]
    assert replay_report["shape"] == cold_run_report["shape"]
    assert abs(cold_run_report["sum"] - replay_report["sum"]) < 1e-2
