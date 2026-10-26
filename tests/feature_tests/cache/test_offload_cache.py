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


@pytest.mark.parametrize(
    "sidecar",
    [
        {0: {"name": "w"}},
        {0: _slot_metadata("w", None)},
        {0: _slot_metadata("w", (2, -1))},
        {0: _slot_metadata("w", (2, 2), dtype=None)},
        {0: _slot_metadata("w", (2, 2), nbytes="32")},
        {0: _slot_metadata("w", (2, 2), resident="False")},
        {0: _slot_metadata("w", (2, 2)), "0": _slot_metadata("other", (2, 2))},
    ],
    ids=[
        "missing-field",
        "missing-shape",
        "negative-dimension",
        "invalid-dtype",
        "invalid-bytes",
        "invalid-resident",
        "duplicate-slot",
    ],
)
def test_invalid_sidecar_entries_are_cache_misses(tmp_path, sidecar):
    """Malformed fields must miss before slot matching or restoring any resident storage."""
    cache = _make_offload_cache(tmp_path)
    cache.host_slots_path.write_text(repr(sidecar))
    assert cache.prepare_graph_replay(fx.symbolic_trace(nn.Identity())) == "offload host_slots sidecar missing"
    assert not cache.can_replay_artifact()


def test_sidecar_normalizes_slot_ids_and_shapes_at_load(tmp_path):
    cache = _make_offload_cache(tmp_path)
    metadata = _slot_metadata("w", [2, 2])
    del metadata["resident"]
    cache.host_slots_path.write_text(repr({"7": metadata}))
    assert cache._load_sidecar() == {7: _slot_metadata("w", (2, 2))}


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


@pytest.mark.parametrize("replace_fails", [False, True])
def test_sidecar_replacement_preserves_old_contents_until_commit(tmp_path, monkeypatch, replace_fails):
    """A failed sidecar update must leave the previous file intact and remove the temporary file."""
    cache = _make_offload_cache(tmp_path)
    previous_contents = repr({7: _slot_metadata("previous_weight", (2, 2))})
    cache.host_slots_path.write_text(previous_contents)
    replace_path = Path.replace

    def inspect_replacement(source, destination):
        assert source.parent == destination.parent
        assert ast.literal_eval(source.read_text()) == {}
        assert destination.read_text() == previous_contents
        if replace_fails:
            raise OSError("simulated replacement failure")
        return replace_path(source, destination)

    monkeypatch.setattr(Path, "replace", inspect_replacement)
    if replace_fails:
        with pytest.raises(OSError, match="simulated replacement failure"):
            cache.save_slot_sidecar()
        assert cache.host_slots_path.read_text() == previous_contents
    else:
        cache.save_slot_sidecar()
        assert ast.literal_eval(cache.host_slots_path.read_text()) == {}
    assert list(tmp_path.iterdir()) == [cache.host_slots_path]


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
@pytest.mark.parametrize("peer_sidecar_miss", [False, True])
def test_residents_restore_only_after_all_required_sidecars_match(tmp_path: Path, monkeypatch, peer_sidecar_miss):
    """A peer's miss must veto residency restoration and slot remapping even when this sidecar matches."""
    from magi_compiler.passes.transfer_overlap import rank_sync
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

    def any_rank_missed(local_miss):
        assert not local_miss
        return peer_sidecar_miss

    monkeypatch.setattr(rank_sync, "any_rank_has_flag", any_rank_missed)
    reason = cache.prepare_graph_replay(_make_slot_tagged_graph(slot), sync_across_ranks=True)
    assert (reason is not None) == peer_sidecar_miss
    assert cache.can_replay_artifact() == (not peer_sidecar_miss)
    assert cache.can_store_artifact() == peer_sidecar_miss
    assert cache._cached_to_current_slots == (None if peer_sidecar_miss else {99: slot})
    assert get_global_host_weight_store().is_resident(slot) == (not peer_sidecar_miss)


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
@pytest.mark.parametrize("sidecar_failure", [None, "missing", "mismatch"])
def test_offload_cache_reuse_across_processes(tmp_path: Path, sidecar_failure: str | None):
    """A sidecar miss must rebuild residency despite warm Torch caches, then allow artifact replay."""
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
    assert any(metadata["resident"] for metadata in sidecar.values())
    cold_run_report = json.loads(cold_run_report_path.read_text())
    assert cold_run_report["resident_bytes"] > 0

    if sidecar_failure is not None:
        if sidecar_failure == "missing":
            sidecars[0].unlink()
        else:
            mismatched_sidecar = {slot: {**metadata, "name": "different_weight"} for slot, metadata in sidecar.items()}
            sidecars[0].write_text(repr(mismatched_sidecar))
        rebuild_report_path = tmp_path / "rebuild.json"
        rebuild_result = subprocess.run(
            [sys.executable, str(helper_path), "--output", str(rebuild_report_path)], env=env, capture_output=True, text=True
        )
        assert (
            rebuild_result.returncode == 0
        ), f"rebuild failed\nstdout:\n{rebuild_result.stdout}\nstderr:\n{rebuild_result.stderr}"
        rebuild_report = json.loads(rebuild_report_path.read_text())
        assert rebuild_report["resident_bytes"] == cold_run_report["resident_bytes"]
        assert rebuild_report["fx_graph_cache_hits"] == 0
        assert rebuild_report["autograd_cache_hits"] == 0
        assert rebuild_report["fx_graph_cache_misses"] > 0
        rebuilt_sidecar = ast.literal_eval(sidecars[0].read_text())
        assert rebuilt_sidecar == sidecar

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

    replay_report = json.loads(replay_report_path.read_text())
    assert replay_report["resident_bytes"] == cold_run_report["resident_bytes"]
    assert cold_run_report["weight_count"] >= 1
    assert replay_report["weight_count"] == cold_run_report["weight_count"]
    assert replay_report["shape"] == cold_run_report["shape"]
    assert abs(cold_run_report["sum"] - replay_report["sum"]) < 1e-2


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize("sidecar_failure", ["missing", "corrupt"])
def test_fsdp_sidecar_miss_recompiles_every_rank(tmp_path: Path, sidecar_failure: str):
    """One rank's sidecar failure must recompile all FSDP ranks and produce reusable artifacts."""
    helper_path = Path(__file__).parent / "cache_reuse_helper" / "offload_cache_reuse_helper.py"
    cache_root = tmp_path / "cache"
    env = {**os.environ, "MAGI_LOGGING_LEVEL": "info", "MAGI_COMPILE_CACHE_ROOT_DIR": str(cache_root)}

    def run_phase(phase: str, *, assert_cache_hit: bool = False) -> list[dict]:
        phase_env = {**env, "MAGI_COMPILE_ASSERT_CACHE_HIT": str(int(assert_cache_hit))}
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=2",
                str(helper_path),
                "--fsdp",
                "--output",
                str(tmp_path / f"{phase}.json"),
            ],
            env=phase_env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, f"{phase} failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        return [json.loads((tmp_path / f"{phase}.rank{rank}.json").read_text()) for rank in range(2)]

    cold_reports = run_phase("cold")
    assert all(report["weight_count"] > 0 and report["artifacts_saved"] > 0 for report in cold_reports)
    [rank1_sidecar] = cache_root.glob("magi_cache/*_rank_1_*/**/host_slots.py")
    if sidecar_failure == "missing":
        rank1_sidecar.unlink()
    else:
        rank1_sidecar.write_text("{0: {'name':")

    rebuilt_reports = run_phase("rebuilt")
    for report in rebuilt_reports:
        assert report["inductor_compiles"] > 0 and report["artifacts_saved"] > 0
        assert report["fx_graph_cache_misses"] > 0
        assert report["fx_graph_cache_hits"] == report["autograd_cache_hits"] == 0

    replay_reports = run_phase("replay", assert_cache_hit=True)
    for cold, rebuilt, replay in zip(cold_reports, rebuilt_reports, replay_reports):
        assert replay["inductor_compiles"] == 0
        assert replay["resident_bytes"] == rebuilt["resident_bytes"] == cold["resident_bytes"] > 0
        assert rebuilt["sum"] == replay["sum"] == cold["sum"]
