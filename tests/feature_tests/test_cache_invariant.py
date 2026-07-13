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

"""Cache invariant tests: every entry in ``CompilerManager.cache`` must have a
valid on-disk artifact directory.  These tests reproduce three classes of stale-
cache bugs present on ``origin/main`` (before fix).

Each test constructs a ``CompilerManager`` with known-bad state and asserts the
invariant that the code *should* maintain.  On unfixed code they FAIL.
"""

from __future__ import annotations

import pprint
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.fx as fx

from magi_compiler.config import CompileConfig
from magi_compiler.magi_backend._cache_data_cls import CacheEntry, CacheHandle
from magi_compiler.magi_backend.magi_backend import CompilerManager


@pytest.fixture(autouse=True)
def cleanup_cache():
    """Override the global autouse cleanup_cache fixture from conftest.py.

    These tests use tmp_path and don't need global cache cleanup, which can
    hang when cache_root_dir points to a large or network-mounted directory.
    """
    yield


def _make_cm(tmp_path: Path) -> CompilerManager:
    saved_argv = sys.argv
    try:
        sys.argv = ["test"]
        cfg = CompileConfig(cache_root_dir=str(tmp_path), backend="eager")
    finally:
        sys.argv = saved_argv
    return CompilerManager(cfg)


def _write_index(cache_dir: Path, data: dict) -> None:
    (cache_dir / "subgraph_indices.py").write_text(pprint.pformat(data))


def _dummy_graph() -> fx.GraphModule:
    return fx.GraphModule(torch.nn.Module(), fx.Graph())


# ---------------------------------------------------------------------------
# Bug 1: load() does not check whether artifact dir exists on disk.
#
# On origin/main, load() blindly passes the handle to compiler.load() which
# calls CompiledArtifact.load() on a non-existent path → crash.
#
# Expected after fix: load() returns None and removes the stale entry.
# ---------------------------------------------------------------------------


def test_load_rejects_missing_artifact_dir(tmp_path: Path) -> None:
    cache_dir = tmp_path / "model_bug1"
    cache_dir.mkdir(parents=True)
    bogus = cache_dir / "artifact_shape_None_subgraph_0"
    assert not bogus.exists()

    data = {(None, 0, "inductor_standalone"): ("artifact_shape_None_subgraph_0", str(bogus), 0)}
    _write_index(cache_dir, data)

    cm = _make_cm(tmp_path)
    cm.initialize_cache(cache_dir)

    entry = CacheEntry(None, 0, "inductor_standalone")
    assert entry in cm.cache, "precondition: entry loaded from index"

    # Patch compiler.load so that if CompilerManager.load reaches it we know
    # the guard is missing (it should never be called for a missing dir).
    with patch.object(cm.compiler, "load", side_effect=AssertionError("compiler.load should not be called")):
        result = cm.load(_dummy_graph(), [], entry)

    assert result is None, "load() must return None for missing artifact dir"
    assert entry not in cm.cache, "stale entry must be removed from cache"


# ---------------------------------------------------------------------------
# Bug 2: _maybe_store_cache_entry(entry, None, ...) does not remove a stale
# handle that was rehydrated from subgraph_indices.py.
#
# On origin/main, it just returns False; the ghost handle survives in memory
# and save_to_file() will re-serialize it.
#
# Expected after fix: the stale entry is deleted from self.cache.
# ---------------------------------------------------------------------------


def test_store_none_handle_removes_stale_entry(tmp_path: Path) -> None:
    cache_dir = tmp_path / "model_bug2"
    cache_dir.mkdir(parents=True)
    bogus = cache_dir / "artifact_shape_None_subgraph_0"

    data = {(None, 0, "inductor_standalone"): ("artifact_shape_None_subgraph_0", str(bogus), 1)}
    _write_index(cache_dir, data)

    cm = _make_cm(tmp_path)
    cm.initialize_cache(cache_dir)

    entry = CacheEntry(None, 0, "inductor_standalone")
    assert entry in cm.cache, "precondition: entry loaded from index"

    cm._maybe_store_cache_entry(entry, None, None, "artifact_shape_None_subgraph_0")

    assert entry not in cm.cache, "stale entry must be removed when handle is None"


# ---------------------------------------------------------------------------
# Bug 3: initialize_cache() does not reset self.cache when
# subgraph_indices.py is absent.
#
# On origin/main, `self.cache = {}` only runs inside the
# `if self.cache_file_path.exists()` branch.  So a second call to
# initialize_cache (same dir, no index file) preserves ghost handles
# injected between the two calls.
#
# Expected after fix: self.cache is always reset to {}.
# Note: _remaining_restart_skips is NOT reset by initialize_cache because
# it is runtime consumption state that must survive across Dynamo
# RestartAnalysis retries (which re-call initialize_cache).
# ---------------------------------------------------------------------------


def test_reinit_without_indices_clears_memory(tmp_path: Path) -> None:
    cache_dir = tmp_path / "model_bug3"
    cache_dir.mkdir(parents=True)

    cm = _make_cm(tmp_path)
    cm.initialize_cache(cache_dir)

    # Simulate a ghost handle left from a partial first compilation.
    bogus = cache_dir / "artifact_shape_None_subgraph_0"
    entry = CacheEntry(None, 0, "inductor_standalone")
    cm.cache[entry] = CacheHandle("artifact_shape_None_subgraph_0", str(bogus), 1)
    cm._remaining_restart_skips[0] = 1

    assert not (cache_dir / "subgraph_indices.py").exists(), "precondition: no index file"

    # Re-initialize same directory — should clear cache dict.
    cm.initialize_cache(cache_dir)

    assert entry not in cm.cache, "ghost handle must be cleared on re-init"
