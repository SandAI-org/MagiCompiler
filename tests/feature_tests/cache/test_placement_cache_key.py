# Copyright (c) 2025 SandAI. All Rights Reserved.
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

"""Changing parameter placements alone must not reuse a compiled artifact from the old placements."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

_HELPER = Path(__file__).parent / "cache_reuse_helper" / "placement_cache_key_helper.py"

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(shutil.which("torchrun") is None, reason="requires torchrun"),
]


def _compiles(cache_root: Path, output_dir: Path, mode: str, port: int, *flags: str) -> int:
    output_dir.mkdir(parents=True)
    env = os.environ.copy()
    env["MAGI_COMPILE_CACHE_ROOT_DIR"] = str(cache_root)
    p = subprocess.run(
        ["torchrun", "--nproc_per_node=1", f"--master_port={port}", str(_HELPER), "--output-dir", str(output_dir), "--mode", mode, *flags],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert p.returncode == 0, f"helper failed:\n{(p.stdout + p.stderr)[-4000:]}"
    payload = json.loads((output_dir / "rank0.json").read_text())
    assert payload["max_abs_diff"] < 1e-2, payload
    return payload["num_inductor_compiles"]


def test_bug_reproduced_without_fix(tmp_path: Path):
    """Without the graph signature in the key, the replicate run loads the fully_shard artifact."""
    cache_root = tmp_path / "cache"
    assert _compiles(cache_root, tmp_path / "shard", "fully_shard", 29690, "--disable-fix") > 0
    assert _compiles(cache_root, tmp_path / "replicate", "replicate", 29691, "--disable-fix") == 0


def test_bug_fixed_placement_change_recompiles(tmp_path: Path):
    """A placement change recompiles; rerunning the same placements still reuses the cache."""
    cache_root = tmp_path / "cache"
    assert _compiles(cache_root, tmp_path / "shard", "fully_shard", 29692) > 0
    assert _compiles(cache_root, tmp_path / "replicate", "replicate", 29693) > 0
    assert _compiles(cache_root, tmp_path / "replicate_warm", "replicate", 29694) == 0
