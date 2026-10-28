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

"""An explicit model_tag names the cache dir on its own, so the cache no longer depends on call order."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

_RUN = """
import json, sys
import torch
from magi_compiler import magi_compile
from magi_compiler.utils import compilation_counter

@magi_compile(model_tag=lambda block: getattr(block, "tag", None), dynamic_arg_dims={"x": 0})
class Block(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale, device="cuda"))

    def forward(self, x):
        return torch.relu(x * self.scale) + 1

blocks = {}
for name in sys.argv[1]:
    blocks[name] = Block({"a": 2.0, "b": 3.0, "_": 4.0}[name])
    if name != "_":
        blocks[name].tag = name
for name in sys.argv[1]:
    blocks[name](torch.randn(8, 4, device="cuda"))
duplicate = None
if "a" in blocks:
    twin = Block(2.0)
    twin.tag = "a"
    try:
        twin(torch.randn(8, 4, device="cuda"))
    except ValueError as e:
        duplicate = str(e)
print(json.dumps({"compiles": compilation_counter.num_inductor_compiles, "duplicate": duplicate}))
"""


def _run(cache_root: Path, order: str) -> dict:
    env = {**os.environ, "MAGI_COMPILE_CACHE_ROOT_DIR": str(cache_root)}
    p = subprocess.run([sys.executable, "-c", _RUN, order], env=env, capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, (p.stdout + p.stderr)[-4000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def _model_dirs(cache_root: Path) -> list[str]:
    return sorted(d.name.rsplit("_rank_", 1)[0] for d in (cache_root / "magi_cache").iterdir())


def test_explicit_tags_hit_whatever_order_the_models_first_run_in(tmp_path: Path):
    cold = _run(tmp_path, "ab")
    assert cold["compiles"] > 0
    assert _model_dirs(tmp_path) == ["model_a", "model_b"]

    warm = _run(tmp_path, "ba")
    assert warm["compiles"] == 0, warm
    assert _model_dirs(tmp_path) == ["model_a", "model_b"]


def test_a_second_live_object_cannot_take_an_explicit_tag(tmp_path: Path):
    assert "model_tag 'a' is already used by another live Block" in _run(tmp_path, "a")["duplicate"]


def test_without_a_tag_the_dir_keeps_the_call_order_index(tmp_path: Path):
    _run(tmp_path, "_")
    assert _model_dirs(tmp_path) == ["model_1_Block"]
