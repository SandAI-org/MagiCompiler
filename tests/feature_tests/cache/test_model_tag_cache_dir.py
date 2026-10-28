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

"""model_tag alone names the cache dir, so a tagged model's cache no longer depends on call order."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

# Each run is its own process, as a bake and the deploy that loads it are. argv: mode, then its argument.
_RUN = """
import json, sys
import torch
from magi_compiler import magi_compile
from magi_compiler.utils import compilation_counter

@magi_compile(model_tag=lambda block: block.tag, dynamic_arg_dims={"x": 0})
class Block(torch.nn.Module):
    def __init__(self, scale, tag=None):
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale, device="cuda"))
        self.tag = tag

    def forward(self, x):
        return torch.relu(x * self.scale) + 1

def call(block):
    block(torch.randn(8, 4, device="cuda"))

# A tag always names the same weights, whatever order the blocks are built and called in.
SCALES = {"a": 2.0, "b": 3.0}
mode, arg = sys.argv[1], sys.argv[2]
result = {}
if mode == "tagged":  # one block per tag in arg, called in that order
    for tag in arg:
        call(Block(SCALES[tag], tag=tag))
elif mode == "twin":  # a second live block asks for the tag in arg
    first = Block(SCALES[arg], tag=arg)
    call(first)
    try:
        call(Block(SCALES[arg], tag=arg))
    except ValueError as e:
        result["error"] = str(e)
elif mode == "untagged":
    call(Block(4.0))
result["compiles"] = compilation_counter.num_inductor_compiles
print(json.dumps(result))
"""


def _run(cache_root: Path, mode: str, arg: str = "") -> dict:
    env = {**os.environ, "MAGI_COMPILE_CACHE_ROOT_DIR": str(cache_root)}
    p = subprocess.run([sys.executable, "-c", _RUN, mode, arg], env=env, capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, (p.stdout + p.stderr)[-4000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def _model_dirs(cache_root: Path) -> list[str]:
    """Cache dir names without the ``_rank_{rank}_{topology}`` suffix."""
    return sorted(d.name.rsplit("_rank_", 1)[0] for d in (cache_root / "magi_cache").iterdir())


def test_tagged_blocks_hit_whatever_order_they_first_run_in(tmp_path: Path):
    cold = _run(tmp_path, "tagged", "ab")
    assert cold["compiles"] > 0
    assert _model_dirs(tmp_path) == ["model_a", "model_b"]

    warm = _run(tmp_path, "tagged", "ba")
    assert warm["compiles"] == 0, warm
    assert _model_dirs(tmp_path) == ["model_a", "model_b"]


def test_a_second_live_block_cannot_take_the_same_tag(tmp_path: Path):
    assert "model_tag 'a' is already used by another live Block" in _run(tmp_path, "twin", "a")["error"]


def test_an_untagged_block_is_named_by_call_order_and_class(tmp_path: Path):
    _run(tmp_path, "untagged")
    assert _model_dirs(tmp_path) == ["model_1_Block"]
