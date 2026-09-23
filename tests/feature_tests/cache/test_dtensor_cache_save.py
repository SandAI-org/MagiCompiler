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

"""SimpleFSDP (DTensor) subgraphs must produce a saveable, reusable compiled artifact.

Bug (400B inference bake): through torch 2.9 Dynamo traces SimpleFSDP's
``param.redistribute(...).to_local(...)`` into ``prim_redistribute`` / ``prim_to_local``
closures. AOTAutograd refuses to key a graph that calls them, records no aot_autograd
artifact, and ``CompiledArtifact.save()`` asserts:

    Failed to save compiled artifact for key 'artifact_shape_None_subgraph_0', skipping cache:
    CacheInfo(artifacts=defaultdict(<class 'list'>, {'inductor': [...], 'aot_autograd': []}))

so every bake recompiles every subgraph.

Fix: ``_nonce_autograd_cache_key_on_keying_failure`` in piecewise_compiler.py.

Driven through torchrun like the fsdp e2e tests: world=1, and world=2 where the weights are
real Shard(0) DTensors all-gathered inside the graph.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

_HELPER = Path(__file__).parent / "cache_reuse_helper" / "dtensor_cache_save_helper.py"

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
requires_torchrun = pytest.mark.skipif(shutil.which("torchrun") is None, reason="requires torchrun")
world_sizes = pytest.mark.parametrize(
    "nproc", [1, pytest.param(2, marks=pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires >=2 GPUs"))]
)


def _run(cache_root: Path, output_dir: Path, nproc: int, port: int, *flags: str) -> tuple[str, list[dict]]:
    output_dir.mkdir(parents=True)
    env = os.environ.copy()
    env["MAGI_LOGGING_LEVEL"] = "info"
    env["MAGI_COMPILE_CACHE_ROOT_DIR"] = str(cache_root)
    p = subprocess.run(
        [
            "torchrun",
            f"--nproc_per_node={nproc}",
            f"--master_port={port}",
            str(_HELPER),
            "--output-dir",
            str(output_dir),
            *flags,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    out = p.stdout + p.stderr
    assert p.returncode == 0, f"helper failed:\n{out[-4000:]}"
    return out, [json.loads((output_dir / f"rank{rank}.json").read_text()) for rank in range(nproc)]


@requires_cuda
@requires_torchrun
@world_sizes
def test_bug_reproduced_without_fix(tmp_path: Path, nproc: int):
    """Without the nonce-key fix: the exact bake error, and no artifact is saved."""
    out, payloads = _run(tmp_path / "cache", tmp_path / "out", nproc, 29660 + nproc, "--disable-fix")
    reasons = [reason for payload in payloads for reason in payload["aot_bypass_reasons"]]
    if not reasons:
        pytest.skip("this PyTorch keys SimpleFSDP graphs (DTensor methods traced as call_method)")

    assert any("prim_redistribute" in r or "prim_to_local" in r for r in reasons), reasons
    assert "Failed to save compiled artifact" in out, out[-4000:]
    assert "'aot_autograd': []" in out, out[-4000:]
    assert all(payload["num_compiled_artifacts_saved"] == 0 for payload in payloads), payloads


@requires_cuda
@requires_torchrun
@world_sizes
def test_bug_fixed_artifact_saved_and_reused(tmp_path: Path, nproc: int):
    """With the fix: every compiled subgraph is saved on the cold run and loaded on the warm run."""
    cache_root = tmp_path / "cache"

    cold_out, cold_payloads = _run(cache_root, tmp_path / "cold", nproc, 29670 + nproc)
    assert "Failed to save compiled artifact" not in cold_out, cold_out[-4000:]
    for payload in cold_payloads:
        assert payload["num_inductor_compiles"] > 0, payload
        assert payload["num_compiled_artifacts_saved"] == payload["num_inductor_compiles"], payload
        assert payload["max_abs_diff"] < 1e-2, payload

    # A nonce key can never be looked up again, so its entry must not stay in the shared cache dir.
    for nonce_key in re.findall(r"using nonce key (a[0-9a-f]{32})", cold_out):
        assert not (cache_root / "inductor_cache" / "aotautograd" / nonce_key).exists(), nonce_key

    warm_out, warm_payloads = _run(cache_root, tmp_path / "warm", nproc, 29680 + nproc)
    assert "Failed to save compiled artifact" not in warm_out, warm_out[-4000:]
    for payload in warm_payloads:
        assert payload["num_inductor_compiles"] == 0, f"artifact not reused: {payload}\n{warm_out[-4000:]}"
        assert payload["max_abs_diff"] < 1e-2, payload
