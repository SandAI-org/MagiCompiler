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

"""End-to-end FSDP-overlap chain test: a small SimpleFSDP-sharded model through
``magi_compile`` with ``fsdp_config.enable_fsdp`` -- exercising the whole
chain (redistribute lowering -> bucketing -> FsdpOverlapReorder) in one real compile.

Driven via a ``torchrun`` subprocess helper (fsdp_overlap_helper/e2e_helper.py); we
assert on stdout markers + returncode.

- SINGLE rank (world=1): chain runs, compiles, output matches eager.  (At world=1 the
  weights are Replicate() so there is nothing to gather -- this only proves the chain
  is wired and does not break the compile.)
- MULTI rank (world=2, needs >=2 GPUs): weights are real Shard(0) DTensors, so the
  graph has weight all-gathers -- the reorder pass actually repositions them (asserted
  via the backend's "repositioned N/M" INFO log).  This is the true full-chain test.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

_HELPER = Path(__file__).parent / "fsdp_overlap_helper" / "e2e_helper.py"

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
requires_torchrun = pytest.mark.skipif(shutil.which("torchrun") is None, reason="requires torchrun")


def _run(nproc: int, cost_mode: str, port: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["MAGI_LOGGING_LEVEL"] = "info"  # so the backend's chain INFO logs are captured
    return subprocess.run(
        ["torchrun", f"--nproc_per_node={nproc}", f"--master_port={port}", str(_HELPER), "--cost-mode", cost_mode],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )


@requires_cuda
@requires_torchrun
def test_e2e_single_rank():
    """world=1: whole overlap chain compiles + runs + output matches eager."""
    p = _run(1, "analytical", "29641")
    out = p.stdout + p.stderr
    assert p.returncode == 0, f"helper failed:\n{out[-4000:]}"
    assert "E2E_COMPILED" in p.stdout, out[-4000:]
    assert "E2E_PASS" in p.stdout, out[-4000:]
    # the chain was actually invoked
    assert "FSDP fullgraph overlap" in out, out[-4000:]


@requires_cuda
@requires_torchrun
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires >=2 GPUs")
def test_e2e_multi_rank_full_chain():
    """world=2: real Shard(0) weights -> the reorder repositions real weight
    all-gathers.  Asserts the full chain ran (no deadlock, output correct, and the
    reorder pass touched >=1 gather)."""
    p = _run(2, "analytical", "29642")
    out = p.stdout + p.stderr
    assert p.returncode == 0, f"helper failed:\n{out[-4000:]}"
    assert "E2E_PASS" in p.stdout, out[-4000:]
    # the reorder pass ran on a graph WITH weight all-gathers (Shard(0) at world>1)
    assert "FSDP overlap reorder: repositioned" in out, (
        "reorder pass did not run on any weight all-gather at world=2\n" + out[-4000:]
    )


@requires_cuda
@requires_torchrun
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires >=2 GPUs")
def test_e2e_multi_rank_profile_sync():
    """world=2 with profile_sync cost mode: exercises warm_and_sync's rank-lockstep
    real measurement path end to end."""
    p = _run(2, "profile_sync", "29643")
    out = p.stdout + p.stderr
    assert p.returncode == 0, f"helper failed:\n{out[-4000:]}"
    assert "E2E_PASS" in p.stdout, out[-4000:]
