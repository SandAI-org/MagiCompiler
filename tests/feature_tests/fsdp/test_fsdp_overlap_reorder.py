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

"""FsdpOverlapReorder integration test.

The reorder pass is an Inductor scheduler callback that needs a process group + a
real ``torch.compile``, so we drive it via a ``torchrun`` subprocess helper
(fsdp_overlap_helper/reorder_helper.py) and assert on its stdout markers -- same
subprocess pattern as test_autograd_function_cache_flag.py.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

_REORDER_HELPER_PATH = Path(__file__).parent / "fsdp_overlap_helper" / "reorder_helper.py"
_RANK_CONSENSUS_HELPER_PATH = Path(__file__).parent / "fsdp_overlap_helper" / "reorder_rank_helper.py"

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
requires_torchrun = pytest.mark.skipif(shutil.which("torchrun") is None, reason="requires torchrun")


def _run_reorder_helper(nproc: int, *extra: str, port: str = "29631") -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["MAGI_LOGGING_LEVEL"] = env.get("MAGI_LOGGING_LEVEL", "info")
    return subprocess.run(
        ["torchrun", f"--nproc_per_node={nproc}", f"--master_port={port}", str(_REORDER_HELPER_PATH), *extra],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


@requires_torchrun
def test_reorder_negotiates_on_a_rank_without_weight_gathers():
    """Two gloo ranks, only one with a weight gather: both join the negotiation and abort, neither hangs."""
    root = str(Path(__file__).resolve().parents[3])
    env = os.environ.copy()
    env["PYTHONPATH"] = root if not env.get("PYTHONPATH") else root + os.pathsep + env["PYTHONPATH"]
    env["MAGI_LOGGING_LEVEL"] = env.get("MAGI_LOGGING_LEVEL", "info")
    process_result = subprocess.run(
        ["torchrun", "--nproc_per_node=2", "--master_port=29637", str(_RANK_CONSENSUS_HELPER_PATH)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    combined_log = process_result.stdout + process_result.stderr
    assert process_result.returncode == 0, f"helper failed:\n{combined_log[-3000:]}"
    assert "REORDER_RANK_PASS" in process_result.stdout, combined_log[-3000:]


@requires_cuda
@requires_torchrun
def test_reorder_single_rank():
    """world=1: the reorder pass runs inside a real compile, sees a weight gather,
    reorders without error, and the compiled output matches eager."""
    process_result = _run_reorder_helper(1)
    combined_log = process_result.stdout + process_result.stderr
    assert process_result.returncode == 0, f"helper failed:\n{combined_log[-3000:]}"
    assert "REORDER_CALLED gathers=1" in process_result.stdout, combined_log[-3000:]
    assert "REORDER_OK ran=True" in process_result.stdout, combined_log[-3000:]
    assert "REORDER_PASS" in process_result.stdout, combined_log[-3000:]


@requires_cuda
@requires_torchrun
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires >=2 GPUs")
def test_reorder_multi_rank():
    """world=2: same, with a real 2-rank all-gather group."""
    process_result = _run_reorder_helper(2)
    combined_log = process_result.stdout + process_result.stderr
    assert process_result.returncode == 0, f"helper failed:\n{combined_log[-3000:]}"
    assert "REORDER_SKELETON ok=True" in process_result.stdout, combined_log[-3000:]
    assert "REORDER_PASS" in process_result.stdout, combined_log[-3000:]


@requires_torchrun
def test_reorder_mode_ladder():
    """Matching skeletons use slots; mismatched skeletons pin, and failed pricing or counts abort."""
    process_result = _run_reorder_helper(2, "--modes-only", port="29633")
    combined_log = process_result.stdout + process_result.stderr
    assert process_result.returncode == 0, f"helper failed:\n{combined_log[-3000:]}"
    assert "REORDER_MODES ok=True" in process_result.stdout, combined_log[-3000:]


@requires_cuda
@requires_torchrun
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires >=2 GPUs")
def test_reorder_graph_mismatch_slot_mode():
    """Different rank-local compute must still emit matching collective order and match eager."""
    process_result = _run_reorder_helper(2, "--mismatch", port="29632")
    combined_log = process_result.stdout + process_result.stderr
    assert process_result.returncode == 0, f"helper failed:\n{combined_log[-3000:]}"
    assert "REORDER_SLOT" in process_result.stdout, combined_log[-3000:]
    assert "REORDER_SKELETON ok=True" in process_result.stdout, combined_log[-3000:]
    assert "REORDER_PASS" in process_result.stdout, combined_log[-3000:]


@requires_cuda
@requires_torchrun
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires >=2 GPUs")
def test_reorder_copy_engine_recognition():
    """world=2, gathers served by the copy engine out of a symmetric buffer.

    The gather is an opaque fallback kernel, so if the pass fails to see through
    it nothing is planned and ``gathers`` comes out 0.  Destinations are fresh
    ``empty``s -- Inductor owns liveness -- so the only extra invariant is that
    the three launches are recognized and the compiled answer matches NCCL.
    """
    process_result = _run_reorder_helper(2, "--copy-engine", port="29634")
    combined_log = process_result.stdout + process_result.stderr
    assert process_result.returncode == 0, f"helper failed:\n{combined_log[-3000:]}"
    assert "REORDER_CALLED gathers=3" in process_result.stdout, combined_log[-3000:]  # all three were recognized
    assert "REORDER_FINITE ok=True" in process_result.stdout, combined_log[-3000:]  # ... and match the NCCL answer
    assert "REORDER_PASS" in process_result.stdout, combined_log[-3000:]
