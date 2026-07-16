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

_HELPER = Path(__file__).parent / "fsdp_overlap_helper" / "reorder_helper.py"

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
requires_torchrun = pytest.mark.skipif(shutil.which("torchrun") is None, reason="requires torchrun")


def _run(nproc: int, *extra: str, port: str = "29631") -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["MAGI_LOGGING_LEVEL"] = env.get("MAGI_LOGGING_LEVEL", "info")
    return subprocess.run(
        ["torchrun", f"--nproc_per_node={nproc}", f"--master_port={port}", str(_HELPER), *extra],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


@requires_cuda
@requires_torchrun
def test_reorder_single_rank():
    """world=1: the reorder pass runs inside a real compile, sees a weight gather,
    reorders without error, and the compiled output matches eager."""
    p = _run(1)
    out = p.stdout + p.stderr
    assert p.returncode == 0, f"helper failed:\n{out[-3000:]}"
    assert "REORDER_CALLED gathers=1" in p.stdout, out[-3000:]
    assert "REORDER_OK ran=True" in p.stdout, out[-3000:]
    assert "REORDER_PASS" in p.stdout, out[-3000:]


@requires_cuda
@requires_torchrun
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires >=2 GPUs")
def test_reorder_multi_rank():
    """world=2: same, with a real 2-rank all-gather group."""
    p = _run(2)
    out = p.stdout + p.stderr
    assert p.returncode == 0, f"helper failed:\n{out[-3000:]}"
    assert "REORDER_PASS" in p.stdout, out[-3000:]


@requires_cuda
@requires_torchrun
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires >=2 GPUs")
def test_reorder_graph_mismatch_fail_fast():
    """world=2 with rank1 compiling a structurally DIFFERENT graph: the cross-rank
    graph-fingerprint check must fire on both ranks (warning), leave the schedule
    unchanged, and complete without deadlock."""
    p = _run(2, "--mismatch", port="29632")
    out = p.stdout + p.stderr
    assert p.returncode == 0, f"helper failed:\n{out[-3000:]}"
    assert "REORDER_MISMATCH unchanged=True" in p.stdout, out[-3000:]
    assert "REORDER_WARNED" in p.stdout, out[-3000:]  # the fail-fast warning fired
    assert "REORDER_PASS" in p.stdout, out[-3000:]
