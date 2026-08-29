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

"""End-to-end guard for ``fsdp_config.transport="copy_engine"``.

The unit tests each stub something out: the arena tests materialize by calling
``_apply`` with a hand-forged lambda, and the gather tests run on one rank where
the peer view is the local shard.  The property that only shows up when the
whole chain runs -- meta build, SimpleFSDP, ``to_empty``, checkpoint load,
compile, rewrite, reorder -- is that the weights are never ordinary tensors at
any point, and that is what the helper script asserts.  It needs a real process
group and two ranks, so it runs as a ``torchrun`` subprocess and this file
asserts on its stdout markers.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

_SCRIPT = Path(__file__).parent / "symm_helper" / "verify_symm_e2e.py"

requires_2gpu = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires >=2 GPUs")
requires_torchrun = pytest.mark.skipif(shutil.which("torchrun") is None, reason="requires torchrun")


def _run(transport: str, port: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["MAGI_LOGGING_LEVEL"] = env.get("MAGI_LOGGING_LEVEL", "info")
    return subprocess.run(
        [
            "torchrun",
            "--nproc_per_node=2",
            f"--master_port={port}",
            str(_SCRIPT),
            "--transport",
            transport,
            # Small enough to compile quickly, big enough that the gather is a
            # real cross-rank transfer rather than a rounding error.
            "--hidden",
            "1024",
            "--n-layers",
            "4",
            "--n-tokens",
            "512",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )


@requires_2gpu
@requires_torchrun
def test_copy_engine_end_to_end():
    """Weights land in symmetric memory, the gathers get retargeted, and two
    consecutive steps both match an unsharded eager model -- the second step
    is where a missing wait would show up."""
    p = _run("copy_engine", "29641")
    out = p.stdout + p.stderr
    assert p.returncode == 0, f"script failed:\n{out[-4000:]}"
    assert "CHECK placement: 4/4 block shards" in p.stdout, out[-4000:]
    assert "CHECK rewrite: 4/4 gathers" in p.stdout, out[-4000:]
    assert "E2E_PASS" in p.stdout, out[-4000:]


@requires_2gpu
@requires_torchrun
def test_nccl_transport_is_untouched():
    """The default transport must allocate no window and rewrite no gather: the
    control that says the copy-engine result above came from the flag."""
    p = _run("nccl", "29642")
    out = p.stdout + p.stderr
    assert p.returncode == 0, f"script failed:\n{out[-4000:]}"
    assert "CHECK placement: 0/4 block shards in 0 window(s)" in p.stdout, out[-4000:]
    assert "CHECK rewrite: 0/0 gathers" in p.stdout, out[-4000:]
    assert "E2E_PASS" in p.stdout, out[-4000:]
