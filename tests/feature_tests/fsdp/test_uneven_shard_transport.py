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

"""Guard: an uneven ``Shard(0)`` weight must get the same transport on every rank.

A weight gather is a collective, so which transport carries it is a joint decision.
An uneven ``Shard(0)`` is where that decision used to split: the lowering pads the
shards of the ranks that own fewer rows, and copy-engine eligibility was decided from
the gather's input, which the pad changes.  Only the trailing ranks refused the copy
engine; the rest moved on without them and their NCCL all-gather never completed.

The divergence is invisible at one rank -- each rank's graph is individually
reasonable -- so the check has to compare graphs ACROSS ranks, which needs a real
process group and two of them.  The helper runs under ``torchrun`` and all-gathers
its own decisions; this file asserts on its markers.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

_SCRIPT = Path(__file__).parent / "fsdp_overlap_helper" / "uneven_shard_helper.py"

requires_2gpu = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires >=2 GPUs")
requires_torchrun = pytest.mark.skipif(shutil.which("torchrun") is None, reason="requires torchrun")


def _run(port: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["MAGI_LOGGING_LEVEL"] = env.get("MAGI_LOGGING_LEVEL", "info")
    return subprocess.run(
        ["torchrun", "--nproc_per_node=2", f"--master_port={port}", str(_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


@requires_2gpu
@requires_torchrun
def test_uneven_shard_transport_is_rank_identical():
    """Both the uneven weight and the even control must be decided unanimously.

    The even case is the control that keeps the fix honest: refusing the copy engine
    for everything would satisfy the uneven assertion and quietly cost the whole
    feature, so ``rows=4`` has to still come out as one copy-engine bucket.

    The mixed graph interleaves the two: even weights stay on the copy engine, uneven
    ones stay on NCCL, each as their own coalesced bucket, and every rank agrees.
    """
    p = _run("29645")
    out = p.stdout + p.stderr
    assert p.returncode == 0, f"helper failed:\n{out[-4000:]}"
    # even control: still bucketed onto the copy engine
    assert "UNEVEN_TRANSPORT rows=4 agree=True targets={'ce_coalesced': 1}" in p.stdout, out[-4000:]
    # uneven: every rank keeps it on NCCL -- but still buckets it there.  Losing the
    # copy engine must not also cost bucketing, or one odd weight turns N gathers
    # into N launches.
    assert "UNEVEN_TRANSPORT rows=3 agree=True targets={'nccl_coalesced': 1}" in p.stdout, out[-4000:]
    assert "UNEVEN_NCCL_BUCKETS rows=3 agree=True" in p.stdout, out[-4000:]
    assert "UNEVEN_SYMM agree=True in_buffer=['even']" in p.stdout, out[-4000:]
    assert (
        "UNEVEN_MIXED agree=True targets={'ce_coalesced': 1, 'nccl_coalesced': 1} "
        "sizes=[('ce_coalesced', 2), ('nccl_coalesced', 2)]" in p.stdout
    ), out[-4000:]
    assert "UNEVEN_PASS" in p.stdout, out[-4000:]
