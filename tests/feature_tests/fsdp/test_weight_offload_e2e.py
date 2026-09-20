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

"""End-to-end compile-time weight offload on a SimpleFSDP model.

Driven through a ``torchrun`` subprocess (fsdp_overlap_helper/offload_e2e_helper.py)
because the chain needs a process group inside a real compile -- same pattern as
test_fsdp_overlap_e2e.py.

Offload tags whatever host-first parked and the redistribute lowering exposed,
so if the installed SimpleFSDP emits a shape the lowering does not match, there
is nothing to offload and the numeric check would pass on an ordinary graph.
The helper prints ``OFFLOAD_SKIPPED`` in that case and these tests skip rather
than report a green run for a chain that never executed.
"""

import os
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path

import pytest
import torch

_HELPER = Path(__file__).parent / "fsdp_overlap_helper" / "offload_e2e_helper.py"

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
requires_torchrun = pytest.mark.skipif(shutil.which("torchrun") is None, reason="requires torchrun")


def _free_port() -> str:
    """A port the kernel just told us is free.

    Hard-coded ports collide with whatever the previous test left in TIME_WAIT,
    which fails as EADDRINUSE and reads exactly like a real regression.
    """
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return str(s.getsockname()[1])


def _run(nproc: int, *extra: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["MAGI_LOGGING_LEVEL"] = env.get("MAGI_LOGGING_LEVEL", "info")
    # A cache root of its own per run. Two runs of this helper produce the same
    # FX graph by design -- host-first changes where the weights live, not what
    # the graph says -- so a shared cache has the second one replay the first
    # one's artifact and skip the scheduler, which is where the placement pass
    # and every log line these tests assert on live.
    with tempfile.TemporaryDirectory(prefix="magi_offload_e2e_") as cache_root:
        env["MAGI_COMPILE_CACHE_ROOT_DIR"] = cache_root
        return subprocess.run(
            ["torchrun", f"--nproc_per_node={nproc}", f"--master_port={_free_port()}", str(_HELPER), *extra],
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )


def _marker(stdout: str, marker: str, field: str) -> float:
    """One ``key=value`` field off a helper marker line."""
    line = next(l for l in stdout.splitlines() if l.startswith(marker))
    return float(line.split(f"{field}=")[1].split()[0])


def _check(p: subprocess.CompletedProcess) -> str:
    out = p.stdout + p.stderr
    if "OFFLOAD_SKIPPED" in p.stdout:
        pytest.skip("the redistribute lowering matched no weight gather; nothing to offload")
    assert p.returncode == 0, f"helper failed:\n{out[-4000:]}"
    return out


@requires_cuda
@requires_torchrun
def test_offload_single_rank():
    """world=1: shards leave the device, the graph loads them back, output matches eager."""
    p = _run(1)
    out = _check(p)
    assert "OFFLOAD_PASS" in p.stdout, out[-4000:]
    assert _marker(p.stdout, "OFFLOAD_FREED", "shards") > 0


@requires_cuda
@requires_torchrun
def test_both_reorder_phases_run():
    """Phase 1 alone is correct and fully exposed; phase 2 is the whole point.

    Asserted on the logs because every way this silently degrades -- a load the
    pass does not recognize, a wait it cannot reach through the unpack, an order
    that fails validation -- leaves the numerics perfect and the overlap absent.
    """
    p = _run(1)
    out = _check(p)
    assert "FSDP overlap reorder: repositioned" in out, out[-4000:]
    assert "h2d load reorder: hoisted" in out, out[-4000:]
    hoisted = next(line for line in out.splitlines() if "h2d load reorder: hoisted" in line)
    moved = int(hoisted.split("hoisted ")[1].split("/")[0])
    assert moved > 0, hoisted


@requires_cuda
@requires_torchrun
def test_offload_with_coalesced_buckets():
    """Bucketing must keep offloaded and resident gathers apart: a bucket is one
    launch, so every member has to have landed before it."""
    p = _run(1, "--bucket-mode", "coalesced")
    out = _check(p)
    assert "OFFLOAD_PASS" in p.stdout, out[-4000:]


@requires_cuda
@requires_torchrun
def test_loads_are_coalesced_to_match_the_buckets():
    """One submission per bucket, not per weight.

    The per-load fixed cost -- stream sync, event, Work registration -- is ~10us
    of CPU sitting directly in front of the first all-gather, and it inflates the
    very window the placement pass is trying to size.
    """
    p = _run(1, "--bucket-mode", "coalesced", "--bucket-size-mib", "4", "--n-layers", "6")
    out = _check(p)
    assert "OFFLOAD_PASS" in p.stdout, out[-4000:]

    buckets = int(next(l for l in out.splitlines() if "Whole-graph FSDP bucketing" in l).split("created ")[1].split()[0])
    loads = int(next(l for l in out.splitlines() if "inserted" in l and "h2d_load" in l).split("inserted ")[1].split()[0])
    shards = _marker(p.stdout, "OFFLOAD_FREED", "shards")
    assert buckets > 1, "this shape is supposed to produce several buckets"
    assert loads == buckets, f"expected one load per bucket, got {loads} for {buckets}"
    assert loads < shards, "coalescing is supposed to be fewer loads than weights"


@requires_cuda
@requires_torchrun
def test_only_one_shard_is_live_at_a_time():
    """The rule that bounds the device cost, end to end.

    With several buckets back to back, the hoists would otherwise stack and put
    most of the model back on the device.  The pass hands the overlapping ones
    back instead, and reports the resulting in-flight peak -- which must be one
    bucket, not the sum of them.
    """
    p = _run(1, "--bucket-mode", "coalesced", "--bucket-size-mib", "4", "--n-layers", "6")
    out = _check(p)
    hoisted = next(line for line in out.splitlines() if "h2d load reorder: hoisted" in line)
    peak = float(hoisted.split("in-flight peak ")[1].split(" MiB")[0])
    assert 0 < peak <= 4.5, f"in-flight peak should be about one 4 MiB bucket: {hoisted}"


@requires_cuda
@requires_torchrun
def test_residency_cap_is_respected():
    """The second lever, end to end: whatever the schedule wants to keep, the cap
    is what it actually gets.

    This shape's loads happen to sit far enough apart that the sweep keeps
    nothing resident, so the cap has nothing to claw back here -- that path is
    unit-tested in ``test_h2d_reorder.py``.  What this covers is that a capped
    run still compiles, still matches eager, and does not exceed the cap.
    """
    p = _run(1, "--bucket-mode", "coalesced", "--bucket-size-mib", "4", "--n-layers", "6", "--max-resident-mib", "1")
    out = _check(p)
    assert "OFFLOAD_PASS" in p.stdout, out[-4000:]
    assert _marker(p.stdout, "OFFLOAD_FREED", "promoted_mib") <= 1, "residency must stay inside the cap"


@requires_cuda
@requires_torchrun
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires >=2 GPUs")
def test_offload_multi_rank():
    """world=2: a real 2-rank gather, fed by per-rank shards that only exist in
    host memory until the graph loads them."""
    p = _run(2)
    out = _check(p)
    assert "OFFLOAD_PASS" in p.stdout, out[-4000:]


# ------------------------------------------------------- host-first loading


@requires_cuda
@requires_torchrun
def test_host_first_never_puts_the_weights_on_the_device():
    """The number offload exists to lower, measured rather than inferred.

    Loading is where peak device memory is decided: the shards are
    materialized in host memory and filled there, so the load phase should
    cost no device memory. That is why it is measured separately here.
    """
    p = _run(1, "--host-first")
    out = _check(p)
    assert "OFFLOAD_PASS" in p.stdout, out[-4000:]

    peak = _marker(p.stdout, "OFFLOAD_LOAD", "peak_mib")
    weights = _marker(p.stdout, "OFFLOAD_LOAD", "weights_mib")
    assert weights > 1, "the shape under test is supposed to have weights worth offloading"
    assert peak < 0.5, f"materializing and filling the model should cost no device memory, cost {peak} MiB"


@requires_cuda
@requires_torchrun
def test_host_first_weights_are_all_claimed_by_the_graph():
    """Nothing may fall in the gap between "parked" and "loaded".

    Parking happens while the model is built, from a weight's placements alone;
    whether the graph actually loads it is only known once the lowering has run.
    A weight in the gap has no bytes behind it, so the backend hands it back and
    says so -- correct, but it means offload bought nothing for that weight, and
    on this model it should never happen.
    """
    p = _run(1, "--host-first")
    out = _check(p)
    assert "are not loaded by any compiled graph" not in out, out[-4000:]


@requires_cuda
@requires_torchrun
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires >=2 GPUs")
def test_host_first_multi_rank():
    """world=2: ranks in a shard group vote per candidate, not WORLD-all-or-nothing.

    A rank-dependent parking decision drops only the weights they do not share;
    the rest stay offloaded.  An empty intersection would log ``nothing to offload``.
    """
    p = _run(2, "--host-first")
    out = _check(p)
    assert "OFFLOAD_PASS" in p.stdout, out[-4000:]
    assert "nothing to offload" not in out, out[-4000:]
    assert _marker(p.stdout, "OFFLOAD_LOAD", "peak_mib") < 0.5
