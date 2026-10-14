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

Driven through a ``torchrun`` subprocess (h2d_helper/offload_e2e_helper.py)
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

_HELPER = Path(__file__).parent / "h2d_helper" / "offload_e2e_helper.py"

# Several 4 MiB buckets back to back, close enough together that the sweep
# cannot hide them all: the shape where placement has to choose between
# residency and exposure, and therefore the one the budget tests are worth
# running on.  At one rank a gather moves no bytes, so the fitted per-byte cost
# is zero and only the cap sizes the buckets.
_MULTI_LOAD_SHAPE = ("--bucket-mode", "auto", "--bucket-size-mib", "4", "--n-layers", "6")

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
    assert _REORDER_REPORT in out, out[-4000:]
    moved = int(_reorder_report(out).split("Hoisted ")[1].split("/")[0])
    assert moved > 0, _reorder_report(out)


@requires_cuda
@requires_torchrun
def test_offload_with_scheduler_stage_buckets():
    """``bucket_mode='auto'`` buckets during scheduling, after the loads exist;
    the coalesced launch still has to wait for every member's own load, and
    bucketing keeps offloaded and resident gathers apart."""
    p = _run(2, "--bucket-mode", "auto", "--n-layers", "6")
    out = _check(p)
    assert "OFFLOAD_PASS" in p.stdout, out[-4000:]
    line = next(l for l in out.splitlines() if "FSDP auto bucket:" in l)
    assert int(line.split("FSDP auto bucket: ")[1].split()[0]) >= 1, line
    loads = int(next(l for l in out.splitlines() if "inserted" in l and "h2d_load" in l).split("inserted ")[1].split()[0])
    assert loads == _marker(p.stdout, "OFFLOAD_FREED", "shards"), "one load per weight"


_REORDER_REPORT = "h2d load reorder: "
"""Prefix of phase 2's summary line, which the assertions below parse."""


def _reorder_report(out: str) -> str:
    """The INFO summary, not the WARNING lines that share the same prefix."""
    return next(l for l in out.splitlines() if _REORDER_REPORT in l and "Hoisted " in l)


def _inflight(out: str) -> tuple[float, float]:
    """The in-flight peak and the budget it was held to, off the reorder's report."""
    peak, rest = _reorder_report(out).split("in-flight peak ")[1].split(" MiB of the ", 1)
    return float(peak), float(rest.split(" MiB")[0])


@requires_cuda
@requires_torchrun
def test_the_inflight_budget_is_never_exceeded():
    """The promise the whole pass rests on, end to end.

    With several 4 MiB buckets back to back the hoists would otherwise stack and
    put most of the model back on the device.  Unasked, the budget is what phase
    1's own unhoisted order already needs plus one bucket, and the emitted
    schedule has to fit inside it -- overshooting reads as a working run until the
    day the model is big enough to OOM on the difference.
    """
    p = _run(1, *_MULTI_LOAD_SHAPE)
    out = _check(p)
    peak, budget = _inflight(out)
    assert 0 < peak <= budget, f"in-flight peak {peak} MiB over the {budget} MiB budget"
    assert budget <= 4.5 * 2, f"unhoisted floor plus one bucket, not more: {budget} MiB"


@requires_cuda
@requires_torchrun
def test_no_residency_budget_keeps_nothing_resident():
    """The default, end to end.

    Handing weights back costs device memory for the whole graph, so it happens
    only when a budget asks for it -- doing it uninvited is how a run OOMs, and
    the numerics have to come out right either way.
    """
    p = _run(1, *_MULTI_LOAD_SHAPE)
    out = _check(p)
    assert "OFFLOAD_PASS" in p.stdout, out[-4000:]
    assert _marker(p.stdout, "OFFLOAD_FREED", "promoted_mib") == 0, "an empty budget must not promote"


@requires_cuda
@requires_torchrun
def test_device_weight_budget_is_spent_and_not_exceeded():
    """The one budget: residency plus in-flight room, split by the pass.

    This shape has more exposed transfer than the in-flight room alone can hide,
    so part of the budget goes to residency, and the two halves together must
    stay inside what was offered.
    """
    p = _run(1, *_MULTI_LOAD_SHAPE, "--max-device-weight-mib", "24")
    out = _check(p)
    promoted = _marker(p.stdout, "OFFLOAD_FREED", "promoted_mib")
    _, inflight_budget = _inflight(out)
    assert "OFFLOAD_PASS" in p.stdout, out[-4000:]
    assert promoted > 0, f"some of the budget has to buy residency: {_reorder_report(out)}"
    assert promoted + inflight_budget <= 24, f"residency {promoted} + in-flight {inflight_budget} MiB over 24 MiB"


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
