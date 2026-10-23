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

_OFFLOAD_HELPER_PATH = Path(__file__).parent / "h2d_helper" / "offload_e2e_helper.py"

# Several 4 MiB buckets back to back, close enough together that the sweep
# cannot hide them all: the shape where placement has to choose between
# residency and exposure, and therefore the one the budget tests are worth
# running on.  At one rank a gather moves no bytes, so the fitted per-byte cost
# is zero and only the cap sizes the buckets.
_MULTI_LOAD_ARGS = ("--bucket-mode", "auto", "--bucket-size-mib", "4", "--n-layers", "6")

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


def _run_offload_e2e(nproc: int, *extra: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["MAGI_LOGGING_LEVEL"] = env.get("MAGI_LOGGING_LEVEL", "info")
    # Isolate the cache so replay cannot skip the scheduler and its asserted reports.
    with tempfile.TemporaryDirectory(prefix="magi_offload_e2e_") as cache_root:
        env["MAGI_COMPILE_CACHE_ROOT_DIR"] = cache_root
        return subprocess.run(
            ["torchrun", f"--nproc_per_node={nproc}", f"--master_port={_free_port()}", str(_OFFLOAD_HELPER_PATH), *extra],
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )


def _parse_marker_float(stdout: str, marker: str, field: str) -> float:
    """One ``key=value`` field off a helper marker line."""
    line = next(l for l in stdout.splitlines() if l.startswith(marker))
    return float(line.split(f"{field}=")[1].split()[0])


def _require_success_or_skip(process_result: subprocess.CompletedProcess) -> str:
    combined_log = process_result.stdout + process_result.stderr
    if "OFFLOAD_SKIPPED" in process_result.stdout:
        pytest.skip("the redistribute lowering matched no weight gather; nothing to offload")
    assert process_result.returncode == 0, f"helper failed:\n{combined_log[-4000:]}"
    return combined_log


@requires_cuda
@requires_torchrun
def test_offload_single_rank():
    """world=1: shards leave the device, the graph loads them back, output matches eager."""
    process_result = _run_offload_e2e(1)
    combined_log = _require_success_or_skip(process_result)
    assert "OFFLOAD_PASS" in process_result.stdout, combined_log[-4000:]
    assert _parse_marker_float(process_result.stdout, "OFFLOAD_FREED", "shards") > 0


@requires_cuda
@requires_torchrun
def test_fsdp_and_h2d_reorders_hoist_loads():
    """Correct outputs alone cannot detect lost overlap: both reorder passes must run,
    and the H2D pass must report at least one hoisted load."""
    process_result = _run_offload_e2e(1)
    combined_log = _require_success_or_skip(process_result)
    assert "FSDP overlap reorder: repositioned" in combined_log, combined_log[-4000:]
    assert _LOAD_SCHEDULE_LOG_PREFIX in combined_log, combined_log[-4000:]
    hoisted_load_count = int(_find_load_schedule_log(combined_log).split("Hoisted ")[1].split("/")[0])
    assert hoisted_load_count > 0, _find_load_schedule_log(combined_log)


@requires_cuda
@requires_torchrun
def test_offload_with_scheduler_stage_buckets():
    """Scheduler bucketing runs after load insertion and must preserve each load dependency.
    Local plans separate offloaded and resident gathers; cross-rank adoption may mix them."""
    process_result = _run_offload_e2e(2, "--bucket-mode", "auto", "--n-layers", "6")
    combined_log = _require_success_or_skip(process_result)
    assert "OFFLOAD_PASS" in process_result.stdout, combined_log[-4000:]
    line = next(l for l in combined_log.splitlines() if "FSDP auto bucket:" in l)
    assert int(line.split("FSDP auto bucket: ")[1].split()[0]) >= 1, line
    inserted_load_count = int(
        next(l for l in combined_log.splitlines() if "inserted" in l and "h2d_load" in l).split("inserted ")[1].split()[0]
    )
    assert inserted_load_count == _parse_marker_float(process_result.stdout, "OFFLOAD_FREED", "shards"), "one load per weight"


_LOAD_SCHEDULE_LOG_PREFIX = "h2d load reorder: "
"""Prefix of the H2D scheduling summary parsed by these assertions."""


def _find_load_schedule_log(combined_log: str) -> str:
    """The INFO summary, not the WARNING lines that share the same prefix."""
    return next(l for l in combined_log.splitlines() if _LOAD_SCHEDULE_LOG_PREFIX in l and "Hoisted " in l)


def _parse_inflight_peak_and_budget_mib(combined_log: str) -> tuple[float, float]:
    """The in-flight peak and the budget it was held to, off the reorder's report."""
    peak_mib_text, budget_log_suffix = (
        _find_load_schedule_log(combined_log).split("in-flight peak ")[1].split(" MiB of the ", 1)
    )
    return float(peak_mib_text), float(budget_log_suffix.split(" MiB")[0])


@requires_cuda
@requires_torchrun
def test_the_inflight_budget_is_never_exceeded():
    """Without residency, hoisted loads must fit the default in-flight budget:
    the unhoisted live-buffer peak plus one largest load."""
    process_result = _run_offload_e2e(1, *_MULTI_LOAD_ARGS)
    combined_log = _require_success_or_skip(process_result)
    inflight_peak_mib, inflight_budget_mib = _parse_inflight_peak_and_budget_mib(combined_log)
    assert (
        0 < inflight_peak_mib <= inflight_budget_mib
    ), f"in-flight peak {inflight_peak_mib} MiB over the {inflight_budget_mib} MiB budget"
    assert inflight_budget_mib <= 4.5 * 2, f"unhoisted floor plus one bucket, not more: {inflight_budget_mib} MiB"


@requires_cuda
@requires_torchrun
def test_no_residency_budget_keeps_nothing_resident():
    """Without a residency budget, weights must stay offloaded and outputs must match eager."""
    process_result = _run_offload_e2e(1, *_MULTI_LOAD_ARGS)
    combined_log = _require_success_or_skip(process_result)
    assert "OFFLOAD_PASS" in process_result.stdout, combined_log[-4000:]
    assert _parse_marker_float(process_result.stdout, "OFFLOAD_FREED", "promoted_mib") == 0, "an empty budget must not promote"


@requires_cuda
@requires_torchrun
def test_residency_and_inflight_fit_weight_budget():
    """The one budget: residency plus in-flight room, split by the pass.

    This shape has more exposed transfer than the in-flight room alone can hide,
    so part of the budget goes to residency, and the two halves together must
    stay inside what was offered.
    """
    process_result = _run_offload_e2e(1, *_MULTI_LOAD_ARGS, "--max-device-weight-mib", "24")
    combined_log = _require_success_or_skip(process_result)
    resident_mib = _parse_marker_float(process_result.stdout, "OFFLOAD_FREED", "promoted_mib")
    _, inflight_budget_mib = _parse_inflight_peak_and_budget_mib(combined_log)
    assert "OFFLOAD_PASS" in process_result.stdout, combined_log[-4000:]
    assert resident_mib > 0, f"some of the budget has to buy residency: {_find_load_schedule_log(combined_log)}"
    assert (
        resident_mib + inflight_budget_mib <= 24
    ), f"residency {resident_mib} + in-flight {inflight_budget_mib} MiB over 24 MiB"


@requires_cuda
@requires_torchrun
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires >=2 GPUs")
def test_offload_multi_rank():
    """world=2: a real 2-rank gather, fed by per-rank shards that only exist in
    host memory until the graph loads them."""
    process_result = _run_offload_e2e(2)
    combined_log = _require_success_or_skip(process_result)
    assert "OFFLOAD_PASS" in process_result.stdout, combined_log[-4000:]


# ------------------------------------------------------- host-first loading


@requires_cuda
@requires_torchrun
def test_host_first_materialization_has_negligible_cuda_allocation():
    """Materializing and filling parked weights must avoid full CUDA weight allocation.
    Measure this phase separately from later handoff and graph execution."""
    process_result = _run_offload_e2e(1, "--host-first")
    combined_log = _require_success_or_skip(process_result)
    assert "OFFLOAD_PASS" in process_result.stdout, combined_log[-4000:]

    materialization_peak_mib = _parse_marker_float(process_result.stdout, "OFFLOAD_LOAD", "peak_mib")
    weight_size_mib = _parse_marker_float(process_result.stdout, "OFFLOAD_LOAD", "weights_mib")
    assert weight_size_mib > 1, "the shape under test is supposed to have weights worth offloading"
    assert (
        materialization_peak_mib < 0.5
    ), f"materializing and filling the model should cost no device memory, cost {materialization_peak_mib} MiB"


@requires_cuda
@requires_torchrun
def test_host_first_weights_are_all_claimed_by_the_graph():
    """All parked weights in this model must be claimed by graph loads.
    Restoring an unclaimed weight is safe but would lose its offload saving."""
    process_result = _run_offload_e2e(1, "--host-first")
    combined_log = _require_success_or_skip(process_result)
    assert _parse_marker_float(process_result.stdout, "OFFLOAD_FREED", "shards") > 0, combined_log[-4000:]
    assert _parse_marker_float(process_result.stdout, "OFFLOAD_FREED", "restored_unclaimed") == 0, combined_log[-4000:]


@requires_cuda
@requires_torchrun
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires >=2 GPUs")
def test_host_first_multi_rank():
    """world=2: ranks in a shard group vote per candidate, not WORLD-all-or-nothing.

    A rank-dependent parking decision drops only the weights they do not share;
    the rest stay offloaded.  An empty intersection would log ``nothing to offload``.
    """
    process_result = _run_offload_e2e(2, "--host-first")
    combined_log = _require_success_or_skip(process_result)
    assert "OFFLOAD_PASS" in process_result.stdout, combined_log[-4000:]
    assert "nothing to offload" not in combined_log, combined_log[-4000:]
    assert _parse_marker_float(process_result.stdout, "OFFLOAD_LOAD", "peak_mib") < 0.5
