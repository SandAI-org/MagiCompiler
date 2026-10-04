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

"""Scheduler-stage weight all-gather bucketing (``bucket_mode='auto'``).

Two layers, tested separately:

* the planner -- ``fit_alpha_beta``, the amortizing target and
  ``plan_segments`` -- on synthetic members, where every case the rule
  distinguishes (alone, merged, capped, folded, illegal) can be built exactly;
  and the placement model in ``bucket_eval``;
* the IR surgery and the pass, end to end on a 2-rank SimpleFSDP model through
  ``fsdp_overlap_helper/ir_coalesce_helper.py``: the compiled output must match
  eager, the wrapper must launch coalesced gathers, and every wait must still
  read a buffer the coalesced launch's unpack produced under its old name.
"""

import os
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path

import pytest
import torch

from magi_compiler.passes.fsdp_overlap.auto_bucket import Member, PlanParams, amortizing_bytes, fit_alpha_beta, plan_segments

MiB = 2**20

_HELPER = Path(__file__).parent / "fsdp_overlap_helper" / "ir_coalesce_helper.py"

requires_2gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2 or shutil.which("torchrun") is None,
    reason="requires 2 GPUs and torchrun",
)


# ------------------------------------------------------------------ helpers


def _layers(n: int, *, local_mib: int = 8, world: int = 8, compute_ns: float = 200e3, lo: int = 0):
    """``n`` gathers, each at ``3k`` with its wait at ``3k+1`` and a consumer of
    ``compute_ns`` at ``3k+2`` -- the shape Inductor hands over, gather just
    above its own matmul."""
    members = [
        Member(
            idx=3 * k,
            lo=lo,
            wait=3 * k + 1,
            deadline=3 * k + 2,
            local_bytes=local_mib * MiB,
            out_bytes=local_mib * MiB * world,
        )
        for k in range(n)
    ]
    prefix = [0.0] * (3 * n + 1)
    for i in range(3 * n):
        prefix[i + 1] = prefix[i] + (compute_ns if i % 3 == 2 else 0.0)
    return members, prefix


def _sizes(segs):
    return [e - s for s, e in segs]


def _covers(segs, n):
    return [i for s, e in segs for i in range(s, e)] == list(range(n))


# ------------------------------------------------------------ alpha / beta


def test_fit_recovers_a_linear_cost():
    alpha, beta = fit_alpha_beta([(1e6, 25e3), (2e6, 30e3), (4e6, 40e3)])
    assert alpha == pytest.approx(20e3)
    assert beta == pytest.approx(5e-3)


def test_fit_with_one_size_splits_by_the_guess():
    """Every weight the same size: the fixed part is not observable, so the guess
    (capped at half the cheapest point) takes it and bytes take the rest."""
    alpha, beta = fit_alpha_beta([(1e6, 30e3), (1e6, 30e3)], alpha_guess_ns=10e3)
    assert alpha == pytest.approx(10e3)
    assert beta == pytest.approx(20e3 / 1e6)


def test_fit_never_goes_negative():
    alpha, beta = fit_alpha_beta([(1e6, 50e3), (2e6, 40e3)])
    assert alpha >= 0 and beta >= 0
    alpha, beta = fit_alpha_beta([(1e6, 1e3), (4e6, 40e3)])
    assert alpha >= 0 and beta >= 0


# --------------------------------------------------------------- the planner

# alpha and beta chosen so that, at a 10% launch share, the target is exactly
# 100 MiB of gathered bytes (and the cap 200 MiB).
_ALPHA = 90e3
_BETA = _ALPHA * 0.9 / (0.1 * 100 * MiB)


def _params(**kw):
    return PlanParams(alpha_ns=_ALPHA, beta_ns_per_byte=_BETA, overhead_ratio=0.1, **kw)


def _members(out_mib, *, world: int = 8):
    """One gather per entry (gathered MiB), each at ``3k`` with its wait at
    ``3k+1`` and its reader at ``3k+2``."""
    return [
        Member(idx=3 * k, lo=0, wait=3 * k + 1, deadline=3 * k + 2, local_bytes=int(o * MiB) // world, out_bytes=int(o * MiB))
        for k, o in enumerate(out_mib)
    ]


def _out(members, segs):
    return [sum(m.out_bytes for m in members[s:e]) / MiB for s, e in segs]


def test_the_target_is_where_the_launch_is_the_given_share():
    target = amortizing_bytes(_ALPHA, _BETA, 0.1)
    assert target == pytest.approx(100 * MiB)
    assert _ALPHA / (_ALPHA + _BETA * target) == pytest.approx(0.1)
    assert amortizing_bytes(0.0, _BETA, 0.1) == 0.0, "no launch cost: nothing to amortize"
    assert amortizing_bytes(_ALPHA, 0.0, 0.1) == float("inf"), "free bytes: merge whatever is legal"


def test_large_gathers_go_alone():
    members = _members([400, 150, 100, 120])
    assert plan_segments(members, _params()) == [(0, 1), (1, 2), (2, 3), (3, 4)]


def test_tiny_gathers_merge_until_their_launch_is_amortized():
    """60 gathers of 4 MiB: buckets of 25 reach the 100 MiB target; the last 10
    (40 MiB, under half the target) fold into the bucket before them."""
    members = _members([4] * 60)
    segs = plan_segments(members, _params())
    assert segs == [(0, 25), (25, 60)]
    assert max(_out(members, segs)) <= 200


def test_a_large_gather_ends_the_bucket_and_is_never_inside_one():
    members = _members([4] * 5 + [300] + [4] * 5)
    assert plan_segments(members, _params()) == [(0, 5), (5, 6), (6, 11)]


def test_a_bucket_never_grows_past_twice_the_target():
    members = _members([90, 90, 90])
    segs = plan_segments(members, _params())
    assert segs == [(0, 2), (2, 3)]
    assert max(_out(members, segs)) <= 200


def test_bucket_size_mib_caps_local_bytes():
    members = _members([4] * 8)  # 0.5 MiB local each
    segs = plan_segments(members, _params(max_local_bytes=1 * MiB))
    assert all(e - s <= 2 for s, e in segs), segs


def test_a_bucket_never_spans_a_wait_its_launch_cannot_precede():
    """Member 3's shard is produced after member 2's wait: no single launch can
    follow the one and precede the other, so the bucket breaks there."""
    members = _members([4] * 6)
    members[3] = Member(idx=9, lo=8, wait=10, deadline=11, local_bytes=MiB // 2, out_bytes=4 * MiB)
    segs = plan_segments(members, _params())
    assert segs == [(0, 3), (3, 6)]
    for s, e in segs:
        seg = members[s:e]
        assert max(seg[0].idx, max(m.lo for m in seg)) <= min(m.wait for m in seg)


def test_a_tail_that_would_break_the_cap_is_left_alone():
    members = _members([60, 60, 40])
    # [60, 60] reaches the target at 120; folding the 40 MiB tail would make 160,
    # still under the cap -- so it folds.  At 90+90 the fold would make 220.
    assert plan_segments(members, _params()) == [(0, 3)]
    members = _members([90, 90, 40])
    assert plan_segments(members, _params()) == [(0, 2), (2, 3)]


def test_the_plan_is_a_function_of_its_inputs():
    """What makes the ranks agree: same graph and costs, same plan, every time."""
    members = _members([4, 90, 2, 300, 7, 7, 7, 60, 1, 1] * 3)
    first = plan_segments(members, _params())
    assert all(plan_segments(list(members), _params()) == first for _ in range(3))


# ----------------------------------------------------- the placement model


def test_placement_launches_as_late_as_the_window_allows():
    """Enough compute: no lateness past the first layer (nothing runs before it),
    and each later launch exactly one layer ahead of its reader -- the latest
    index whose compute still covers the transfer."""
    from magi_compiler.passes.fsdp_overlap.bucket_eval import place

    members, prefix = _layers(4, compute_ns=1e6)
    params = PlanParams(alpha_ns=0.0, beta_ns_per_byte=0.5e6 / (64 * MiB), margin_ns=0.0)
    placed = place(members, [(i, i + 1) for i in range(4)], prefix, params)
    assert placed.late_ns == [0.5e6, 0.0, 0.0, 0.0]
    # Layer k's gather reads compute at 3k+2; half a layer of transfer fits behind
    # layer k-1's compute, which runs at 3(k-1)+2 = 3k-1.
    assert placed.launches[1:] == [3 * k - 1 for k in range(1, 4)]


def test_placement_lateness_is_the_least_a_serial_stream_allows():
    from magi_compiler.passes.fsdp_overlap.bucket_eval import place

    members, prefix = _layers(3, compute_ns=1e5)
    params = PlanParams(alpha_ns=0.0, beta_ns_per_byte=1e6 / (64 * MiB), margin_ns=0.0)
    placed = place(members, [(0, 1), (1, 2), (2, 3)], prefix, params)
    # Released at t=0, 1 ms each, back to back: finishes at 1, 2, 3 ms against
    # readers at 0, 0.1, 0.2 ms.
    assert placed.late_ns == pytest.approx([1e6, 2e6 - 1e5, 3e6 - 2e5])


def test_a_merged_member_is_allocated_at_the_bucket_launch():
    from magi_compiler.passes.fsdp_overlap.bucket_eval import peak_with, place

    members, prefix = _layers(4, compute_ns=1e6)
    params = PlanParams(alpha_ns=0.0, beta_ns_per_byte=0.5e6 / (64 * MiB), margin_ns=0.0)
    base = [0] * (len(prefix))
    singles = [(i, i + 1) for i in range(4)]
    merged = [(0, 2), (2, 4)]
    one = peak_with(base, [(members, singles, place(members, singles, prefix, params))])
    two = peak_with(base, [(members, merged, place(members, merged, prefix, params))])
    # One layer ahead, a gather holds at most its own 64 MiB early; a bucket of
    # two holds its second member across the first member's layer as well.
    assert one[0] == 64 * MiB
    assert two[0] == 2 * 64 * MiB


# ---------------------------------------------------------------- end to end


def _free_port() -> str:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return str(s.getsockname()[1])


def _run(*extra: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["MAGI_LOGGING_LEVEL"] = env.get("MAGI_LOGGING_LEVEL", "info")
    # The FX graph is the same with and without the scheduler pass, so a shared
    # cache would replay another run's artifact and skip the pass under test.
    env["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
    with tempfile.TemporaryDirectory(prefix="magi_auto_bucket_") as cache_root:
        env["MAGI_COMPILE_CACHE_ROOT_DIR"] = cache_root
        return subprocess.run(
            ["torchrun", "--nproc_per_node=2", f"--master_port={_free_port()}", str(_HELPER), *extra],
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )


def _field(stdout: str, marker: str, field: str) -> int:
    line = next(l for l in stdout.splitlines() if l.startswith(marker))
    return int(line.split(f"{field}=")[1].split()[0])


@requires_2gpu
def test_fixed_pairs_coalesce_under_their_old_names():
    """The surgery alone: every pair merged, output equal to eager, and each wait
    reading the unpack that took over the replaced gather's buffer name."""
    p = _run("--pair", "2")
    out = p.stdout + p.stderr
    assert p.returncode == 0 and "IRC_PASS" in p.stdout, out[-4000:]
    buckets = _field(p.stdout, "IRC_COALESCED", "buckets")
    assert buckets == 4, out[-4000:]
    assert _field(p.stdout, "IRC_CODE", "coalesced_calls") == buckets
    assert _field(p.stdout, "IRC_CODE", "plain_calls") == 0
    waits = _field(p.stdout, "IRC_WAITS", "total")
    assert waits == 8 and _field(p.stdout, "IRC_WAITS", "on_unpacks") == waits, out[-4000:]


@requires_2gpu
def test_auto_bucketing_end_to_end():
    p = _run("--pair", "0", "--n-layers", "8", "--hidden", "1024")
    out = p.stdout + p.stderr
    assert p.returncode == 0 and "IRC_PASS" in p.stdout, out[-4000:]
    assert "FSDP auto bucket:" in out, out[-4000:]
    assert _field(p.stdout, "IRC_CODE", "coalesced_calls") >= 1
    assert "snode cost table:" not in out, "the pass must price the snodes it builds"


@requires_2gpu
def test_auto_bucketing_respects_the_cap_under_rank_synchronized_costs():
    """profile_sync measures on each rank; the plan digest check is what keeps
    the buckets identical, and the cap bounds whatever the planner prefers."""
    p = _run("--pair", "0", "--n-layers", "8", "--hidden", "1024", "--cost-mode", "profile_sync", "--bucket-size-mib", "4")
    out = p.stdout + p.stderr
    assert p.returncode == 0 and "IRC_PASS" in p.stdout, out[-4000:]
    assert "unbucketed" not in out, out[-4000:]
    line = next(l for l in out.splitlines() if "FSDP auto bucket:" in l)
    biggest = float(line.split("min/median/max ")[1].split(";")[0].split("/")[2])
    assert biggest <= 4.0, line


@requires_2gpu
def test_memory_probe_reports_every_stage_and_changes_nothing():
    p = _run("--pair", "0", "--n-layers", "8", "--hidden", "1024", "--memory-probe")
    out = p.stdout + p.stderr
    assert p.returncode == 0 and "IRC_PASS" in p.stdout, out[-4000:]
    for tag in ("baseline", "after auto bucket", "after FSDP reorder"):
        assert f"memory probe [{tag}]: est. peak" in out, (tag, out[-4000:])
    assert "FSDP auto bucket model: base peak" in out, out[-4000:]
    assert "could not estimate" not in out and "model report failed" not in out, out[-4000:]


@requires_2gpu
def test_a_rank_whose_own_plan_differs_takes_rank_0s():
    """Per-rank measured costs can flip a near-tie plan (wan: 4 plans on 8 ranks).

    Falling back to no buckets there made the model slower than the manual
    size; instead every rank must apply rank 0's plan, and the collectives,
    being the same sequence everywhere, must still produce eager's output.
    """
    p = _run("--pair", "0", "--n-layers", "8", "--hidden", "1024", "--bucket-size-mib", "4", "--diverge")
    out = p.stdout + p.stderr
    assert p.returncode == 0 and "IRC_PASS" in p.stdout, out[-4000:]
    assert "every rank takes rank 0's" in out, out[-4000:]
    assert "unbucketed" not in out, out[-4000:]
    per_rank = next(l for l in p.stdout.splitlines() if l.startswith("IRC_RANK_BUCKETS")).split()[1:]
    assert len(set(per_rank)) == 1 and per_rank[0].split("/")[0] != "0", per_rank
