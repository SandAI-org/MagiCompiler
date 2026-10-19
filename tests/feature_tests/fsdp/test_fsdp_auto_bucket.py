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

Tested separately:

* the planner -- ``fit_alpha_beta``, the amortizing target and
  ``plan_segments`` -- on synthetic members, where every case the rule
  distinguishes (alone, merged, capped, illegal) can be built exactly;
* the IR surgery and the pass, end to end on a 2-rank SimpleFSDP model through
  ``fsdp_overlap_helper/ir_coalesce_helper.py``: the compiled output must match
  eager, the wrapper must launch coalesced gathers, and every wait must still
  read a buffer the coalesced launch's unpack produced under its old name;
* the cross-rank skip, on gloo without a compile: a rank with fewer than two
  candidates still joins mode negotiation, then every rank leaves the graph alone;
* ``slot`` mode, on gloo without a compile: rank 0's plan is applied only where
  every rank's merge keeps the collective order.
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
_RANK_HELPER = Path(__file__).parent / "fsdp_overlap_helper" / "auto_bucket_rank_helper.py"

requires_2gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2 or shutil.which("torchrun") is None,
    reason="requires 2 GPUs and torchrun",
)


# ------------------------------------------------------------ alpha / beta


@pytest.mark.parametrize(
    "points,kwargs,expected",
    [
        pytest.param([(1e6, 25e3), (2e6, 30e3), (4e6, 40e3)], {}, (20e3, 5e-3), id="linear"),
        pytest.param([(1e6, 30e3), (1e6, 30e3)], {"alpha_guess_ns": 10e3}, (10e3, 20e3 / 1e6), id="one-size-guess"),
        pytest.param([(1e6, 50e3), (2e6, 40e3)], {}, (45e3, 0.0), id="negative-slope"),
        # A negative intercept falls back to a least-squares fit through the origin.
        pytest.param([(1e6, 1e3), (4e6, 40e3)], {}, (0.0, 161e9 / 17e12), id="negative-intercept"),
    ],
)
def test_fit_alpha_beta(points, kwargs, expected):
    alpha, beta = fit_alpha_beta(points, **kwargs)
    assert (alpha, beta) == pytest.approx(expected)
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
        Member(idx=3 * k, lo=0, wait=3 * k + 1, local_bytes=int(o * MiB) // world, out_bytes=int(o * MiB))
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


def test_the_ratio_defaults_per_transport():
    """A copy-engine launch costs far more fixed time and far less per byte than an
    NCCL one: one ratio would give it a target several times the NCCL one."""
    from magi_compiler.passes.fsdp_overlap.auto_bucket import DEFAULT_OVERHEAD_RATIOS, FsdpAutoBucket

    assert DEFAULT_OVERHEAD_RATIOS == {"nccl": 0.1, "ce": 0.5}
    assert FsdpAutoBucket(cost_fn=lambda s: 0.0).overhead_ratio is None
    assert FsdpAutoBucket(cost_fn=lambda s: 0.0, overhead_ratio=0.3).overhead_ratio == 0.3
    with pytest.raises(ValueError):
        FsdpAutoBucket(cost_fn=lambda s: 0.0, overhead_ratio=1.5)


def test_each_process_group_gets_its_own_fit(monkeypatch):
    """Fit single-rank local copies separately from multi-rank gathers;
    pooling their timings must not underprice collective transfers."""
    from magi_compiler.passes.fsdp_overlap import auto_bucket
    from magi_compiler.passes.fsdp_overlap.ir_coalesce import GatherInfo
    from magi_compiler.passes.overlap import CostView

    class _Gather:
        def __init__(self, group, world, local_mib, us_per_gathered_mib, fixed_us):
            self.info = GatherInfo(group, world, torch.bfloat16, int(local_mib * MiB))
            self.ns = (fixed_us + us_per_gathered_mib * local_mib * world) * 1e3

    dense = [_Gather("dense", 8, mib, 3.9, 44.0) for mib in (22.5, 15.0, 144.0, 9.0, 36.0)]
    expert = [_Gather("expert", 1, mib, 0.32, 10.0) for mib in (1920.0, 960.0)]
    cands = dense + expert
    monkeypatch.setattr(auto_bucket, "gather_info", lambda s: s.info)

    params = auto_bucket.FsdpAutoBucket(cost_fn=lambda s: s.ns)._params_by_group(cands, CostView(lambda s: s.ns))

    assert set(params) == {("nccl", "dense", 8), ("nccl", "expert", 1)}
    assert params[("nccl", "dense", 8)].beta_ns_per_byte * MiB == pytest.approx(3.9e3)
    assert params[("nccl", "dense", 8)].target_bytes / MiB == pytest.approx(44 * 0.9 / (0.1 * 3.9), rel=1e-6)
    assert params[("nccl", "expert", 1)].beta_ns_per_byte * MiB == pytest.approx(0.32e3)

    _, pooled_beta = fit_alpha_beta([(s.info.shard_bytes * s.info.world, s.ns) for s in cands])
    assert pooled_beta < 0.2 * params[("nccl", "dense", 8)].beta_ns_per_byte, "pooled, the copies set beta for all"


@pytest.mark.parametrize(
    "out_mib,expected",
    [
        pytest.param([400, 150, 100, 120], [(0, 1), (1, 2), (2, 3), (3, 4)], id="large-and-target-sized-alone"),
        # Two 100 MiB buckets, then the 40 MiB left over goes alone.
        pytest.param([4] * 60, [(0, 25), (25, 50), (50, 60)], id="tiny-gathers-fill-to-target"),
        pytest.param([4] * 5 + [300] + [4] * 5, [(0, 5), (5, 6), (6, 11)], id="large-gather-splits-run"),
        pytest.param([60, 60, 40], [(0, 2), (2, 3)], id="bucket-closes-at-target"),
    ],
)
def test_plan_segments(out_mib, expected):
    members = _members(out_mib)
    params = _params()
    segs = plan_segments(members, params)
    assert segs == expected
    # An already-large singleton may exceed twice the target; a merged bucket may not.
    for (start, end), size in zip(segs, _out(members, segs)):
        if end - start > 1:
            assert size * MiB < 2 * params.target_bytes


def test_bucket_size_mib_caps_local_bytes():
    members = _members([4] * 8)  # 0.5 MiB local each
    segs = plan_segments(members, _params(max_local_bytes=1 * MiB))
    assert segs == [(0, 2), (2, 4), (4, 6), (6, 8)]


def test_a_bucket_never_spans_a_wait_its_launch_cannot_precede():
    """Member 3's shard is produced after member 2's wait: no single launch can
    follow the one and precede the other, so the bucket breaks there."""
    members = _members([4] * 6)
    members[3] = Member(idx=9, lo=8, wait=10, local_bytes=MiB // 2, out_bytes=4 * MiB)
    segs = plan_segments(members, _params())
    assert segs == [(0, 3), (3, 6)]
    for s, e in segs:
        seg = members[s:e]
        assert max(seg[0].idx, max(m.lo for m in seg)) <= min(m.wait for m in seg)


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
@pytest.mark.parametrize(
    "extra,cap_mib",
    [
        pytest.param((), None, id="analytical"),
        pytest.param(("--cost-mode", "profile_sync", "--bucket-size-mib", "4"), 4.0, id="profile-sync-cap"),
        # Inject a missing bucket on rank 1; both ranks must adopt rank 0's plan.
        pytest.param(("--bucket-size-mib", "4", "--diverge"), 4.0, id="divergent-plan"),
        # Rank 1 adds compute but keeps the collective skeleton: exercise SLOT consensus.
        pytest.param(("--mismatch",), None, id="different-compute-graphs"),
    ],
)
def test_auto_bucketing_end_to_end(extra, cap_mib):
    p = _run("--pair", "0", "--n-layers", "8", "--hidden", "1024", *extra)
    out = p.stdout + p.stderr
    assert p.returncode == 0 and "IRC_PASS" in p.stdout, out[-4000:]
    assert "FSDP auto bucket:" in out, out[-4000:]
    assert _field(p.stdout, "IRC_CODE", "coalesced_calls") >= 1
    assert "unbucketed" not in out, out[-4000:]
    assert "snode cost table:" not in out, "the pass must price the snodes it builds"

    if cap_mib is not None:
        # Plan-consensus messages share the prefix; select the size report explicitly.
        line = next(l for l in out.splitlines() if "FSDP auto bucket:" in l and "min/median/max " in l)
        biggest = float(line.split("min/median/max ")[1].split(";")[0].split("/")[2])
        assert biggest <= cap_mib, line

    # This marker reports bucket/member counts, not the full membership plan.
    per_rank = next(l for l in p.stdout.splitlines() if l.startswith("IRC_RANK_BUCKETS")).split()[1:]
    assert len(per_rank) == 2 and len(set(per_rank)) == 1, per_rank
    assert int(per_rank[0].split("/")[0]) > 0, per_rank


requires_torchrun = pytest.mark.skipif(shutil.which("torchrun") is None, reason="requires torchrun")


def _run_gloo_helper(helper: Path) -> subprocess.CompletedProcess:
    root = str(Path(__file__).resolve().parents[3])
    env = os.environ.copy()
    prev = env.get("PYTHONPATH")
    env["PYTHONPATH"] = root if not prev else root + os.pathsep + prev
    env["PYTHONUNBUFFERED"] = "1"
    env["MAGI_LOGGING_LEVEL"] = env.get("MAGI_LOGGING_LEVEL", "info")
    return subprocess.run(
        ["torchrun", "--nproc_per_node=2", f"--master_port={_free_port()}", str(helper)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


@requires_torchrun
def test_auto_bucket_rank_consensus():
    """Check unequal candidate counts and slot-plan adoption in one two-rank run.

    A rank with one candidate must still negotiate abort with its peer. Then
    check matching slots, changed collective order, mismatched signatures, and
    illegal merges even when both ranks planned the same buckets.
    """
    p = _run_gloo_helper(_RANK_HELPER)
    out = p.stdout + p.stderr
    assert p.returncode == 0, out[-4000:]
    # These markers report verdicts reduced across both ranks. Per-rank lines
    # remain diagnostic only, because concurrent prints can share a line.
    assert "BUCKET_RANK_PASS" in p.stdout, out[-4000:]
    assert "SLOT_PASS" in p.stdout, out[-4000:]
