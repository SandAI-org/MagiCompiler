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

* the planner -- ``fit_gather_cost_model``, the amortizing target and
  ``plan_bucket_segments`` -- on synthetic members, where every case the rule
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

from magi_compiler.passes.fsdp_overlap.auto_bucket import (
    BucketPlanParams,
    GatherMember,
    fit_gather_cost_model,
    gather_bytes_for_overhead_ratio,
    plan_bucket_segments,
)

BYTES_PER_MIB = 2**20

_COALESCE_HELPER_PATH = Path(__file__).parent / "fsdp_overlap_helper" / "ir_coalesce_helper.py"
_RANK_CONSENSUS_HELPER_PATH = Path(__file__).parent / "fsdp_overlap_helper" / "auto_bucket_rank_helper.py"

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
def test_fit_gather_cost_model(points, kwargs, expected):
    alpha, beta = fit_gather_cost_model(points, **kwargs)
    assert (alpha, beta) == pytest.approx(expected)
    assert alpha >= 0 and beta >= 0


# --------------------------------------------------------------- the planner

# alpha and beta chosen so that, at a 10% launch share, the target is exactly
# 100 MiB of gathered bytes (and the cap 200 MiB).
_LAUNCH_OVERHEAD_NS = 90e3
_TRANSFER_NS_PER_BYTE = _LAUNCH_OVERHEAD_NS * 0.9 / (0.1 * 100 * BYTES_PER_MIB)


def _bucket_params(**kwargs):
    return BucketPlanParams(alpha_ns=_LAUNCH_OVERHEAD_NS, beta_ns_per_byte=_TRANSFER_NS_PER_BYTE, overhead_ratio=0.1, **kwargs)


def _gather_members(out_mib, *, group_size: int = 8):
    """One gather per entry (gathered MiB), each at ``3k`` with its wait at
    ``3k+1`` and its reader at ``3k+2``."""
    return [
        GatherMember(
            launch_index=3 * k,
            earliest_launch_index=0,
            first_wait_index=3 * k + 1,
            shard_bytes=int(o * BYTES_PER_MIB) // group_size,
            gathered_bytes=int(o * BYTES_PER_MIB),
        )
        for k, o in enumerate(out_mib)
    ]


def _bucket_sizes_mib(members, segs):
    return [sum(m.gathered_bytes for m in members[s:e]) / BYTES_PER_MIB for s, e in segs]


def test_bucket_target_matches_launch_overhead_ratio():
    target = gather_bytes_for_overhead_ratio(_LAUNCH_OVERHEAD_NS, _TRANSFER_NS_PER_BYTE, 0.1)
    assert target == pytest.approx(100 * BYTES_PER_MIB)
    assert _LAUNCH_OVERHEAD_NS / (_LAUNCH_OVERHEAD_NS + _TRANSFER_NS_PER_BYTE * target) == pytest.approx(0.1)
    assert gather_bytes_for_overhead_ratio(0.0, _TRANSFER_NS_PER_BYTE, 0.1) == 0.0, "no launch cost: nothing to amortize"
    assert gather_bytes_for_overhead_ratio(_LAUNCH_OVERHEAD_NS, 0.0, 0.1) == float(
        "inf"
    ), "free bytes: merge whatever is legal"


def test_the_ratio_defaults_per_transport():
    """A copy-engine launch costs far more fixed time and far less per byte than an
    NCCL one: one ratio would give it a target several times the NCCL one."""
    from magi_compiler.passes.fsdp_overlap.auto_bucket import DEFAULT_OVERHEAD_RATIOS, FsdpAutoBucket

    assert DEFAULT_OVERHEAD_RATIOS == {"nccl": 0.1, "ce": 0.5}
    assert FsdpAutoBucket(runtime_ns_fn=lambda s: 0.0).overhead_ratio is None
    assert FsdpAutoBucket(runtime_ns_fn=lambda s: 0.0, overhead_ratio=0.3).overhead_ratio == 0.3

    from magi_compiler.config import FSDPConfig

    with pytest.raises(ValueError):
        FSDPConfig(auto_bucket_overhead_ratio=1.5)


def test_each_process_group_gets_its_own_fit(monkeypatch):
    """Fit single-rank local copies separately from multi-rank gathers;
    pooling their timings must not underprice collective transfers."""
    from magi_compiler.passes.fsdp_overlap import auto_bucket
    from magi_compiler.passes.fsdp_overlap.ir_coalesce import GatherInfo
    from magi_compiler.passes.transfer_overlap import CostView

    class _Gather:
        def __init__(self, group, group_size, local_mib, us_per_gathered_mib, fixed_us):
            self.info = GatherInfo(group, group_size, torch.bfloat16, int(local_mib * BYTES_PER_MIB))
            self.ns = (fixed_us + us_per_gathered_mib * local_mib * group_size) * 1e3

    dense = [_Gather("dense", 8, mib, 3.9, 44.0) for mib in (22.5, 15.0, 144.0, 9.0, 36.0)]
    expert = [_Gather("expert", 1, mib, 0.32, 10.0) for mib in (1920.0, 960.0)]
    gather_candidates = dense + expert
    monkeypatch.setattr(auto_bucket, "parse_single_gather", lambda s: s.info)

    params = auto_bucket.FsdpAutoBucket(runtime_ns_fn=lambda s: s.ns)._fit_bucket_params(
        gather_candidates, CostView(lambda s: s.ns)
    )

    assert set(params) == {("nccl", "dense", 8), ("nccl", "expert", 1)}
    assert params[("nccl", "dense", 8)].beta_ns_per_byte * BYTES_PER_MIB == pytest.approx(3.9e3)
    assert params[("nccl", "dense", 8)].target_gathered_bytes / BYTES_PER_MIB == pytest.approx(
        44 * 0.9 / (0.1 * 3.9), rel=1e-6
    )
    assert params[("nccl", "expert", 1)].beta_ns_per_byte * BYTES_PER_MIB == pytest.approx(0.32e3)

    _, pooled_beta = fit_gather_cost_model([(s.info.shard_bytes * s.info.world_size, s.ns) for s in gather_candidates])
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
def test_plan_bucket_segments(out_mib, expected):
    members = _gather_members(out_mib)
    params = _bucket_params()
    segs = plan_bucket_segments(members, params)
    assert segs == expected
    # An already-large singleton may exceed twice the target; a merged bucket may not.
    for (start, end), size in zip(segs, _bucket_sizes_mib(members, segs)):
        if end - start > 1:
            assert size * BYTES_PER_MIB < 2 * params.target_gathered_bytes


def test_bucket_size_mib_caps_local_bytes():
    members = _gather_members([4] * 8)  # 0.5 MiB local each
    segs = plan_bucket_segments(members, _bucket_params(max_shard_bytes=1 * BYTES_PER_MIB))
    assert segs == [(0, 2), (2, 4), (4, 6), (6, 8)]


def test_a_bucket_never_spans_a_wait_its_launch_cannot_precede():
    """Member 3's shard is produced after member 2's wait: no single launch can
    follow the one and precede the other, so the bucket breaks there."""
    members = _gather_members([4] * 6)
    members[3] = GatherMember(
        launch_index=9,
        earliest_launch_index=8,
        first_wait_index=10,
        shard_bytes=BYTES_PER_MIB // 2,
        gathered_bytes=4 * BYTES_PER_MIB,
    )
    segs = plan_bucket_segments(members, _bucket_params())
    assert segs == [(0, 3), (3, 6)]
    for s, e in segs:
        seg = members[s:e]
        segment_launch_index = max(seg[0].launch_index, max(m.earliest_launch_index for m in seg))
        assert segment_launch_index <= min(m.first_wait_index for m in seg)


# ------------------------------------------------- weight gathers and skeleton


def _gather_snode(*, tagged: bool, dtype=torch.bfloat16):
    """A single NCCL gather launch with just the surface the predicates read."""
    from types import SimpleNamespace

    from torch._inductor import ir

    from magi_compiler.passes.fsdp_overlap.node_meta import WEIGHT_AG
    from magi_compiler.passes.snode_utils import ALL_GATHER_OP

    node = object.__new__(ir._CollectiveKernel)
    origin = SimpleNamespace(meta={"custom": {WEIGHT_AG: True}} if tagged else {})
    for key, value in {
        "op_overload": ALL_GATHER_OP,
        "origins": [origin],
        "get_size": lambda: [8, 8],
        "get_dtype": lambda: dtype,
    }.items():
        object.__setattr__(node, key, value)
    return SimpleNamespace(snodes=None, node=node)


def test_weight_gathers_prefer_tags_and_fallback_to_all():
    from magi_compiler.passes.snode_utils import weight_gathers

    weight, activation = _gather_snode(tagged=True), _gather_snode(tagged=False)
    assert weight_gathers([activation, weight]) == [weight], "a tagged graph leaves its activation gathers alone"
    other = _gather_snode(tagged=False)
    assert weight_gathers([activation, other]) == [activation, other], "an untagged graph treats every gather as a weight"


def test_collective_signature_includes_weight_tag_and_dtype():
    """Slot mode, which auto-bucket requires, compares these keys; rank 0's bucket
    indices only name the same collectives on a peer whose keys match."""
    from magi_compiler.passes.transfer_overlap.rank_sync import _collective_signature

    bf16 = _gather_snode(tagged=True)
    assert _collective_signature(bf16, True) == _collective_signature(_gather_snode(tagged=True), True)
    assert _collective_signature(bf16, True) != _collective_signature(_gather_snode(tagged=True, dtype=torch.float32), True)
    assert _collective_signature(bf16, True) != _collective_signature(bf16, False)


# ---------------------------------------------------------------- end to end


def _free_port() -> str:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return str(s.getsockname()[1])


def _run_coalesce_e2e(*extra: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["MAGI_LOGGING_LEVEL"] = env.get("MAGI_LOGGING_LEVEL", "info")
    # The FX graph is the same with and without the scheduler pass, so a shared
    # cache would replay another run's artifact and skip the pass under test.
    env["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
    with tempfile.TemporaryDirectory(prefix="magi_auto_bucket_") as cache_root:
        env["MAGI_COMPILE_CACHE_ROOT_DIR"] = cache_root
        return subprocess.run(
            ["torchrun", "--nproc_per_node=2", f"--master_port={_free_port()}", str(_COALESCE_HELPER_PATH), *extra],
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )


def _parse_marker_int(stdout: str, marker: str, field: str) -> int:
    line = next(l for l in stdout.splitlines() if l.startswith(marker))
    return int(line.split(f"{field}=")[1].split()[0])


@requires_2gpu
def test_fixed_pairs_coalesce_under_their_old_names():
    """The surgery alone: every pair merged, output equal to eager, and each wait
    reading the unpack that took over the replaced gather's buffer name."""
    process_result = _run_coalesce_e2e("--pair", "2")
    combined_log = process_result.stdout + process_result.stderr
    assert process_result.returncode == 0 and "IRC_PASS" in process_result.stdout, combined_log[-4000:]
    buckets = _parse_marker_int(process_result.stdout, "IRC_COALESCED", "buckets")
    assert buckets == 4, combined_log[-4000:]
    assert _parse_marker_int(process_result.stdout, "IRC_CODE", "coalesced_calls") == buckets
    assert _parse_marker_int(process_result.stdout, "IRC_CODE", "plain_calls") == 0
    waits = _parse_marker_int(process_result.stdout, "IRC_WAITS", "total")
    assert waits == 8 and _parse_marker_int(process_result.stdout, "IRC_WAITS", "on_unpacks") == waits, combined_log[-4000:]


@requires_2gpu
@pytest.mark.parametrize(
    "extra,cap_mib",
    [
        pytest.param((), None, id="analytical"),
        # Profiled costs, but a fixed 64 MiB target: whether to merge must not depend on
        # measured latency, and the 4 MiB cap, not the target, has to size the buckets.
        pytest.param(
            ("--cost-mode", "profile_sync", "--bucket-size-mib", "4", "--bucket-target-mib", "64"), 4.0, id="profile-sync-cap"
        ),
        # Inject a missing bucket on rank 1; both ranks must adopt rank 0's plan.
        pytest.param(("--bucket-size-mib", "4", "--diverge"), 4.0, id="divergent-plan"),
        # Rank 1 adds compute but keeps the collective skeleton: exercise SLOT consensus.
        pytest.param(("--mismatch",), None, id="different-compute-graphs"),
    ],
)
def test_auto_bucketing_end_to_end(extra, cap_mib):
    process_result = _run_coalesce_e2e("--pair", "0", "--n-layers", "8", "--hidden", "1024", *extra)
    combined_log = process_result.stdout + process_result.stderr
    assert process_result.returncode == 0 and "IRC_PASS" in process_result.stdout, combined_log[-4000:]
    assert "FSDP auto bucket:" in combined_log, combined_log[-4000:]
    assert _parse_marker_int(process_result.stdout, "IRC_CODE", "coalesced_calls") >= 1
    assert "unbucketed" not in combined_log, combined_log[-4000:]
    assert "snode cost table:" not in combined_log, "the pass must price the snodes it builds"

    if cap_mib is not None:
        # Plan-consensus messages share the prefix; select the size report explicitly.
        line = next(l for l in combined_log.splitlines() if "FSDP auto bucket:" in l and "min/median/max " in l)
        biggest = float(line.split("min/median/max ")[1].split(";")[0].split("/")[2])
        assert biggest <= cap_mib, line

    # This marker reports bucket/member counts, not the full membership plan.
    per_rank = next(l for l in process_result.stdout.splitlines() if l.startswith("IRC_RANK_BUCKETS")).split()[1:]
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
    process_result = _run_gloo_helper(_RANK_CONSENSUS_HELPER_PATH)
    combined_log = process_result.stdout + process_result.stderr
    assert process_result.returncode == 0, combined_log[-4000:]
    # These markers report verdicts reduced across both ranks. Per-rank lines
    # remain diagnostic only, because concurrent prints can share a line.
    assert "BUCKET_RANK_PASS" in process_result.stdout, combined_log[-4000:]
    assert "SLOT_PASS" in process_result.stdout, combined_log[-4000:]
