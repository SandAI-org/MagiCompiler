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

"""Two gloo ranks exercise auto-bucket consensus without CUDA or compilation.

First, rank 0 discovers one candidate and rank 1 discovers two, with matching
collective skeletons. Both must negotiate abort and leave their schedules unchanged;
an early return hangs the peer and is caught by the process-group timeout.

Then both ranks exercise adoption of rank 0's bucket plan:
* same: matching collective order -> applied;
* hop: rank 1 moves the bucket across another collective -> unbucketed;
* same-plan-hop: both ranks planned the same buckets, but their merged orders differ -> unbucketed;
* same-plan-illegal: rank 0's shared plan is illegal on rank 1 -> unbucketed.

Each stage reports a verdict reduced across both ranks. All patches are scoped
so that stages share only the process group, not substituted implementations.

Run: torchrun --nproc_per_node=2 tests/feature_tests/fsdp/fsdp_overlap_helper/auto_bucket_rank_helper.py
"""

from __future__ import annotations

import os
from datetime import timedelta
from unittest.mock import patch

os.environ.setdefault("MAGI_LOGGING_LEVEL", "info")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from magi_compiler.passes.fsdp_overlap.auto_bucket import FsdpAutoBucket  # noqa: E402
from magi_compiler.passes.transfer_overlap import rank_sync  # noqa: E402


class FakeCollectiveIR:
    op_overload = "fake.op"
    origins = None

    def get_size(self):
        return [8, 8]


class _CandidateSnode:
    """Enough of a schedule node for the skeleton walk."""

    snodes = None

    def __init__(self) -> None:
        self.node = FakeCollectiveIR()


class _SlotSnode:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name


def _report_rank_consensus(ok: bool, marker: str) -> bool:
    verdict = torch.tensor([int(ok)], dtype=torch.int32)
    dist.all_reduce(verdict, op=dist.ReduceOp.MIN)
    all_ok = bool(verdict.item())
    if dist.get_rank() == 0:
        print(f"{marker}_{'PASS' if all_ok else 'FAIL'}", flush=True)
    return all_ok


def _check_candidate_negotiation() -> bool:
    rank = dist.get_rank()
    n_cands = 1 if rank == 0 else 2

    def candidates(order):
        return list(order[:n_cands])

    seen: dict = {}
    real_negotiate = rank_sync.negotiate_placement_mode

    def negotiate(candidate_count, collective_signatures, is_profiling_valid=True, pass_name="FSDP overlap"):
        mode, group, world_size = real_negotiate(
            candidate_count=candidate_count,
            collective_signatures=collective_signatures,
            is_profiling_valid=is_profiling_valid,
            pass_name=pass_name,
        )
        seen["mode"] = mode
        seen["n"] = candidate_count
        return mode, group, world_size

    with patch.object(FsdpAutoBucket, "_collect_candidates", staticmethod(candidates)), patch.object(
        rank_sync, "negotiate_placement_mode", negotiate
    ):
        snodes = [_CandidateSnode() for _ in range(3)]
        out = FsdpAutoBucket(runtime_ns_fn=lambda s: 0.0)(snodes)

    unchanged = out is snodes
    ok = seen.get("mode") == "abort" and seen.get("n") == n_cands and unchanged
    print(f"BUCKET_NEGOTIATED rank={rank} mode={seen.get('mode')} n={seen.get('n')} unchanged={unchanged}", flush=True)
    return _report_rank_consensus(ok, "BUCKET_RANK")


def _check_slot_consensus() -> bool:
    rank, world_size = dist.get_rank(), dist.get_world_size()
    g0, x, g1, g2 = (_SlotSnode(n) for n in ("g0", "x", "g1", "g2"))
    order, gather_candidates, skeleton = [g0, x, g1, g2], [g0, g1, g2], [g0, x, g1, g2]
    before_x, after_x = [g0, g1, x, g2], [x, g0, g1, g2]

    cases = [
        # (name, rank 1's merged order, same plan, expected)
        ("same", before_x, False, "applied"),
        ("hop", after_x, False, "unbucketed"),
        ("same-plan-hop", after_x, True, "unbucketed"),
        ("same-plan-illegal", None, True, "unbucketed"),
    ]
    ok = True
    for name, rank_one_order, same_plan, expected in cases:
        merged = rank_one_order if rank == 1 else before_x
        with patch.object(FsdpAutoBucket, "_simulate_bucket_merges", staticmethod(lambda order, buckets, m=merged: m)):
            p = FsdpAutoBucket(runtime_ns_fn=lambda s: 0.0)
            # Every rank adopts rank 0's plan, even when its own plan differs.
            buckets = [[g0, g1]] if rank == 0 or same_plan else []
            out = p._adopt_rank_zero_buckets(
                is_local_plan_valid=True,
                buckets=buckets,
                order=order,
                candidates=gather_candidates,
                sync_group=None,
                world_size=world_size,
                skeleton_nodes=skeleton,
            )
        adoption_status = "unbucketed" if out is None else "applied"
        if out is not None and [[s.name for s in b] for b in out] != [["g0", "g1"]]:
            adoption_status = f"wrong:{out}"
        ok = ok and adoption_status == expected
        print(f"SLOT_CASE rank={rank} case={name} expected={expected} got={adoption_status}", flush=True)
    return _report_rank_consensus(ok, "SLOT")


def main() -> None:
    dist.init_process_group("gloo", timeout=timedelta(seconds=15))
    try:
        negotiation_ok = _check_candidate_negotiation()
        # Always run both stages, even if the first stage reports a failure.
        slot_ok = _check_slot_consensus()
        dist.barrier()
    finally:
        dist.destroy_process_group()
    raise SystemExit(0 if negotiation_ok and slot_ok else 1)


if __name__ == "__main__":
    main()
