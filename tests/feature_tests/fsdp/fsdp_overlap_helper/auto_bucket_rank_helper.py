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

First, rank 0 discovers one candidate and rank 1 discovers two, with different
fingerprints. Both must negotiate abort and leave their schedules unchanged;
an early return hangs the peer and is caught by the process-group timeout.

Then both ranks exercise adoption of rank 0's bucket plan:
* same: matching collective order and signatures -> applied;
* hop: rank 1 moves the bucket across another collective -> unbucketed;
* signature: rank 1 has a different candidate dtype -> unbucketed;
* identical: identical-graph mode bypasses the slot-order check -> applied.

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
from magi_compiler.passes.overlap import rank_sync  # noqa: E402


class _IR:
    op_overload = "fake.op"
    origins = None

    def get_size(self):
        return [8, 8]


class _CandidateSnode:
    """Enough of a schedule node for the skeleton walk."""

    snodes = None

    def __init__(self) -> None:
        self.node = _IR()


class _SlotSnode:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name


def _report(ok: bool, marker: str) -> bool:
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

    def fingerprint(order):
        return f"fp-{rank}"

    seen: dict = {}
    real_negotiate = rank_sync.negotiate_mode

    def negotiate(order, anchors, skel_kinds, costs_ok=True, who="FSDP overlap"):
        mode, group, world = real_negotiate(order, anchors, skel_kinds, costs_ok, who=who)
        seen["mode"] = mode
        seen["n"] = len(anchors)
        return mode, group, world

    with (
        patch.object(FsdpAutoBucket, "_candidates", staticmethod(candidates)),
        patch.object(rank_sync, "graph_fingerprint", fingerprint),
        patch.object(rank_sync, "negotiate_mode", negotiate),
    ):
        snodes = [_CandidateSnode() for _ in range(3)]
        out = FsdpAutoBucket(cost_fn=lambda s: 0.0)(snodes)

    unchanged = out is snodes
    ok = seen.get("mode") == "abort" and seen.get("n") == n_cands and unchanged
    print(f"BUCKET_NEGOTIATED rank={rank} mode={seen.get('mode')} n={seen.get('n')} unchanged={unchanged}", flush=True)
    return _report(ok, "BUCKET_RANK")


def _check_slot_consensus() -> bool:
    rank, world = dist.get_rank(), dist.get_world_size()
    g0, x, g1, g2 = (_SlotSnode(n) for n in ("g0", "x", "g1", "g2"))
    order, cands, skeleton = [g0, x, g1, g2], [g0, g1, g2], [g0, x, g1, g2]
    before_x, after_x = [g0, g1, x, g2], [x, g0, g1, g2]
    sig = [("nccl", 2, "torch.bfloat16")] * 3
    odd_sig = [sig[0], ("nccl", 2, "torch.float32"), sig[2]]

    cases = [
        # (name, mode, rank 1's merged order, rank 1's signature, expected)
        ("same", "slot", before_x, sig, "applied"),
        ("hop", "slot", after_x, sig, "unbucketed"),
        ("signature", "slot", before_x, odd_sig, "unbucketed"),
        ("identical", "identical", after_x, sig, "applied"),
    ]
    ok = True
    for name, mode, odd_merged, odd_signature, expected in cases:
        merged = odd_merged if rank == 1 else before_x
        signature = odd_signature if rank == 1 else sig
        with patch.object(FsdpAutoBucket, "_merged", staticmethod(lambda order, buckets, m=merged: m)), patch.object(
            FsdpAutoBucket, "_signature", staticmethod(lambda cands, s=signature: s)
        ):
            p = FsdpAutoBucket(cost_fn=lambda s: 0.0)
            # Rank 1 planned no buckets of its own, so it must take rank 0's.
            buckets = [[g0, g1]] if rank == 0 else []
            out = p._adopt(True, buckets, order, cands, None, world, mode, skeleton)
        got = "unbucketed" if out is None else "applied"
        if out is not None and [[s.name for s in b] for b in out] != [["g0", "g1"]]:
            got = f"wrong:{out}"
        ok = ok and got == expected
        print(f"SLOT_CASE rank={rank} case={name} expected={expected} got={got}", flush=True)
    return _report(ok, "SLOT")


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
