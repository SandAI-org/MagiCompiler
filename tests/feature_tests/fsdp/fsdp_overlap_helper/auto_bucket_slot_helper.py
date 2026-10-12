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

"""torchrun entrypoint: ``FsdpAutoBucket._adopt`` under ``slot`` mode.

Two gloo ranks, no CUDA and no compile.  Each rank holds the order
``[g0, x, g1, g2]`` -- three weight gathers and one CP / EP collective ``x`` --
and rank 0's plan merges ``g0`` and ``g1``.  Only the per-rank merge (where the
bucket's launch lands) and the candidate signature are stubbed; the exchanges
are real.  Every rank walks the cases in the same order, so the collectives
inside ``_adopt`` stay lockstep.

* same:      both ranks land the bucket before ``x``           -> applied
* hop:       rank 1 lands it after ``x``                       -> unbucketed
* signature: rank 1's second gather is another dtype           -> unbucketed
* identical: rank 1 lands it after ``x``, but graphs are equal -> applied
  (the slot check is only for ``slot``; identical graphs merge identically)

Markers (every rank, then rank 0):
  SLOT_CASE rank=<n> case=<name> expected=<applied|unbucketed> got=<...>
  SLOT_PASS / SLOT_FAIL

Run: torchrun --nproc_per_node=2 tests/feature_tests/fsdp/fsdp_overlap_helper/auto_bucket_slot_helper.py
"""

from __future__ import annotations

import os
from datetime import timedelta

os.environ.setdefault("MAGI_LOGGING_LEVEL", "info")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from magi_compiler.passes.fsdp_overlap.auto_bucket import FsdpAutoBucket  # noqa: E402


class _Snode:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name


def main() -> None:
    dist.init_process_group("gloo", timeout=timedelta(seconds=15))
    rank = dist.get_rank()
    world = dist.get_world_size()
    all_ok = False
    try:
        g0, x, g1, g2 = (_Snode(n) for n in ("g0", "x", "g1", "g2"))
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
            FsdpAutoBucket._merged = staticmethod(lambda order, buckets, m=merged: m)
            FsdpAutoBucket._signature = staticmethod(lambda cands, s=signature: s)
            p = FsdpAutoBucket(cost_fn=lambda s: 0.0)
            # Rank 1 planned no buckets of its own, so it must take rank 0's.
            buckets = [[g0, g1]] if rank == 0 else []
            out = p._adopt(True, buckets, order, cands, None, world, mode, skeleton)
            got = "unbucketed" if out is None else "applied"
            if out is not None and [[s.name for s in b] for b in out] != ["g0 g1".split()]:
                got = f"wrong:{out}"
            ok = ok and got == expected
            print(f"SLOT_CASE rank={rank} case={name} expected={expected} got={got}", flush=True)

        verdict = torch.tensor([1 if ok else 0], dtype=torch.int32)
        dist.all_reduce(verdict, op=dist.ReduceOp.MIN)
        all_ok = bool(verdict.item())
        if rank == 0:
            print("SLOT_PASS" if all_ok else "SLOT_FAIL", flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
