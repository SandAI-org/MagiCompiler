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

"""torchrun entrypoint: heterogeneous candidate counts must still negotiate.

Two gloo ranks, no CUDA and no compile.  Rank 0 discovers one bucket candidate
and rank 1 discovers two; their graph fingerprints differ too.  Only those two
lookups are replaced.  ``FsdpAutoBucket`` then runs for real, including
``negotiate_mode``'s all_gather.  Both ranks must take ``abort`` and return the
schedule unchanged.  A rank that returns before the exchange deadlocks the other
(the process group timeout is what turns that hang into a failed run).

Markers (every rank, then rank 0):
  BUCKET_NEGOTIATED rank=<n> mode=<mode> n=<candidates> unchanged=<bool>
  BUCKET_RANK_PASS / BUCKET_RANK_FAIL

Run: torchrun --nproc_per_node=2 tests/feature_tests/fsdp/fsdp_overlap_helper/auto_bucket_rank_helper.py
"""

from __future__ import annotations

import os
from datetime import timedelta

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


class _Snode:
    """Enough of a schedule node for the skeleton walk. Candidates and the fingerprint are stubbed."""

    snodes = None

    def __init__(self) -> None:
        self.node = _IR()


def main() -> None:
    dist.init_process_group("gloo", timeout=timedelta(seconds=15))
    rank = dist.get_rank()
    n_cands = 1 if rank == 0 else 2
    all_ok = False
    try:

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

        FsdpAutoBucket._candidates = staticmethod(candidates)
        rank_sync.graph_fingerprint = fingerprint
        rank_sync.negotiate_mode = negotiate

        snodes = [_Snode() for _ in range(3)]
        out = FsdpAutoBucket(cost_fn=lambda s: 0.0)(snodes)
        unchanged = out is snodes
        ok = seen.get("mode") == "abort" and seen.get("n") == n_cands and unchanged
        print(f"BUCKET_NEGOTIATED rank={rank} mode={seen.get('mode')} n={seen.get('n')} unchanged={unchanged}", flush=True)
        verdict = torch.tensor([1 if ok else 0], dtype=torch.int32)
        dist.all_reduce(verdict, op=dist.ReduceOp.MIN)
        all_ok = bool(verdict.item())
        if rank == 0:
            print("BUCKET_RANK_PASS" if all_ok else "BUCKET_RANK_FAIL", flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
