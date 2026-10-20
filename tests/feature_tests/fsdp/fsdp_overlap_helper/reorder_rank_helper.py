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

"""Two gloo ranks check that the FSDP reorder negotiates on every rank, without CUDA or compilation.

Rank 0 has one weight gather and rank 1 has none. Both must negotiate abort and
leave their schedules unchanged; a rank that returned before negotiating would
hang its peer, which the process-group timeout turns into a failure.

Run: torchrun --nproc_per_node=2 tests/feature_tests/fsdp/fsdp_overlap_helper/reorder_rank_helper.py
"""

from __future__ import annotations

import os
from datetime import timedelta
from unittest.mock import patch

os.environ.setdefault("MAGI_LOGGING_LEVEL", "info")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from magi_compiler.passes.fsdp_overlap import reorder  # noqa: E402
from magi_compiler.passes.overlap import rank_sync  # noqa: E402


class _IR:
    op_overload = "fake.op"
    origins = None

    def get_size(self):
        return [8, 8]


class _Snode:
    """Enough of a schedule node for the graph index and the skeleton walk."""

    snodes = None
    unmet_dependencies = ()

    def __init__(self, name: str, gather: bool) -> None:
        self.name = name
        self.gather = gather
        self.node = _IR()

    def get_name(self) -> str:
        return self.name

    def get_buffer_names(self) -> set[str]:
        return {self.name}


def _check_rank_without_gathers_negotiates() -> bool:
    rank = dist.get_rank()
    seen: dict = {}
    real_negotiate = rank_sync.negotiate_mode

    def negotiate(n_anchors, skel_kinds, costs_ok=True, who="FSDP overlap"):
        mode, group, world = real_negotiate(n_anchors=n_anchors, skel_kinds=skel_kinds, costs_ok=costs_ok, who=who)
        seen["mode"] = mode
        return mode, group, world

    snodes = [_Snode("buf0", gather=rank == 0), _Snode("buf1", gather=False)]
    gathers = lambda order: [s for s in order if s.gather]  # noqa: E731
    with patch.object(reorder, "weight_gathers", gathers), patch.object(rank_sync, "negotiate_mode", negotiate):
        out = reorder.FsdpOverlapReorder(cost_fn=lambda s: 0.0)(list(snodes))

    unchanged = out == snodes
    ok = seen.get("mode") == "abort" and unchanged
    print(f"REORDER_NEGOTIATED rank={rank} mode={seen.get('mode')} unchanged={unchanged}", flush=True)
    verdict = torch.tensor([int(ok)], dtype=torch.int32)
    dist.all_reduce(verdict, op=dist.ReduceOp.MIN)
    all_ok = bool(verdict.item())
    if rank == 0:
        print(f"REORDER_RANK_{'PASS' if all_ok else 'FAIL'}", flush=True)
    return all_ok


def main() -> None:
    dist.init_process_group("gloo", timeout=timedelta(seconds=15))
    try:
        ok = _check_rank_without_gathers_negotiates()
        dist.barrier()
    finally:
        dist.destroy_process_group()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
