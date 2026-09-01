# Copyright (c) 2025 SandAI. All Rights Reserved.
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

"""torchrun entrypoint: drive ``FsdpOverlapReorder`` through a real ``torch.compile``
on a tiny graph that contains upstream compute + a weight all-gather + a consumer,
and print markers for the pytest driver (test_fsdp_overlap_reorder.py).

The reorder pass is an Inductor ``reorder_for_compute_comm_overlap_passes`` callback;
it needs a process group (its multi-rank-determinism warmup calls dist.get_rank()).
We build a fn:  y = (x @ w0).relu()            # upstream compute
                y = all_reduce(y) ; wait        # a NON-weight collective in between
                g = all_gather(shard) ; wait    # a weight gather to hoist
                out = y @ gathered_use          # consumer after the compute
and wrap the pass so we can assert it RAN and returned a valid schedule.  The
all_reduce stands in for a CP / EP kernel: the gather's compute window reaches past
it, so hoisting requires hopping another collective.

With ``--mismatch`` (needs >=2 ranks): rank 1's fn gets an EXTRA compute op so the
per-rank graphs are structurally DIFFERENT.  The reorder pass's cross-rank
graph-fingerprint check must fire on EVERY rank (symmetric all_gather), warn, and
continue in SLOT-consensus mode (the collective skeleton still matches) -- the
gather may hop the all_reduce as long as EVERY rank hops it.  The invariant that
actually matters is checked directly: the collective sequence of the FINAL schedule
is all_gathered and compared across ranks.

With ``--modes-only`` (>=2 ranks, gloo, no CUDA / no compile): drive
``_negotiate_mode`` directly with synthetic per-rank inputs and assert it returns
the expected mode for each rung of the ladder (identical / slot / pinned / abort).

Run: torchrun --nproc_per_node=1 tests/feature_tests/fsdp_overlap_helper/reorder_helper.py
     torchrun --nproc_per_node=2 ... reorder_helper.py --mismatch
     torchrun --nproc_per_node=2 ... reorder_helper.py --modes-only

Markers (rank 0):
  REORDER_CALLED gathers=<n>
  REORDER_OK moved=<n>          (pass returned; N launches repositioned)
  REORDER_FINITE ok=<bool>      (compiled output finite + matches eager)
  REORDER_SKELETON ok=<bool>    (final collective sequence identical on all ranks)
  REORDER_MISMATCH local=<bool>   (--mismatch only: divergent-graph path taken)
  REORDER_SLOT rank=<n>           (--mismatch only: SLOT-consensus mode chosen)
  REORDER_MODES ok=<bool>         (--modes-only: the mode ladder returned as expected)
  REORDER_PASS / REORDER_FAIL
"""

from __future__ import annotations

import argparse
import os

import torch
import torch._inductor.config as inductor_config
import torch.distributed as dist

from magi_compiler.passes.fsdp_overlap import FsdpOverlapReorder
from magi_compiler.passes.fsdp_overlap import reorder as _ro


class _FakeIR:
    op_overload = "fake.op"
    origins = None

    def get_size(self):
        return [8, 8]


class _FakeSnode:
    """Enough of a snode for ``_graph_fingerprint`` (the only thing the mode
    negotiation reads out of the schedule)."""

    snodes = None

    def __init__(self) -> None:
        self.node = _FakeIR()


def _mode_ladder_selfcheck(rank: int) -> bool:
    """Assert every rung of ``_negotiate_mode``'s ladder, with rank 1 feeding the
    divergent input.  All ranks walk the cases in the same order, so the symmetric
    all_gather inside each call stays lockstep."""
    negotiate = FsdpOverlapReorder._negotiate_mode
    odd = rank == 1
    ag, other = (True, "ag", (8, 8)), (False, "cp", (4,))
    cases = {
        # (n_snodes, weight-AG count, skeleton kinds) -> expected mode
        "identical": (4, 2, [ag, other, ag]),
        "slot": (5 if odd else 4, 2, [ag, other, ag]),  # graphs differ, skeleton does not
        "pinned": (5 if odd else 4, 2, [ag, other, ag] if odd else [ag, ag, other]),
        "abort": (5 if odd else 4, 3 if odd else 2, [ag, other, ag]),
    }
    ok = True
    for expected, (n_snodes, n_ag, kinds) in cases.items():
        got = negotiate([_FakeSnode() for _ in range(n_snodes)], [None] * n_ag, kinds)[0]
        ok = ok and got == expected
        print(f"REORDER_MODE_CASE rank={rank} expected={expected} got={got}", flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mismatch", action="store_true", help="rank1 compiles a structurally different graph")
    ap.add_argument("--modes-only", action="store_true", help="only self-check the mode ladder (gloo, no compile)")
    args = ap.parse_args()

    if args.modes_only:
        dist.init_process_group("gloo")
        my_rank = dist.get_rank()
        t = torch.tensor([1 if _mode_ladder_selfcheck(my_rank) else 0])
        dist.all_reduce(t, op=dist.ReduceOp.MIN)  # every rank sees the same verdict
        all_ok = bool(t.item())
        if my_rank == 0:
            print(f"REORDER_MODES ok={all_ok}", flush=True)
            print("REORDER_PASS" if all_ok else "REORDER_FAIL", flush=True)
        dist.barrier()
        dist.destroy_process_group()
        raise SystemExit(0 if all_ok else 1)

    dist.init_process_group("cpu:gloo,cuda:nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.cuda.current_device()
    grp = dist.group.WORLD.group_name
    torch.manual_seed(0)

    _AG = torch.ops._c10d_functional.all_gather_into_tensor.default
    _AR = torch.ops._c10d_functional.all_reduce.default
    _WAIT = torch.ops._c10d_functional.wait_tensor.default

    H = 512
    w0 = torch.randn(H, H, device=dev, dtype=torch.bfloat16)
    shard = torch.randn(H, H, device=dev, dtype=torch.bfloat16)

    extra_op = args.mismatch and rank == 1  # structural per-rank divergence on demand

    def fn(x, w0, shard):
        y = (x @ w0).relu()  # upstream compute the gather can hide behind
        y = _WAIT(_AR(y, "sum", grp))  # non-weight collective the gather must hop
        if extra_op:
            y = y.sin()  # rank1-only node -> graphs differ across ranks
        g = _WAIT(_AG(shard, world, grp))  # weight all-gather + wait
        gathered = g.reshape(world * H, H)[:H]  # use the gathered weight
        return y @ gathered

    # instrument the pass: count how many times it runs, how many launches move,
    # and whether the returned schedule is identical to the input (LOCAL path).
    calls = {"n": 0, "gathers": 0, "moved": 0, "unchanged": True, "warned_mismatch": False, "slot_mode": False}
    skeletons: list = []  # collective sequence of every schedule the pass returned
    orig_call = FsdpOverlapReorder.__call__

    # magi_logger output from inside an Inductor compile does not reliably reach the
    # subprocess streams; intercept the warning call itself to detect the mode taken.
    orig_warning = _ro.magi_logger.warning

    def spy_warning(msg, *a, **kw):
        if "NOT structurally identical" in str(msg):
            calls["warned_mismatch"] = True
            print(f"REORDER_WARNED rank={rank}", flush=True)
        if "SLOT-consensus" in str(msg):
            calls["slot_mode"] = True
            print(f"REORDER_SLOT rank={rank}", flush=True)
        return orig_warning(msg, *a, **kw)

    _ro.magi_logger.warning = spy_warning

    def spy(self, snodes):
        calls["n"] += 1
        calls["gathers"] += sum(1 for s in snodes if _ro._is_weight_gather(s))
        before = list(snodes)
        out = orig_call(self, snodes)
        calls["unchanged"] = len(out) == len(before) and all(a is b for a, b in zip(out, before))
        skeletons.append(_ro._collective_skeleton(out)[1])
        return out

    FsdpOverlapReorder.__call__ = spy

    reorder = FsdpOverlapReorder(comm_overlap_window_margin_ns=5000.0)  # default cost_fn (Inductor analytical)
    prev_flag = inductor_config.reorder_for_compute_comm_overlap
    prev_passes = inductor_config.reorder_for_compute_comm_overlap_passes
    prev_cache = inductor_config.force_disable_caches
    inductor_config.reorder_for_compute_comm_overlap = True
    inductor_config.reorder_for_compute_comm_overlap_passes = [reorder]
    inductor_config.force_disable_caches = True
    try:
        torch._dynamo.reset()
        x = torch.randn(H, H, device=dev, dtype=torch.bfloat16)
        eager = fn(x, w0, shard)
        compiled = torch.compile(fn, dynamic=False)
        out = compiled(x, w0, shard)
        torch.cuda.synchronize()
    finally:
        inductor_config.reorder_for_compute_comm_overlap = prev_flag
        inductor_config.reorder_for_compute_comm_overlap_passes = prev_passes
        inductor_config.force_disable_caches = prev_cache
        FsdpOverlapReorder.__call__ = orig_call
        _ro.magi_logger.warning = orig_warning

    finite = bool(torch.isfinite(out).all().item())
    rel = ((out.float() - eager.float()).norm() / (eager.float().norm() + 1e-6)).item()
    numeric_ok = finite and rel < 5e-2

    # THE invariant: whatever each rank decided, the collective sequence of the
    # schedules it emitted must be identical on every rank -- that (not identical
    # absolute placement) is what keeps NCCL's positional matching intact.
    peer_skeletons: list = [None] * world
    dist.all_gather_object(peer_skeletons, skeletons)
    skeleton_ok = all(s == peer_skeletons[0] for s in peer_skeletons[1:])

    # In --mismatch mode the divergent-graph path must warn on EVERY rank; agree
    # across ranks before printing.  Schedule may change under SLOT reordering.
    ok_local = calls["n"] > 0 and calls["gathers"] >= 1 and numeric_ok and skeleton_ok
    if args.mismatch:
        ok_local = ok_local and calls["warned_mismatch"] and calls["slot_mode"]
    t = torch.tensor([1 if ok_local else 0], device=dev)
    dist.all_reduce(t)
    all_ok = int(t.item()) == world

    if rank == 0:
        print(f"REORDER_CALLED gathers={calls['gathers']}", flush=True)
        print(f"REORDER_OK ran={calls['n'] > 0}", flush=True)
        print(f"REORDER_FINITE ok={numeric_ok} rel={rel:.5f}", flush=True)
        print(f"REORDER_SKELETON ok={skeleton_ok}", flush=True)
        if args.mismatch:
            print(f"REORDER_MISMATCH local={calls['warned_mismatch']} unchanged={calls['unchanged']}", flush=True)
        print("REORDER_PASS" if all_ok else "REORDER_FAIL", flush=True)
        rc = 0 if all_ok else 1
    else:
        rc = 0

    dist.barrier()
    dist.destroy_process_group()
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
