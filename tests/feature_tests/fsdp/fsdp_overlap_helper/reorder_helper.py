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
per-rank graphs are structurally DIFFERENT. Every rank must negotiate SLOT
consensus because the collective skeleton still matches -- the
gather may hop the all_reduce as long as EVERY rank hops it.  The invariant that
actually matters is checked directly: the collective sequence of the FINAL schedule
is all_gathered and compared across ranks.

With ``--modes-only`` (>=2 ranks, gloo, no CUDA / no compile): drive
``rank_sync.negotiate_mode`` directly with synthetic per-rank inputs and assert it returns
the expected mode for each rung of the ladder (slot / pinned / abort).

With ``--copy-engine``: the same shape, but the gathers are
``magi::ce_all_gather`` reading a symmetric buffer.  Two things are checked that
NCCL does not exercise.  First, recognition: the gather is a plain fallback
kernel with an alias node between it and its wait, so the pass has to see through
that or it silently plans nothing.  Second, slot safety: the gathers cycle
through a small set of RESIDENT destination buffers, and a launch hoisted above
the last read of the buffer it is about to overwrite would corrupt a weight in
flight.  Inductor cannot infer that constraint -- the reuse is invisible in the
graph -- so ``REORDER_SLOTS`` asserts it directly on the emitted schedule.

Run: torchrun --nproc_per_node=1 tests/feature_tests/fsdp_overlap_helper/reorder_helper.py
     torchrun --nproc_per_node=2 ... reorder_helper.py --mismatch
     torchrun --nproc_per_node=2 ... reorder_helper.py --modes-only

Markers (rank 0):
  REORDER_CALLED gathers=<n>
  REORDER_OK ran=<bool>         (pass returned)
  REORDER_FINITE ok=<bool>      (compiled output finite + matches eager)
  REORDER_SKELETON ok=<bool>    (final collective sequence identical on all ranks)
  REORDER_SLOT rank=<n>           (SLOT-consensus mode chosen)
  REORDER_MODES ok=<bool>         (--modes-only: the mode ladder returned as expected)
  REORDER_SLOTS ok=<bool>         (--copy-engine: no launch overwrites a live slot)
  REORDER_PASS / REORDER_FAIL
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("TORCH_SYMM_MEM_DISABLE_MULTICAST", "1")

import torch  # noqa: E402
import torch._inductor.config as inductor_config  # noqa: E402
import torch.distributed as dist  # noqa: E402

from magi_compiler.passes import snode_utils  # noqa: E402
from magi_compiler.passes.fsdp_overlap import FsdpOverlapReorder  # noqa: E402
from magi_compiler.passes.overlap import rank_sync  # noqa: E402


def _mode_ladder_selfcheck(rank: int) -> bool:
    """Assert every rung of ``negotiate_mode``'s ladder, with rank 1 feeding the
    divergent input.  All ranks walk the cases in the same order, so the symmetric
    all_gather inside each call stays lockstep."""
    negotiate = rank_sync.negotiate_mode
    odd = rank == 1
    ag, other = (True, "ag", (8, 8)), (False, "cp", (4,))
    cases = [
        # (expected mode, weight-AG count, skeleton kinds, costs_ok)
        ("slot", 2, [ag, other, ag], True),
        ("pinned", 2, [ag, other, ag] if odd else [ag, ag, other], True),
        ("abort", 3 if odd else 2, [ag, other, ag], True),
        # Matching skeletons, but rank 1 failed to price its graph: every rank aborts.
        ("abort", 2, [ag, other, ag], not odd),
    ]
    ok = True
    for expected, n_ag, kinds, costs_ok in cases:
        got = negotiate(n_anchors=n_ag, skel_kinds=kinds, costs_ok=costs_ok)[0]
        ok = ok and got == expected
        print(f"REORDER_MODE_CASE rank={rank} expected={expected} got={got}", flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mismatch", action="store_true", help="rank1 compiles a structurally different graph")
    ap.add_argument("--modes-only", action="store_true", help="only self-check the mode ladder (gloo, no compile)")
    ap.add_argument("--copy-engine", action="store_true", help="gather from a symmetric buffer instead of NCCL")
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
    N_CE_LAYERS = 3
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

    ce_shards: list = []
    if args.copy_engine:
        from magi_compiler.symm_mem import alloc_shard
        from magi_compiler.symm_mem.all_gather import CE_ALL_GATHER

        for i in range(N_CE_LAYERS):
            s = alloc_shard((H, H), torch.bfloat16, torch.device("cuda", dev), grp)
            s.normal_(0.0, H**-0.5).add_(0.01 * i)
            ce_shards.append(s)
        # A peer read is only legal once that peer has written its shard, and
        # these shards are filled here, so the barrier belongs here too.
        torch.cuda.synchronize()
        dist.barrier()

        def fn(x, w0, shards):  # noqa: F811 - deliberately replaces the NCCL variant
            y = (x @ w0).relu()
            y = _WAIT(_AR(y, "sum", grp))
            acc = None
            for i, sh in enumerate(shards):
                g = _WAIT(CE_ALL_GATHER(sh, world, grp))
                z = y @ g.reshape(world * H, H)[:H]
                acc = z if acc is None else acc + z
            return acc

        def ref_fn(x, w0, shards):
            """Same arithmetic over NCCL: an independent answer, so the numeric
            check is a real cross-transport comparison rather than the copy
            engine grading its own homework."""
            y = (x @ w0).relu()
            y = _WAIT(_AR(y, "sum", grp))
            acc = None
            for sh in shards:
                g = _WAIT(_AG(sh, world, grp))
                z = y @ g.reshape(world * H, H)[:H]
                acc = z if acc is None else acc + z
            return acc

    # Instrument pass invocations, gather counts, and consensus mode.
    calls = {"n": 0, "gathers": 0, "slot_mode": False}
    skeletons: list = []  # collective sequence of every schedule the pass returned
    orig_call = FsdpOverlapReorder.__call__

    orig_negotiate = rank_sync.negotiate_mode

    def spy_negotiate(*a, **kw):
        result = orig_negotiate(*a, **kw)
        if result[0] == "slot":
            calls["slot_mode"] = True
            print(f"REORDER_SLOT rank={rank}", flush=True)
        return result

    rank_sync.negotiate_mode = spy_negotiate

    def spy(self, snodes):
        calls["n"] += 1
        calls["gathers"] += sum(1 for s in snodes if snode_utils.is_all_gather_launch(s))
        out = orig_call(self, snodes)
        skeletons.append(rank_sync.collective_skeleton(out)[1])
        return out

    FsdpOverlapReorder.__call__ = spy

    def greedy_cost(snode) -> float:
        """A gather nobody can hide: Inductor's analytical model prices a
        fallback kernel at 0us, so without this the launches barely move."""
        from torch._inductor.comms import estimate_op_runtime

        if snode_utils.is_ce_ag_ir(snode_utils.leaf_comm_node(snode)):
            return 1e7  # 10ms, far more than the whole graph's compute
        return estimate_op_runtime(snode)

    reorder = FsdpOverlapReorder(comm_overlap_window_margin_ns=5000.0, cost_fn=greedy_cost if args.copy_engine else None)
    prev_flag = inductor_config.reorder_for_compute_comm_overlap
    prev_passes = inductor_config.reorder_for_compute_comm_overlap_passes
    prev_cache = inductor_config.force_disable_caches
    inductor_config.reorder_for_compute_comm_overlap = True
    inductor_config.reorder_for_compute_comm_overlap_passes = [reorder]
    inductor_config.force_disable_caches = True
    try:
        torch._dynamo.reset()
        x = torch.randn(H, H, device=dev, dtype=torch.bfloat16)
        weights = ce_shards if args.copy_engine else shard
        eager = (ref_fn if args.copy_engine else fn)(x, w0, weights)
        torch.cuda.synchronize()
        compiled = torch.compile(fn, dynamic=False)
        out = compiled(x, w0, weights)
        torch.cuda.synchronize()
    finally:
        inductor_config.reorder_for_compute_comm_overlap = prev_flag
        inductor_config.reorder_for_compute_comm_overlap_passes = prev_passes
        inductor_config.force_disable_caches = prev_cache
        FsdpOverlapReorder.__call__ = orig_call
        rank_sync.negotiate_mode = orig_negotiate

    finite = bool(torch.isfinite(out).all().item())
    rel = ((out.float() - eager.float()).norm() / (eager.float().norm() + 1e-6)).item()
    numeric_ok = finite and rel < 5e-2

    # THE invariant: whatever each rank decided, the collective sequence of the
    # schedules it emitted must be identical on every rank -- that (not identical
    # absolute placement) is what keeps NCCL's positional matching intact.
    peer_skeletons: list = [None] * world
    dist.all_gather_object(peer_skeletons, skeletons)
    skeleton_ok = all(s == peer_skeletons[0] for s in peer_skeletons[1:])

    # Matching skeletons must use slot consensus on every rank, even when compute differs.
    ok_local = calls["n"] > 0 and calls["gathers"] >= 1 and numeric_ok and skeleton_ok and calls["slot_mode"]
    if args.copy_engine:
        ok_local = ok_local and calls["gathers"] >= N_CE_LAYERS
    t = torch.tensor([1 if ok_local else 0], device=dev)
    dist.all_reduce(t)
    all_ok = int(t.item()) == world

    if rank == 0:
        print(f"REORDER_CALLED gathers={calls['gathers']}", flush=True)
        print(f"REORDER_OK ran={calls['n'] > 0}", flush=True)
        print(f"REORDER_FINITE ok={numeric_ok} rel={rel:.5f}", flush=True)
        print(f"REORDER_SKELETON ok={skeleton_ok}", flush=True)
        print("REORDER_PASS" if all_ok else "REORDER_FAIL", flush=True)
        rc = 0 if all_ok else 1
    else:
        rc = 0

    dist.barrier()
    dist.destroy_process_group()
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
