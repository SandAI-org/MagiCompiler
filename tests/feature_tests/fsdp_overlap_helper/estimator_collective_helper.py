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

"""torchrun entrypoint: verify the profile_sync COLLECTIVE measurement accuracy on a
real multi-rank all-gather, for the pytest driver (test_profiling_estimator.py).

profile_sync measures collectives with ``_measure_collective_op`` (barrier + fixed
iters, rank-lockstep).  Here we:
  1. compile a tiny all_gather fn and capture its real _CollectiveKernel snode (via the
     same Inductor reorder hook the pass uses);
  2. run ``_measure_collective_op`` on that snode -> the estimator's collective time;
  3. INDEPENDENTLY time the same all_gather with CUDA events (rank-lockstep);
  4. assert the estimate is within a tolerance band of the independent measurement.
Also exercises ``ProfilingRuntimeEstimator.warm_and_sync`` (the profile_sync entry) to
confirm it runs rank-lockstep without deadlock and reconciles a collective entry, and
the KEY-SET MISMATCH fail-fast: when one rank's table has an extra key (simulating a
per-rank structural divergence), warm_and_sync must NOT enter the barrier loop (which
would deadlock) -- it must detect the mismatch on every rank, warn, and degrade every
stashed entry to the analytical estimate (measured=False).

Run: torchrun --nproc_per_node=2 .../estimator_collective_helper.py

Markers (rank 0):
  COLL_MEASURED est_us=<f> real_us=<f> ratio=<f>
  COLL_ACCURATE ok=<bool>
  COLL_WARMSYNC reconciled=<n> ok=<bool>
  COLL_MISMATCH ok=<bool>
  COLL_PASS / COLL_FAIL
"""

from __future__ import annotations

import os

import torch
import torch._inductor.config as inductor_config
import torch.distributed as dist
from torch._inductor.utils import contains_collective

from magi_compiler.profiling import ProfilingRuntimeEstimator
from magi_compiler.profiling.runtime_estimator import ProfileEntry, _measure_collective_op

_AG = torch.ops._c10d_functional.all_gather_into_tensor.default
_WAIT = torch.ops._c10d_functional.wait_tensor.default


class _CaptureColl:
    """Grab the collective snode(s) of one compiled graph."""

    def __init__(self):
        self.snodes = []

    def __call__(self, snodes):
        for s in snodes:
            if contains_collective(s):
                self.snodes.append(s)
        return snodes


def _real_allgather_ns(shard, world, grp_name, iters=20, warmup=5):
    """Independent rank-lockstep CUDA-event timing of all_gather(+wait)."""

    def fn():
        _WAIT(_AG(shard, world, grp_name))

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return (s.elapsed_time(e) / iters) * 1e6  # ns


def main() -> None:
    dist.init_process_group("cpu:gloo,cuda:nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.cuda.current_device()
    grp_name = dist.group.WORLD.group_name

    numel = 4 * 1024 * 1024  # 4M bf16 = 8 MiB local shard -- big enough to time stably
    shard = torch.randn(numel, device=dev, dtype=torch.bfloat16)

    # 1. capture the real all_gather snode from a compiled fn.
    cap = _CaptureColl()
    prev = (
        inductor_config.reorder_for_compute_comm_overlap,
        inductor_config.reorder_for_compute_comm_overlap_passes,
        inductor_config.force_disable_caches,
    )
    inductor_config.reorder_for_compute_comm_overlap = True
    inductor_config.reorder_for_compute_comm_overlap_passes = [cap]
    inductor_config.force_disable_caches = True
    try:
        torch._dynamo.reset()
        torch.compile(lambda s: _WAIT(_AG(s, world, grp_name)), dynamic=False)(shard)
        torch.cuda.synchronize()
    finally:
        (
            inductor_config.reorder_for_compute_comm_overlap,
            inductor_config.reorder_for_compute_comm_overlap_passes,
            inductor_config.force_disable_caches,
        ) = prev

    assert cap.snodes, "no collective snode captured"
    coll_snode = cap.snodes[0]

    # 2. estimator's profile_sync collective measurement (barrier + fixed iters).
    dist.barrier()
    est_ns = _measure_collective_op(coll_snode)
    dist.barrier()

    # 3. independent CUDA-event ground truth of the SAME all_gather.
    real_ns = _real_allgather_ns(shard, world, grp_name)
    dist.barrier()

    ratio = est_ns / real_ns if real_ns > 0 else 0.0
    # both measure the same op with fixed-iter CUDA events; expect close (allow
    # 0.5-2x for iteration-count / warmup / noise differences).
    accurate = est_ns > 0 and real_ns > 0 and 0.5 <= ratio <= 2.0

    # 4. warm_and_sync: put the collective in the table (mimic the reorder warm-up),
    #    stash its snode, then reconcile across ranks -- must not deadlock.
    est = ProfilingRuntimeEstimator()
    est._sync_across_ranks = True
    est(coll_snode)  # __call__ seeds the table + stashes the snode (sync mode)
    n_reconciled = est.warm_and_sync()
    # the collective entry should now be marked measured (real value)
    coll_entries = [e for e in est.table.values() if e.kind == "collective"]
    warmsync_ok = len(coll_entries) >= 1 and all(e.measured for e in coll_entries)

    # 5. key-set MISMATCH fail-fast: rank 1 injects an extra table entry so the
    #    cross-rank key sets differ.  warm_and_sync must detect this on EVERY rank
    #    (symmetric all_gather_object), skip the per-key barrier loop entirely (which
    #    would deadlock on the count mismatch), and degrade this compile's entries to
    #    the analytical estimate (measured=False).  Completing at all proves no hang.
    est2 = ProfilingRuntimeEstimator()
    est2._sync_across_ranks = True
    est2(coll_snode)  # both ranks: seed the shared collective entry
    if rank == 1:
        fake_key = ("mismatch_only_on_rank1",)
        est2._table[fake_key] = ProfileEntry(ns=1.0, kind="extern", label="fake", measured=True)
        est2._key_snode[fake_key] = coll_snode
    est2.warm_and_sync()
    mismatch_ok = all(not e.measured for e in est2.table.values()) and not est2._key_snode
    dist.barrier()

    # gather agreement across ranks
    ok_local = accurate and warmsync_ok and mismatch_ok
    t = torch.tensor([1 if ok_local else 0], device=dev)
    dist.all_reduce(t)
    all_ok = int(t.item()) == world

    if rank == 0:
        print(f"COLL_MEASURED est_us={est_ns/1e3:.1f} real_us={real_ns/1e3:.1f} ratio={ratio:.2f}", flush=True)
        print(f"COLL_ACCURATE ok={accurate}", flush=True)
        print(f"COLL_WARMSYNC reconciled={n_reconciled} ok={warmsync_ok}", flush=True)
        print(f"COLL_MISMATCH ok={mismatch_ok}", flush=True)
        print("COLL_PASS" if all_ok else "COLL_FAIL", flush=True)
        rc = 0 if all_ok else 1
    else:
        rc = 0

    dist.barrier()
    dist.destroy_process_group()
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
