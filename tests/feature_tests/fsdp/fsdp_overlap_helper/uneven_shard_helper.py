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

"""torchrun entrypoint: every rank must make the SAME transport decision for an
uneven ``Shard(0)`` weight.

A collective is a joint operation: if one rank keeps a weight gather on NCCL while
its peers move it to the copy engine, the NCCL all-gather waits for peers that will
never arrive and the step hangs.  Nothing in a single rank's graph reveals this, so
it cannot be tested in-process at one rank -- which is why it lives here.

``Shard(0)`` gives ``ceil(F / world)`` rows to the leading ranks and the remainder
to the trailing ones, so with ``F % world != 0`` the ranks do not agree on the local
shard length.  Two decisions were derived from that length and diverged because of
it:

  1. **Transport.**  The lowering pads the short shards up to a
     full chunk, and copy-engine eligibility was decided by asking whether the
     gather's input was the shard itself -- which a pad makes false.  So the trailing
     ranks kept the gather on NCCL and every other rank moved it to the copy engine.
  2. **Symmetric placement.**  A shard's allocation is sized from the local numel, so
     an uneven shard makes it rank-dependent -- and the gather copies one fixed-size
     slab per peer.

A mixed graph (even and uneven weights interleaved, copy-engine transport) is the
production shape: one odd weight must not drag its even neighbours onto NCCL, and
the even ones must not drag the odd one onto the copy engine.

Checked below, at ``world`` ranks, for an evenly and an unevenly divisible weight:
the full set of gather targets is all-gathered and compared, and so is the buffer
window size.  (Bucketing happens later, during scheduling, where every rank adopts
rank 0's plan.)

Driven by ``tests/feature_tests/test_uneven_shard_transport.py``.  Run directly with::

    torchrun --nproc_per_node=2 tests/feature_tests/fsdp_overlap_helper/uneven_shard_helper.py

Markers (rank 0):
  UNEVEN_TRANSPORT rows=<F> agree=<bool> targets=<counts>
  UNEVEN_SYMM agree=<bool> in_buffer=<n> skipped=<n>
  UNEVEN_MIXED agree=<bool> targets=<counts>
  UNEVEN_PASS
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from typing import Sequence

os.environ.setdefault("TORCH_SYMM_MEM_DISABLE_MULTICAST", "1")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.fx as fx  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.distributed.device_mesh import init_device_mesh  # noqa: E402

_NCCL_AG = torch.ops._c10d_functional.all_gather_into_tensor.default


def _build_graph(mesh, row_counts: Sequence[int], cols: int, dtype=torch.bfloat16) -> fx.GraphModule:
    """One SimpleFSDP weight redistribute per entry of ``row_counts``.

    Every placeholder is declared before the first redistribute, as in a real traced
    graph, so a hoisted coalesced launch stays topologically valid.  The metas are real
    DTensors on ``mesh``, which is the point: the local shard shape is whatever this
    rank actually owns, and mixed even/uneven rows keep that per weight.
    """
    from torch.distributed.tensor import Partial, Replicate, Shard, distribute_tensor

    g = fx.Graph()
    weights = []
    replicated_metas = []
    for i, rows in enumerate(row_counts):
        full = torch.zeros(rows, cols, device="cuda", dtype=dtype)
        w = g.placeholder(f"layer_{i}_weight_parameter")
        w.meta["example_value"] = distribute_tensor(full, mesh, [Shard(0)])
        weights.append(w)
        replicated_metas.append(distribute_tensor(full, mesh, [Replicate()]))

    outs = []
    for w, replicated in zip(weights, replicated_metas):
        rd = g.call_method("redistribute", (w,), {"placements": [Replicate()], "forward_dtype": None, "backward_dtype": None})
        rd.meta["example_value"] = replicated
        tl = g.call_method("to_local", (rd,), {"grad_placements": [Partial()]})
        tl.meta["example_value"] = replicated._local_tensor
        outs.append(tl)
    g.output(tuple(outs))
    return fx.GraphModule(nn.Module(), g)


def _example_inputs(gm) -> list[object]:
    """What Dynamo would hand the backend: the live weights, in placeholder order.

    Read off the placeholder metas here, since this graph is hand-built rather
    than traced -- the metas are the real DTensors, so binding sees the same
    rank-dependent local shapes it would in production.
    """
    return [n.meta["example_value"] for n in gm.graph.find_nodes(op="placeholder")]


def _gather_targets(gm) -> dict[str, int]:
    """How many gathers ended up on each transport."""
    from magi_compiler.symm_mem.all_gather import CE_ALL_GATHER

    names = {_NCCL_AG: "nccl", CE_ALL_GATHER: "ce"}
    counts: Counter[str] = Counter()
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target in names:
            counts[names[node.target]] += 1
    return dict(sorted(counts.items()))


def _agree(value) -> bool:
    """True when every rank produced the same value."""
    seen = [None] * dist.get_world_size()
    dist.all_gather_object(seen, value)
    return all(v == seen[0] for v in seen)


def _pipeline(gm, *, transport: str) -> None:
    """``WeightPipeline.run`` with host offload off, inlined.

    Lower, bind, retarget what bound -- the order is what decides transport per
    weight, which is the thing every rank has to agree on.
    """
    from magi_compiler.passes.fsdp_overlap import (
        bind_weights_for_copy_engine,
        lower_prim_redistribute_to_collectives,
        rewrite_weight_ag_to_copy_engine,
    )

    examples = _example_inputs(gm)
    lower_prim_redistribute_to_collectives(gm)
    if transport == "copy_engine":
        bind_weights_for_copy_engine(gm, examples, 0)
        rewrite_weight_ag_to_copy_engine(gm)


def _check_transport(mesh, rows: int, *, n: int, cols: int, say) -> bool:
    from magi_compiler.symm_mem import reset_registry

    reset_registry()  # each check accounts for its own windows
    gm = _build_graph(mesh, [rows] * n, cols)
    _pipeline(gm, transport="copy_engine")
    targets = _gather_targets(gm)
    ok = _agree(targets)
    say(f"UNEVEN_TRANSPORT rows={rows} agree={ok} targets={targets}")
    return ok


def _check_binding(mesh, *, even_rows: int, uneven_rows: int, cols: int, say) -> bool:
    """An uneven shard must not be moved into symmetric memory.

    A shard's allocation is sized from the local numel, so an uneven weight makes it
    rank-dependent, and the gather copies one fixed-size slab per peer: a rank whose
    peers own fewer rows reads past the end of theirs.  Nothing at run time
    re-derives that, so it has to be excluded here.
    """
    from torch.distributed.tensor import Shard, distribute_tensor

    from magi_compiler.symm_mem import bind_parameters, lookup_shard, registered_buffers, reset_registry

    reset_registry()

    def param(rows: int) -> nn.Parameter:
        t = torch.zeros(rows, cols, device="cuda", dtype=torch.bfloat16)
        return nn.Parameter(distribute_tensor(t, mesh, [Shard(0)]))

    named = [("even", param(even_rows)), ("uneven", param(uneven_rows))]
    bind_parameters([p for _name, p in named])

    window_bytes = sorted(w.nbytes for w in registered_buffers())
    bound = [name for name, p in named if lookup_shard(p._local_tensor.data_ptr())]
    ok = _agree((window_bytes, bound)) and bound == ["even"]
    say(f"UNEVEN_SYMM agree={ok} in_buffer={bound} window_bytes={window_bytes}")
    return ok


def _check_mixed(mesh, *, even_rows: int, uneven_rows: int, cols: int, say) -> bool:
    """Even and uneven weights in one copy-engine graph must split by transport.

    Interleaved: the even pair goes to the copy engine, the uneven pair stays on
    NCCL.  All ranks must report the same split.
    """
    from magi_compiler.symm_mem import reset_registry

    reset_registry()
    gm = _build_graph(mesh, [even_rows, uneven_rows, even_rows, uneven_rows], cols)
    _pipeline(gm, transport="copy_engine")
    targets = _gather_targets(gm)
    ok = _agree(targets) and targets == {"ce": 2, "nccl": 2}
    say(f"UNEVEN_MIXED agree={ok} targets={targets}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-weights", type=int, default=4)
    ap.add_argument("--cols", type=int, default=256)
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    assert world >= 2, "an uneven Shard(0) needs at least 2 ranks to be uneven"
    mesh = init_device_mesh("cuda", (world,))

    def say(msg: str) -> None:
        if rank == 0:
            print(msg, flush=True)

    # world+1 rows never divides by world (for world >= 2), so the last rank owns 1 row
    # where the others own 2.  world*2 rows is the even control.
    uneven_rows, even_rows = world + 1, world * 2

    ok = _check_transport(mesh, even_rows, n=args.n_weights, cols=args.cols, say=say)
    ok &= _check_transport(mesh, uneven_rows, n=args.n_weights, cols=args.cols, say=say)
    ok &= _check_binding(mesh, even_rows=even_rows, uneven_rows=uneven_rows, cols=args.cols, say=say)
    ok &= _check_mixed(mesh, even_rows=even_rows, uneven_rows=uneven_rows, cols=args.cols, say=say)

    if ok:
        say("UNEVEN_PASS")
    dist.destroy_process_group()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
