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

"""torchrun entrypoint: scheduler-level weight all-gather coalescing, end to end.

A SimpleFSDP model is compiled with FX bucketing off, and a reorder pass that
merges every ``--pair`` consecutive weight gathers through
``ir_coalesce.coalesce_all_gathers`` is installed right after the cost profile
(``--pair 0``: ``bucket_mode='auto'``, so ``FsdpAutoBucket`` decides instead).
What is checked is the IR surgery itself, not a bucketing policy: the compiled
output must match eager, and the generated wrapper must launch coalesced
gathers while still waiting on the original buffer names.

Markers printed on rank 0:
  IRC_COALESCED buckets=<n> members=<m>
  IRC_RANK_BUCKETS <buckets>/<members> per rank
  IRC_TAGGED <weight-tagged candidates>/<candidates>   (fixed-pair mode only)
  IRC_CODE coalesced_calls=<n> plain_calls=<n>
  IRC_WAITS on_unpacks=<waits reading a coalesced unpack> total=<waits>
  IRC_NUMERIC rel=<f> ok=<bool>
  IRC_PASS / IRC_FAIL
"""

from __future__ import annotations

import argparse
import os
import re

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh

from magi_compiler import magi_compile
from magi_compiler.config import CompileMode, CudaGraphMode

_STATS = {"buckets": 0, "members": 0}
_AG_COALESCED = torch.ops._c10d_functional.all_gather_into_tensor_coalesced.default
_CODE: list[str] = []


class Block(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class TinyModel(nn.Module):
    def __init__(self, hidden: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(Block(hidden) for _ in range(n_layers))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PairCoalesce:
    """Merge every ``pair`` consecutive weight gathers that can legally merge."""

    def __init__(self, pair: int) -> None:
        self.pair = pair

    def __deepcopy__(self, memo):
        return self

    def __call__(self, snodes):
        from magi_compiler.passes.fsdp_overlap.ir_coalesce import (
            coalesce_all_gathers,
            coalesce_placement,
            gather_info,
            refresh_memory_planning,
        )
        from magi_compiler.passes.fsdp_overlap.node_meta import is_weight_ag

        order = list(snodes)
        cands = [s for s in order if gather_info(s) is not None]
        _STATS["tagged"] = sum(1 for s in cands if any(is_weight_ag(o) for o in s.node.origins))
        _STATS["cands"] = len(cands)
        for i in range(0, len(cands) - self.pair + 1, self.pair):
            members = cands[i : i + self.pair]
            if coalesce_placement(order, members) is None:
                continue
            order, _, _ = coalesce_all_gathers(order, members)
            _STATS["buckets"] += 1
            _STATS["members"] += len(members)
        snodes[:] = order
        refresh_memory_planning(snodes)
        return snodes


def _install(pair: int, diverge: bool = False) -> None:
    """``pair > 0``: fixed pairs through PairCoalesce; 0: the config's own passes.

    ``diverge``: ranks other than 0 drop their last planned bucket, standing in
    for a plan flipped by per-rank cost noise; rank 0's plan must still win.
    """
    from magi_compiler.magi_backend import magi_backend as mb

    orig = mb.MagiBackend._configure_overlap_passes

    def patched(self, **kw):
        orig(self, **kw)
        passes = self.inductor_compile_config.get("reorder_for_compute_comm_overlap_passes") or []
        if passes and pair > 0:
            passes.insert(1, PairCoalesce(pair))

    mb.MagiBackend._configure_overlap_passes = patched

    from magi_compiler.passes.fsdp_overlap import auto_bucket

    orig_call = auto_bucket.FsdpAutoBucket.__call__

    def counted(self, snodes):
        before = sum(1 for s in snodes if auto_bucket.gather_info(s) is not None)
        out = orig_call(self, snodes)
        after = sum(1 for s in out if auto_bucket.gather_info(s) is not None)
        n_packed = sum(1 for s in out if getattr(getattr(s, "node", None), "op_overload", None) is _AG_COALESCED)
        _STATS["buckets"] += n_packed
        _STATS["members"] += before - after
        return out

    auto_bucket.FsdpAutoBucket.__call__ = counted

    if diverge and dist.get_rank() != 0:
        orig_plan = auto_bucket.FsdpAutoBucket._plan

        def diverged(self, order, cands, cost):
            buckets, report = orig_plan(self, order, cands, cost)
            return buckets[:-1], report

        auto_bucket.FsdpAutoBucket._plan = diverged

    from torch._inductor.graph import GraphLowering

    orig_codegen = GraphLowering.codegen

    def codegen(self, *a, **k):
        out = orig_codegen(self, *a, **k)
        try:
            code = out[0].value if hasattr(out[0], "value") else str(out[0])
        except Exception:  # noqa: BLE001
            code = str(out)
        _CODE.append(code)
        return out

    GraphLowering.codegen = codegen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", type=int, default=2, help="0 = bucket_mode='auto' instead of fixed pairs")
    ap.add_argument("--bucket-size-mib", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--cost-mode", default="analytical", choices=["analytical", "profile_sync"])
    ap.add_argument("--memory-probe", action="store_true")
    ap.add_argument("--diverge", action="store_true", help="non-zero ranks plan one bucket fewer")
    args = ap.parse_args()

    dist.init_process_group("cpu:gloo,cuda:nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.cuda.current_device()
    torch.manual_seed(0)
    os.environ.setdefault("MAGI_LOGGING_LEVEL", "INFO")
    _install(args.pair, args.diverge)

    from torchtitan.experiments.simple_fsdp.simple_fsdp import data_parallel

    mesh = init_device_mesh("cuda", (world,))
    ref = TinyModel(args.hidden, args.n_layers).to(dev).to(torch.bfloat16)
    x = torch.randn(64, args.hidden, device=dev, dtype=torch.bfloat16)
    with torch.no_grad():
        eager_out = ref(x)

    model = TinyModel(args.hidden, args.n_layers).to(dev).to(torch.bfloat16)
    with torch.no_grad():
        for (_, dst), (_, src) in zip(model.named_parameters(), ref.named_parameters()):
            dst.copy_(src)
    model = data_parallel(model, mesh, mode="fully_shard", ac_mode="full")

    def _patch(cfg):
        cfg.compile_mode = CompileMode.MAGI_COMPILE
        cfg.cudagraph_mode = CudaGraphMode.NONE
        cfg.disable_graph_split = True
        cfg.fsdp_config.enable_fsdp = True
        cfg.fsdp_config.bucket_mode = "none" if args.pair > 0 else "auto"
        cfg.fsdp_config.bucket_size_mib = args.bucket_size_mib
        cfg.fsdp_config.cost_mode = args.cost_mode
        cfg.fsdp_config.memory_probe = args.memory_probe
        return cfg

    compiled = magi_compile(model, config_patch=_patch, dynamic_arg_dims={"x": 0})
    with torch.no_grad():
        out = compiled(x)
        again = compiled(x)
        torch.cuda.synchronize()

    code = "\n".join(_CODE)
    n_coal = len(re.findall(r"all_gather_into_tensor_coalesced", code))
    n_plain = len(re.findall(r"all_gather_into_tensor\.default\(", code))
    packed = set(re.findall(r"(\w+) = torch\.ops\._c10d_functional\.all_gather_into_tensor_coalesced", code))
    unpacked = {m.group(1) for m in re.finditer(r"(\w+) = (\w+)\[\d+\]", code) if m.group(2) in packed}
    waited = re.findall(r"wait_tensor\.default\((\w+)\)", code)
    rel = ((out.float() - eager_out.float()).norm() / (eager_out.float().norm() + 1e-6)).item()
    ok = bool(torch.isfinite(out).all().item()) and rel < 5e-2 and torch.equal(out, again)
    ok = ok and (world == 1 or _STATS["buckets"] > 0)
    ok_t = torch.tensor([1 if ok else 0], device=dev)
    dist.all_reduce(ok_t)
    all_ok = int(ok_t.item()) == world
    per_rank: list = [None] * world
    dist.all_gather_object(per_rank, (_STATS["buckets"], _STATS["members"]))
    if rank == 0:
        print(f"IRC_COALESCED buckets={_STATS['buckets']} members={_STATS['members']}", flush=True)
        print(f"IRC_RANK_BUCKETS {' '.join(f'{b}/{m}' for b, m in per_rank)}", flush=True)
        print(f"IRC_TAGGED {_STATS.get('tagged')}/{_STATS.get('cands')}", flush=True)
        print(f"IRC_CODE coalesced_calls={n_coal} plain_calls={n_plain}", flush=True)
        print(f"IRC_WAITS on_unpacks={sum(1 for w in waited if w in unpacked)} total={len(waited)}", flush=True)
        print(f"IRC_NUMERIC rel={rel:.5f} ok={ok}", flush=True)
        print("IRC_PASS" if all_ok else "IRC_FAIL", flush=True)
        if os.environ.get("IRC_DUMP_CODE"):
            print(code, flush=True)
    dist.barrier()
    dist.destroy_process_group()
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
