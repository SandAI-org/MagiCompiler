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
from torch._inductor import ir
from torch.distributed.device_mesh import init_device_mesh

from magi_compiler import magi_compile
from magi_compiler.config import CompileMode, CudaGraphMode

_COALESCE_COUNTS = {"buckets": 0, "members": 0}
_COALESCED_ALL_GATHER_OP = torch.ops._c10d_functional.all_gather_into_tensor_coalesced.default
_GENERATED_SOURCES: list[str] = []


class Block(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class TinyModel(nn.Module):
    def __init__(self, hidden: int, n_layers: int, extra_op: bool = False):
        super().__init__()
        self.layers = nn.ModuleList(Block(hidden) for _ in range(n_layers))
        self.extra_op = extra_op

    def forward(self, x):
        if self.extra_op:
            # Same values, different graph: rank-private compute for ``slot`` mode
            # (``sin(x) * 0`` would be folded away).
            x = torch.where(torch.isnan(x), torch.sin(x), x)
        for layer in self.layers:
            x = layer(x)
        return x


class FixedSizeCoalesce:
    """Merge every ``bucket_member_count`` consecutive weight gathers that can legally merge."""

    def __init__(self, bucket_member_count: int) -> None:
        self.bucket_member_count = bucket_member_count

    def __deepcopy__(self, memo):
        return self

    def __call__(self, snodes):
        from magi_compiler.passes.fsdp_overlap.ir_coalesce import (
            coalesce_all_gathers,
            coalesce_placement,
            parse_single_gather,
            refresh_memory_planning,
        )
        from magi_compiler.passes.fsdp_overlap.node_meta import is_weight_ag

        order = list(snodes)
        gather_candidates = [s for s in order if parse_single_gather(s) is not None]
        _COALESCE_COUNTS["tagged"] = sum(1 for s in gather_candidates if any(is_weight_ag(o) for o in s.node.origins))
        _COALESCE_COUNTS["cands"] = len(gather_candidates)
        for i in range(0, len(gather_candidates) - self.bucket_member_count + 1, self.bucket_member_count):
            members = gather_candidates[i : i + self.bucket_member_count]
            if coalesce_placement(order, members) is None:
                continue
            order, _, _ = coalesce_all_gathers(order, members)
            _COALESCE_COUNTS["buckets"] += 1
            _COALESCE_COUNTS["members"] += len(members)
        snodes[:] = order
        refresh_memory_planning(snodes)
        return snodes


def _install_coalesce_test_hooks(bucket_member_count: int, diverge: bool = False) -> None:
    """``bucket_member_count > 0``: fixed groups through FixedSizeCoalesce; 0: the config's own passes.

    ``diverge``: ranks other than 0 drop their last planned bucket, standing in
    for a plan flipped by per-rank cost noise; rank 0's plan must still win.
    """
    from magi_compiler.magi_backend import weight_pipeline

    original_build_passes = weight_pipeline.build_scheduler_passes

    def build_test_passes(**kwargs):
        passes = original_build_passes(**kwargs)
        if passes and bucket_member_count > 0:
            passes.insert(1, FixedSizeCoalesce(bucket_member_count))
        return passes

    weight_pipeline.build_scheduler_passes = build_test_passes

    from magi_compiler.passes.fsdp_overlap import auto_bucket

    original_bucket_pass = auto_bucket.FsdpAutoBucket.__call__

    def count_coalesced_gathers(self, snodes):
        gather_count_before = sum(1 for s in snodes if auto_bucket.parse_single_gather(s) is not None)
        reordered_nodes = original_bucket_pass(self, snodes)
        gather_count_after = sum(1 for s in reordered_nodes if auto_bucket.parse_single_gather(s) is not None)
        coalesced_gather_count = sum(
            1
            for s in reordered_nodes
            if isinstance(s.node, ir.ExternKernel) and s.node.op_overload is _COALESCED_ALL_GATHER_OP
        )
        _COALESCE_COUNTS["buckets"] += coalesced_gather_count
        _COALESCE_COUNTS["members"] += gather_count_before - gather_count_after
        return reordered_nodes

    auto_bucket.FsdpAutoBucket.__call__ = count_coalesced_gathers

    if diverge and dist.get_rank() != 0:
        original_plan_buckets = auto_bucket.FsdpAutoBucket._plan_local_buckets

        def drop_last_bucket(self, order, gather_candidates, cost):
            buckets, params_by_group = original_plan_buckets(self, order, gather_candidates, cost)
            return buckets[:-1], params_by_group

        auto_bucket.FsdpAutoBucket._plan_local_buckets = drop_last_bucket

    from torch._inductor.graph import GraphLowering

    original_codegen = GraphLowering.codegen

    def capture_codegen(self, *args, **kwargs):
        codegen_result = original_codegen(self, *args, **kwargs)
        try:
            generated_source = codegen_result[0].value if hasattr(codegen_result[0], "value") else str(codegen_result[0])
        except Exception:  # noqa: BLE001
            generated_source = str(codegen_result)
        _GENERATED_SOURCES.append(generated_source)
        return codegen_result

    GraphLowering.codegen = capture_codegen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", type=int, default=2, help="0 = bucket_mode='auto' instead of fixed pairs")
    ap.add_argument("--bucket-size-mib", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--cost-mode", default="analytical", choices=["analytical", "profile_sync"])
    ap.add_argument("--diverge", action="store_true", help="non-zero ranks plan one bucket fewer")
    ap.add_argument("--mismatch", action="store_true", help="rank 1 compiles a structurally different graph")
    ap.add_argument(
        "--bucket-target-mib",
        type=float,
        default=0.0,
        help="fixed gathered MiB per auto bucket instead of the measured fit (0 = measured)",
    )
    args = ap.parse_args()

    dist.init_process_group("cpu:gloo,cuda:nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.cuda.current_device()
    torch.manual_seed(0)
    os.environ.setdefault("MAGI_LOGGING_LEVEL", "INFO")
    _install_coalesce_test_hooks(args.pair, args.diverge)
    if args.bucket_target_mib > 0:
        from magi_compiler.passes.fsdp_overlap import auto_bucket

        # Measured gather costs vary run to run; a fixed target makes the merge count a function of the shape.
        auto_bucket.BucketPlanParams.target_gathered_bytes = property(lambda self: args.bucket_target_mib * 2**20)

    from torchtitan.experiments.simple_fsdp.simple_fsdp import data_parallel

    mesh = init_device_mesh("cuda", (world_size,))
    extra_op = args.mismatch and rank == 1
    ref = TinyModel(args.hidden, args.n_layers, extra_op).to(dev).to(torch.bfloat16)
    x = torch.randn(64, args.hidden, device=dev, dtype=torch.bfloat16)
    with torch.no_grad():
        eager_out = ref(x)

    model = TinyModel(args.hidden, args.n_layers, extra_op).to(dev).to(torch.bfloat16)
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
        return cfg

    compiled = magi_compile(model, config_patch=_patch, dynamic_arg_dims={"x": 0})
    with torch.no_grad():
        out = compiled(x)
        repeated_output = compiled(x)
        torch.cuda.synchronize()

    code = "\n".join(_GENERATED_SOURCES)
    n_coal = len(re.findall(r"all_gather_into_tensor_coalesced", code))
    n_plain = len(re.findall(r"all_gather_into_tensor\.default\(", code))
    packed = set(re.findall(r"(\w+) = torch\.ops\._c10d_functional\.all_gather_into_tensor_coalesced", code))
    unpacked = {m.group(1) for m in re.finditer(r"(\w+) = (\w+)\[\d+\]", code) if m.group(2) in packed}
    waited = re.findall(r"wait_tensor\.default\((\w+)\)", code)
    relative_error = ((out.float() - eager_out.float()).norm() / (eager_out.float().norm() + 1e-6)).item()
    is_output_correct = bool(torch.isfinite(out).all().item()) and relative_error < 5e-2 and torch.equal(out, repeated_output)
    is_output_correct = is_output_correct and (world_size == 1 or _COALESCE_COUNTS["buckets"] > 0)
    correctness_vote = torch.tensor([1 if is_output_correct else 0], device=dev)
    dist.all_reduce(correctness_vote)
    all_ok = int(correctness_vote.item()) == world_size
    per_rank: list = [None] * world_size
    dist.all_gather_object(per_rank, (_COALESCE_COUNTS["buckets"], _COALESCE_COUNTS["members"]))
    if rank == 0:
        print(f"IRC_COALESCED buckets={_COALESCE_COUNTS['buckets']} members={_COALESCE_COUNTS['members']}", flush=True)
        print(f"IRC_RANK_BUCKETS {' '.join(f'{b}/{m}' for b, m in per_rank)}", flush=True)
        print(f"IRC_TAGGED {_COALESCE_COUNTS.get('tagged')}/{_COALESCE_COUNTS.get('cands')}", flush=True)
        print(f"IRC_CODE coalesced_calls={n_coal} plain_calls={n_plain}", flush=True)
        print(f"IRC_WAITS on_unpacks={sum(1 for w in waited if w in unpacked)} total={len(waited)}", flush=True)
        print(f"IRC_NUMERIC rel={relative_error:.5f} ok={is_output_correct}", flush=True)
        print("IRC_PASS" if all_ok else "IRC_FAIL", flush=True)
        if os.environ.get("IRC_DUMP_CODE"):
            print(code, flush=True)
    dist.barrier()
    dist.destroy_process_group()
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
