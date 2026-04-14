# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from magi_compiler import magi_compile, magi_register_custom_op
from magi_compiler.config import get_compile_config

DEVICE = "cuda"
DTYPE = torch.bfloat16
HIDDEN = 4
SEQ_LEN = 4


@magi_register_custom_op("test::boundary_identity_restart_cache_shared", infer_output_meta_fn=["x"], is_subgraph_boundary=True)
def _boundary_identity(x: torch.Tensor) -> torch.Tensor:
    # Custom op outputs must not alias inputs.
    return x.clone()


class TwoGroupDispatcher:
    def __init__(self, routing: torch.Tensor):
        self.group_size_cpu = torch.bincount(routing, minlength=2).to(torch.int32).cpu().tolist()
        self.permute = torch.argsort(routing)
        self.inv_permute = torch.argsort(self.permute)

    def dispatch(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x[self.permute]
        a, b = torch.split(x, self.group_size_cpu, dim=0)
        return a, b

    def undispatch(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat([a, b], dim=0)[self.inv_permute]


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(HIDDEN, dtype=DTYPE, device=DEVICE)
        self.fc = nn.Linear(HIDDEN, HIDDEN, dtype=DTYPE, device=DEVICE)

    def forward(self, x: torch.Tensor, use_boundary: bool) -> torch.Tensor:
        y = self.fc(self.norm(x))
        if use_boundary:
            y = _boundary_identity(y)
        return x + y


@magi_compile(dynamic_arg_dims={"x": 0, "permute": 0})
class CompiledBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = TinyBlock()

    def forward(
        self, x: torch.Tensor, permute: torch.Tensor, dispatcher: TwoGroupDispatcher, use_boundary: bool
    ) -> torch.Tensor:
        group0, group1 = dispatcher.dispatch(x)
        group0 = self.layer(group0, use_boundary)
        group1 = self.layer(group1, use_boundary)
        return dispatcher.undispatch(group0, group1)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = CompiledBlock()

    def forward(self, x: torch.Tensor, routing: torch.Tensor, use_boundary: bool = True) -> torch.Tensor:
        dispatcher = TwoGroupDispatcher(routing)
        return self.block(x, dispatcher.permute, dispatcher, use_boundary)


def _collect_subgraph0_events_from_run(run_dir: Path) -> dict[str, list[dict[str, object]]]:
    tracked = {
        "fullgraph_before_compiler_manager_load": [],
        "fullgraph_after_compiler_manager_load": [],
        "fullgraph_before_compiler_compile": [],
        "fullgraph_failed_compiler_compile": [],
    }
    for timeline_jsonl in sorted(run_dir.rglob("timeline_events/timeline.jsonl")):
        lines = [line for line in timeline_jsonl.read_text().splitlines() if line.strip()]
        events = [json.loads(line) for line in lines]
        for ev in events:
            name = ev.get("name")
            attrs = ev.get("attributes", {}) or {}
            if attrs.get("graph_index") != 0:
                continue
            if name in tracked:
                tracked[name].append({"index": int(ev.get("index", -1)), "attributes": attrs})
    return tracked


def _collect_new_run_dirs(cache_root: Path, before: set[str]) -> list[Path]:
    all_runs = sorted([p for p in cache_root.rglob("run_*") if p.is_dir()])
    return [p for p in all_runs if str(p) not in before]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = get_compile_config()
    cache_root = Path(config.cache_root_dir)
    run_dirs_before = {str(p) for p in cache_root.rglob("run_*") if p.is_dir()}

    torch._dynamo.reset()
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    model = Model().eval()
    x = torch.randn(SEQ_LEN, HIDDEN, device=DEVICE, dtype=DTYPE)
    group0 = SEQ_LEN // 2
    group1 = SEQ_LEN - group0
    routing = torch.tensor([0] * group0 + [1] * group1, device=DEVICE, dtype=torch.long)

    with torch.no_grad():
        out = model(x, routing, True)

    new_run_dirs = _collect_new_run_dirs(cache_root, run_dirs_before)
    run_events = {}
    for run_dir in new_run_dirs:
        run_events[str(run_dir)] = _collect_subgraph0_events_from_run(run_dir)

    payload = {
        "shape": list(out.shape),
        "sum": float(out.float().sum().item()),
        "mean": float(out.float().mean().item()),
        "new_run_dirs": [str(p) for p in new_run_dirs],
        "subgraph0_events_by_run": run_events,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f)


if __name__ == "__main__":
    main()
