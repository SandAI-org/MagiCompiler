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

"""
Regression tests for the EP shared-memory weight corruption bug.

Root cause: _materialize_shm_weights with per_rank=False creates a single
shared-memory file from local_rank=0 and has ALL ranks map it.  With expert
parallelism (EP > 1), each rank holds a different expert shard; sharing one
mmap means the last writer's data overwrites everyone else's views.

Fix: when ep_size > 1, the caller passes per_rank=True so each rank writes
its OWN shared-memory file, preserving expert shard diversity.

These tests use torch.multiprocessing.spawn with 2 workers and the gloo
backend to reproduce the exact multi-rank scenario without a real model.
"""

import os
import tempfile
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from magi_compiler._api import _materialize_shm_weights


class FakeExpertBlock(nn.Module):
    """Tiny module simulating an EP-sharded expert block."""

    def __init__(self, num_experts: int, dim: int):
        super().__init__()
        self.expert_weight = nn.Parameter(torch.randn(num_experts, dim, dtype=torch.bfloat16))

    def forward(self, x):
        return x @ self.expert_weight.T


def _group_params(module: nn.Module) -> dict[torch.dtype, list[tuple[str, torch.Tensor]]]:
    grouped: dict[torch.dtype, list[tuple[str, torch.Tensor]]] = {}
    for name, param in module.named_parameters():
        grouped.setdefault(param.dtype, []).append((name, param.data))
    return grouped


def _run_materialize(model, local_rank, shared_dir, per_rank):
    """Call production _materialize_shm_weights with env patched to use tmpdir."""
    grouped = _group_params(model)
    with patch("magi_compiler.utils.envs.MAGI_SHARED_BIN_PATH", shared_dir), patch(
        "magi_compiler._api.pin_memory_in_place", lambda t: t
    ):
        _materialize_shm_weights(model, grouped, local_rank=local_rank, per_rank=per_rank)


# ───────────────────────────────────────────────────────────────
#  Worker functions for spawn
# ───────────────────────────────────────────────────────────────


def _worker_bug_repro(rank, world_size, shared_dir, seed_per_rank, result_file):
    """Reproduces the bug: EP>1 but per_rank=False → shared mmap corrupts weights."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    torch.manual_seed(seed_per_rank[rank])
    model = FakeExpertBlock(num_experts=4, dim=8)
    original_weight = model.expert_weight.data.clone()

    _run_materialize(model, local_rank=rank, shared_dir=shared_dir, per_rank=False)

    weight_after = model.state_dict()["expert_weight"]
    matches_own = torch.equal(weight_after, original_weight)

    torch.save({"rank": rank, "matches_own": matches_own, "weight": weight_after.clone()}, f"{result_file}_{rank}.pt")
    dist.destroy_process_group()


def _worker_fix_verified(rank, world_size, shared_dir, seed_per_rank, result_file):
    """Verifies the fix: per_rank=True → each rank keeps its own expert shard."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29502"
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    torch.manual_seed(seed_per_rank[rank])
    model = FakeExpertBlock(num_experts=4, dim=8)
    original_weight = model.expert_weight.data.clone()

    _run_materialize(model, local_rank=rank, shared_dir=shared_dir, per_rank=True)

    weight_after = model.state_dict()["expert_weight"]
    matches_own = torch.equal(weight_after, original_weight)

    torch.save({"rank": rank, "matches_own": matches_own, "weight": weight_after.clone()}, f"{result_file}_{rank}.pt")
    dist.destroy_process_group()


# ───────────────────────────────────────────────────────────────
#  Tests
# ───────────────────────────────────────────────────────────────


def test_shared_memory_overwrites_ep_shards():
    """
    BUG REPRO: with per_rank=False on EP>1 ranks, the shared mmap causes
    at least one rank to lose its unique expert weights.
    """
    world_size = 2
    seeds = {0: 42, 1: 123}

    with tempfile.TemporaryDirectory() as tmpdir:
        result_file = os.path.join(tmpdir, "result")
        mp.spawn(_worker_bug_repro, args=(world_size, tmpdir, seeds, result_file), nprocs=world_size, join=True)

        r0 = torch.load(f"{result_file}_0.pt", weights_only=True)
        r1 = torch.load(f"{result_file}_1.pt", weights_only=True)

        assert not (r0["matches_own"] and r1["matches_own"]), (
            "BUG REPRO FAILED: both ranks kept their own weights with per_rank=False. "
            "EP>1 with shared mmap should corrupt at least one rank's weights."
        )
        assert torch.equal(r0["weight"], r1["weight"]), (
            "After shared-memory dedup with per_rank=False, both ranks should end up "
            "with identical weights — this is the core of the EP corruption bug."
        )


def test_ep_fix_preserves_per_rank_shards():
    """
    FIX VERIFIED: with per_rank=True, each rank writes its own mmap file and
    keeps its unique expert shard intact.
    """
    world_size = 2
    seeds = {0: 42, 1: 123}

    with tempfile.TemporaryDirectory() as tmpdir:
        result_file = os.path.join(tmpdir, "result")
        mp.spawn(_worker_fix_verified, args=(world_size, tmpdir, seeds, result_file), nprocs=world_size, join=True)

        r0 = torch.load(f"{result_file}_0.pt", weights_only=True)
        r1 = torch.load(f"{result_file}_1.pt", weights_only=True)

        assert r0["matches_own"], "rank 0 should keep its own weights"
        assert r1["matches_own"], (
            "FIX FAILED: rank 1 should keep its own weights when per_rank=True, " "but they were overwritten."
        )
        assert not torch.equal(r0["weight"], r1["weight"]), (
            "With per_rank=True, each rank should have DIFFERENT expert weights. "
            "If they're equal, the per-rank shm path did not work correctly."
        )
