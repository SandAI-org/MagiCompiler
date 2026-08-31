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

"""
Regression tests for the EP shared-memory weight corruption bug.

Root cause: _patch_cpu_offload_apply created a single shared-memory file
from local_rank=0 and had ALL ranks read it.  With expert parallelism
(EP > 1), each rank holds a different expert shard; reading rank-0's data
on every rank destroyed expert weight diversity and produced garbled output.

Fix: when EP_SIZE > 1, each rank writes its OWN shared-memory file so that
expert shards are preserved.  When EP_SIZE <= 1, the original rank-0-writes
all-read scheme is safe (weights are identical across ranks).

These tests use torch.multiprocessing.spawn with 2 workers and the gloo
backend to reproduce the exact multi-rank scenario without a real model.
"""

import gc
import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from magi_compiler._api import (
    _create_empty_shm,
    _pack_params_flat,
    _split_flat_to_params,
    _stream_copy_and_replace,
)

class FakeExpertBlock(nn.Module):
    """Tiny module simulating an EP-sharded expert block."""

    def __init__(self, num_experts: int, dim: int):
        super().__init__()
        self.expert_weight = nn.Parameter(torch.randn(num_experts, dim, dtype=torch.bfloat16))

    def forward(self, x):
        return x @ self.expert_weight.T


def _group_params(module: nn.Module) -> dict[torch.dtype, list[tuple[str, torch.Tensor]]]:
    grouped: dict[torch.dtype, list[tuple[str, torch.Tensor]]] = {}
    for name, tensor in module.state_dict().items():
        grouped.setdefault(tensor.dtype, []).append((name, tensor))
    return grouped


def _shm_write_read(module, local_rank, shared_dir, per_rank):
    """Mirrors _materialize_shm_weights using production helpers.

    per_rank=False → original buggy path (rank 0 writes, all read same file
                     via load_state_dict — every rank gets rank 0's weights).
    per_rank=True  → fixed path (each rank writes its own file, uses
                     _stream_copy_and_replace to keep its own weights).
    """
    grouped = _group_params(module)

    if per_rank:
        for dtype, param_list in grouped.items():
            suffix = f"_rank{local_rank}"
            shared_path = os.path.join(shared_dir, f"shared_{str(dtype).split('.')[-1]}{suffix}.bin")
            total_numel = sum(t.numel() for _, t in param_list)
            giant = _create_empty_shm(shared_path, total_numel, dtype)
            _stream_copy_and_replace(module, giant, param_list)
            dist.barrier()
            if os.path.exists(shared_path):
                os.remove(shared_path)
    else:
        shared_state = {}
        for dtype, param_list in grouped.items():
            shared_path = os.path.join(shared_dir, f"shared_{str(dtype).split('.')[-1]}.bin")
            total_numel = sum(t.numel() for _, t in param_list)
            if local_rank == 0:
                giant = _create_empty_shm(shared_path, total_numel, dtype)
                _pack_params_flat(giant, param_list)
            dist.barrier()
            if local_rank != 0:
                giant = torch.from_file(shared_path, shared=True, size=total_numel, dtype=dtype, device="cpu")
            shared_state.update(_split_flat_to_params(giant, param_list))
            dist.barrier()
            if local_rank == 0 and os.path.exists(shared_path):
                os.remove(shared_path)
        module.load_state_dict(shared_state, assign=True)


# ───────────────────────────────────────────────────────────────
#  Worker functions for spawn
# ───────────────────────────────────────────────────────────────


def _worker_bug_repro(rank, world_size, shared_dir, seed_per_rank, result_file):
    """Reproduces the bug: all ranks get rank-0's weights after shared-memory dedup."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    torch.manual_seed(seed_per_rank[rank])
    model = FakeExpertBlock(num_experts=4, dim=8)
    original_weight = model.expert_weight.data.clone()

    _shm_write_read(model, local_rank=rank, shared_dir=shared_dir, per_rank=False)

    weight_after = model.state_dict()["expert_weight"]
    matches_own = torch.equal(weight_after, original_weight)

    torch.save({"rank": rank, "matches_own": matches_own, "weight": weight_after.clone()}, f"{result_file}_{rank}.pt")
    dist.destroy_process_group()


def _worker_fix_verified(rank, world_size, shared_dir, seed_per_rank, result_file):
    """Verifies the fix: EP>1 uses per-rank shm files, each rank keeps its own weights."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29502"
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MAGI_COMPILE_TOPOLOGY_KEY"] = f"cp1_dp1_ep{world_size}_tp1"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    torch.manual_seed(seed_per_rank[rank])
    model = FakeExpertBlock(num_experts=4, dim=8)
    original_weight = model.expert_weight.data.clone()

    from magi_compiler.config import get_topology_dim
    ep_size = get_topology_dim("ep")
    _shm_write_read(model, local_rank=rank, shared_dir=shared_dir, per_rank=(ep_size > 1))

    weight_after = model.state_dict()["expert_weight"]
    matches_own = torch.equal(weight_after, original_weight)

    torch.save({"rank": rank, "matches_own": matches_own, "weight": weight_after.clone()}, f"{result_file}_{rank}.pt")
    dist.destroy_process_group()


# ───────────────────────────────────────────────────────────────
#  Tests
# ───────────────────────────────────────────────────────────────


def test_shared_memory_overwrites_ep_shards():
    """
    BUG REPRO: with original shared-memory logic, rank 1's expert weights
    are silently overwritten by rank 0's data.
    """
    world_size = 2
    seeds = {0: 42, 1: 123}

    with tempfile.TemporaryDirectory() as tmpdir:
        result_file = os.path.join(tmpdir, "result")
        mp.spawn(_worker_bug_repro, args=(world_size, tmpdir, seeds, result_file), nprocs=world_size, join=True)

        r0 = torch.load(f"{result_file}_0.pt", weights_only=True)
        r1 = torch.load(f"{result_file}_1.pt", weights_only=True)

        assert r0["matches_own"], "rank 0 should keep its own weights (it wrote the file)"
        assert not r1["matches_own"], (
            "BUG REPRO FAILED: rank 1 should have LOST its weights "
            "(overwritten by rank 0's shared-memory file), but it still matches. "
            "The bug may have been fixed upstream — update this test."
        )
        assert torch.equal(r0["weight"], r1["weight"]), (
            "After shared-memory dedup, both ranks should have identical weights "
            "(rank 0's data). This is the core of the EP corruption bug."
        )


def test_ep_fix_preserves_per_rank_shards():
    """
    FIX VERIFIED: with EP_SIZE > 1, per-rank shared-memory files ensure each
    rank keeps its own expert shard intact.
    """
    world_size = 2
    seeds = {0: 42, 1: 123}

    with tempfile.TemporaryDirectory() as tmpdir:
        result_file = os.path.join(tmpdir, "result")
        mp.spawn(_worker_fix_verified, args=(world_size, tmpdir, seeds, result_file), nprocs=world_size, join=True)

        r0 = torch.load(f"{result_file}_0.pt", weights_only=True)
        r1 = torch.load(f"{result_file}_1.pt", weights_only=True)

        assert r0["matches_own"], "rank 0 should keep its own weights"
        assert r1["matches_own"], "FIX FAILED: rank 1 should keep its own weights when EP > 1, " "but they were overwritten."
        assert not torch.equal(r0["weight"], r1["weight"]), (
            "With EP > 1, each rank should have DIFFERENT expert weights. "
            "If they're equal, the per-rank shm path did not work correctly."
        )
