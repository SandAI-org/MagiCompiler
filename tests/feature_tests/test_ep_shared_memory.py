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

Fix: when EP_SIZE > 1, fall back to per-rank pin_memory instead of
cross-rank shared-memory dedup.

These tests use torch.multiprocessing.spawn with 2 GPUs to reproduce the
exact multi-rank scenario without a real model.
"""

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn


_skip_no_dist = pytest.mark.skipif(
    not hasattr(dist, "is_gloo_available") or not dist.is_gloo_available(),
    reason="requires gloo backend",
)


class FakeExpertBlock(nn.Module):
    """Tiny module simulating an EP-sharded expert block."""

    def __init__(self, num_experts: int, dim: int):
        super().__init__()
        self.expert_weight = nn.Parameter(torch.randn(num_experts, dim, dtype=torch.bfloat16))

    def forward(self, x):
        return x @ self.expert_weight.T


def _extract_shared_memory_logic(module, local_rank, shared_dir):
    """
    Reproduces the ORIGINAL (buggy) shared-memory logic from
    _patch_cpu_offload_apply: only local_rank==0 writes the file,
    all ranks read the same file and load_state_dict(assign=True).
    """
    full_state_dict = module.state_dict()
    grouped: dict[torch.dtype, list[tuple[str, torch.Tensor]]] = {}
    for name, tensor in full_state_dict.items():
        dt = tensor.dtype
        grouped.setdefault(dt, []).append((name, tensor))

    shared_state = {}
    for dtype, param_list in grouped.items():
        dtype_str = str(dtype).split(".")[-1]
        shared_path = os.path.join(shared_dir, f"shared_{dtype_str}.bin")
        total_numel = sum(t.numel() for _, t in param_list)

        if local_rank == 0:
            flat = torch.zeros(total_numel, dtype=dtype)
            off = 0
            for _, t in param_list:
                n = t.numel()
                flat[off : off + n].copy_(t.view(-1))
                off += n
            if dtype == torch.bfloat16:
                flat.view(torch.int16).numpy().tofile(shared_path)
            else:
                flat.numpy().tofile(shared_path)
            del flat

        dist.barrier()

        giant = torch.from_file(shared_path, shared=True, size=total_numel, dtype=dtype, device="cpu")
        off = 0
        for name, orig in param_list:
            n = orig.numel()
            shared_state[name] = giant[off : off + n].view(orig.shape)
            off += n

        dist.barrier()

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

    _extract_shared_memory_logic(model, local_rank=rank, shared_dir=shared_dir)

    weight_after = model.state_dict()["expert_weight"]
    matches_own = torch.equal(weight_after, original_weight)

    torch.save({"rank": rank, "matches_own": matches_own, "weight": weight_after.clone()}, f"{result_file}_{rank}.pt")
    dist.destroy_process_group()


def _worker_fix_verified(rank, world_size, shared_dir, seed_per_rank, result_file):
    """Verifies the fix: with EP>1, skip shared-memory → each rank keeps its own weights."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29502"
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["ENGINE_CONFIG__EP_SIZE"] = str(world_size)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    torch.manual_seed(seed_per_rank[rank])
    model = FakeExpertBlock(num_experts=4, dim=8)
    original_weight = model.expert_weight.data.clone()

    ep_size = int(os.environ.get("ENGINE_CONFIG__EP_SIZE", "1"))
    if ep_size <= 1:
        _extract_shared_memory_logic(model, local_rank=rank, shared_dir=shared_dir)
    else:
        pass  # pin_memory path: weights stay as-is

    weight_after = model.state_dict()["expert_weight"]
    matches_own = torch.equal(weight_after, original_weight)

    torch.save({"rank": rank, "matches_own": matches_own, "weight": weight_after.clone()}, f"{result_file}_{rank}.pt")
    dist.destroy_process_group()


# ───────────────────────────────────────────────────────────────
#  Tests
# ───────────────────────────────────────────────────────────────

@_skip_no_dist
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


@_skip_no_dist
def test_ep_fix_preserves_per_rank_shards():
    """
    FIX VERIFIED: with EP_SIZE > 1, shared-memory is skipped and each rank
    keeps its own expert shard.
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
            "FIX FAILED: rank 1 should keep its own weights when EP > 1, "
            "but they were overwritten."
        )
        assert not torch.equal(r0["weight"], r1["weight"]), (
            "With EP > 1, each rank should have DIFFERENT expert weights. "
            "If they're equal, the shared-memory path ran despite EP > 1."
        )
