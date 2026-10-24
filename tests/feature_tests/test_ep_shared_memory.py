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

Fix: weights are compared one by one (per-weight fingerprints across the ranks
of a host); only the identical ones share a file, each rank keeps its own
expert shard.

These tests use torch.multiprocessing.spawn with 2 workers and the gloo
backend to reproduce the exact multi-rank scenario without a real model.
"""

import os
import tempfile
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from magi_compiler._api import _materialize_shm_weights, _param_fingerprint, _rank_identical_params


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
    """Call production _materialize_shm_weights with env patched to use tmpdir.

    per_rank=True keeps every weight per rank, False shares all of them, None shares what the ranks hold identically.
    """
    grouped = _group_params(model)
    with patch("magi_compiler.utils.envs.MAGI_SHARED_BIN_PATH", shared_dir), patch(
        "magi_compiler._api.pin_memory_in_place", lambda t: t
    ):
        if per_rank is None:
            shared, dtypes = _rank_identical_params(grouped)
        else:
            shared = set() if per_rank else {name for params in grouped.values() for name, _ in params}
            dtypes = [] if per_rank else sorted(grouped, key=str)
        _materialize_shm_weights(model, grouped, local_rank, shared, dtypes)
    return shared


# ───────────────────────────────────────────────────────────────
#  Worker functions for spawn
# ───────────────────────────────────────────────────────────────


def _worker_forced_sharing(rank, world_size, shared_dir, seed_per_rank, result_file):
    """EP>1 but per_rank=False: every weight goes through the shared mmap although the ranks' differ."""
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


class FakeMoEBlock(nn.Module):
    """A dense weight every EP rank shares, an expert shard each rank owns, and a dense weight of another dtype."""

    def __init__(self, rank: int):
        super().__init__()
        torch.manual_seed(0)
        self.dense = nn.Parameter(torch.randn(64, 96, dtype=torch.bfloat16))
        self.norm = nn.Parameter(torch.randn(96))
        torch.manual_seed(1000 + rank)
        self.expert_weight = nn.Parameter(torch.randn(4, 64, dtype=torch.bfloat16))


def _worker_mixed(rank, world_size, shared_dir, result_file, port="29503", hosts=1, free_shm=None):
    """Auto-detect: the dense weights share one file per host, each rank keeps its own experts.

    The ranks are split evenly over *hosts* fake hosts; ``free_shm[i]`` is the free shared memory reported on host i.
    """
    per_host = world_size // hosts
    host = f"host{rank // per_host}"
    host_dir = os.path.join(shared_dir, host)
    os.makedirs(host_dir, exist_ok=True)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["LOCAL_RANK"] = str(rank % per_host)
    os.environ["LOCAL_WORLD_SIZE"] = str(per_host)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    model = FakeMoEBlock(rank)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    with ExitStack() as stack:
        stack.enter_context(patch("magi_compiler._api.socket.gethostname", return_value=host))
        if free_shm is not None:
            usage = SimpleNamespace(free=free_shm[rank // per_host])
            stack.enter_context(patch("magi_compiler._api.shutil.disk_usage", return_value=usage))
        shared = _run_materialize(model, local_rank=rank % per_host, shared_dir=host_dir, per_rank=None)

    after = model.state_dict()
    dist.barrier()  # rank 0 removes the file after the last barrier inside
    torch.save(
        {
            "shared": sorted(shared),
            "matches_own": all(torch.equal(after[k], original[k]) for k in original),
            "buffers": sorted((str(b.dtype), b.numel()) for b in model._magi_giant_buffers),
            "leftover": sorted(os.listdir(host_dir)),
        },
        f"{result_file}_{rank}.pt",
    )
    dist.destroy_process_group()


# ───────────────────────────────────────────────────────────────
#  Tests
# ───────────────────────────────────────────────────────────────


def test_only_rank_identical_weights_are_shared():
    world_size = 2
    with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as shared_dir:
        result_file = os.path.join(tmpdir, "result")
        mp.spawn(_worker_mixed, args=(world_size, shared_dir, result_file), nprocs=world_size, join=True)

        for r in range(world_size):
            got = torch.load(f"{result_file}_{r}.pt", weights_only=True)
            assert got["shared"] == ["dense", "norm"], got
            assert got["matches_own"], f"rank {r} lost its own weights"
            # One per-rank bf16 buffer (experts), one shared buffer per dtype (dense bf16, norm fp32).
            assert got["buffers"] == sorted([("torch.bfloat16", 4 * 64), ("torch.bfloat16", 64 * 96), ("torch.float32", 96)])
            assert got["leftover"] == [], got


_PER_RANK_BUFFERS = sorted([("torch.bfloat16", 4 * 64 + 64 * 96), ("torch.float32", 96)])


def test_full_shared_memory_keeps_every_weight_per_rank():
    """Writing past a full tmpfs is SIGBUS, so a host without room shares nothing."""
    world_size = 2
    with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as shared_dir:
        result_file = os.path.join(tmpdir, "result")
        args = (world_size, shared_dir, result_file, "29504", 1, [0])
        mp.spawn(_worker_mixed, args=args, nprocs=world_size, join=True)

        for r in range(world_size):
            got = torch.load(f"{result_file}_{r}.pt", weights_only=True)
            assert got["shared"] == [], got
            assert got["matches_own"], f"rank {r} lost its own weights"
            assert got["buffers"] == _PER_RANK_BUFFERS, got
            assert got["leftover"] == [], got


def test_host_without_room_does_not_stall_the_others():
    """Each host decides alone; the one without room shares nothing while the other still shares."""
    world_size = 4
    with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as shared_dir:
        result_file = os.path.join(tmpdir, "result")
        args = (world_size, shared_dir, result_file, "29505", 2, [1 << 40, 0])
        mp.spawn(_worker_mixed, args=args, nprocs=world_size, join=True)

        for r in range(world_size):
            got = torch.load(f"{result_file}_{r}.pt", weights_only=True)
            assert got["matches_own"], f"rank {r} lost its own weights"
            assert got["leftover"] == [], got
            if r < 2:
                assert got["shared"] == ["dense", "norm"], got
            else:
                assert got["shared"] == [], got
                assert got["buffers"] == _PER_RANK_BUFFERS, got


def test_forced_sharing_keeps_differing_weights_per_rank():
    """
    With per_rank=False on EP>1 ranks, the other ranks find their experts differ
    from local rank 0's in the shared mmap and keep their own copy.
    """
    world_size = 2
    seeds = {0: 42, 1: 123}

    with tempfile.TemporaryDirectory() as tmpdir:
        result_file = os.path.join(tmpdir, "result")
        mp.spawn(_worker_forced_sharing, args=(world_size, tmpdir, seeds, result_file), nprocs=world_size, join=True)

        r0 = torch.load(f"{result_file}_0.pt", weights_only=True)
        r1 = torch.load(f"{result_file}_1.pt", weights_only=True)

        assert r0["matches_own"], "rank 0 should keep its own weights"
        assert r1["matches_own"], "rank 1 took rank 0's experts from the shared mmap"
        assert not torch.equal(r0["weight"], r1["weight"])


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


# ───────────────────────────────────────────────────────────────
#  Fingerprint tests
# ───────────────────────────────────────────────────────────────


def test_fingerprint_identical_models():
    """Two models with the same seed produce the same fingerprint."""
    torch.manual_seed(42)
    m1 = FakeExpertBlock(num_experts=4, dim=8)
    torch.manual_seed(42)
    m2 = FakeExpertBlock(num_experts=4, dim=8)

    assert _param_fingerprint(m1.expert_weight.data) == _param_fingerprint(m2.expert_weight.data)


def test_fingerprint_different_models():
    """Two models with different seeds produce different fingerprints."""
    torch.manual_seed(42)
    m1 = FakeExpertBlock(num_experts=4, dim=8)
    torch.manual_seed(123)
    m2 = FakeExpertBlock(num_experts=4, dim=8)

    assert _param_fingerprint(m1.expert_weight.data) != _param_fingerprint(m2.expert_weight.data)


def test_fingerprint_detects_single_element_change():
    """Changing one element changes the fingerprint (small tensors are hashed whole)."""
    torch.manual_seed(42)
    w = torch.randn(4, 8, dtype=torch.bfloat16)
    changed = w.clone()
    changed[2, 3] += 1.0

    assert _param_fingerprint(w) != _param_fingerprint(changed)


def test_fingerprint_samples_the_middle_of_large_tensors():
    """Large tensors equal at both ends but different in between still differ."""
    a = torch.zeros(1 << 20, dtype=torch.bfloat16)
    b = a.clone()
    b[1000:-1000] = 1.0

    assert _param_fingerprint(a) != _param_fingerprint(b)
    assert _param_fingerprint(a) == _param_fingerprint(a.clone())
    assert _param_fingerprint(a) != _param_fingerprint(a.view(1024, 1024))
