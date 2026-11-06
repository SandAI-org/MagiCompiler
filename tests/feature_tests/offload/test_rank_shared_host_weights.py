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

"""Shared-weight tests check cross-rank writes and NUMA placement via /proc/self/numa_maps."""

import errno
import json
import os
import re
import socket
import tempfile
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from magi_compiler.passes.weight_offload.runtime.rank_shared_host_region import (
    SHARED_MEMORY_HEADROOM_BYTES,
    _plan_sharing_group_regions,
    _RankSharingPlanInput,
    _WeightDescriptor,
)
from magi_compiler.utils.numa import online_numa_nodes

requires_cuda_and_tmpfs = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.isdir("/dev/shm"), reason="requires CUDA and /dev/shm"
)

_OVERWRITE_VALUE = 7.0


def _writer_failure_worker(rank: int, port: str, result_dir: str, failure_stage: str) -> None:
    from unittest.mock import patch

    from magi_compiler.passes.weight_offload.runtime import rank_shared_host_region as region_module
    from magi_compiler.passes.weight_offload.weight_lifecycle import _park_with_shared_region

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group("gloo", rank=rank, world_size=2, timeout=timedelta(seconds=15))
    region_plan = region_module._SharingGroupRegionPlan(
        writer_rank=0, hostname="host", region_path=os.path.join(result_dir, "region"), slice_by_weight_name={}, region_bytes=8
    )
    try:

        def create_region(**kwargs):
            if failure_stage == "create":
                raise OSError("injected create failure")

        def park_weight(**kwargs):
            if failure_stage == "fill":
                raise RuntimeError("injected fill failure")
            return 0

        def map_region(**kwargs):
            if (failure_stage == "map" and rank == 0) or (failure_stage == "reader_map" and rank == 1):
                raise RuntimeError(f"injected {failure_stage} failure")
            return torch.empty(8)

        with (
            patch.object(region_module, "_create_shared_memory_file", create_region),
            patch.object(region_module, "_map_and_pin_shared_memory_file", map_region),
            patch("magi_compiler.passes.weight_offload.weight_lifecycle._park_as_shared_region_writer", park_weight),
        ):
            region = region_module.RankSharedHostRegion(
                region_plan=region_plan, is_writer=rank == 0, region_numa_policy=None, gloo_group=dist.group.WORLD
            )
            try:
                _park_with_shared_region(cpu_parameters=[(["weight"], None)], shared_region=region)
            except RuntimeError as error:
                with open(os.path.join(result_dir, f"{rank}.txt"), "w") as result_file:
                    result_file.write(str(error))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("failure_stage", ["create", "map", "fill", "reader_map"])
def test_publication_failure_reaches_every_rank(tmp_path, failure_stage):
    """A writer or reader mapping failure must reach every rank without a barrier timeout."""
    mp.spawn(_writer_failure_worker, args=(_free_port(), str(tmp_path), failure_stage), nprocs=2, join=True)
    for rank in range(2):
        failed_rank = 1 if failure_stage == "reader_map" else 0
        assert f"rank {failed_rank}: " in (tmp_path / f"{rank}.txt").read_text()
        assert f"injected {failure_stage} failure" in (tmp_path / f"{rank}.txt").read_text()


def _preparation_failure_worker(rank: int, port: str, result_dir: str, failure_stage: str) -> None:
    from unittest.mock import patch

    from magi_compiler.passes.weight_offload.runtime import rank_shared_host_region as region_module

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group("gloo", rank=rank, world_size=2, timeout=timedelta(seconds=15))
    target = "shutil.disk_usage" if failure_stage == "disk_usage" else "compute_tensor_fingerprint"
    original = region_module.shutil.disk_usage if failure_stage == "disk_usage" else region_module.compute_tensor_fingerprint

    def prepare_value(*args):
        if rank == 1:
            raise OSError(f"injected {failure_stage} failure")
        return original(*args)

    try:
        with (
            patch.object(region_module, "get_cpu_gloo_group", return_value=dist.group.WORLD),
            patch("magi_compiler.utils.envs.MAGI_SHARED_BIN_PATH", result_dir),
            patch(f"{region_module.__name__}.{target}", side_effect=prepare_value),
        ):
            try:
                region_module.open_rank_shared_host_region(
                    tensor_by_weight_name={"weight": torch.ones(8)}, region_label="test", sharing_scope="per_host_interleaved"
                )
            except RuntimeError as error:
                with open(os.path.join(result_dir, f"{rank}.txt"), "w") as result_file:
                    result_file.write(str(error))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("failure_stage", ["disk_usage", "fingerprint"])
def test_plan_input_collection_failure_reaches_every_rank(tmp_path, failure_stage: str) -> None:
    """A single rank's local collection failure must not strand peers in the plan input gather."""
    mp.spawn(_preparation_failure_worker, args=(_free_port(), str(tmp_path), failure_stage), nprocs=2, join=True)
    for rank in range(2):
        error_message = (tmp_path / f"{rank}.txt").read_text()
        assert f"rank 1: OSError: injected {failure_stage} failure" in error_message


class _ReplicatedModelWithExpert(nn.Module):
    """``dense`` is identical on every rank, ``expert`` differs per rank (like an EP shard)."""

    def __init__(self, rank: int):
        super().__init__()
        torch.manual_seed(0)
        self.dense = nn.Linear(256, 256, bias=False)
        torch.manual_seed(1 + rank)
        self.expert = nn.Linear(256, 256, bias=False)


def _free_port() -> str:
    with socket.socket() as probe_socket:
        probe_socket.bind(("127.0.0.1", 0))
        return str(probe_socket.getsockname()[1])


def _scenario_patches(scenario: str, shared_dir: str) -> list:
    from types import SimpleNamespace
    from unittest.mock import patch

    region_module = "magi_compiler.passes.weight_offload.runtime.rank_shared_host_region"
    scenario_patches = [patch("magi_compiler.utils.envs.MAGI_SHARED_BIN_PATH", shared_dir)]
    if scenario == "fingerprint_collision":
        scenario_patches.append(patch(f"{region_module}.compute_tensor_fingerprint", lambda tensor: b"collision"))
    elif scenario == "no_room":
        scenario_patches.append(patch(f"{region_module}.shutil.disk_usage", lambda path: SimpleNamespace(free=0)))
    elif scenario == "allocation_no_room":
        scenario_patches.append(
            patch(f"{region_module}.os.posix_fallocate", side_effect=OSError(errno.ENOSPC, "injected allocation failure"))
        )
    elif scenario == "numa_policy_denied":
        scenario_patches.append(patch("magi_compiler.utils.numa._get_thread_memory_policy", return_value=None))
    return scenario_patches


def _numa_pages_of(tensor: torch.Tensor) -> dict[int, int]:
    """Pages per NUMA node that this process has mapped of the mapping holding ``tensor``."""
    address = tensor.data_ptr()
    with open("/proc/self/maps") as maps_file:
        mapping_starts = [int(line.split("-")[0], 16) for line in maps_file if _mapping_contains(line, address)]
    with open("/proc/self/numa_maps") as numa_maps_file:
        for line in numa_maps_file:
            fields = line.split()
            if int(fields[0], 16) in mapping_starts:
                node_page_matches = (re.fullmatch(r"N(\d+)=(\d+)", field) for field in fields)
                return {int(match[1]): int(match[2]) for match in node_page_matches if match}
    return {}


def _mapping_contains(maps_line: str, address: int) -> bool:
    start, end = (int(bound, 16) for bound in maps_line.split()[0].split("-"))
    return start <= address < end


def _park_and_probe_sharing_worker(
    rank: int, world_size: int, port: str, shared_dir: str, result_dir: str, scenario: str, sharing_scope: str
) -> None:
    from unittest.mock import patch

    from magi_compiler.passes.weight_offload import get_global_host_weight_store, install_host_first_hooks
    from magi_compiler.utils import numa

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=60))
    # Spread the ranks over the GPUs, so on a multi-socket machine they span NUMA nodes.
    device_index = rank * torch.cuda.device_count() // world_size
    torch.cuda.set_device(device_index)
    active_patches = _scenario_patches(scenario, shared_dir)
    policy_results = []
    replace_policy = numa._replace_thread_memory_policy

    def record_policy_result(*, policy, nodes):
        saved_policy = replace_policy(policy=policy, nodes=nodes)
        policy_results.append(saved_policy is not None)
        return saved_policy

    active_patches.append(patch.object(numa, "_replace_thread_memory_policy", side_effect=record_policy_result))
    for active_patch in active_patches:
        active_patch.start()
    try:
        model = _ReplicatedModelWithExpert(rank)
        expected_weight_by_name = {name: param.detach().clone() for name, param in model.named_parameters()}
        install_host_first_hooks(model, min_weight_bytes=0, host_weight_sharing=sharing_scope)
        model.cuda()

        store = get_global_host_weight_store()
        host_buffer_by_name = {
            name: store.host_buffer(store.slot_for_tensor(param)) for name, param in model.named_parameters()
        }
        result = {
            "gpu_numa_node": numa.cuda_device_numa_node(device_index),
            "dense_page_numa_nodes": sorted(_numa_pages_of(host_buffer_by_name["dense.weight"])),
            "matches_own_weights": {
                name: torch.equal(host_buffer, expected_weight_by_name[name])
                for name, host_buffer in host_buffer_by_name.items()
            },
            "is_pinned": {name: host_buffer.is_pinned() for name, host_buffer in host_buffer_by_name.items()},
            "leftover_files": sorted(os.listdir(shared_dir)),
        }

        policies_by_rank = [None] * world_size
        dist.all_gather_object(policies_by_rank, policy_results)
        result["policy_results_by_rank"] = policies_by_rank

        dist.barrier()
        if rank == 0:
            for host_buffer in host_buffer_by_name.values():
                host_buffer.fill_(_OVERWRITE_VALUE)
        dist.barrier()
        result["sees_rank0_writes"] = {
            name: bool((host_buffer == _OVERWRITE_VALUE).all()) for name, host_buffer in host_buffer_by_name.items()
        }
        with open(os.path.join(result_dir, f"{rank}.json"), "w") as result_file:
            json.dump(result, result_file)
    finally:
        for active_patch in active_patches:
            active_patch.stop()
        dist.destroy_process_group()


def _run_ranks(tmp_path, *, world_size: int, scenario: str, sharing_scope: str) -> list[dict]:
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    # cudaHostRegister refuses mappings of overlay-backed files, so the region must live on tmpfs.
    with tempfile.TemporaryDirectory(dir="/dev/shm") as shared_dir:
        mp.spawn(
            _park_and_probe_sharing_worker,
            args=(world_size, _free_port(), shared_dir, str(result_dir), scenario, sharing_scope),
            nprocs=world_size,
            join=True,
        )
    return [json.loads((result_dir / f"{rank}.json").read_text()) for rank in range(world_size)]


def _assert_every_rank_keeps_its_own_pinned_weights(results: list[dict]) -> None:
    for rank, result in enumerate(results):
        assert all(result["matches_own_weights"].values()), f"rank {rank}: {result}"
        assert all(result["is_pinned"].values()), f"rank {rank}: {result}"
        assert result["leftover_files"] == [], f"rank {rank}: {result}"


@requires_cuda_and_tmpfs
@pytest.mark.parametrize("scenario", ["default", "numa_policy_denied"])
def test_per_numa_node_shares_one_local_copy_per_node(tmp_path, scenario):
    world_size = 4 if torch.cuda.device_count() >= 4 else 2
    results = _run_ranks(tmp_path, world_size=world_size, scenario=scenario, sharing_scope="per_numa_node")

    _assert_every_rank_keeps_its_own_pinned_weights(results)
    rank0_numa_node = results[0]["gpu_numa_node"]
    for rank, result in enumerate(results[1:], start=1):
        shares_rank0_copy = result["gpu_numa_node"] == rank0_numa_node
        assert result["sees_rank0_writes"] == {"dense.weight": shares_rank0_copy, "expert.weight": False}, f"rank {rank}"
    for rank, result in enumerate(results):
        writer_rank = next(i for i, peer in enumerate(results) if peer["gpu_numa_node"] == result["gpu_numa_node"])
        writer_policies = result["policy_results_by_rank"][writer_rank]
        group_size = sum(peer["gpu_numa_node"] == result["gpu_numa_node"] for peer in results)
        if scenario == "numa_policy_denied" and result["gpu_numa_node"] is not None and group_size > 1:
            assert writer_policies == [False]
        if writer_policies == [True]:
            assert result["dense_page_numa_nodes"] == [result["gpu_numa_node"]], f"rank {rank}: {result}"


@requires_cuda_and_tmpfs
@pytest.mark.parametrize("scenario", ["default", "numa_policy_denied"])
def test_per_host_interleaved_shares_one_copy_spread_over_every_node(tmp_path, scenario):
    world_size = 4 if torch.cuda.device_count() >= 4 else 2
    results = _run_ranks(tmp_path, world_size=world_size, scenario=scenario, sharing_scope="per_host_interleaved")

    _assert_every_rank_keeps_its_own_pinned_weights(results)
    for rank, result in enumerate(results[1:], start=1):
        assert result["sees_rank0_writes"] == {"dense.weight": True, "expert.weight": False}, f"rank {rank}"
    if scenario == "numa_policy_denied" and len(online_numa_nodes()) > 1:
        assert results[0]["policy_results_by_rank"][0] == [False]
    if len(online_numa_nodes()) > 1 and results[0]["policy_results_by_rank"][0] == [True]:
        for rank, result in enumerate(results):
            assert result["dense_page_numa_nodes"] == online_numa_nodes(), f"rank {rank}: {result}"


@requires_cuda_and_tmpfs
def test_a_fingerprint_collision_never_hands_a_rank_another_ranks_weight(tmp_path):
    """The reader compares bytes, so a weight whose sampled fingerprint matches but whose bytes differ stays private."""
    results = _run_ranks(tmp_path, world_size=2, scenario="fingerprint_collision", sharing_scope="per_host_interleaved")

    _assert_every_rank_keeps_its_own_pinned_weights(results)
    assert results[1]["sees_rank0_writes"] == {"dense.weight": True, "expert.weight": False}


@requires_cuda_and_tmpfs
@pytest.mark.parametrize("scenario", ["no_room", "allocation_no_room"])
def test_a_host_without_shared_memory_room_keeps_every_weight_private(tmp_path, scenario):
    """Writing past a full tmpfs is SIGBUS, so a host without room shares nothing."""
    results = _run_ranks(tmp_path, world_size=2, scenario=scenario, sharing_scope="per_host_interleaved")

    _assert_every_rank_keeps_its_own_pinned_weights(results)
    assert results[1]["sees_rank0_writes"] == {"dense.weight": False, "expert.weight": False}


def _plan_input(
    sharing_group: str, fingerprint_by_weight_name: dict[str, bytes], free_bytes: int = 1 << 40, hostname: str | None = None
) -> _RankSharingPlanInput:
    return _RankSharingPlanInput(
        hostname=hostname or sharing_group,
        sharing_group=sharing_group,
        region_path=f"/dev/shm/{sharing_group}_{len(fingerprint_by_weight_name)}",
        free_shared_memory_bytes=free_bytes,
        descriptor_by_weight_name={
            name: _WeightDescriptor(shape=(10,), dtype=torch.float32, nbytes=40, fingerprint=fingerprint)
            for name, fingerprint in fingerprint_by_weight_name.items()
        },
    )


def test_each_host_plans_its_own_region_from_its_own_ranks():
    """Hosts decide alone: one without room shares nothing while the other shares what its ranks agree on."""
    plan_input_by_rank = [
        _plan_input("host_a", {"dense": b"d", "norm": b"n", "expert": b"e0"}),
        _plan_input("host_a", {"dense": b"d", "norm": b"n", "expert": b"e1"}),
        _plan_input("host_b", {"dense": b"d"}, free_bytes=0),
        _plan_input("host_b", {"dense": b"d"}),
    ]

    region_plan_by_sharing_group = _plan_sharing_group_regions(plan_input_by_rank)

    host_a_plan = region_plan_by_sharing_group["host_a"]
    assert host_a_plan.writer_rank == 0
    assert list(host_a_plan.slice_by_weight_name) == ["dense", "norm"]
    # Every weight starts on the private pool's alignment.
    assert [weight_slice.byte_offset for weight_slice in host_a_plan.slice_by_weight_name.values()] == [0, 512]
    assert host_a_plan.region_bytes == 1024

    host_b_plan = region_plan_by_sharing_group["host_b"]
    assert host_b_plan.writer_rank == 2
    assert host_b_plan.slice_by_weight_name == {} and host_b_plan.region_bytes == 0
    assert host_b_plan.rejected_region_bytes == 512


def test_numa_groups_of_one_host_must_fit_its_shared_memory_together():
    """Each node's region fits the host's tmpfs alone, but both do not, so neither node shares."""
    free_bytes = SHARED_MEMORY_HEADROOM_BYTES + 512 + 511
    plan_input_by_rank = [
        _plan_input("host/numa0", {"dense": b"d"}, free_bytes=free_bytes, hostname="host"),
        _plan_input("host/numa0", {"dense": b"d"}, free_bytes=free_bytes, hostname="host"),
        _plan_input("host/numa1", {"dense": b"d"}, free_bytes=free_bytes, hostname="host"),
        _plan_input("host/numa1", {"dense": b"d"}, free_bytes=free_bytes, hostname="host"),
    ]

    region_plan_by_sharing_group = _plan_sharing_group_regions(plan_input_by_rank)

    assert {group: region_plan.writer_rank for group, region_plan in region_plan_by_sharing_group.items()} == {
        "host/numa0": 0,
        "host/numa1": 2,
    }
    assert all(region_plan.region_bytes == 0 for region_plan in region_plan_by_sharing_group.values())
    assert all(region_plan.rejected_region_bytes == 512 for region_plan in region_plan_by_sharing_group.values())


def test_a_rank_alone_in_its_group_shares_nothing():
    region_plan_by_sharing_group = _plan_sharing_group_regions(
        [_plan_input("host_a", {"dense": b"d"}), _plan_input("host_b", {"dense": b"d"})]
    )

    assert all(region_plan.region_bytes == 0 for region_plan in region_plan_by_sharing_group.values())


def _asymmetric_parking_worker(rank: int, port: str, result_dir: str, scenario: str) -> None:
    from unittest.mock import patch

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Replicate

    from magi_compiler.passes.weight_offload import get_global_host_weight_store, install_host_first_hooks

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group("gloo", rank=rank, world_size=2, timeout=timedelta(seconds=30))
    torch.cuda.set_device(rank % torch.cuda.device_count())
    torch.manual_seed(0)
    model = nn.Linear(8, 8, bias=False)
    try:
        if scenario == "dtensor":
            mesh = init_device_mesh("cuda", (2,))
            if rank == 1:
                model.weight = nn.Parameter(DTensor.from_local(model.weight.cuda(), mesh, [Replicate()], run_check=False))
        elif rank == 1:
            if scenario == "already_cuda":
                model.cuda()
            else:
                model = nn.Identity()
        with patch("magi_compiler.utils.envs.MAGI_SHARED_BIN_PATH", result_dir):
            install_host_first_hooks(model, min_weight_bytes=0, host_weight_sharing="per_host_interleaved")
            model.cuda()
        store = get_global_host_weight_store()
        parked_weights = [store.slot_for_tensor(param) is not None for param in model.parameters()]
        with open(os.path.join(result_dir, f"{rank}.json"), "w") as result_file:
            json.dump(parked_weights, result_file)
    finally:
        get_global_host_weight_store().reset()
        dist.destroy_process_group()


@requires_cuda_and_tmpfs
@pytest.mark.parametrize("scenario", ["empty", "already_cuda", "dtensor"])
def test_ineligible_rank_still_joins_parking(tmp_path, scenario: str) -> None:
    """Ranks without eligible CPU parameters must join negotiation without parking DTensors or CUDA weights."""
    mp.spawn(_asymmetric_parking_worker, args=(_free_port(), str(tmp_path), scenario), nprocs=2, join=True)
    assert json.loads((tmp_path / "0.json").read_text()) == [True]
    assert not any(json.loads((tmp_path / "1.json").read_text()))


def _compiled_inference_worker(rank: int, port: str, shared_dir: str, result_dir: str, sharing_scope: str) -> None:
    from unittest.mock import patch

    from magi_compiler import magi_compile
    from magi_compiler.config import CompileConfig, CompileMode, CudaGraphMode
    from magi_compiler.passes.weight_offload import get_global_host_weight_store

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=2, timeout=timedelta(seconds=180))
    torch.cuda.set_device(rank % torch.cuda.device_count())
    # Compare CPU reference and CUDA compilation without TF32 rounding differences.
    torch.set_float32_matmul_precision("highest")
    store = get_global_host_weight_store()
    outputs = []
    try:
        with patch("magi_compiler.utils.envs.MAGI_SHARED_BIN_PATH", shared_dir), torch.no_grad():
            for mode in ("off", sharing_scope):
                torch.manual_seed(0)
                model = nn.Sequential(nn.Linear(64, 64, bias=False), nn.GELU(), nn.Linear(64, 64, bias=False))
                inputs = torch.randn(8, 64)
                expected_output = model(inputs).cuda()

                def configure(config: CompileConfig) -> CompileConfig:
                    config.compile_mode = CompileMode.MAGI_COMPILE
                    config.cudagraph_mode = CudaGraphMode.NONE
                    config.disable_graph_split = True
                    config.fsdp_config.cost_mode = "analytical"
                    config.offload_config.graph_weight_offload = True
                    config.offload_config.offload_min_shard_mib = 0
                    config.offload_config.offload_h2d_bandwidth_gbps = 20.0
                    config.offload_config.offload_host_weight_sharing = mode
                    return config

                compiled = magi_compile(model, config_patch=configure, dynamic_arg_dims={"input": 0})
                model.cuda()
                device_inputs = inputs.cuda()
                for _ in range(2):
                    actual_output = compiled(device_inputs)
                    torch.testing.assert_close(actual_output, expected_output, rtol=1e-4, atol=1e-5)
                assert store.offloaded_bytes() > 0
                outputs.append(actual_output.cpu())
                del actual_output, compiled, model
                torch._dynamo.reset()
                store.reset()
        for output in outputs[1:]:
            torch.testing.assert_close(output, outputs[0], rtol=0, atol=0)
        with open(os.path.join(result_dir, f"{rank}.txt"), "w") as result_file:
            result_file.write("shared and private inference match")
    finally:
        store.reset()
        dist.destroy_process_group()


@requires_cuda_and_tmpfs
@pytest.mark.parametrize("sharing_scope", ["per_numa_node", "per_host_interleaved"])
def test_public_compile_config_shared_inference_matches_private(tmp_path, sharing_scope: str) -> None:
    """Both sharing modes must preserve compiled H2D execution and match the private-copy output on every rank."""
    with tempfile.TemporaryDirectory(dir="/dev/shm") as shared_dir:
        mp.spawn(
            _compiled_inference_worker, args=(_free_port(), shared_dir, str(tmp_path), sharing_scope), nprocs=2, join=True
        )
    for rank in range(2):
        assert (tmp_path / f"{rank}.txt").read_text() == "shared and private inference match"
