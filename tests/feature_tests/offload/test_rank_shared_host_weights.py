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
from contextlib import ExitStack
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from magi_compiler.passes.weight_offload.runtime import rank_shared_host_region as region_module
from magi_compiler.passes.weight_offload.runtime.rank_shared_host_region import (
    _plan_sharing_groups,
    _RankPlanInput,
    _WeightDescriptor,
)
from magi_compiler.utils.numa import online_numa_nodes

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

_OVERWRITE_VALUE = 7.0


def _free_port() -> str:
    with socket.socket() as probe_socket:
        probe_socket.bind(("127.0.0.1", 0))
        return str(probe_socket.getsockname()[1])


def _init_process_group(*, rank: int, world_size: int, port: str, backend: str = "gloo", timeout_seconds: int = 60) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group(backend, rank=rank, world_size=world_size, timeout=timedelta(seconds=timeout_seconds))


def _failure_worker(rank: int, port: str, result_dir: str, failure_stage: str) -> None:
    from unittest.mock import patch

    from magi_compiler.passes.weight_offload import install_host_first_hooks, weight_lifecycle

    _init_process_group(rank=rank, world_size=2, port=port, timeout_seconds=30)
    torch.cuda.set_device(rank % torch.cuda.device_count())
    failing_rank = 1 if failure_stage in ("plan", "reader_map", "reader_handoff") else 0
    module, attribute = {
        "plan": (region_module, "compute_tensor_fingerprint"),
        "create": (region_module, "_create_shared_memory_file"),
        "map": (region_module, "_map_and_pin_shared_memory_file"),
        "reader_map": (region_module, "_map_and_pin_shared_memory_file"),
        "fill": (weight_lifecycle, "_install_weight_standin"),
        "reader_handoff": (weight_lifecycle, "_install_weight_standin"),
    }[failure_stage]
    original = getattr(module, attribute)

    def fail_on_one_rank(*args, **kwargs):
        if rank == failing_rank:
            raise RuntimeError(f"injected {failure_stage} failure")
        return original(*args, **kwargs)

    torch.manual_seed(0)
    model = nn.Linear(256, 256, bias=False)
    try:
        with patch.object(module, attribute, fail_on_one_rank):
            install_host_first_hooks(model, min_weight_bytes=0, host_weight_sharing="per_host_interleaved")
            with pytest.raises(RuntimeError) as error:
                model.cuda()
        with open(os.path.join(result_dir, f"{rank}.txt"), "w") as result_file:
            result_file.write(str(error.value))
    finally:
        dist.destroy_process_group()


@requires_cuda
@pytest.mark.parametrize("failure_stage", ["plan", "create", "map", "fill", "reader_map", "reader_handoff"])
def test_a_single_rank_failure_reaches_every_rank(tmp_path, failure_stage: str) -> None:
    """Planning, writer and reader failures on one rank must reach every rank without a barrier timeout."""
    mp.spawn(_failure_worker, args=(_free_port(), str(tmp_path), failure_stage), nprocs=2, join=True)
    failing_rank = 1 if failure_stage in ("plan", "reader_map", "reader_handoff") else 0
    for rank in range(2):
        error_message = (tmp_path / f"{rank}.txt").read_text()
        assert f"rank {failing_rank}: RuntimeError: injected {failure_stage} failure" in error_message


class _ReplicatedModelWithExpert(nn.Module):
    """``dense`` is identical on every rank, ``expert`` differs per rank (like an EP shard)."""

    def __init__(self, rank: int):
        super().__init__()
        torch.manual_seed(0)
        self.dense = nn.Linear(256, 256, bias=False)
        torch.manual_seed(1 + rank)
        self.expert = nn.Linear(256, 256, bias=False)


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
    rank: int, world_size: int, port: str, result_dir: str, scenario: str, sharing_scope: str
) -> None:
    from unittest.mock import patch

    from magi_compiler.passes.weight_offload import get_global_host_weight_store, install_host_first_hooks
    from magi_compiler.utils import numa

    _init_process_group(rank=rank, world_size=world_size, port=port)
    # Spread the ranks over the GPUs, so on a multi-socket machine they span NUMA nodes.
    device_index = rank * torch.cuda.device_count() // world_size
    torch.cuda.set_device(device_index)
    policy_results = []
    replace_policy = numa._replace_thread_memory_policy

    def record_policy_result(*, policy, nodes):
        saved_policy = None if scenario == "numa_policy_denied" else replace_policy(policy=policy, nodes=nodes)
        policy_results.append(saved_policy is not None)
        return saved_policy

    scenario_patches = [patch.object(numa, "_replace_thread_memory_policy", side_effect=record_policy_result)]
    if scenario == "fingerprint_collision":
        scenario_patches.append(patch.object(region_module, "compute_tensor_fingerprint", lambda tensor: b"collision"))
    elif scenario == "allocation_no_room":
        enospc = OSError(errno.ENOSPC, "injected allocation failure")
        scenario_patches.append(patch.object(region_module.os, "posix_fallocate", side_effect=enospc))
    try:
        with ExitStack() as patches:
            for scenario_patch in scenario_patches:
                patches.enter_context(scenario_patch)
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
        dist.destroy_process_group()


def _run_ranks(tmp_path, *, world_size: int, scenario: str, sharing_scope: str) -> list[dict]:
    mp.spawn(
        _park_and_probe_sharing_worker,
        args=(world_size, _free_port(), str(tmp_path), scenario, sharing_scope),
        nprocs=world_size,
        join=True,
    )
    results = [json.loads((tmp_path / f"{rank}.json").read_text()) for rank in range(world_size)]
    for rank, result in enumerate(results):
        assert all(result["matches_own_weights"].values()), f"rank {rank}: {result}"
        assert all(result["is_pinned"].values()), f"rank {rank}: {result}"
    return results


@requires_cuda
@pytest.mark.parametrize("scenario", ["default", "numa_policy_denied"])
def test_per_numa_node_shares_one_local_copy_per_node(tmp_path, scenario):
    world_size = 4 if torch.cuda.device_count() >= 4 else 2
    results = _run_ranks(tmp_path, world_size=world_size, scenario=scenario, sharing_scope="per_numa_node")

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


@requires_cuda
@pytest.mark.parametrize("scenario", ["default", "numa_policy_denied"])
def test_per_host_interleaved_shares_one_copy_spread_over_every_node(tmp_path, scenario):
    world_size = 4 if torch.cuda.device_count() >= 4 else 2
    results = _run_ranks(tmp_path, world_size=world_size, scenario=scenario, sharing_scope="per_host_interleaved")

    for rank, result in enumerate(results[1:], start=1):
        assert result["sees_rank0_writes"] == {"dense.weight": True, "expert.weight": False}, f"rank {rank}"
    if scenario == "numa_policy_denied" and len(online_numa_nodes()) > 1:
        assert results[0]["policy_results_by_rank"][0] == [False]
    if len(online_numa_nodes()) > 1 and results[0]["policy_results_by_rank"][0] == [True]:
        for rank, result in enumerate(results):
            assert result["dense_page_numa_nodes"] == online_numa_nodes(), f"rank {rank}: {result}"


@requires_cuda
def test_a_fingerprint_collision_never_hands_a_rank_another_ranks_weight(tmp_path):
    """The reader compares bytes, so a weight whose sampled fingerprint matches but whose bytes differ stays private."""
    results = _run_ranks(tmp_path, world_size=2, scenario="fingerprint_collision", sharing_scope="per_host_interleaved")

    assert results[1]["sees_rank0_writes"] == {"dense.weight": True, "expert.weight": False}


@requires_cuda
def test_allocation_enospc_keeps_every_weight_private(tmp_path):
    results = _run_ranks(tmp_path, world_size=2, scenario="allocation_no_room", sharing_scope="per_host_interleaved")

    assert results[1]["sees_rank0_writes"] == {"dense.weight": False, "expert.weight": False}


def _plan_input(sharing_group: str, nbytes_by_weight_name: dict[str, int], expert_fingerprint: bytes = b"") -> _RankPlanInput:
    return _RankPlanInput(
        hostname=sharing_group,
        sharing_group=sharing_group,
        descriptor_by_weight_name={
            name: _WeightDescriptor(
                shape=(nbytes,),
                dtype=torch.uint8,
                nbytes=nbytes,
                fingerprint=name.encode() + (expert_fingerprint if name == "expert" else b""),
            )
            for name, nbytes in nbytes_by_weight_name.items()
        },
    )


def test_each_group_shares_only_what_all_its_ranks_hold_identically():
    weights = {"dense": 40, "norm": 40, "expert": 40}
    group_plan_by_sharing_group = _plan_sharing_groups(
        [
            _plan_input("host_a", weights, expert_fingerprint=b"0"),
            _plan_input("host_a", weights, expert_fingerprint=b"1"),
            _plan_input("host_b", weights),
            _plan_input("host_c", weights),
        ]
    )

    assert group_plan_by_sharing_group["host_a"].ranks == (0, 1)
    (chunk,) = group_plan_by_sharing_group["host_a"].chunks
    # Every weight starts on the private pool's alignment.
    assert {name: weight_slice.byte_offset for name, weight_slice in chunk.slice_by_weight_name.items()} == {
        "dense": 0,
        "norm": 512,
    }
    assert chunk.nbytes == 1024
    # A rank alone in its group shares nothing.
    assert group_plan_by_sharing_group["host_b"].chunks == () and group_plan_by_sharing_group["host_c"].chunks == ()


def test_chunks_split_at_the_chunk_size_and_larger_weights_stand_alone(monkeypatch):
    monkeypatch.setattr(region_module, "SHARED_MEMORY_CHUNK_BYTES", 1024)
    plan_input = _plan_input("host", {"small_a": 512, "small_b": 512, "large": 1536, "small_c": 512})

    (group_plan,) = _plan_sharing_groups([plan_input, plan_input]).values()

    assert [chunk.nbytes for chunk in group_plan.chunks] == [1024, 1536, 512]
    assert [
        {name: weight_slice.byte_offset for name, weight_slice in chunk.slice_by_weight_name.items()}
        for chunk in group_plan.chunks
    ] == [{"small_a": 0, "small_b": 512}, {"large": 0}, {"small_c": 0}]


_WEIGHT_BYTES = 4 << 20
_GROUP_COUNT = 2


def _chunk_peak_worker(rank: int, port: str, result_dir: str, weight_counts: tuple[int, ...], shared_state) -> None:
    from unittest.mock import patch

    from magi_compiler.passes.weight_offload import weight_lifecycle

    remaining_bytes, allocated_bytes, peak_extra_bytes, lock = shared_state
    _init_process_group(rank=rank, world_size=len(weight_counts), port=port, timeout_seconds=30)
    torch.manual_seed(0)
    model = nn.ParameterList([nn.Parameter(torch.randn(1024, 1024)) for _ in range(weight_counts[rank])])
    parameters = [([name], param) for name, param in model.named_parameters()]
    initial_bytes = sum(weight_counts) * _WEIGHT_BYTES
    allocation_sizes = []
    create_region = region_module._create_shared_memory_file
    previous_threads = torch.get_num_threads()

    def create_and_count(**kwargs) -> int:
        file_descriptor = create_region(**kwargs)
        with lock:
            allocated_bytes.value += kwargs["nbytes"]
            peak_extra_bytes.value = max(peak_extra_bytes.value, sum(remaining_bytes) + allocated_bytes.value - initial_bytes)
        allocation_sizes.append(kwargs["nbytes"])
        return file_descriptor

    def release_original(*, names: list[str], param: nn.Parameter, host_buffer: torch.Tensor) -> int:
        torch.testing.assert_close(host_buffer, param.detach(), rtol=0, atol=0)
        with lock:
            remaining_bytes[rank] -= param.numel() * param.element_size()
        param.data = torch.empty(0)
        return 0

    sharing_group = f"host/numa{rank * _GROUP_COUNT // len(weight_counts)}"
    try:
        with (
            patch.object(region_module, "SHARED_MEMORY_CHUNK_BYTES", _WEIGHT_BYTES),
            patch.object(region_module, "_sharing_group_and_numa_policy", return_value=(sharing_group, None)),
            patch.object(region_module, "_create_shared_memory_file", side_effect=create_and_count),
            patch.object(
                region_module,
                "_map_and_pin_shared_memory_file",
                side_effect=lambda path, nbytes: torch.from_file(path, shared=True, size=nbytes, dtype=torch.uint8),
            ),
            patch.object(weight_lifecycle, "_install_weight_standin", side_effect=release_original),
        ):
            parking_plan = region_module.plan_shared_host_parking(
                tensor_by_weight_name={names[0]: param.detach() for names, param in parameters},
                region_label="peak_test",
                sharing_scope="per_numa_node",
            )
            assert allocation_sizes == [], "planning must not allocate shared pages"
            _, shared_bytes = weight_lifecycle._park_parameters(cpu_parameters=parameters, parking_plan=parking_plan)
        assert shared_bytes == weight_counts[rank] * _WEIGHT_BYTES
        assert torch.get_num_threads() == previous_threads
        with open(os.path.join(result_dir, f"{rank}.json"), "w") as result_file:
            json.dump(allocation_sizes, result_file)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("weight_counts", [(6, 6, 6, 6), (6, 6, 3, 3)])
def test_groups_park_concurrently_within_one_chunk_each(tmp_path, weight_counts: tuple[int, ...]) -> None:
    """Each group allocates its next chunk only after its readers released the last one's originals."""
    context = mp.get_context("spawn")
    shared_state = (
        context.RawArray("q", [count * _WEIGHT_BYTES for count in weight_counts]),
        context.Value("q", 0, lock=False),
        context.Value("q", 0, lock=False),
        context.Lock(),
    )
    mp.spawn(_chunk_peak_worker, args=(_free_port(), str(tmp_path), weight_counts, shared_state), nprocs=4, join=True)

    remaining_bytes, _, peak_extra_bytes, _ = shared_state
    assert list(remaining_bytes) == [0] * 4
    assert peak_extra_bytes.value <= _GROUP_COUNT * _WEIGHT_BYTES
    allocation_sizes = [size for rank in range(4) for size in json.loads((tmp_path / f"{rank}.json").read_text())]
    assert allocation_sizes == [_WEIGHT_BYTES] * (weight_counts[0] + weight_counts[2])


def _asymmetric_parking_worker(rank: int, port: str, result_dir: str, scenario: str) -> None:
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Replicate

    from magi_compiler.passes.weight_offload import get_global_host_weight_store, install_host_first_hooks

    _init_process_group(rank=rank, world_size=2, port=port, timeout_seconds=30)
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
        install_host_first_hooks(model, min_weight_bytes=0, host_weight_sharing="per_host_interleaved")
        model.cuda()
        store = get_global_host_weight_store()
        parked_weights = [store.slot_for_tensor(param) is not None for param in model.parameters()]
        with open(os.path.join(result_dir, f"{rank}.json"), "w") as result_file:
            json.dump(parked_weights, result_file)
    finally:
        get_global_host_weight_store().reset()
        dist.destroy_process_group()


@requires_cuda
@pytest.mark.parametrize("scenario", ["empty", "already_cuda", "dtensor"])
def test_ineligible_rank_still_joins_parking(tmp_path, scenario: str) -> None:
    """Ranks without eligible CPU parameters must join negotiation without parking DTensors or CUDA weights."""
    mp.spawn(_asymmetric_parking_worker, args=(_free_port(), str(tmp_path), scenario), nprocs=2, join=True)
    assert json.loads((tmp_path / "0.json").read_text()) == [True]
    assert not any(json.loads((tmp_path / "1.json").read_text()))


def _compiled_inference_worker(rank: int, port: str, result_dir: str, sharing_scope: str) -> None:
    from magi_compiler import magi_compile
    from magi_compiler.config import CompileConfig, CompileMode, CudaGraphMode
    from magi_compiler.passes.weight_offload import get_global_host_weight_store

    _init_process_group(rank=rank, world_size=2, port=port, backend="cpu:gloo,cuda:nccl", timeout_seconds=180)
    torch.cuda.set_device(rank % torch.cuda.device_count())
    # Compare CPU reference and CUDA compilation without TF32 rounding differences.
    torch.set_float32_matmul_precision("highest")
    store = get_global_host_weight_store()
    outputs = []
    try:
        with torch.no_grad():
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


@requires_cuda
@pytest.mark.parametrize("sharing_scope", ["per_numa_node", "per_host_interleaved"])
def test_public_compile_config_shared_inference_matches_private(tmp_path, sharing_scope: str) -> None:
    """Both sharing modes must preserve compiled H2D execution and match the private-copy output on every rank."""
    mp.spawn(_compiled_inference_worker, args=(_free_port(), str(tmp_path), sharing_scope), nprocs=2, join=True)
    for rank in range(2):
        assert (tmp_path / f"{rank}.txt").read_text() == "shared and private inference match"
