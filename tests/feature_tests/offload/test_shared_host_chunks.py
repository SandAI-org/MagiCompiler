# Copyright (c) 2026 SandAI. All Rights Reserved.

import ctypes
import errno
import json
import os
import socket
from contextlib import contextmanager
from datetime import timedelta
from multiprocessing.synchronize import Lock
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

_WEIGHT_BYTES = 4 << 20


def _chunk_peak_worker(
    rank: int,
    port: str,
    result_dir: str,
    weight_counts: tuple[int, ...],
    remaining_bytes: ctypes.Array,
    allocated_bytes: ctypes.c_longlong,
    peak_extra: ctypes.c_longlong,
    lock: Lock,
    failure_mode: str | None = None,
) -> None:
    from magi_compiler.passes.weight_offload import weight_lifecycle
    from magi_compiler.passes.weight_offload.runtime import rank_shared_host_region as region_module
    from magi_compiler.passes.weight_offload.weight_lifecycle import _park_with_shared_plan

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group('gloo', rank=rank, world_size=4, timeout=timedelta(seconds=30))
    torch.manual_seed(0)
    model = nn.ParameterList([nn.Parameter(torch.randn(1024, 1024)) for _ in range(weight_counts[rank])])
    parameters = [([name], param) for name, param in model.named_parameters()]
    host_buffers = []
    allocation_sizes = []
    allocate_region = region_module._create_shared_memory_file
    initial_bytes = sum(weight_counts) * _WEIGHT_BYTES
    previous_threads = torch.get_num_threads()
    compare_thread_counts = []
    compare_weights = region_module.RankSharedHostRegion.holds_identical_copy

    def compare_region(region, weight_name: str, tensor: torch.Tensor) -> bool:
        compare_thread_counts.append(torch.get_num_threads())
        return compare_weights(region, weight_name, tensor)

    @contextmanager
    def allocate_chunk(**kwargs):
        if failure_mode == 'enospc' and rank == 0 and allocation_sizes:
            raise OSError(errno.ENOSPC, 'injected second-chunk failure')
        with allocate_region(**kwargs) as descriptor:
            with lock:
                allocated_bytes.value += kwargs['nbytes']
                peak_extra.value = max(peak_extra.value, sum(remaining_bytes) + allocated_bytes.value - initial_bytes)
            allocation_sizes.append(kwargs['nbytes'])
            yield descriptor

    def install_standin(*, names: list[str], param: nn.Parameter, host_buffer: torch.Tensor) -> int:
        torch.testing.assert_close(host_buffer, param.detach(), rtol=0, atol=0)
        original_bytes = param.numel() * param.element_size()
        param.data = torch.empty(0)
        host_buffers.append(host_buffer)
        with lock:
            remaining_bytes[rank] -= original_bytes
        return len(host_buffers) - 1

    def park_private(*, names: list[str], param: nn.Parameter) -> int:
        host_buffer = param.detach().clone()
        with lock:
            allocated_bytes.value += param.numel() * param.element_size()
        return install_standin(names=names, param=param, host_buffer=host_buffer)

    def available_memory() -> int:
        if failure_mode == 'headroom' and allocated_bytes.value:
            return region_module.SHARED_MEMORY_HEADROOM_BYTES
        return 1 << 40

    def map_region(*, path: str, nbytes: int) -> torch.Tensor:
        return torch.from_file(path, shared=True, size=nbytes, dtype=torch.uint8)

    try:
        with (
            patch.object(region_module, 'SHARED_MEMORY_CHUNK_BYTES', _WEIGHT_BYTES),
            patch.object(region_module.RankSharedHostRegion, 'holds_identical_copy', compare_region),
            patch.object(weight_lifecycle.magi_logger, 'info') as comparison_log,
            patch.object(region_module, 'get_available_host_memory_bytes', side_effect=available_memory),
            patch.object(region_module.shutil, 'disk_usage', return_value=SimpleNamespace(free=1 << 40)),
            patch.object(
                region_module, '_local_sharing_group_and_region_numa_policy', return_value=(f'host/numa{rank // 2}', None)
            ),
            patch.object(region_module, '_create_shared_memory_file', side_effect=allocate_chunk),
            patch.object(region_module, '_map_and_pin_shared_memory_file', side_effect=map_region),
            patch('magi_compiler.passes.weight_offload.weight_lifecycle._install_weight_standin', side_effect=install_standin),
            patch(
                'magi_compiler.passes.weight_offload.weight_lifecycle._park_in_private_host_buffer', side_effect=park_private
            ),
        ):
            parking_plan = region_module.open_rank_shared_host_region(
                tensor_by_weight_name={names[0]: param.detach() for names, param in parameters},
                region_label='peak_test',
                sharing_scope='per_numa_node',
            )
            assert parking_plan is not None
            assert allocation_sizes == [], 'planning must not allocate shared pages'
            slots = _park_with_shared_plan(cpu_parameters=parameters, parking_plan=parking_plan)
            assert len(slots) == weight_counts[rank]
            expected_shared_bytes = weight_counts[rank] * _WEIGHT_BYTES
            if failure_mode is not None:
                expected_shared_bytes = _WEIGHT_BYTES if rank < 2 else 0
            assert parking_plan.shared_bytes == expected_shared_bytes
            _assert_reader_comparison_metrics(
                thread_counts=compare_thread_counts,
                previous_threads=previous_threads,
                expected_shared_bytes=expected_shared_bytes,
                comparison_log=comparison_log,
            )
            Path(result_dir, f'{rank}.json').write_text(json.dumps(allocation_sizes))
    finally:
        dist.destroy_process_group()


def _assert_reader_comparison_metrics(
    *, thread_counts: list[int], previous_threads: int, expected_shared_bytes: int, comparison_log: Mock
) -> None:
    assert torch.get_num_threads() == previous_threads
    if not thread_counts:
        assert comparison_log.call_count == 0
        return
    expected_threads = min(previous_threads, max(1, len(os.sched_getaffinity(0)) // 4))
    assert set(thread_counts) == {expected_threads}
    assert comparison_log.call_count == 1
    metrics = comparison_log.call_args.kwargs
    assert metrics['rank'] == 'all'
    assert metrics['extra']['compared_bytes'] == expected_shared_bytes
    assert metrics['extra']['compare_seconds'] >= 0
    assert metrics['extra']['compare_threads'] == expected_threads


@pytest.mark.parametrize(
    'weight_counts, failure_mode',
    [((6, 6, 6, 6), None), ((6, 6, 3, 3), None), ((6, 6, 6, 6), 'enospc'), ((6, 6, 6, 6), 'headroom')],
)
def test_multiple_groups_release_originals_before_allocating_next_chunk(
    tmp_path: Path, weight_counts: tuple[int, ...], failure_mode: str | None
) -> None:
    """The live pageable-plus-shared weight peak must grow by at most one chunk across all groups."""
    context = mp.get_context('spawn')
    remaining_bytes = context.RawArray('q', [count * _WEIGHT_BYTES for count in weight_counts])
    allocated_bytes = context.Value('q', 0, lock=False)
    peak_extra = context.Value('q', 0, lock=False)
    lock = context.Lock()
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = str(probe.getsockname()[1])
    mp.spawn(
        _chunk_peak_worker,
        args=(port, str(tmp_path), weight_counts, remaining_bytes, allocated_bytes, peak_extra, lock, failure_mode),
        nprocs=4,
        join=True,
    )
    assert list(remaining_bytes) == [0] * 4
    if failure_mode is None:
        assert peak_extra.value <= _WEIGHT_BYTES
    allocation_sizes = [size for rank in range(4) for size in json.loads((tmp_path / f'{rank}.json').read_text())]
    expected_chunks = 1 if failure_mode is not None else weight_counts[0] + weight_counts[2]
    assert allocation_sizes == [_WEIGHT_BYTES] * expected_chunks


def test_oversized_weight_stays_whole_in_its_own_chunk(monkeypatch) -> None:
    from magi_compiler.passes.weight_offload.runtime import rank_shared_host_region as region_module

    monkeypatch.setattr(region_module, 'SHARED_MEMORY_CHUNK_BYTES', 512)
    plan = region_module._SharingGroupRegionPlan(
        writer_rank=0,
        hostname='host',
        region_bytes=2560,
        slice_by_weight_name={
            name: region_module._RegionWeightSlice(byte_offset=0, shape=(size,), dtype=torch.uint8, nbytes=size)
            for name, size in [('small_a', 512), ('large', 1536), ('small_b', 512)]
        },
    )
    chunks = list(region_module._split_region_plan(plan))
    assert [chunk.region_bytes for chunk in chunks] == [512, 1536, 512]
    assert [tuple(chunk.slice_by_weight_name) for chunk in chunks] == [('small_a',), ('large',), ('small_b',)]
    assert all(weight.byte_offset == 0 for chunk in chunks for weight in chunk.slice_by_weight_name.values())
