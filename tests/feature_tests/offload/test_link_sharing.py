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

"""Transfers taking turns on one shared link: H2D loads, NCCL all-gathers and all-to-alls.

Graph nodes are real Inductor IR classes built with ``object.__new__``, so waits,
collectives and loads are classified by the production predicates.
"""

import tempfile

import pytest
import torch
from torch._inductor import ir

from magi_compiler.passes.fsdp_overlap.reorder import FsdpOverlapCallState, FsdpOverlapReorder
from magi_compiler.passes.transfer_overlap import (
    CostView,
    LinkSharing,
    SnodeGraph,
    TransferTask,
    free_time_axis,
    link_busy_spans_ns,
    schedule_transfers_alap,
)
from magi_compiler.passes.weight_offload import H2dLoadReorder, RouteCollectivesToLinkStream

BYTES_PER_MIB = 1 << 20
_ALL_GATHER_OP = torch.ops._c10d_functional.all_gather_into_tensor.default
_COALESCED_ALL_GATHER_OP = torch.ops._c10d_functional.all_gather_into_tensor_coalesced.default
_ALL_TO_ALL_OP = torch.ops._c10d_functional.all_to_all_single.default


class _Dep:
    def __init__(self, name):
        self.name = name


class _Compute:
    def get_size(self):
        return [1]

    def get_dtype(self):
        return torch.bfloat16


class FakeSnode:
    snodes = None

    def __init__(self, name, node, deps=(), runtime_ns=0.0):
        self.name = name
        self.node = node
        self.runtime_ns = runtime_ns
        self.unmet_dependencies = [_Dep(d) for d in deps]

    def get_name(self):
        return self.name

    def get_buffer_names(self):
        return [self.name]

    def __repr__(self):
        return f"<{self.name}>"


def _ir(cls, **fields):
    node = object.__new__(cls)
    defaults = {"op_overload": None, "origins": None, "constant_args": (), "get_size": lambda: [0]}
    for key, value in {**defaults, **fields}.items():
        object.__setattr__(node, key, value)
    object.__setattr__(node, "get_dtype", lambda: torch.bfloat16)
    return node


def _compute(name, runtime_ns, deps=()):
    return FakeSnode(name, _Compute(), deps, runtime_ns)


def _wait(name, dep):
    return FakeSnode(name, _ir(ir._WaitKernel), [dep])


def _gather(name, runtime_ns, op=_ALL_GATHER_OP):
    return FakeSnode(name, _ir(ir._CollectiveKernel, op_overload=op), (), runtime_ns)


def _all_to_all(name, runtime_ns, deps=()):
    return FakeSnode(name, _ir(ir._CollectiveKernel, op_overload=_ALL_TO_ALL_OP), deps, runtime_ns)


def _load(name, mib, slot=0):
    from magi_compiler.passes.weight_offload.runtime.h2d_op import H2D_LOAD

    numel = mib * BYTES_PER_MIB // 2
    return FakeSnode(name, _ir(ir.FallbackKernel, op_overload=H2D_LOAD, constant_args=(slot,), get_size=lambda: [numel]))


def _runtime_ns(snode):
    return snode.runtime_ns


def _place_loads(order, *, share_link_with_collectives):
    reorder = H2dLoadReorder(
        bandwidth_bytes_per_ns=10.0,
        window_margin_ns=0.0,
        max_inflight_bytes=64 * BYTES_PER_MIB,
        runtime_ns_fn=_runtime_ns,
        share_link_with_collectives=share_link_with_collectives,
    )
    return [s.name for s in reorder(order)]


# ------------------------------------------------ the time axis


def test_the_free_axis_cuts_out_busy_time_and_keeps_each_span_as_a_point():
    free_prefix_ns, blocked_windows_ns = free_time_axis([0.0, 100.0, 200.0, 400.0], [(50.0, 150.0), (300.0, 350.0)])

    assert free_prefix_ns == [0.0, 50.0, 100.0, 250.0]
    assert blocked_windows_ns == [(50.0, 50.0), (200.0, 200.0)]


def test_a_collective_occupies_the_link_only_until_compute_stalls_on_its_wait():
    """An all-to-all whose result is needed 0.4 ms after launch holds 0.4 ms of compute;
    the rest of it runs while compute waits."""
    order = [
        _compute("c0", 1e6),
        _all_to_all("a2a", 1e6),
        _compute("independent", 0.4e6),
        _wait("w", "a2a"),
        _compute("attn", 1e6, deps=["w"]),
        _gather("ag", 0.5e6),
        _compute("c1", 2e6),
        _wait("wg", "ag"),
    ]
    graph, cost = SnodeGraph(order), CostView(_runtime_ns)

    busy_spans = link_busy_spans_ns(graph, cost, cost.compute_prefix_ns(order))

    assert busy_spans == [(1e6, 1.4e6), (2.4e6, 2.9e6)]
    assert link_busy_spans_ns(graph, cost, cost.compute_prefix_ns(order), excluded_snodes={order[5]}) == [(1e6, 1.4e6)]


def _task(name, *, deadline_index, required_overlap_ns, earliest_launch_index=0):
    return TransferTask(
        anchor=FakeSnode(name, _Compute()),
        moving_nodes=(),
        earliest_launch_index=earliest_launch_index,
        deadline_index=deadline_index,
        required_overlap_ns=required_overlap_ns,
        last_user_index=deadline_index,
    )


def test_a_transfer_finishes_before_a_blocked_window_rather_than_straddling_it():
    task = _task("ld", deadline_index=4, required_overlap_ns=100.0)
    prefix_ns = [0.0, 700.0, 850.0, 960.0, 1000.0]
    original_index_by_node = {task.anchor: 3}

    unblocked = schedule_transfers_alap([task], prefix_ns, original_index_by_node, float("inf"))
    blocked = schedule_transfers_alap(
        [task], prefix_ns, original_index_by_node, float("inf"), blocked_windows_ns=[(920.0, 920.0)]
    )

    assert unblocked.launch_index_by_anchor[task.anchor] == 2, "ALAP start 900 sits after the 850 boundary"
    assert blocked.launch_index_by_anchor[task.anchor] == 1, "ending by 920 means starting by 820"
    assert blocked.total_unhidden_ns == 0


def test_a_transfer_released_inside_a_blocked_window_starts_after_it():
    task = _task("ld", deadline_index=4, required_overlap_ns=100.0, earliest_launch_index=2)
    prefix_ns = [0.0, 700.0, 920.0, 960.0, 1000.0]

    result = schedule_transfers_alap([task], prefix_ns, {task.anchor: 3}, float("inf"), blocked_windows_ns=[(900.0, 950.0)])

    assert result.unhidden_ns_by_anchor[task.anchor] == pytest.approx(50.0), "starts at 950, done at 1050"


# ------------------------------------------------ H2D loads around collectives


def _load_behind_a_gather_with_little_free_time():
    """The gather (1 ms) runs inside the 1.2 ms the load (10 MiB, ~1.05 ms at 10 GB/s) would
    otherwise hide behind; only 0.2 ms of it is left free."""
    return [
        _compute("c0", 5e6),
        _compute("c1", 1e6),
        _gather("ag", 1e6),
        _compute("c2", 1.2e6),
        _wait("wg", "ag"),
        _compute("use_g", 1e5, deps=["wg"]),
        _load("ld", 10),
        _wait("w", "ld"),
        _compute("user", 1e6, deps=["w"]),
    ]


def test_without_link_sharing_the_load_counts_the_gathers_time_as_its_own():
    names = _place_loads(_load_behind_a_gather_with_little_free_time(), share_link_with_collectives=False)

    assert names.index("ag") < names.index("ld") < names.index("c2")


def test_a_load_moves_to_compute_no_gather_occupies():
    names = _place_loads(_load_behind_a_gather_with_little_free_time(), share_link_with_collectives=True)

    assert names.index("ld") < names.index("c1"), "done before the gather starts, not straddling it"


def test_a_load_behind_a_gather_at_the_same_boundary_is_issued_after_it():
    """One long kernel after a gather: the load can only launch at that kernel's boundary,
    so it must be issued after the gather and queue behind it on the link stream."""
    order = [
        _compute("c0", 1e5),
        _gather("ag", 1e6),
        _compute("long_kernel", 10e6),
        _wait("wg", "ag"),
        _load("ld", 10),
        _wait("w", "ld"),
        _compute("user", 1e6, deps=["w", "wg"]),
    ]

    names = _place_loads(order, share_link_with_collectives=True)

    assert names.index("ag") < names.index("ld") < names.index("long_kernel")


def _load_across_an_all_to_all():
    """ALAP, the load (~1.05 ms) would run 5.3-6.35 ms, across the all-to-all compute stalls on at 6 ms."""
    return [
        _compute("c0", 5e6),
        _compute("c1", 0.3e6),
        _compute("c1b", 0.7e6),
        _all_to_all("a2a", 1e6, deps=["c1b"]),
        _wait("w_a2a", "a2a"),
        _compute("attn", 0.5e6, deps=["w_a2a"]),
        _load("ld", 10),
        _wait("w", "ld"),
        _compute("user", 1e6, deps=["w"]),
    ]


def test_without_link_sharing_a_load_runs_across_an_all_to_all():
    names = _place_loads(_load_across_an_all_to_all(), share_link_with_collectives=False)

    assert names.index("c1") < names.index("ld") < names.index("c1b")


def test_a_load_finishes_before_the_all_to_all_it_shares_the_link_with():
    names = _place_loads(_load_across_an_all_to_all(), share_link_with_collectives=True)

    assert names.index("ld") < names.index("c1"), "ends by 6 ms, so it starts by 4.95 ms"


# ------------------------------------------------ FSDP gathers around all-to-alls


def _place_gather_across_an_all_to_all(link_sharing: LinkSharing) -> int:
    """The gather's ALAP slot (start 5.5 ms for a 6.5 ms deadline) straddles the all-to-all at 6 ms."""
    order = [
        _compute("c0", 5e6),
        _compute("c1", 0.3e6),
        _compute("c1b", 0.7e6),
        _all_to_all("a2a", 1e6, deps=["c1b"]),
        _wait("w_a2a", "a2a"),
        _compute("attn", 0.5e6, deps=["w_a2a"]),
        _gather("ag", 1e6),
        _wait("wg", "ag"),
        _compute("use", 1e6, deps=["wg"]),
    ]
    gather = order[6]
    task = TransferTask(
        anchor=gather, moving_nodes=(gather,), earliest_launch_index=0, deadline_index=7, required_overlap_ns=1e6
    )
    reorder = FsdpOverlapReorder(comm_overlap_window_margin_ns=0.0, runtime_ns_fn=_runtime_ns, link_sharing=link_sharing)
    graph = SnodeGraph(order)
    launch_index_by_anchor = reorder.assign_launch_indices(graph, CostView(_runtime_ns), [task], FsdpOverlapCallState())
    return launch_index_by_anchor[gather]


def test_without_link_sharing_a_gather_runs_across_an_all_to_all():
    assert _place_gather_across_an_all_to_all(LinkSharing.OFF) == 2, "launched before c1b, at 5.3 ms"


@pytest.mark.parametrize("link_sharing", [LinkSharing.PCIE, LinkSharing.NVLINK])
def test_a_gather_finishes_before_the_all_to_all_it_shares_the_link_with(link_sharing):
    assert _place_gather_across_an_all_to_all(link_sharing) == 1, "ends by 6 ms, so it launches before c1"


# ------------------------------------------------ routing and the link-stream ops


@pytest.mark.parametrize("link_sharing", [LinkSharing.PCIE, LinkSharing.NVLINK])
def test_routing_rewrites_only_the_emitted_kernel_name_of_link_collectives(link_sharing):
    single = _gather("ag", 1e6)
    coalesced = _gather("agc", 1e6, op=_COALESCED_ALL_GATHER_OP)
    all_to_all = _all_to_all("a2a", 1e6)
    order = [single, _compute("c", 1e6), coalesced, all_to_all]

    assert RouteCollectivesToLinkStream(link_sharing)(order) == order
    link = link_sharing.value
    assert single.node.python_kernel_name == f"torch.ops.magi.link_all_gather.{link}"
    assert coalesced.node.python_kernel_name == f"torch.ops.magi.link_all_gather_coalesced.{link}"
    assert all_to_all.node.python_kernel_name == f"torch.ops.magi.link_all_to_all_single.{link}"
    assert isinstance(single.node, ir._CollectiveKernel), "every pass still sees a functional collective"


_LINK_SLEEP_CYCLES = 50_000_000
"""Arbitrary; tens of milliseconds, far above a one-rank collective."""


def _time_link_sleep_ms(link_stream) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(link_stream):
        start.record(link_stream)
        torch.cuda._sleep(_LINK_SLEEP_CYCLES)
        end.record(link_stream)
    end.synchronize()
    return start.elapsed_time(end)


@pytest.fixture(scope="module")
def nccl_world_of_one():
    import torch.distributed as dist

    if dist.is_initialized():
        pytest.skip("a process group is already initialized in this process")
    store = dist.FileStore(tempfile.mktemp(prefix="link_collective_"), 1)
    dist.init_process_group("nccl", store=store, rank=0, world_size=1, device_id=torch.device("cuda", 0))
    yield dist.group.WORLD.group_name
    dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NCCL collectives need CUDA")
@pytest.mark.parametrize("link_sharing", [LinkSharing.PCIE, LinkSharing.NVLINK])
def test_link_collectives_match_the_functional_ones_and_queue_behind_the_link(nccl_world_of_one, link_sharing):
    from magi_compiler.passes.weight_offload.runtime.h2d_op import get_h2d_stream
    from magi_compiler.passes.weight_offload.runtime.link_collective_op import get_nvlink_stream

    link = link_sharing.value
    link_all_gather = getattr(torch.ops.magi.link_all_gather, link)
    link_all_gather_coalesced = getattr(torch.ops.magi.link_all_gather_coalesced, link)
    link_all_to_all = getattr(torch.ops.magi.link_all_to_all_single, link)
    wait_tensor = torch.ops._c10d_functional.wait_tensor.default
    shard = torch.randn(64, 32, device="cuda")
    other_shard = torch.randn(16, 8, device="cuda")
    link_stream = get_h2d_stream() if link_sharing is LinkSharing.PCIE else get_nvlink_stream()
    sleep_ms = _time_link_sleep_ms(link_stream)

    compute_start = torch.cuda.Event(enable_timing=True)
    compute_start.record()
    with torch.cuda.stream(link_stream):
        torch.cuda._sleep(_LINK_SLEEP_CYCLES)  # stands in for a transfer already queued on the link
    exchanged = wait_tensor(link_all_to_all(shard, [64], [64], nccl_world_of_one))
    compute_released = torch.cuda.Event(enable_timing=True)
    compute_released.record()
    gathered = wait_tensor(link_all_gather(shard, 1, nccl_world_of_one))
    coalesced = [wait_tensor(t) for t in link_all_gather_coalesced([shard, other_shard], 1, nccl_world_of_one)]
    torch.cuda.synchronize()

    assert compute_start.elapsed_time(compute_released) >= 0.8 * sleep_ms, "the all-to-all queued behind the link"
    torch.testing.assert_close(exchanged, shard)
    torch.testing.assert_close(gathered, shard)
    torch.testing.assert_close(coalesced[0], shard)
    torch.testing.assert_close(coalesced[1], other_shard)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="link streams need CUDA")
def test_collectives_sharing_nvlink_do_not_queue_behind_loads():
    from magi_compiler.passes.weight_offload.runtime.h2d_op import get_h2d_stream
    from magi_compiler.passes.weight_offload.runtime.link_collective_op import get_nvlink_stream

    assert get_nvlink_stream() != get_h2d_stream()
