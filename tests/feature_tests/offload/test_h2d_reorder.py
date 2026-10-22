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

"""``H2dLoadReorder`` placement, on synthetic snode lists.

Phase 1 leaves every ``h2d_load`` directly in front of the wait that guards it,
which is correct and fully exposed; this pass is the one that opens a compute
window in between.  What is worth testing here is the arithmetic of that window
and the in-flight budget that bounds it -- both are pure functions of the snode
list, so the list is synthetic and only the wait predicate is stubbed (Inductor
decides it by ``isinstance``, which no synthetic node can satisfy).

The cost model is deliberately NOT stubbed: a load's window comes from
``bytes / bandwidth``, and getting that unit wrong by 1e9 is the failure mode
this suite exists to catch.
"""

import pytest
import torch
from torch._inductor import ir

from magi_compiler.passes.overlap import CostView, SnodeGraph, memory, scheduler
from magi_compiler.passes.weight_offload.schedule.h2d_reorder import H2dLoadReorder, LoadPlan

_MIB = 1 << 20
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="promotion moves real bytes")


@pytest.fixture
def clean_host_weights():
    from magi_compiler.passes.weight_offload import get_global_host_weight_store

    get_global_host_weight_store().reset()
    try:
        yield
    finally:
        get_global_host_weight_store().reset()


def _park(tensor: torch.Tensor, name: str = "") -> int:
    """Put ``tensor`` in the host pool the production way: reserve, fill, empty, adopt."""
    from magi_compiler.passes.weight_offload import get_global_host_weight_store

    host = get_global_host_weight_store().reserve(tuple(tensor.shape), tensor.dtype, name=name)
    host.copy_(tensor.detach())
    tensor.untyped_storage().resize_(0)
    return get_global_host_weight_store().adopt(host, tensor, name=name)


class _Dep:
    def __init__(self, name):
        self.name = name


def _make_ir(op_overload, numel, dtype=torch.bfloat16, slot=None):
    node = object.__new__(ir.FallbackKernel)
    node.op_overload = op_overload
    node.layout = ir.FixedLayout(device=torch.device("cpu"), dtype=dtype, size=[numel])
    # Inductor flattens a custom op's non-tensor args here; for the load op
    # that is exactly the host-pool slot.
    node.constant_args = () if slot is None else (slot,)
    return node


class _Snode:
    """A stand-in with just the surface the pass touches."""

    snodes = None

    def __init__(self, name, kind, *, deps=(), cost=0.0, numel=0, slot=None):
        from magi_compiler.passes.weight_offload.runtime.h2d_op import H2D_LOAD

        self.name = name
        self.kind = kind
        self.cost = cost
        self.unmet_dependencies = [_Dep(d) for d in deps]
        if kind == "unpack":
            # Scheduling only needs the layout; construction normally registers with V.graph.
            self.node = object.__new__(ir.MultiOutput)
            self.node.layout = ir.FixedLayout(device=torch.device("cpu"), dtype=torch.bfloat16, size=[numel])
        else:
            self.node = _make_ir(H2D_LOAD if kind == "load" else "fake.op", numel, slot=slot)

    def get_name(self):
        return self.name

    def get_buffer_names(self):
        return [self.name]

    def __repr__(self):
        return f"<{self.kind} {self.name}>"


@pytest.fixture(autouse=True)
def stub_wait_predicate(monkeypatch):
    """Inductor decides ``contains_wait`` by isinstance against ``ir._WaitKernel``,
    which a synthetic node cannot be."""
    import magi_compiler.passes.overlap.graph as overlap_graph
    from magi_compiler.passes import snode_utils

    is_wait = lambda s: getattr(s, "kind", None) == "wait"  # noqa: E731
    monkeypatch.setattr(overlap_graph, "contains_wait", is_wait)
    monkeypatch.setattr(snode_utils, "contains_wait", is_wait)


def _compute(name, cost_ns, deps=()):
    return _Snode(name, "compute", cost=cost_ns, deps=deps)


def _load(name, mib, deps=(), slot=0):
    return _Snode(name, "load", deps=deps, numel=mib * _MIB // 2, slot=slot)  # bf16: 2 bytes/elem


def _wait(name, load_name):
    return _Snode(name, "wait", deps=[load_name])


def _unpack(name, load_name, mib):
    """The unpack carries the output's layout -- that is where the pass reads the
    transfer size from, since a load lowers to a FallbackKernel whose own size
    describes no tensor."""
    return _Snode(name, "unpack", deps=[load_name], numel=mib * _MIB // 2)


def _reorder(
    order,
    *,
    bandwidth=10.0,
    margin=0.0,
    max_resident_bytes=0,
    max_inflight_bytes=0,
    max_device_weight_bytes=0,
    bus_utilization=0.9,
):
    """Run the pass; bandwidth is bytes/ns, i.e. GB/s.

    Residency defaults to nothing, and the in-flight budget to the pass's own
    default of one load on top of what the unhoisted order already needs.  Tests
    that want a sharper bound pass the bytes: with buckets this small, the
    difference between "one load of slack" and "none" is the whole behaviour.
    """
    p = H2dLoadReorder(
        bandwidth_bytes_per_ns=bandwidth,
        window_margin_ns=margin,
        max_resident_bytes=max_resident_bytes,
        max_inflight_bytes=max_inflight_bytes,
        max_device_weight_bytes=max_device_weight_bytes,
        bus_utilization=bus_utilization,
        cost_fn=lambda s: s.cost,
    )
    return [s.name for s in p(order)]


def test_cost_query_failure_leaves_schedule_unchanged():
    """An estimator bug must stop scheduling instead of turning compute into zero-cost work."""
    order = [_compute("c", 2e6), _load("ld", 10), _wait("w", "ld"), _compute("user", 1e6, deps=["w"])]
    before = list(order)

    def failed_cost(snode):
        raise RuntimeError("cost query failed")

    reorder = H2dLoadReorder(bandwidth_bytes_per_ns=10.0, cost_fn=failed_cost)
    with pytest.raises(RuntimeError, match="cost query failed"):
        reorder(order)
    assert order == before


@pytest.mark.parametrize(
    "compute_ns,expected",
    [
        pytest.param([2e6, 2e6], ["c0", "ld", "c1", "w", "user"], id="nearest-kernel-covers-transfer"),
        pytest.param([2e5] * 3, ["ld", "c0", "c1", "c2", "w", "user"], id="all-compute-still-insufficient"),
    ],
)
def test_load_hoists_only_as_far_as_the_transfer_needs(compute_ns, expected):
    """10 MiB at 10 GB/s needs about 1.05 ms of preceding compute."""
    order = [_compute(f"c{i}", cost) for i, cost in enumerate(compute_ns)]
    order += [_load("ld", 10), _wait("w", "ld"), _compute("user", 1e6, deps=["w"])]
    assert _reorder(order) == expected


def test_two_loads_do_not_spend_the_same_compute():
    """Each transfer fits a 1 ms window alone, but their serial sum does not."""
    order = [
        _compute("c0", 1e6),
        _load("ld0", 6),
        _wait("w0", "ld0"),
        _load("ld1", 6),
        _wait("w1", "ld1"),
        _compute("user", 1e6, deps=["w0", "w1"]),
    ]
    graph = SnodeGraph(order)
    p = H2dLoadReorder(bandwidth_bytes_per_ns=10.0, window_margin_ns=0.0, max_inflight_bytes=12 * _MIB)
    plans = p._plan(graph)
    prefix = CostView(lambda s: s.cost).prefix(order)
    p._schedule(plans, order, graph.index_of, prefix)

    assert len(plans) == 2
    assert [plan.need for plan in plans] == pytest.approx([629145.6, 629145.6])
    assert plans[0].exposed == 0
    assert plans[1].exposed == pytest.approx(258291.2), "the shared bus cannot hide 1.258 ms in 1 ms"
    assert not any(plan.budget_bound for plan in plans), "both buffers fit; the shortage is on the bus"

    out = _reorder(order, max_inflight_bytes=12 * _MIB)
    assert out == ["ld0", "ld1", "c0", "w0", "w1", "user"]


def test_load_never_crosses_its_own_producer():
    """The real data dep is the floor; a load cannot outrun the shard it reads."""
    order = [
        _compute("c0", 5e6),
        _compute("shard_prep", 1e3),
        _load("ld", 10, deps=["shard_prep"]),
        _wait("w", "ld"),
        _compute("user", 1e6, deps=["w"]),
    ]
    out = _reorder(order)
    assert out.index("shard_prep") < out.index("ld")


def _two_conflicting_loads(slots=(0, 0)):
    """The topology one load of in-flight budget cannot hoist both of: both loads
    are still read at ``user``, so the earlier one's bytes are live where the
    later one lands."""
    return [
        _compute("c0", 5e6),
        _compute("c1", 5e6),
        _compute("c2", 5e6),
        _load("ld0", 8, slot=slots[0]),
        _wait("w0", "ld0"),
        _load("ld1", 8, slot=slots[1]),
        _wait("w1", "ld1"),
        _compute("user", 1e6, deps=["w0", "w1"]),
    ]


def _two_loads_one_window():
    """Two loads whose only worthwhile compute is the same three kernels.

    Unhoisted, their live ranges are disjoint -- each dies at its own consumer --
    so the memory floor here is one load and hoisting the second one is a real
    purchase against the in-flight budget rather than something phase 1 already
    paid for.
    """
    return [
        _compute("c0", 5e6),
        _compute("c1", 5e6),
        _compute("c2", 5e6),
        _load("ld0", 8),
        _wait("w0", "ld0"),
        _compute("u0", 1e3, deps=["w0"]),
        _load("ld1", 8),
        _wait("w1", "ld1"),
        _compute("u1", 1e3, deps=["w1"]),
    ]


def _serialized_loads(slots, *, mib=10):
    """One compute window and several loads that all want it.

    Each load's bytes die at its own consumer, so live ranges never collide and
    the in-flight budget is not what binds -- the bus is.  With only the first
    kernel worth hiding behind, the loads have to take turns, which makes this the
    topology where what the schedule is short of is transfer rather than memory.
    """
    order = [_compute("c0", 1e6)]
    for i, slot in enumerate(slots):
        order += [_load(f"ld{i}", mib, slot=slot), _wait(f"w{i}", f"ld{i}"), _compute(f"u{i}", 1e3, deps=[f"w{i}"])]
    return order


@pytest.mark.parametrize(
    "configured_mib,expected_mib",
    [
        pytest.param(0, 24, id="default-adds-one-load"),
        pytest.param(8, 16, id="below-unhoisted-floor"),
        pytest.param(20, 20, id="explicit-budget"),
    ],
)
def test_inflight_budget_preserves_the_unhoisted_floor(configured_mib, expected_mib):
    """Two 8 MiB weights share their last user, so even unhoisted they need 16 MiB.

    The default adds one largest load; an explicit budget is floored at 16 MiB.
    All three budgets let both transfers use the nearest sufficient kernel.
    """
    order = _two_conflicting_loads()
    p = H2dLoadReorder(bandwidth_bytes_per_ns=10.0, max_inflight_bytes=configured_mib * _MIB)
    assert p._inflight_budget(p._plan(SnodeGraph(order))) == expected_mib * _MIB

    out = _reorder(order, max_inflight_bytes=configured_mib * _MIB)
    assert out.index("ld0") < out.index("w0") and out.index("ld1") < out.index("w1")
    assert out.index("ld1") < out.index("w0"), "the original order already needs both buffers together"
    assert out.index("c1") < out.index("ld0") < out.index("ld1") < out.index("c2")


@pytest.mark.parametrize(
    "budget_mib,can_overlap",
    [
        pytest.param(8, False, id="one-buffer"),
        pytest.param(15, False, id="just-too-small"),
        pytest.param(16, True, id="two-buffers"),
    ],
)
def test_inflight_bytes_determine_whether_loads_share_a_window(budget_mib, can_overlap):
    """Two 8 MiB buffers can share a compute window only when both fit."""
    out = _reorder(_two_loads_one_window(), max_inflight_bytes=budget_mib * _MIB)
    assert out.index("ld0") < out.index("w0") and out.index("ld1") < out.index("w1")
    if can_overlap:
        assert out.index("ld1") < out.index("w0")
        assert out.index("c1") < out.index("ld0") < out.index("ld1") < out.index("c2")
    else:
        assert out.index("ld1") + 1 == out.index("w1"), "the second load must remain against its wait"


@requires_cuda
@pytest.mark.usefixtures("clean_host_weights")
@pytest.mark.parametrize(
    "resident_mib,expected_resident", [pytest.param(0, False, id="disabled"), pytest.param(64, True, id="enabled")]
)
def test_residency_budget_applies_even_to_fully_hideable_loads(resident_mib, expected_resident):
    """Zero budget keeps offload enabled; ample budget removes even hidden PCIe traffic."""
    from magi_compiler.passes.weight_offload import get_global_host_weight_store

    shards = [torch.randn(8 * _MIB // 2, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    slots = [_park(s) for s in shards]
    _reorder(_two_conflicting_loads(slots), max_resident_bytes=resident_mib * _MIB)
    assert [get_global_host_weight_store().is_resident(slot) for slot in slots] == [expected_resident] * 2
    assert get_global_host_weight_store().resident_bytes() == (16 * _MIB if expected_resident else 0)


@requires_cuda
@pytest.mark.usefixtures("clean_host_weights")
def test_the_residency_budget_buys_out_the_transfer_nothing_can_hide():
    """When two loads exceed one compute window, residency must remove exposed H2D work.
    Without residency, the later load stays at its wait when hoisting cannot help."""
    from magi_compiler.passes.weight_offload import get_global_host_weight_store

    def run(resident):
        # Use fresh tensors because _park releases their device storage.
        get_global_host_weight_store().reset()
        shards = [torch.randn(10 * _MIB // 2, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        slots = [_park(s) for s in shards]
        out = _reorder(_serialized_loads(slots), max_resident_bytes=resident, max_inflight_bytes=20 * _MIB)
        return out, [get_global_host_weight_store().is_resident(s) for s in slots]

    out, resident = run(0)
    assert not any(resident), "an empty budget buys nothing"
    assert out.index("ld0") < out.index("c0"), "the earlier deadline claims the only window"
    assert out.index("ld1") + 1 == out.index("w1"), "the one that loses it is not hoisted for nothing"

    out, resident = run(10 * _MIB)
    assert resident == [True, False], "the budget takes one of the two off the bus"


@requires_cuda
@pytest.mark.usefixtures("clean_host_weights")
def test_the_residency_budget_is_a_ceiling_not_a_target():
    """Three 10 MiB weights compete for a 10 MiB budget: restore exactly one."""
    from magi_compiler.passes.weight_offload import get_global_host_weight_store

    shards = [torch.randn(10 * _MIB // 2, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    slots = [_park(s) for s in shards]
    _reorder(_serialized_loads(slots), max_resident_bytes=10 * _MIB, max_inflight_bytes=40 * _MIB)
    assert get_global_host_weight_store().resident_bytes() == 10 * _MIB


def _layered_graph(slots, *, layers=6, big_mib=8, small_mib=4, compute_ns=1e6):
    """Repeated blocks with one compute kernel and two loads each.
    At the defaults, 12 MiB at 10 GB/s exceeds 1 ms of compute and requires residency."""
    order = []
    for i in range(layers):
        order.append(_compute(f"c{i}", compute_ns))
        for tag, mib, slot in (("big", big_mib, slots[2 * i]), ("small", small_mib, slots[2 * i + 1])):
            order.append(_load(f"l{i}_{tag}", mib, slot=slot))
            order.append(_wait(f"w{i}_{tag}", f"l{i}_{tag}"))
            order.append(_compute(f"u{i}_{tag}", 0.0, deps=[f"w{i}_{tag}"]))
    return order


@requires_cuda
@pytest.mark.usefixtures("clean_host_weights")
def test_residency_closes_each_windows_own_deficit():
    """A 4 MiB promotion must cover a 3.4 MiB window deficit.
    Promoting 8 MiB instead would exhaust the budget needed by later windows."""
    from magi_compiler.passes.weight_offload import get_global_host_weight_store

    slots, sizes = [], [8, 4] * 6
    for mib in sizes:
        slots.append(_park(torch.randn(mib * _MIB // 2, device="cuda", dtype=torch.bfloat16)))
    _reorder(_layered_graph(slots), max_device_weight_bytes=(24 + 16) * _MIB)

    resident = {i for i, s in enumerate(slots) if get_global_host_weight_store().is_resident(s)}
    assert resident == {1, 3, 5, 7, 9, 11}, "every block gives back its own smaller weight"
    assert get_global_host_weight_store().resident_bytes() == 24 * _MIB


@requires_cuda
@pytest.mark.usefixtures("clean_host_weights")
def test_residency_is_spread_across_the_graph_not_stacked_at_the_front():
    """Distribute residency across overloaded windows rather than exhausting it at the front."""
    from magi_compiler.passes.weight_offload import get_global_host_weight_store

    slots = []
    for mib in [8, 4] * 6:
        slots.append(_park(torch.randn(mib * _MIB // 2, device="cuda", dtype=torch.bfloat16)))
    _reorder(_layered_graph(slots), max_device_weight_bytes=64 * _MIB)

    blocks_touched = {i // 2 for i, s in enumerate(slots) if get_global_host_weight_store().is_resident(s)}
    assert len(blocks_touched) >= 5, f"residency only reached blocks {sorted(blocks_touched)}"


@requires_cuda
@pytest.mark.usefixtures("clean_host_weights")
def test_device_weight_budget_reserves_inflight_room_before_residency():
    """One budget, split by the pass: residency never eats the in-flight floor."""
    from magi_compiler.passes.weight_offload import get_global_host_weight_store

    slots = []
    for mib in [8, 4] * 6:
        slots.append(_park(torch.randn(mib * _MIB // 2, device="cuda", dtype=torch.bfloat16)))
    _reorder(_layered_graph(slots), max_device_weight_bytes=30 * _MIB)

    # 2 x the largest bucket is reserved for in-flight before residency is
    # handed anything, so a 30 MiB budget can buy at most 14 MiB back.
    assert 0 < get_global_host_weight_store().resident_bytes() <= 14 * _MIB


def test_last_user_is_the_consumer_not_the_wait():
    """The wait is only the floor; last_user is whoever still reads the bytes."""
    order = [_compute("c0", 5e6), _load("ld", 4), _wait("w", "ld"), _compute("matmul", 1e6, deps=["w"])]
    graph = SnodeGraph(order)
    plans = H2dLoadReorder(bandwidth_bytes_per_ns=10.0, window_margin_ns=0.0, cost_fn=lambda s: s.cost)._plan(graph)
    assert len(plans) == 1
    assert plans[0].deadline == graph.index_of[order[2]]
    assert plans[0].last_user == graph.index_of[order[3]]


def test_load_without_a_wait_is_left_alone():
    """Moving a transfer away from a synchronization we cannot see is how you get a
    race that only shows up once the hoist is long enough."""
    order = [_compute("c0", 5e6), _load("orphan", 10), _compute("user", 1e6)]
    assert _reorder(order) == ["c0", "orphan", "user"]


def test_bandwidth_sets_the_window_size():
    """A faster bus needs a smaller transfer window, so the same load stops
    at a nearer kernel.  This is the check that a GB/s-vs-bytes/ns slip fails."""

    def build():
        return [
            _compute("c0", 1e5),
            _compute("c1", 1e5),
            _compute("c2", 1e5),
            _load("ld", 4),
            _wait("w", "ld"),
            _compute("user", 1e6, deps=["w"]),
        ]

    slow = _reorder(build(), bandwidth=1.0)  # 4 MiB / 1 GB/s ~ 4.2ms: nothing covers it
    fast = _reorder(build(), bandwidth=100.0)  # ~42us: one 0.1ms kernel is plenty
    assert slow.index("ld") < fast.index("ld")
    assert fast == ["c0", "c1", "ld", "c2", "w", "user"]


def test_graph_without_loads_is_returned_untouched():
    order = [_compute("c0", 1e6), _compute("c1", 1e6)]
    assert _reorder(order) == ["c0", "c1"]


def test_wait_is_found_through_the_multioutput_unpack():
    """This is the shape a real compile produces.

    A custom op's result reaches its consumers through a ``MultiOutput`` unpack,
    so the wait is two hops from the load.  Looking only at direct readers finds
    the unpack, decides it is not a wait, and drops the load from the plan --
    which is not an error, just an overlap that never happens.
    """
    order = [
        _compute("c0", 2e6),
        _compute("c1", 2e6),
        _load("ld", 10),
        _unpack("mo", "ld", 10),
        _wait("w", "mo"),
        _compute("user", 1e6, deps=["w"]),
    ]
    out = _reorder(order)
    assert out.index("ld") < out.index("c1"), "the load must be hoisted, not skipped"
    # The unpack reads only the load's buffer, so it travels with it and stays adjacent.
    assert out.index("mo") == out.index("ld") + 1


def test_live_ranges_never_overlap_however_tight_the_chain():
    """A one-buffer budget keeps all four closed live ranges disjoint.

    No residency is requested, so synthetic loads need no host slots or CUDA.
    Computes do not read the waits here; each live range ends at its wait.
    """
    order = [_compute("c0", 9e6)]
    for i in range(4):
        order += [_load(f"ld{i}", 4), _wait(f"w{i}", f"ld{i}"), _compute(f"m{i}", 3e6)]
    out = _reorder(order, max_inflight_bytes=4 * _MIB)

    live = sorted((out.index(f"ld{i}"), out.index(f"w{i}")) for i in range(4))
    assert all(start < end for start, end in live)
    for (s0, e0), (s1, e1) in zip(live, live[1:]):
        assert e0 < s1, f"two buffers live at once: [{s0},{e0}] and [{s1},{e1}]"


def test_the_budget_charges_for_the_last_user_not_the_last_wait():
    """Charge a load through its last reader, even after its wait.
    The second load may share that interval only when both buffers fit the budget."""

    def build():
        return [
            _compute("c0", 5e6),
            _compute("c1", 5e6),
            _load("ld0", 8),
            _wait("w0", "ld0"),
            _compute("mid", 5e6),
            _compute("late0", 1e3, deps=["w0"]),
            _load("ld1", 8),
            _wait("w1", "ld1"),
            _compute("u1", 1e3, deps=["w1"]),
        ]

    tight = _reorder(build(), max_inflight_bytes=8 * _MIB)
    assert tight.index("ld1") + 1 == tight.index("w1"), "late0 still reads ld0 where ld1 wants to start"

    wide = _reorder(build(), max_inflight_bytes=16 * _MIB)
    assert wide.index("ld1") < wide.index("mid"), "given room for both, ld1 hides behind mid"


@requires_cuda
@pytest.mark.usefixtures("clean_host_weights")
def test_a_lone_unhideable_load_is_bought_out_only_when_asked():
    """A load with no preceding compute becomes resident only when given a residency budget."""
    from magi_compiler.passes.weight_offload import get_global_host_weight_store

    def run(resident):
        get_global_host_weight_store().reset()
        shard = torch.randn(2 * _MIB // 2, device="cuda", dtype=torch.bfloat16)
        slot = _park(shard)
        order = [_load("ld", 2, slot=slot), _wait("w", "ld"), _compute("user", 1e6, deps=["w"])]
        _reorder(order, max_resident_bytes=resident)
        return get_global_host_weight_store().is_resident(slot), shard

    resident, shard = run(0)
    assert not resident and shard.untyped_storage().nbytes() == 0, "it should still be offloaded"
    resident, _ = run(2 * _MIB)
    assert resident, "given the budget, the transfer stops happening"


def _equal_boundary_order():
    """``ld0.last_user`` lands exactly on where ld1 wants to start."""
    return [
        _compute("c0", 5e6),
        _load("ld0", 4),
        _wait("w0", "ld0"),
        _compute("user0", 1e6, deps=["w0"]),
        _load("ld1", 4),
        _wait("w1", "ld1"),
        _compute("user1", 1e6, deps=["w1"]),
    ]


@pytest.mark.parametrize(
    "budget_mib,expected",
    [
        pytest.param(4, ["ld0", "c0", "w0", "user0", "ld1", "w1", "user1"], id="one-buffer"),
        pytest.param(8, ["ld0", "c0", "w0", "ld1", "user0", "w1", "user1"], id="two-buffers"),
    ],
)
def test_equal_boundary_overlap_requires_room_for_both_buffers(budget_mib, expected):
    """A load inserted before the previous buffer's last user makes both live.

    The shared boundary is legal when the budget pays for both 4 MiB buffers.
    """
    assert _reorder(_equal_boundary_order(), max_inflight_bytes=budget_mib * _MIB) == expected


def test_adjacent_closed_ranges_both_stay_offloaded():
    """[at, last_user] and [last_user+1, ...] touch but do not overlap."""
    order = [
        _compute("c0", 5e6),
        _load("ld0", 4),
        _wait("w0", "ld0"),
        _compute("user0", 1e6, deps=["w0"]),
        _compute("mid", 1e6),
        _load("ld1", 4),
        _wait("w1", "ld1"),
        _compute("user1", 1e6, deps=["w1"]),
    ]
    out = _reorder(order)
    assert out.index("ld0") < out.index("c0"), "ld0 is still hoisted"
    assert out.index("ld1") < out.index("mid"), "ld1 is still hoisted"
    assert out.index("user0") < out.index("ld1"), "closed ranges may touch, not overlap"


def _plan_stub(name, *, deadline, last_user, nbytes, need=0.0, lower=0, inactive=False):
    """A ``LoadPlan`` carrying only the fields the memory and pricing helpers read."""
    return LoadPlan(
        anchor=_Snode(name, "load"),
        group=[],
        slot=0,
        deadline=deadline,
        last_user=last_user,
        need=need,
        lower=lower,
        nbytes=nbytes,
        exposed=need,
        inactive=inactive,
    )


def test_latest_finish_chains_the_deadlines_backwards():
    """A transfer may not end after its own wait, nor after the next one starts."""
    prefix = [0.0, 100.0, 300.0, 600.0, 1000.0]
    early = _plan_stub("early", deadline=2, last_user=4, nbytes=_MIB, need=100.0)
    late = _plan_stub("late", deadline=4, last_user=5, nbytes=_MIB, need=250.0)

    limits = scheduler.latest_finish([early, late], prefix)
    assert limits[late.anchor] == 1000.0, "the last one is held only by its own wait"
    assert limits[early.anchor] == 300.0, "its own wait is tighter than 1000 - 250"

    # Grow the later transfer until it is what binds the earlier one.
    late.need = 800.0
    limits = scheduler.latest_finish([early, late], prefix)
    assert limits[early.anchor] == 200.0, "it now has to be done before `late` starts"


def test_issue_index_is_just_in_time():
    """The DMA starts at ``t_bus`` whatever index we pick, so pick the latest."""
    prefix = [0.0, 100.0, 300.0, 600.0, 1000.0]

    assert scheduler.issue_index(prefix, 300.0, 0, 4, -1) == 2
    assert scheduler.issue_index(prefix, 599.0, 0, 4, -1) == 2, "600 is later than the slot"
    assert scheduler.issue_index(prefix, 0.0, 0, 4, -1) == 0

    assert scheduler.issue_index(prefix, 1000.0, 0, 3, -1) == 2, "never past its own wait"
    assert scheduler.issue_index(prefix, 0.0, 2, 4, -1) == 2, "never before its producers"
    assert scheduler.issue_index(prefix, 0.0, 0, 4, 3) == 3, "never before the load ahead of it"


def test_peak_counts_a_promoted_weight_twice_where_its_copy_runs():
    """Promotion is not a memory saving: the load still allocates an output.

    ``h2d_load`` copies even a resident slot into a fresh buffer, so a promoted
    weight costs its bytes for the whole graph AND again where its load runs.
    """
    kept = _plan_stub("kept", deadline=4, last_user=3, nbytes=4 * _MIB)
    resident = _plan_stub("resident", deadline=11, last_user=11, nbytes=8 * _MIB, inactive=True)
    plans = [kept, resident]
    index_of = {kept.anchor: 0, resident.anchor: 10}

    assert memory.device_peak(plans, {kept.anchor: 0}, index_of) == 8 * _MIB + 8 * _MIB

    resident.last_user = 3  # its buffer now overlaps the kept load's
    index_of[resident.anchor] = 1
    assert memory.device_peak(plans, {kept.anchor: 0}, index_of) == 8 * _MIB + 12 * _MIB


def test_inflight_occupancy_finds_the_earliest_fitting_start():
    """Find the earliest launch whose closed live range fits the remaining byte budget."""
    live = memory.InflightMap(budget=10)
    live.add(4, 8, 6)  # 6 bytes live over the closed range [4, 8]

    assert live.earliest_start(12, 4) == 0, "4 fits alongside 6 in a 10-byte budget"
    assert live.earliest_start(12, 5) == 9, "5 does not, so it has to start past the range"
    assert live.earliest_start(3, 5) == 0, "a range ending before the occupied one never meets it"
    assert live.earliest_start(4, 5) == 5, "closed ranges touching at one index is an overlap"

    # The sweep books every load unhoisted and releases one only to place it, so
    # removal has to undo an add exactly -- a leaked delta silently shrinks the
    # budget for every load after it.
    live.remove(4, 8, 6)
    assert live.earliest_start(12, 10) == 0


def test_inflight_peak_counts_closed_interval_touch():
    """Same-index start/end is overlap under a closed interval; peak is the sum."""

    class _Fake:
        def __init__(self, last_user, nbytes):
            self.anchor = object()
            self.last_user = last_user
            self.nbytes = nbytes

    earlier = _Fake(last_user=4, nbytes=100)
    later = _Fake(last_user=7, nbytes=50)
    peak = memory.inflight_peak([earlier, later], {earlier.anchor: 0, later.anchor: 4})
    assert peak == 150

    later_after = _Fake(last_user=7, nbytes=50)
    peak_adjacent = memory.inflight_peak([earlier, later_after], {earlier.anchor: 0, later_after.anchor: 5})
    assert peak_adjacent == 100


def test_a_load_window_is_sized_from_its_unpack():
    """The load itself carries no size (a FallbackKernel); reading it there would
    size every window as zero and leave every load where phase 1 parked it."""

    def build(mib):
        order = [_compute(f"c{i}", 3e5) for i in range(6)]
        order.append(_load("ld", 0))
        order.append(_unpack("mo", "ld", mib))
        order.append(_wait("w", "mo"))
        order.append(_compute("user", 1e6, deps=["w"]))
        return order

    small = _reorder(build(1), bandwidth=10.0)
    big = _reorder(build(4), bandwidth=10.0)
    assert big.index("ld") < small.index("ld") < small.index("c5"), "a bigger load needs a longer window"
    assert big.index("mo") == big.index("ld") + 1


def test_a_late_small_load_is_not_dragged_to_the_front_of_the_graph():
    """A late small transfer must stay near its deadline after earlier large loads.

    Moving it to the front would hold its buffer across unrelated compute."""
    order = [_compute("c0", 6e5)]
    for i in range(3):
        order += [_load(f"big{i}", 4), _wait(f"wb{i}", f"big{i}"), _compute(f"ub{i}", 6e5, deps=[f"wb{i}"])]
    order += [_load("small", 1), _wait("ws", "small"), _compute("us", 1e3, deps=["ws"])]

    out = _reorder(order, max_inflight_bytes=64 * _MIB)

    assert out.index("small") > out.index("wb2"), "it belongs to the block its deadline is in"
    assert out.index("small") < out.index("ub2"), "one boundary earlier, so the DMA still starts in time"
    # The big ones pipeline one block ahead of their own waits, which is the
    # whole prefetch depth this graph has room for.
    assert [n for n in out if n.startswith(("big", "small"))] == ["big0", "big1", "big2", "small"]


def test_large_transfer_keeps_the_window_under_inflight_pressure():
    """The early large transfer uses the long kernel; the late small one stays local.

    Hoisting the small load across the large one needlessly extends its lifetime."""
    # 20 MiB at 10 GB/s ~ 2.1ms; 2 MiB ~ 0.21ms.  One 5ms kernel hides either.
    order = [
        _compute("moe", 5e6),
        _load("big", 20),
        _wait("wb", "big"),
        _compute("after_big", 1e3, deps=["wb"]),
        _compute("gap", 1e6),
        _load("small", 2),
        _wait("ws", "small"),
        _compute("after_small", 1e3, deps=["ws"]),
    ]
    # 22 MiB budget: both can be live together, so memory is not the alibi.
    out = _reorder(order, max_inflight_bytes=22 * _MIB)
    assert out.index("big") < out.index("moe"), "the large transfer claims the fat kernel"
    assert out.index("small") > out.index("after_big"), "the small one is not dragged across it"
    assert out.index("small") < out.index("ws")
