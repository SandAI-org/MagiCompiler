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

from magi_compiler.passes.weight_offload import h2d_reorder
from magi_compiler.passes.weight_offload.h2d_reorder import H2dLoadReorder

_MIB = 1 << 20


def _park(tensor: torch.Tensor, name: str = "") -> int:
    """Put ``tensor`` in the host pool the production way: reserve, fill, empty, adopt."""
    from magi_compiler.passes.weight_offload import host_pool

    host = host_pool.reserve(tuple(tensor.shape), tensor.dtype, name=name)
    host.copy_(tensor.detach())
    tensor.untyped_storage().resize_(0)
    return host_pool.adopt(host, tensor, name=name)


class _Dep:
    def __init__(self, name):
        self.name = name


class _IR:
    def __init__(self, op_overload, numel, dtype=torch.bfloat16, slots=()):
        self.op_overload = op_overload
        self._numel = numel
        self._dtype = dtype
        # Inductor flattens a custom op's non-tensor args here; for the load ops
        # that is exactly the host-pool slot(s).
        self.constant_args = tuple(slots)

    def get_size(self):
        return [self._numel]

    def get_dtype(self):
        return self._dtype


class MultiOutput(_IR):
    """Named to match Inductor's unpack node: the pass recognizes it by class name."""


class _Snode:
    """A stand-in with just the surface the pass touches."""

    snodes = None

    def __init__(self, name, kind, *, deps=(), cost=0.0, numel=0, slots=()):
        from magi_compiler.passes.weight_offload.h2d_op import H2D_LOAD

        self.name = name
        self.kind = kind
        self.cost = cost
        self.unmet_dependencies = [_Dep(d) for d in deps]
        cls = MultiOutput if kind == "unpack" else _IR
        self.node = cls(H2D_LOAD if kind == "load" else "fake.op", numel, slots=slots)

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
    from magi_compiler.passes import snode_utils

    is_wait = lambda s: getattr(s, "kind", None) == "wait"  # noqa: E731
    monkeypatch.setattr(h2d_reorder, "contains_wait", is_wait)
    monkeypatch.setattr(snode_utils, "contains_wait", is_wait)


def _compute(name, cost_ns, deps=()):
    return _Snode(name, "compute", cost=cost_ns, deps=deps)


def _load(name, mib, deps=(), slots=()):
    return _Snode(name, "load", deps=deps, numel=mib * _MIB // 2, slots=slots)  # bf16: 2 bytes/elem


def _wait(name, load_name):
    return _Snode(name, "wait", deps=[load_name])


def _unpack(name, load_name, mib):
    """The unpack carries the member's layout -- that is where the pass reads the
    transfer size from, since a coalesced load's own size describes no one tensor."""
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


def test_load_is_hoisted_until_the_compute_covers_the_transfer():
    # 10 MiB at 10 GB/s is ~1.05ms; one 2ms kernel is more than enough, so the
    # load should stop just in front of it rather than walking to the top.
    order = [
        _compute("c0", 2e6),
        _compute("c1", 2e6),
        _load("ld", 10),
        _wait("w", "ld"),
        _compute("gather_user", 1e6, deps=["w"]),
    ]
    assert _reorder(order) == ["c0", "ld", "c1", "w", "gather_user"]


def test_load_walks_further_when_one_kernel_is_not_enough():
    order = [
        _compute("c0", 2e5),
        _compute("c1", 2e5),
        _compute("c2", 2e5),
        _load("ld", 10),  # ~1.05ms needs all three 0.2ms kernels and still is not covered
        _wait("w", "ld"),
        _compute("user", 1e6, deps=["w"]),
    ]
    assert _reorder(order) == ["ld", "c0", "c1", "c2", "w", "user"]


def test_two_loads_do_not_spend_the_same_compute():
    """One PCIe stream means the loads are serialized against each other, so the
    compute that hides one is gone as far as the other is concerned."""
    order = [
        _compute("c0", 1e6),
        _compute("c1", 1e6),
        _load("ld0", 5),  # ~0.52ms
        _wait("w0", "ld0"),
        # No dep on w0: last_user is w0, so the live ranges stay adjacent
        # ([ld0, w0] / [ld1, ...]) and this test only checks compute claiming.
        _compute("mid", 1e6),
        _load("ld1", 5),
        _wait("w1", "ld1"),
        _compute("user", 1e6, deps=["w1"]),
    ]
    out = _reorder(order)
    # ld1 claims mid, so ld0 has to fall back to c1 -- not share mid.
    assert out.index("ld1") < out.index("mid")
    assert out.index("ld0") < out.index("c1")
    # Closed [ld0, last_user]: last_user must be strictly before the next load.
    assert out.index("ld0") < out.index("w0") < out.index("ld1")


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


def _two_conflicting_loads(slots=(None, None)):
    """The topology one load of in-flight budget cannot hoist both of: both loads
    are still read at ``user``, so the earlier one's bytes are live where the
    later one lands."""
    return [
        _compute("c0", 5e6),
        _compute("c1", 5e6),
        _compute("c2", 5e6),
        _load("ld0", 8, slots=[slots[0]] if slots[0] is not None else ()),
        _wait("w0", "ld0"),
        _load("ld1", 8, slots=[slots[1]] if slots[1] is not None else ()),
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
        order += [_load(f"ld{i}", mib, slots=[slot]), _wait(f"w{i}", f"ld{i}"), _compute(f"u{i}", 1e3, deps=[f"w{i}"])]
    return order


def test_the_default_budget_is_what_the_unhoisted_order_already_needs():
    """An unasked-for budget makes the peak no worse, and spends all of that.

    Both loads here are still read at ``user``, so even unhoisted -- each against
    its own wait, the shortest live range it can have -- their bytes are live
    together.  16 MiB is therefore the floor of this problem, not a hoist's fault,
    and declining to use it would leave two transfers exposed to save memory that
    is spent either way.  Both stay hidden; the bus/live-range split then places
    them as late as that hide allows, so the live ranges stay short.
    """
    out = _reorder(_two_conflicting_loads())
    assert out.index("ld0") < out.index("w0") and out.index("ld1") < out.index("w1")
    assert out.index("ld1") < out.index("w0"), "the overlap phase 1 already pays for"
    # Shortest hide: both fit behind c2 rather than being chain-pushed into c0.
    assert out.index("c1") < out.index("ld0") <= out.index("ld1") < out.index("c2")


def test_the_inflight_budget_is_what_lets_two_loads_share_a_window():
    """What widening the budget buys, and the only thing that can buy it.

    Unhoisted these two never coexist, so one load is the floor and ld1 has to
    stay against its wait, fully exposed, however much compute sits upstream.
    16 MiB puts both of them under the same three kernels: 8 MiB more at the peak
    for a transfer that stops costing anything.
    """
    tight = _reorder(_two_loads_one_window(), max_inflight_bytes=8 * _MIB)
    assert tight.index("ld1") + 1 == tight.index("w1"), "one load of budget leaves ld1 where it was"

    wide = _reorder(_two_loads_one_window(), max_inflight_bytes=16 * _MIB)
    assert wide.index("ld0") < wide.index("w0") and wide.index("ld1") < wide.index("w1")
    assert wide.index("ld1") < wide.index("w0"), "the overlap is the point, not an accident"
    assert wide.index("c1") < wide.index("ld0") <= wide.index("ld1") < wide.index("c2")


def test_the_inflight_budget_counts_bytes_not_loads():
    """15 MiB does not hold two 8 MiB loads.

    A budget counting loads would allow the second hoist and hand back a schedule
    needing 16 MiB, which is a budget that does not bound anything.
    """
    out = _reorder(_two_loads_one_window(), max_inflight_bytes=15 * _MIB)
    assert out.index("ld1") + 1 == out.index("w1")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="promotion moves real bytes")
def test_an_empty_residency_budget_keeps_nothing_on_the_device():
    """The default, and the reason it is the default.

    Handing a weight back is the one decision this pass makes that costs device
    memory for the whole graph rather than for a window, so it happens only when
    asked for.  Doing it by default is how a 50 GiB budget ended up as 46 GiB of
    permanently resident weight with one bucket of overlap left over.
    """
    from magi_compiler.passes.weight_offload import host_pool

    host_pool.reset()
    try:
        shards = [torch.randn(8 * _MIB // 2, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        slots = [_park(s) for s in shards]
        _reorder(_two_conflicting_loads(slots))
        assert not any(host_pool.is_resident(s) for s in slots)
    finally:
        host_pool.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="promotion moves real bytes")
def test_the_residency_budget_buys_out_the_transfer_nothing_can_hide():
    """The lever that changes how much traffic the bus carries at all.

    One compute window, two loads that both want it: whatever the in-flight
    budget, one of them is exposed -- the graph is short of compute, not memory.
    The window goes to the earlier deadline, and the load that loses it stays
    against its own wait instead of being hoisted somewhere it gains nothing.
    Residency is the only thing that helps: the bytes go back on the device and
    the transfer stops happening.
    """
    from magi_compiler.passes.weight_offload import host_pool

    def run(resident):
        # Fresh shards each time: adopt empties their storage, so a tensor can
        # only be parked once.
        host_pool.reset()
        shards = [torch.randn(10 * _MIB // 2, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        slots = [_park(s) for s in shards]
        out = _reorder(_serialized_loads(slots), max_resident_bytes=resident, max_inflight_bytes=20 * _MIB)
        return out, [host_pool.is_resident(s) for s in slots]

    try:
        out, resident = run(0)
        assert not any(resident), "an empty budget buys nothing"
        assert out.index("ld0") < out.index("c0"), "the earlier deadline claims the only window"
        assert out.index("ld1") + 1 == out.index("w1"), "the one that loses it is not hoisted for nothing"

        out, resident = run(10 * _MIB)
        assert resident == [True, False], "the budget takes one of the two off the bus"
    finally:
        host_pool.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="promotion moves real bytes")
def test_the_residency_budget_is_a_ceiling_not_a_target():
    """Three loads worth buying out and room for one.

    Spending past the budget is what OOM'd 400B, and stopping early would leave
    exposure the caller has already paid memory for.
    """
    from magi_compiler.passes.weight_offload import host_pool

    host_pool.reset()
    try:
        shards = [torch.randn(10 * _MIB // 2, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        slots = [_park(s) for s in shards]
        _reorder(_serialized_loads(slots), max_resident_bytes=10 * _MIB, max_inflight_bytes=40 * _MIB)
        assert host_pool.resident_bytes() == 10 * _MIB
    finally:
        host_pool.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="promotion moves real bytes")
def test_residency_is_spent_even_where_the_sweep_hides_everything():
    """A hidden transfer is still a transfer, and it happens every step.

    The sweep hides both loads here, so neither purchase shortens the *modelled*
    schedule -- and for a while that was reason enough not to make either one
    resident.  400B showed what that costs: exposure modelled at zero, 45 GiB of
    the residency budget left unspent, the full weight set crossing PCIe every
    step, and a step 35% slower than the run that had spent it.  The model prices
    one bus at a nominal bandwidth; what the budget is actually for is making
    bytes stop crossing it.
    """
    from magi_compiler.passes.weight_offload import host_pool

    host_pool.reset()
    try:
        shards = [torch.randn(8 * _MIB // 2, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        slots = [_park(s) for s in shards]
        _reorder(_two_conflicting_loads(slots), max_resident_bytes=64 * _MIB)
        assert all(host_pool.is_resident(s) for s in slots), "room for both, so both come off the bus"
    finally:
        host_pool.reset()


def _layered_graph(slots, *, layers=6, big_mib=8, small_mib=4, compute_ns=1e6):
    """``layers`` identical blocks, each one compute kernel then two loads.

    Sized so a block's transfers do not fit its compute: 12 MiB at 10 GB/s is
    1.26ms against 1.0ms of compute, so every block runs a deficit and residency
    is the only way to close it.  This is the shape of a real transformer under
    weight offload, and the shape the placement sweep alone cannot fix.
    """
    order = []
    for i in range(layers):
        order.append(_compute(f"c{i}", compute_ns))
        for tag, mib, slot in (("big", big_mib, slots[2 * i]), ("small", small_mib, slots[2 * i + 1])):
            order.append(_load(f"l{i}_{tag}", mib, slots=(slot,)))
            order.append(_wait(f"w{i}_{tag}", f"l{i}_{tag}"))
            order.append(_compute(f"u{i}_{tag}", 0.0, deps=[f"w{i}_{tag}"]))
    return order


@pytest.mark.skipif(not torch.cuda.is_available(), reason="promotion moves real bytes")
def test_residency_closes_each_windows_own_deficit():
    """Best fit, block by block: the 4 MiB weight covers a 3.4 MiB overrun.

    The 8 MiB one covers it too, and taking that instead is what a largest-first
    fill does -- it overshoots by 4.6 MiB, leaves the block with bus to spare and
    the budget short for a later block.  With the budget set to exactly what the
    target density costs, only the best-fit choice fits at all.
    """
    from magi_compiler.passes.weight_offload import host_pool

    host_pool.reset()
    try:
        slots, sizes = [], [8, 4] * 6
        for mib in sizes:
            slots.append(_park(torch.randn(mib * _MIB // 2, device="cuda", dtype=torch.bfloat16)))
        _reorder(_layered_graph(slots), max_device_weight_bytes=(24 + 16) * _MIB)

        resident = {i for i, s in enumerate(slots) if host_pool.is_resident(s)}
        assert resident == {1, 3, 5, 7, 9, 11}, "every block gives back its own smaller weight"
        assert host_pool.resident_bytes() == 24 * _MIB
    finally:
        host_pool.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="promotion moves real bytes")
def test_residency_is_spread_across_the_graph_not_stacked_at_the_front():
    """The budget is spent where the graph is over-subscribed, which is everywhere.

    Spending it on the largest weights globally would buy out the first blocks
    and leave the last ones at their original density -- which is then what the
    step time follows, because a load can only be hoisted so far before the
    in-flight budget stops it.
    """
    from magi_compiler.passes.weight_offload import host_pool

    host_pool.reset()
    try:
        slots = []
        for mib in [8, 4] * 6:
            slots.append(_park(torch.randn(mib * _MIB // 2, device="cuda", dtype=torch.bfloat16)))
        _reorder(_layered_graph(slots), max_device_weight_bytes=64 * _MIB)

        blocks_touched = {i // 2 for i, s in enumerate(slots) if host_pool.is_resident(s)}
        assert len(blocks_touched) >= 5, f"residency only reached blocks {sorted(blocks_touched)}"
    finally:
        host_pool.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="promotion moves real bytes")
def test_device_weight_budget_bounds_residency_plus_inflight():
    """One budget, split by the pass: residency never eats the in-flight floor."""
    from magi_compiler.passes.weight_offload import host_pool

    host_pool.reset()
    try:
        slots = []
        for mib in [8, 4] * 6:
            slots.append(_park(torch.randn(mib * _MIB // 2, device="cuda", dtype=torch.bfloat16)))
        _reorder(_layered_graph(slots), max_device_weight_bytes=30 * _MIB)

        # 2 x the largest bucket is reserved for in-flight before residency is
        # handed anything, so a 30 MiB budget can buy at most 14 MiB back.
        assert 0 < host_pool.resident_bytes() <= 14 * _MIB
    finally:
        host_pool.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="promotion moves real bytes")
def test_loads_far_enough_apart_both_survive():
    """The rule only fires on an actual overlap.

    With ld0's last user (``mid0``) well upstream of where ld1 lands, the two
    loads are never live together and both stay offloaded -- the rule must not
    cost residency it does not have to.
    """
    from magi_compiler.passes.weight_offload import host_pool

    host_pool.reset()
    try:
        shards = [torch.randn(2 * _MIB // 2, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        s0, s1 = _park(shards[0]), _park(shards[1])
        order = [
            _compute("c0", 5e6),
            _load("ld0", 2, slots=[s0]),
            _wait("w0", "ld0"),
            _compute("mid0", 5e6, deps=["w0"]),
            _compute("mid1", 5e6),
            _load("ld1", 2, slots=[s1]),
            _wait("w1", "ld1"),
            _compute("user", 1e6, deps=["w1"]),
        ]
        _reorder(order)
        assert not host_pool.is_resident(s0)
        assert not host_pool.is_resident(s1)
    finally:
        host_pool.reset()


def test_last_user_is_the_consumer_not_the_wait():
    """The wait is only the floor; last_user is whoever still reads the bytes."""
    from collections import defaultdict

    from torch._inductor.comms import _is_fake_dep

    order = [_compute("c0", 5e6), _load("ld", 4), _wait("w", "ld"), _compute("matmul", 1e6, deps=["w"])]
    users = defaultdict(set)
    for s in order:
        for d in s.unmet_dependencies:
            if not _is_fake_dep(d):
                users[d.name].add(s)
    index_of = {s: i for i, s in enumerate(order)}
    load = next(s for s in order if s.name == "ld")
    plans = H2dLoadReorder(bandwidth_bytes_per_ns=10.0, window_margin_ns=0.0, cost_fn=lambda s: s.cost)._plan(
        [load], order, index_of, {}, users
    )
    assert len(plans) == 1
    assert plans[0].wait_idx == index_of[order[2]]
    assert plans[0].last_user == index_of[order[3]]


def test_load_without_a_wait_is_left_alone():
    """Moving a transfer away from a synchronization we cannot see is how you get a
    race that only shows up once the hoist is long enough."""
    order = [_compute("c0", 5e6), _load("orphan", 10), _compute("user", 1e6)]
    assert _reorder(order) == ["c0", "orphan", "user"]


def test_bandwidth_sets_the_window_size():
    """A tenfold faster bus needs a tenfold smaller window, so the same load stops
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="promotion moves real bytes")
def test_live_ranges_never_overlap_however_tight_the_chain():
    """What a one-load budget buys, asserted directly on the emitted order.

    Whatever the pass decides to hoist, no two loads may be live at the same time
    -- a load is live from where it runs until its last user.  These computes have
    no dep on the wait, so last_user falls back to the wait.  Checking the
    invariant beats checking how many loads got hoisted: the count depends on
    where the sweep happens to land, the invariant is the contract.
    """
    from magi_compiler.passes.weight_offload import host_pool

    host_pool.reset()
    try:
        shards = [torch.randn(4 * _MIB // 2, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
        slots = [_park(s) for s in shards]
        order = [_compute("c0", 9e6)]
        for i, slot in enumerate(slots):
            order += [_load(f"ld{i}", 4, slots=[slot]), _wait(f"w{i}", f"ld{i}"), _compute(f"m{i}", 3e6)]
        out = _reorder(order, max_inflight_bytes=4 * _MIB)

        live = [(out.index(f"ld{i}"), out.index(f"w{i}")) for i, slot in enumerate(slots) if not host_pool.is_resident(slot)]
        live.sort()
        for (s0, e0), (s1, e1) in zip(live, live[1:]):
            assert e0 < s1, f"two shards live at once: [{s0},{e0}] and [{s1},{e1}]"
        assert live, "the rule must not promote everything"
    finally:
        host_pool.reset()


def test_the_budget_charges_for_the_last_user_not_the_last_wait():
    """ld0's wait is at index 3, but late0 still reads its bytes at index 5.

    ld1 wants the window in front of ``mid``.  Charging ld0's live range to its
    wait would leave index 4 free and let ld1 have it, with both buffers alive
    across ``mid`` and ``late0`` -- twice the peak the budget was told to hold.
    ``last_user`` blocks it, and doubling the budget then hands it over, which is
    what shows the wait was never what stood in the way.
    """

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="promotion moves real bytes")
def test_a_lone_unhideable_load_is_bought_out_only_when_asked():
    """No compute upstream at all, so no budget makes this one overlap.

    Loading a weight nothing can hide is pure cost every forward, and residency
    is the only answer -- but it is still device memory for the whole graph, so
    it waits for a budget to be handed one.
    """
    from magi_compiler.passes.weight_offload import host_pool

    def run(resident):
        host_pool.reset()
        shard = torch.randn(2 * _MIB // 2, device="cuda", dtype=torch.bfloat16)
        slot = _park(shard)
        order = [_load("ld", 2, slots=[slot]), _wait("w", "ld"), _compute("user", 1e6, deps=["w"])]
        _reorder(order, max_resident_bytes=resident)
        return host_pool.is_resident(slot), shard

    try:
        resident, shard = run(0)
        assert not resident and shard.untyped_storage().nbytes() == 0, "it should still be offloaded"
        resident, _ = run(2 * _MIB)
        assert resident, "given the budget, the transfer stops happening"
    finally:
        host_pool.reset()


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


def test_equal_last_user_and_next_start_counts_as_an_overlap():
    """Closed-interval equality: last_user == next start is an overlap.

    Rebuild inserts the later load *before* its target, so a target equal to the
    earlier load's last_user has both buffers alive while that node runs.  Under
    a one-load budget ld1 must therefore stop short of ``user0`` -- reading the
    boundary as free is how the emitted order ends up needing twice the peak the
    pass reported.
    """
    out = _reorder(_equal_boundary_order(), max_inflight_bytes=4 * _MIB)
    assert out == ["ld0", "c0", "w0", "user0", "ld1", "w1", "user1"]
    assert out.index("user0") < out.index("ld1"), "ld1 may not come alive where user0 reads ld0"


def test_equal_boundary_is_an_overlap_the_budget_can_afford():
    """The flip side: the equality is a byte count, not an illegality.

    Two loads of budget pay for the one node where both buffers are alive, and
    both transfers then hide -- so the boundary has to be priced rather than
    forbidden.
    """
    out = _reorder(_equal_boundary_order(), max_inflight_bytes=8 * _MIB)
    assert out.index("ld0") < out.index("c0"), "ld0 hides behind c0 now"
    assert out.index("ld1") < out.index("user0"), "and ld1 reaches the window past it"


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


def _plan_stub(name, *, wait_idx, last_user, nbytes, need=0.0, lower=0, promoted=False):
    """A ``_Plan`` carrying only the fields the memory and pricing helpers read."""
    return h2d_reorder._Plan(
        load=_Snode(name, "load"),
        group=[],
        slots=[],
        wait_idx=wait_idx,
        last_user=last_user,
        need=need,
        lower=lower,
        nbytes=nbytes,
        exposed=need,
        promoted=promoted,
    )


def test_latest_finish_chains_the_deadlines_backwards():
    """A transfer may not end after its own wait, nor after the next one starts."""
    prefix = [0.0, 100.0, 300.0, 600.0, 1000.0]
    early = _plan_stub("early", wait_idx=2, last_user=4, nbytes=_MIB, need=100.0)
    late = _plan_stub("late", wait_idx=4, last_user=5, nbytes=_MIB, need=250.0)

    limits = H2dLoadReorder._latest_finish([early, late], prefix)
    assert limits[late.load] == 1000.0, "the last one is held only by its own wait"
    assert limits[early.load] == 300.0, "its own wait is tighter than 1000 - 250"

    # Grow the later transfer until it is what binds the earlier one.
    late.need = 800.0
    limits = H2dLoadReorder._latest_finish([early, late], prefix)
    assert limits[early.load] == 200.0, "it now has to be done before `late` starts"


def test_issue_index_is_just_in_time():
    """The DMA starts at ``t_bus`` whatever index we pick, so pick the latest."""
    prefix = [0.0, 100.0, 300.0, 600.0, 1000.0]

    assert H2dLoadReorder._issue_index(prefix, 300.0, 0, 4, -1) == 2
    assert H2dLoadReorder._issue_index(prefix, 599.0, 0, 4, -1) == 2, "600 is later than the slot"
    assert H2dLoadReorder._issue_index(prefix, 0.0, 0, 4, -1) == 0

    assert H2dLoadReorder._issue_index(prefix, 1000.0, 0, 3, -1) == 2, "never past its own wait"
    assert H2dLoadReorder._issue_index(prefix, 0.0, 2, 4, -1) == 2, "never before its producers"
    assert H2dLoadReorder._issue_index(prefix, 0.0, 0, 4, 3) == 3, "never before the load ahead of it"


def test_peak_counts_a_promoted_weight_twice_where_its_copy_runs():
    """Promotion is not a memory saving: the load still allocates an output.

    ``h2d_load`` copies even a resident slot into a fresh buffer, so a promoted
    weight costs its bytes for the whole graph AND again where its load runs.
    """
    kept = _plan_stub("kept", wait_idx=4, last_user=3, nbytes=4 * _MIB)
    resident = _plan_stub("resident", wait_idx=11, last_user=11, nbytes=8 * _MIB, promoted=True)
    plans = [kept, resident]
    index_of = {kept.load: 0, resident.load: 10}

    assert H2dLoadReorder._peak(plans, {kept.load: 0}, index_of) == 8 * _MIB + 8 * _MIB

    resident.last_user = 3  # its buffer now overlaps the kept load's
    index_of[resident.load] = 1
    assert H2dLoadReorder._peak(plans, {kept.load: 0}, index_of) == 8 * _MIB + 12 * _MIB


def test_inflight_occupancy_finds_the_earliest_fitting_start():
    """The one question the sweep asks per load, answered on a byte budget.

    A range that fits alongside what is already live may start at the top of the
    graph; one that does not has to start past whatever is in its way, which is
    what turns a byte budget into a placement floor.
    """
    live = h2d_reorder._Inflight(budget=10)
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
            self.load = object()
            self.last_user = last_user
            self.nbytes = nbytes

    earlier = _Fake(last_user=4, nbytes=100)
    later = _Fake(last_user=7, nbytes=50)
    peak = H2dLoadReorder._inflight_peak([earlier, later], {earlier.load: 0, later.load: 4})
    assert peak == 150

    later_after = _Fake(last_user=7, nbytes=50)
    peak_adjacent = H2dLoadReorder._inflight_peak([earlier, later_after], {earlier.load: 0, later_after.load: 5})
    assert peak_adjacent == 100


def test_coalesced_load_window_covers_the_whole_bucket():
    """A bucket's window is sized from every member, not from one of them.

    Reading the size off the coalesced load itself, or off a single unpack, would
    under-size the window by the bucket factor -- and the symptom is not a wrong
    answer, it is a hoist that stops several kernels too late.
    """

    def build(members):
        order = [_compute(f"c{i}", 3e5) for i in range(6)]
        order.append(_load("ld", 0))  # a coalesced load carries no size of its own
        for i in range(members):
            order.append(_unpack(f"mo{i}", "ld", 4))
        for i in range(members):
            order.append(_wait(f"w{i}", f"mo{i}"))
        order.append(_compute("user", 1e6, deps=[f"w{i}" for i in range(members)]))
        return order

    one = _reorder(build(1), bandwidth=10.0)
    four = _reorder(build(4), bandwidth=10.0)
    assert four.index("ld") < one.index("ld"), "four members need four times the window"
    # Every unpack travels with the load, so the bucket stays one block.
    assert [n for n in four if n.startswith("mo")] == ["mo0", "mo1", "mo2", "mo3"]
    assert four.index("mo3") == four.index("ld") + 4


def test_a_late_small_load_is_not_dragged_to_the_front_of_the_graph():
    """The gaga4 400B failure mode, in miniature.

    Big transfers ahead of it leave the bus busy through the middle of the graph.
    Scored by exposure alone the only index that looks fully hidden is the gap at
    the top, so the small load used to be emitted there and then hold memory for
    the length of the graph -- 4% of the bytes for 37% of the in-flight peak, on
    the real model.  Its deadline is last, so it belongs last.
    """
    order = [_compute("c0", 6e5)]
    for i in range(3):
        order += [_load(f"big{i}", 4, slots=()), _wait(f"wb{i}", f"big{i}"), _compute(f"ub{i}", 6e5, deps=[f"wb{i}"])]
    order += [_load("small", 1), _wait("ws", "small"), _compute("us", 1e3, deps=["ws"])]

    out = _reorder(order, max_inflight_bytes=64 * _MIB)

    assert out.index("small") > out.index("wb2"), "it belongs to the block its deadline is in"
    assert out.index("small") < out.index("ub2"), "one boundary earlier, so the DMA still starts in time"
    # The big ones pipeline one block ahead of their own waits, which is the
    # whole prefetch depth this graph has room for.
    assert [n for n in out if n.startswith(("big", "small"))] == ["big0", "big1", "big2", "small"]


def test_large_transfer_keeps_the_window_under_inflight_pressure():
    """The 400B mid-graph failure mode, on a toy graph.

    A small late load that only needs a sliver of compute used to be chain-pushed
    a thousand nodes upstream by ``window_end``, pin an in-flight byte for the
    whole middle of the graph, and leave the large early transfer stuck against
    its wait with the bus idle under a fat compute kernel.  Largest-need-first
    plus a bus timeline that does not move live ranges puts the 1920-class load
    under the kernel and keeps the small one's live range short.
    """
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
