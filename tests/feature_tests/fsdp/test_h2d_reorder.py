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
    monkeypatch.setattr(h2d_reorder, "contains_wait", lambda s: getattr(s, "kind", None) == "wait")


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


def _reorder(order, *, bandwidth=10.0, margin=0.0, max_resident_bytes=0):
    """Run the pass; bandwidth is bytes/ns, i.e. GB/s."""
    p = H2dLoadReorder(
        bandwidth_bytes_per_ns=bandwidth,
        window_margin_ns=margin,
        max_resident_bytes=max_resident_bytes,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="promotion moves real bytes")
def test_only_one_shard_is_ever_live():
    """The rule that bounds the device cost.

    Both loads are still read at ``user``, so ld0's last user is after where ld1
    lands.  Hoisting both would put two loads on the device at once, and the
    hoists compound -- which is how an unconstrained sweep rebuilds, on the
    device, the residency that offloading just paid PCIe to remove.  ld0 is
    handed back instead.
    """
    from magi_compiler.passes.weight_offload import host_pool

    host_pool.reset()
    try:
        shards = [torch.randn(8 * _MIB // 2, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        s0, s1 = _park(shards[0]), _park(shards[1])
        order = [
            _compute("c0", 5e6),
            _compute("c1", 5e6),
            _compute("c2", 5e6),
            _load("ld0", 8, slots=[s0]),
            _wait("w0", "ld0"),
            _load("ld1", 8, slots=[s1]),
            _wait("w1", "ld1"),
            _compute("user", 1e6, deps=["w0", "w1"]),
        ]
        out = _reorder(order)

        # ld1 is placed first (the sweep runs back to front) and lands above w0;
        # ld0's last user (``user``) is then still live, so ld0 goes resident.
        assert host_pool.is_resident(s0) and not host_pool.is_resident(s1)
        assert out.index("ld1") < out.index("c2"), "the surviving load is still hoisted"
        # A load handed back is left exactly where phase 1 put it: against its wait.
        assert out.index("ld0") + 1 == out.index("w0")
    finally:
        host_pool.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="promotion moves real bytes")
def test_residency_cap_pulls_a_weight_back_into_an_idle_window():
    """The second lever: the cap buys memory back with speed.

    The sweep keeps ld0 resident because its last user overlaps ld1's placement.
    Under a cap that forbids keeping it, it must come back into the offload plan
    -- into a window the sweep already left idle, where the bus is free.
    """
    from magi_compiler.passes.weight_offload import host_pool

    def run(cap):
        # Fresh shards each time: adopt empties their storage, so a tensor can
        # only be parked once.
        host_pool.reset()
        shards = [torch.randn(8 * _MIB // 2, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        s0, s1 = _park(shards[0]), _park(shards[1])
        order = [
            _compute("c0", 5e6),
            _compute("c1", 5e6),
            _compute("c2", 5e6),
            _load("ld0", 8, slots=[s0]),
            _wait("w0", "ld0"),
            _load("ld1", 8, slots=[s1]),
            _wait("w1", "ld1"),
            _compute("user", 1e6, deps=["w0", "w1"]),
        ]
        return _reorder(order, max_resident_bytes=cap), host_pool.is_resident(s0)

    try:
        _, resident = run(0)  # 0 means "no cap", not "no residency"
        assert resident, "the fastest schedule keeps the weight it cannot hide"

        out, resident = run(1)  # a cap nothing can satisfy
        assert not resident, "the cap must pull the weight back into the offload plan"
        assert out.index("ld0") < out.index("w0"), "and it must still land before its own wait"
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
    """The invariant, asserted directly on the emitted order.

    Whatever the pass decides to hoist, no two surviving loads may be live at
    the same time -- a load is live from where it runs until its last user.
    These computes have no dep on the wait, so last_user falls back to the wait.
    Checking the invariant beats checking a promotion count: the count depends
    on where the sweep happens to land, the invariant is the contract.
    """
    from magi_compiler.passes.weight_offload import host_pool

    host_pool.reset()
    try:
        shards = [torch.randn(4 * _MIB // 2, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
        slots = [_park(s) for s in shards]
        order = [_compute("c0", 9e6)]
        for i, slot in enumerate(slots):
            order += [_load(f"ld{i}", 4, slots=[slot]), _wait(f"w{i}", f"ld{i}"), _compute(f"m{i}", 3e6)]
        out = _reorder(order)

        live = [(out.index(f"ld{i}"), out.index(f"w{i}")) for i, slot in enumerate(slots) if not host_pool.is_resident(slot)]
        live.sort()
        for (s0, e0), (s1, e1) in zip(live, live[1:]):
            assert e0 < s1, f"two shards live at once: [{s0},{e0}] and [{s1},{e1}]"
        assert live, "the rule must not promote everything"
    finally:
        host_pool.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="promotion moves real bytes")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="promotion moves real bytes")
def test_frontier_uses_last_user_not_last_wait():
    """ld0's wait is before ld1, but a matmul after ld1 still reads ld0.

    A wait-based frontier would hoist both: w0 sits upstream of where ld1 lands.
    last_user sees late0 and hands ld0 back.
    """
    from magi_compiler.passes.weight_offload import host_pool

    host_pool.reset()
    try:
        shards = [torch.randn(4 * _MIB // 2, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        s0, s1 = _park(shards[0]), _park(shards[1])
        order = [
            _compute("c0", 5e6),
            _load("ld0", 4, slots=[s0]),
            _wait("w0", "ld0"),
            _compute("mid", 5e6),
            _load("ld1", 4, slots=[s1]),
            _wait("w1", "ld1"),
            _compute("late0", 1e6, deps=["w0"]),
            _compute("late1", 1e6, deps=["w1"]),
        ]
        _reorder(order)
        assert host_pool.is_resident(s0), "ld0 is still read at late0, after ld1 starts"
        assert not host_pool.is_resident(s1)
    finally:
        host_pool.reset()


def test_a_lone_load_is_never_promoted():
    """With nothing to overlap, there is nothing to give back -- even if the load
    cannot be hidden.  Exposure alone does not buy residency any more; only an
    overlapping live range does."""
    from magi_compiler.passes.weight_offload import host_pool

    host_pool.reset()
    try:
        shard = torch.randn(2 * _MIB // 2, device="cuda", dtype=torch.bfloat16)
        slot = _park(shard)
        # No compute upstream at all: unhideable, but alone.
        order = [_load("ld", 2, slots=[slot]), _wait("w", "ld"), _compute("user", 1e6, deps=["w"])]
        _reorder(order)
        assert not host_pool.is_resident(slot)
        assert shard.untyped_storage().nbytes() == 0, "it should still be offloaded"
    finally:
        host_pool.reset()


def test_equal_last_user_and_frontier_promotes_earlier_load():
    """Closed-interval equality: last_user == next target must not hoist both.

    Rebuild inserts the later load *before* its target.  If that target is the
    earlier load's last_user, both buffers are alive while that node runs.
    The sweep must promote the earlier load instead of emitting ``ld0 ... ld1, user0``.
    """
    order = [
        _compute("c0", 5e6),
        _load("ld0", 4),
        _wait("w0", "ld0"),
        _compute("user0", 1e6, deps=["w0"]),
        _load("ld1", 4),
        _wait("w1", "ld1"),
        _compute("user1", 1e6, deps=["w1"]),
    ]
    out = _reorder(order)
    assert out.index("ld0") + 1 == out.index("w0"), "ld0 stays against its wait (promoted)"
    assert out.index("ld1") < out.index("user0"), "ld1 is still hoisted to hide behind user0"
    assert out == ["c0", "ld0", "w0", "ld1", "user0", "w1", "user1"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="promotion moves real bytes")
def test_equal_last_user_and_frontier_hands_earlier_weight_back():
    """Same topology as the order-only equality test, with real host-pool slots."""
    from magi_compiler.passes.weight_offload import host_pool

    host_pool.reset()
    try:
        shards = [torch.randn(4 * _MIB // 2, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        s0, s1 = _park(shards[0]), _park(shards[1])
        order = [
            _compute("c0", 5e6),
            _load("ld0", 4, slots=[s0]),
            _wait("w0", "ld0"),
            _compute("user0", 1e6, deps=["w0"]),
            _load("ld1", 4, slots=[s1]),
            _wait("w1", "ld1"),
            _compute("user1", 1e6, deps=["w1"]),
        ]
        out = _reorder(order)
        assert host_pool.is_resident(s0), "last_user == frontier must hand ld0 back"
        assert not host_pool.is_resident(s1)
        assert out.index("ld0") + 1 == out.index("w0")
        assert out.index("ld1") < out.index("user0")
    finally:
        host_pool.reset()


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


def test_bubble_after_closed_range_starts_at_last_user_plus_one():
    """A later load at last_user would overlap; the idle window starts after it."""

    class _Fake:
        def __init__(self, last_user):
            self.load = object()
            self.last_user = last_user

    first = _Fake(last_user=4)
    second = _Fake(last_user=8)
    by_load = {first.load: first, second.load: second}
    assert H2dLoadReorder._bubbles({first.load: 0, second.load: 5}, by_load) == []
    assert H2dLoadReorder._bubbles({first.load: 0, second.load: 6}, by_load) == [(5, 6)]
    assert H2dLoadReorder._bubbles({second.load: 3}, {second.load: second}) == [(0, 3)]


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
