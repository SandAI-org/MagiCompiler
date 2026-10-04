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

"""Frozen placements of both overlap passes on seeded synthetic graphs.

The unit suites pin individual rules; this one pins the whole schedule, so a
refactor of the shared scheduling core cannot shift a single snode unnoticed.
Every node is built from the real Inductor IR classes (``object.__new__``, no
constructor), so the passes classify waits, collectives and unpacks through
Inductor's own predicates and nothing is stubbed.

Regenerate after an INTENDED behaviour change with ``MAGI_REGEN_GOLDEN=1``.
"""

import json
import os
import random
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch._inductor import ir

_GOLDEN = Path(__file__).parent / "golden" / "overlap_golden.json"
_REGEN = os.environ.get("MAGI_REGEN_GOLDEN") == "1"
_MIB = 1 << 20
_AG = torch.ops._c10d_functional.all_gather_into_tensor.default
_AG_COALESCED = torch.ops._c10d_functional.all_gather_into_tensor_coalesced.default


class _Dep:
    def __init__(self, name):
        self.name = name


class _Snode:
    snodes = None

    def __init__(self, name, node, deps=(), cost=0.0):
        self.name = name
        self.node = node
        self.cost = cost
        self.unmet_dependencies = [_Dep(d) for d in deps]

    def get_name(self):
        return self.name

    def get_buffer_names(self):
        return [self.name]

    def get_operation_names(self):
        return [self.name]

    def __repr__(self):
        return f"<{self.name}>"


def _ir(cls, *, op=None, numel=0, slots=()):
    node = object.__new__(cls)
    for key, value in {
        "op_overload": op,
        "origins": None,
        "constant_args": tuple(slots),
        "get_size": lambda: [numel],
        "get_dtype": lambda: torch.bfloat16,
    }.items():
        object.__setattr__(node, key, value)
    return node


class _Compute:
    """Anything that is not a wait, collective, unpack or load is compute."""

    def __init__(self, numel=1):
        self.op_overload = "fake.compute"
        self.origins = None
        self._numel = numel

    def get_size(self):
        return [self._numel]

    def get_dtype(self):
        return torch.bfloat16


def _compute(name, cost, deps=()):
    return _Snode(name, _Compute(), deps, cost)


def _wait(name, dep):
    return _Snode(name, _ir(ir._WaitKernel), [dep])


def _unpack(name, dep, numel=0):
    return _Snode(name, _ir(ir.MultiOutput, numel=numel), [dep])


def _gather(name, deps, cost, coalesced=False):
    return _Snode(name, _ir(ir._CollectiveKernel, op=_AG_COALESCED if coalesced else _AG), deps, cost)


def _load(name, mib, deps=(), slots=()):
    from magi_compiler.passes.weight_offload.runtime.h2d_op import H2D_LOAD

    return _Snode(name, _ir(_Compute, op=H2D_LOAD, numel=mib * _MIB // 2, slots=slots), deps)


# -- graph builders -----------------------------------------------------------


def _fsdp_graph(seed, *, form):
    """A transformer-ish chain with one weight gather per layer, in phase-0 order.

    ``form``: ``plain`` (1 launch / 1 wait), ``coalesced`` (packed launch + N
    unpacks + N waits) or ``prep`` (an h2d load and a cast feeding the gather,
    the chain ``move_prep_chain`` has to carry).
    """
    rng = random.Random(seed)
    order = [_compute("embed", rng.uniform(1e4, 4e6))]
    act = "embed"
    for i in range(rng.randint(5, 9)):
        comm = rng.uniform(5e4, 3e6)
        consumers = []
        if form == "plain":
            order.append(_gather(f"ag{i}", [], comm))
            order.append(_wait(f"w{i}", f"ag{i}"))
            consumers = [f"w{i}"]
        elif form == "coalesced":
            members = rng.randint(2, 3)
            order.append(_gather(f"ag{i}", [], comm, coalesced=True))
            for k in range(members):
                order.append(_unpack(f"mo{i}_{k}", f"ag{i}"))
            for k in range(members):
                order.append(_wait(f"w{i}_{k}", f"mo{i}_{k}"))
            consumers = [f"w{i}_{k}" for k in range(members)]
        else:
            order.append(_load(f"ld{i}", 4))
            order.append(_wait(f"lw{i}", f"ld{i}"))
            order.append(_compute(f"cast{i}", 0.0, [f"lw{i}"]))
            order.append(_gather(f"ag{i}", [f"cast{i}"], comm))
            order.append(_wait(f"w{i}", f"ag{i}"))
            consumers = [f"w{i}"]
        # A free view between the wait and its real consumer: transparent.
        order.append(_compute(f"view{i}", 0.0, consumers))
        order.append(_compute(f"mm{i}", rng.uniform(1e4, 5e6), [f"view{i}", act]))
        order.append(_compute(f"pw{i}", rng.uniform(1e3, 3e5), [f"mm{i}"]))
        act = f"pw{i}"
        if rng.random() < 0.4:
            order.append(_compute(f"attn{i}", rng.uniform(1e6, 3e7), [act]))
            act = f"attn{i}"
    order.append(_compute("head", rng.uniform(1e4, 1e6), [act]))
    return order


def _fsdp_heavy_graph(seed, comm_ratio):
    """Gathers about as long as the matmul they feed, with the odd long attention:
    the stream saturates, and a long kernel's leftover has to carry."""
    rng = random.Random(seed)
    order = [_compute("embed", rng.uniform(1e5, 1e6))]
    act = "embed"
    for i in range(10):
        mm = rng.uniform(2e5, 4e6)
        order.append(_gather(f"ag{i}", [], comm_ratio * rng.uniform(0.3, 1.6) * mm))
        order.append(_wait(f"w{i}", f"ag{i}"))
        order.append(_compute(f"view{i}", 0.0, [f"w{i}"]))
        order.append(_compute(f"mm{i}", mm, [f"view{i}", act]))
        act = f"mm{i}"
        if rng.random() < 0.35:
            order.append(_compute(f"attn{i}", rng.uniform(5e6, 3e7), [act]))
            act = f"attn{i}"
        for k in range(rng.randint(0, 3)):
            order.append(_compute(f"pw{i}_{k}", rng.uniform(1e3, 5e4), [act]))
            act = f"pw{i}_{k}"
    order.append(_compute("head", 1e5, [act]))
    return order


def _h2d_graph(seed, slots):
    """One load per layer, parked directly in front of its wait (phase 1)."""
    rng = random.Random(seed)
    order = [_compute("embed", rng.uniform(1e5, 4e6))]
    act = "embed"
    for i, slot in enumerate(slots):
        mib = 4 + 4 * (i % 3)
        order.append(_load(f"ld{i}", mib, slots=[slot]))
        order.append(_wait(f"w{i}", f"ld{i}"))
        order.append(_compute(f"mm{i}", rng.uniform(1e5, 3e6), [f"w{i}", act]))
        order.append(_compute(f"pw{i}", rng.uniform(1e3, 5e5), [f"mm{i}"]))
        act = f"pw{i}"
        if rng.random() < 0.3:
            order.append(_compute(f"attn{i}", rng.uniform(1e6, 8e6), [act]))
            act = f"attn{i}"
    order.append(_compute("head", rng.uniform(1e5, 1e6), [act]))
    return order


# -- cases --------------------------------------------------------------------

_FSDP_CASES = [
    (f"fsdp_{form}_s{seed}_{scale}_{margin:.0f}", form, seed, scale, margin)
    for form in ("plain", "coalesced", "prep")
    for seed in (1, 2, 3)
    for scale, margin in ((1.0, 5000.0), (1.5, 0.0))
]

_H2D_CASES = [
    ("h2d_default_s1", 1, 8, {}),
    ("h2d_default_s2", 2, 10, {}),
    ("h2d_inflight_s3", 3, 10, {"max_inflight_bytes": 24 * _MIB}),
    ("h2d_inflight_s4", 4, 12, {"max_inflight_bytes": 40 * _MIB}),
    ("h2d_resident_s5", 5, 10, {"max_resident_bytes": 24 * _MIB}),
    ("h2d_resident_s6", 6, 12, {"max_resident_bytes": 48 * _MIB, "max_inflight_bytes": 32 * _MIB}),
    ("h2d_device_s7", 7, 12, {"max_device_weight_bytes": 64 * _MIB}),
    ("h2d_device_s8", 8, 14, {"max_device_weight_bytes": 40 * _MIB}),
    ("h2d_util_s9", 9, 10, {"bus_utilization": 0.6, "max_resident_bytes": 32 * _MIB}),
    ("h2d_slowbus_s10", 10, 10, {"bandwidth_bytes_per_ns": 2.0, "max_resident_bytes": 40 * _MIB}),
    ("h2d_margin_s11", 11, 10, {"window_margin_ns": 2e5, "window_scale": 1.3}),
]


@pytest.fixture(scope="module")
def world_of_one():
    """The FSDP pass negotiates its placement mode across ranks even at world 1."""
    from magi_compiler.utils import dist_utils

    owned = not dist.is_initialized()
    if owned:
        store = dist.FileStore(tempfile.mktemp(prefix="overlap_golden_"), 1)
        dist.init_process_group("gloo", store=store, rank=0, world_size=1)
    yield
    if owned:
        dist.destroy_process_group()
        dist_utils._CPU_GLOO_GROUP = "uninit"


@pytest.fixture(scope="module")
def golden():
    data = {} if _REGEN or not _GOLDEN.exists() else json.loads(_GOLDEN.read_text())
    yield data
    if _REGEN:
        _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


def _check(golden, key, value):
    if _REGEN:
        golden[key] = value
        return
    assert key in golden, f"no golden entry for {key}; regenerate with MAGI_REGEN_GOLDEN=1"
    assert value == golden[key], f"{key} diverged from the frozen schedule"


@pytest.mark.parametrize("key,form,seed,scale,margin", _FSDP_CASES, ids=[c[0] for c in _FSDP_CASES])
def test_fsdp_schedule_is_frozen(world_of_one, golden, key, form, seed, scale, margin):
    from magi_compiler.passes.fsdp_overlap import FsdpOverlapReorder

    order = _fsdp_graph(seed, form=form)
    reorder = FsdpOverlapReorder(
        comm_overlap_window_margin_ns=margin,
        comm_overlap_window_scale=scale,
        cost_fn=lambda s: s.cost,
        move_prep_chain=form == "prep",
    )
    _check(golden, key, [s.name for s in reorder(order)])


_ALAP_CASES = [
    (f"golden_{form}_s{seed}", form, seed, None) for form in ("plain", "coalesced", "prep") for seed in (1, 2, 3)
] + [(f"heavy_r{ratio}_s{seed}", None, seed, ratio) for ratio in (0.5, 1.0, 1.5) for seed in range(6)]


@pytest.mark.parametrize("key,form,seed,ratio", _ALAP_CASES, ids=[c[0] for c in _ALAP_CASES])
def test_fsdp_alap_is_the_index_sweep_on_the_time_axis(world_of_one, key, form, seed, ratio):
    """``index_sweep`` claims compute a snode at a time and carries a long
    kernel's remainder by hand; the time-axis deadline chain gets the same
    placement with nothing to carry.  If they ever part, one of them changed."""
    from magi_compiler.passes.fsdp_overlap import FsdpOverlapReorder

    def placed(placement):
        order = _fsdp_graph(seed, form=form) if form else _fsdp_heavy_graph(seed, ratio)
        reorder = FsdpOverlapReorder(cost_fn=lambda s: s.cost, move_prep_chain=form == "prep", placement=placement)
        return [s.name for s in reorder(order)]

    assert placed("alap") == placed("index_sweep")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="promotion moves real bytes")
@pytest.mark.parametrize("key,seed,n,kwargs", _H2D_CASES, ids=[c[0] for c in _H2D_CASES])
def test_h2d_schedule_is_frozen(golden, key, seed, n, kwargs):
    from magi_compiler.passes.weight_offload import H2dLoadReorder, host_pool

    host_pool.reset()
    try:
        slots = []
        for i in range(n):
            mib = 4 + 4 * (i % 3)
            shard = torch.randn(mib * _MIB // 2, device="cuda", dtype=torch.bfloat16)
            host = host_pool.reserve(tuple(shard.shape), shard.dtype, name=f"w{i}")
            host.copy_(shard)
            shard.untyped_storage().resize_(0)
            slots.append(host_pool.adopt(host, shard, name=f"w{i}"))
        order = _h2d_graph(seed, slots)
        params = {"bandwidth_bytes_per_ns": 10.0, "window_margin_ns": 0.0, "bus_utilization": 0.9, **kwargs}
        reorder = H2dLoadReorder(cost_fn=lambda s: s.cost, **params)
        names = [s.name for s in reorder(order)]
        resident = [i for i, slot in enumerate(slots) if host_pool.is_resident(slot)]
        _check(golden, key, {"order": names, "resident": resident})
    finally:
        host_pool.reset()
