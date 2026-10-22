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

The frozen schedules live at the bottom of this file, one line per case, so an
intended change shows up in review as exactly the cases it moved.  Regenerate
them after an INTENDED behaviour change with ``MAGI_REGEN_GOLDEN=1``.
"""

import json
import os
import random
import re
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch._inductor import ir

_REGEN = os.environ.get("MAGI_REGEN_GOLDEN") == "1"
# Anchored to line starts: the pattern's own source line must not match.
_GOLDEN_BLOCK = re.compile(r"^(# BEGIN GOLDEN.*\n)(?:.*\n)*?(# END GOLDEN)$", re.MULTILINE)
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


def _ir(cls, *, op=None, numel=0, slot=None):
    node = object.__new__(cls)
    for key, value in {
        "op_overload": op,
        "origins": None,
        "constant_args": () if slot is None else (slot,),
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


def _load(name, mib, deps=(), slot=0):
    from magi_compiler.passes.weight_offload.runtime.h2d_op import H2D_LOAD

    return _Snode(name, _ir(ir.FallbackKernel, op=H2D_LOAD, numel=mib * _MIB // 2, slot=slot), deps)


# -- graph builders -----------------------------------------------------------


def _fsdp_graph(seed, *, form):
    """A repeated compute chain with one weight gather per layer, in phase-0 order.

    ``form``: ``plain`` (1 launch / 1 wait), ``coalesced`` (packed launch + N
    unpacks + N waits) or ``prep`` (an h2d load and a cast feeding the gather,
    the chain the launch has to carry).
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


def _h2d_graph(seed, slots):
    """One load per layer, parked directly in front of its wait (phase 1)."""
    rng = random.Random(seed)
    order = [_compute("embed", rng.uniform(1e5, 4e6))]
    act = "embed"
    for i, slot in enumerate(slots):
        mib = 4 + 4 * (i % 3)
        order.append(_load(f"ld{i}", mib, slot=slot))
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

# One case per graph form and per knob setting.  Extra seeds only reshape the
# random graph: they reach no branch the first seed misses.
_FSDP_CASES = [
    (f"fsdp_{form}_s1_{scale}_{margin:.0f}", form, 1, scale, margin)
    for form in ("plain", "coalesced", "prep")
    for scale, margin in ((1.0, 5000.0), (1.5, 0.0))
]

_H2D_CASES = [
    ("h2d_default_s1", 1, 8, {}),
    ("h2d_inflight_s3", 3, 10, {"max_inflight_bytes": 24 * _MIB}),
    ("h2d_resident_s5", 5, 10, {"max_resident_bytes": 24 * _MIB}),
    ("h2d_device_s7", 7, 12, {"max_device_weight_bytes": 64 * _MIB}),
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


def _render_golden(data):
    lines = ["_GOLDEN = {"]
    for key in sorted(data):
        entry = data[key]
        value = json.dumps(entry) if isinstance(entry, str) else f"({json.dumps(entry[0])}, {entry[1]!r})"
        lines.append(f"    {json.dumps(key)}: {value},")
    lines.append("}")
    return "\n".join(lines)


@pytest.fixture(scope="module")
def golden():
    # Cases this run skipped or deselected keep their entry; cases that no
    # longer exist are dropped.
    known = {c[0] for c in _FSDP_CASES} | {c[0] for c in _H2D_CASES}
    data = {k: v for k, v in _GOLDEN.items() if k in known}
    yield data
    if _REGEN:
        path = Path(__file__)
        text, n = _GOLDEN_BLOCK.subn(lambda m: m.group(1) + _render_golden(data) + "\n" + m.group(2), path.read_text())
        assert n == 1, "the BEGIN/END GOLDEN markers at the bottom of this file are missing"
        path.write_text(text)


def _check(golden, key, order, resident=None):
    """``order`` is the snode names in schedule order; ``resident`` the promoted load indices (h2d only)."""
    entry = " ".join(order) if resident is None else (" ".join(order), resident)
    if _REGEN:
        golden[key] = entry
        return
    assert key in golden, f"no golden entry for {key}; regenerate with MAGI_REGEN_GOLDEN=1"
    want_order, want_resident = (golden[key], None) if resident is None else golden[key]
    assert order == want_order.split(), f"{key}: schedule diverged from the frozen one"
    assert resident == want_resident, f"{key}: resident set diverged from the frozen one"


@pytest.mark.parametrize("key,form,seed,scale,margin", _FSDP_CASES, ids=[c[0] for c in _FSDP_CASES])
def test_fsdp_schedule_is_frozen(world_of_one, golden, key, form, seed, scale, margin):
    from magi_compiler.passes.fsdp_overlap import FsdpOverlapReorder

    order = _fsdp_graph(seed, form=form)
    reorder = FsdpOverlapReorder(
        comm_overlap_window_margin_ns=margin, comm_overlap_window_scale=scale, cost_fn=lambda s: s.cost
    )
    _check(golden, key, [s.name for s in reorder(order)])


def test_fsdp_prep_cost_hides_nothing(world_of_one):
    """The cast travels with its gather, so it is never compute in front of it:
    counted, it alone would cover the gather and leave it where it is."""
    from magi_compiler.passes.fsdp_overlap import FsdpOverlapReorder

    order = [
        _compute("embed", 1e6),
        _compute("mm_a", 1e6, ["embed"]),
        _compute("cast", 5e6),
        _gather("ag", ["cast"], 3e6),
        _wait("w", "ag"),
        _compute("mm", 1e6, ["w", "mm_a"]),
    ]
    reorder = FsdpOverlapReorder(comm_overlap_window_margin_ns=0.0, cost_fn=lambda s: s.cost)
    names = [s.name for s in reorder(order)]
    assert names == ["cast", "ag", "embed", "mm_a", "w", "mm"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="promotion moves real bytes")
@pytest.mark.parametrize("key,seed,n,kwargs", _H2D_CASES, ids=[c[0] for c in _H2D_CASES])
def test_h2d_schedule_is_frozen(golden, key, seed, n, kwargs):
    from magi_compiler.passes.weight_offload import H2dLoadReorder, get_global_host_weight_store

    get_global_host_weight_store().reset()
    try:
        slots = []
        for i in range(n):
            mib = 4 + 4 * (i % 3)
            shard = torch.randn(mib * _MIB // 2, device="cuda", dtype=torch.bfloat16)
            host = get_global_host_weight_store().reserve(tuple(shard.shape), shard.dtype, name=f"w{i}")
            host.copy_(shard)
            shard.untyped_storage().resize_(0)
            slots.append(get_global_host_weight_store().adopt(host, shard, name=f"w{i}"))
        order = _h2d_graph(seed, slots)
        params = {"bandwidth_bytes_per_ns": 10.0, "window_margin_ns": 0.0, "bus_utilization": 0.9, **kwargs}
        reorder = H2dLoadReorder(cost_fn=lambda s: s.cost, **params)
        names = [s.name for s in reorder(order)]
        resident = [i for i, slot in enumerate(slots) if get_global_host_weight_store().is_resident(slot)]
        _check(golden, key, names, resident)
    finally:
        get_global_host_weight_store().reset()


# -- frozen schedules ---------------------------------------------------------

# fmt: off
# BEGIN GOLDEN -- written by MAGI_REGEN_GOLDEN=1; do not edit by hand.
_GOLDEN = {
    "fsdp_coalesced_s1_1.0_5000": "ag0 mo0_0 mo0_1 mo0_2 embed w0_0 w0_1 w0_2 view0 mm0 pw0 ag1 mo1_0 mo1_1 mo1_2 attn0 w1_0 w1_1 w1_2 view1 mm1 pw1 ag2 mo2_0 mo2_1 attn1 w2_0 w2_1 view2 ag3 mo3_0 mo3_1 mm2 pw2 attn2 w3_0 w3_1 view3 mm3 pw3 ag4 mo4_0 mo4_1 attn3 w4_0 w4_1 view4 mm4 pw4 head",
    "fsdp_coalesced_s1_1.5_0": "ag0 mo0_0 mo0_1 mo0_2 embed w0_0 w0_1 w0_2 view0 mm0 pw0 ag1 mo1_0 mo1_1 mo1_2 attn0 w1_0 w1_1 w1_2 view1 mm1 pw1 ag2 mo2_0 mo2_1 attn1 w2_0 w2_1 view2 ag3 mo3_0 mo3_1 mm2 pw2 attn2 w3_0 w3_1 view3 mm3 pw3 ag4 mo4_0 mo4_1 attn3 w4_0 w4_1 view4 mm4 pw4 head",
    "fsdp_plain_s1_1.0_5000": "ag0 embed w0 view0 ag1 mm0 pw0 w1 view1 ag2 mm1 pw1 w2 view2 ag3 mm2 pw2 w3 view3 mm3 ag4 pw3 w4 view4 mm4 pw4 head",
    "fsdp_plain_s1_1.5_0": "ag0 embed w0 view0 ag1 mm0 pw0 w1 view1 mm1 pw1 ag2 w2 view2 ag3 mm2 pw2 w3 view3 mm3 pw3 ag4 w4 view4 mm4 pw4 head",
    "fsdp_prep_s1_1.0_5000": "ld0 lw0 cast0 ag0 embed w0 view0 ld1 lw1 cast1 ag1 mm0 pw0 w1 view1 ld2 lw2 cast2 ag2 mm1 pw1 w2 view2 ld3 lw3 cast3 ag3 mm2 pw2 w3 view3 mm3 ld4 lw4 cast4 ag4 pw3 w4 view4 mm4 pw4 head",
    "fsdp_prep_s1_1.5_0": "ld0 lw0 cast0 ag0 embed w0 view0 ld1 lw1 cast1 ag1 mm0 pw0 w1 view1 mm1 pw1 ld2 lw2 cast2 ag2 w2 view2 ld3 lw3 cast3 ag3 mm2 pw2 w3 view3 mm3 pw3 ld4 lw4 cast4 ag4 w4 view4 mm4 pw4 head",
    "h2d_default_s1": ("ld0 embed w0 mm0 pw0 ld1 attn0 w1 ld2 mm1 ld3 pw1 w2 mm2 pw2 w3 mm3 pw3 ld4 attn3 w4 ld5 mm4 pw4 w5 mm5 pw5 ld6 attn5 w6 mm6 pw6 ld7 attn6 w7 mm7 pw7 head", []),
    "h2d_device_s7": ("embed ld0 w0 mm0 pw0 attn0 ld1 w1 mm1 pw1 ld2 w2 mm2 pw2 attn2 ld3 w3 mm3 pw3 attn3 ld4 w4 ld5 mm4 pw4 w5 mm5 pw5 attn5 ld6 w6 mm6 pw6 ld7 attn6 w7 ld8 mm7 pw7 w8 ld9 mm8 pw8 w9 mm9 pw9 ld10 attn9 w10 ld11 mm10 pw10 w11 mm11 pw11 head", [0, 1, 2, 3, 4, 6]),
    "h2d_inflight_s3": ("ld0 embed w0 ld1 mm0 pw0 w1 mm1 pw1 ld2 attn1 w2 ld3 mm2 pw2 w3 ld4 mm3 pw3 w4 ld5 mm4 pw4 w5 ld6 mm5 pw5 w6 ld7 mm6 pw6 w7 mm7 pw7 ld8 attn7 w8 ld9 mm8 pw8 w9 mm9 pw9 head", []),
    "h2d_margin_s11": ("ld0 embed w0 ld1 mm0 pw0 w1 mm1 pw1 ld2 attn1 w2 mm2 pw2 ld3 ld4 ld5 attn2 w3 mm3 pw3 w4 mm4 pw4 w5 mm5 pw5 ld6 attn5 w6 mm6 ld7 pw6 ld8 attn6 w7 mm7 pw7 w8 ld9 mm8 pw8 w9 mm9 pw9 head", []),
    "h2d_resident_s5": ("embed ld0 w0 mm0 pw0 ld1 w1 mm1 pw1 ld2 attn1 w2 mm2 pw2 ld3 w3 mm3 pw3 attn3 ld4 w4 mm4 pw4 ld5 attn4 w5 mm5 pw5 ld6 attn5 w6 mm6 pw6 ld7 attn6 w7 mm7 pw7 ld8 attn7 w8 ld9 mm8 pw8 w9 mm9 pw9 attn9 head", [0, 1, 3, 4]),
    "h2d_slowbus_s10": ("embed ld0 w0 mm0 pw0 attn0 ld1 w1 ld2 mm1 pw1 attn1 w2 mm2 pw2 ld3 w3 mm3 pw3 ld4 w4 ld5 mm4 pw4 attn4 w5 ld8 mm5 pw5 ld6 w6 mm6 pw6 ld7 w7 mm7 pw7 attn7 w8 mm8 pw8 ld9 w9 mm9 pw9 head", [0, 1, 3, 4, 6, 7, 9]),
    "h2d_util_s9": ("embed ld0 w0 mm0 pw0 ld1 w1 mm1 pw1 ld2 w2 ld3 mm2 pw2 w3 mm3 pw3 ld4 w4 mm4 pw4 ld5 attn4 w5 ld6 mm5 pw5 w6 ld7 mm6 pw6 w7 mm7 pw7 ld8 attn7 w8 mm8 pw8 ld9 attn8 w9 mm9 pw9 head", [0, 1, 2, 4]),
}
# END GOLDEN
# fmt: on
