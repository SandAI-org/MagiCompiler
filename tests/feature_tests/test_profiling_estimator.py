# Copyright (c) 2025 SandAI. All Rights Reserved.
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

"""Unit tests for the profiling runtime estimator
(``magi_compiler.profiling.runtime_estimator``): the arg-realizer, the cache-key
shape helper, the estimator's memoization/deepcopy, and the extern replay measure.
"""

import copy
import os

import pytest
import torch
import torch.fx as fx
from torch.fx.immutable_collections import immutable_list

from magi_compiler.profiling import ProfilingRuntimeEstimator
from magi_compiler.profiling.runtime_estimator import ProfileEntry, _measure_extern, _realize_arg, _static
from magi_compiler.utils.envs import TORCH_VERSION

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


# ---------------------------------------------------------------------------
# _realize_arg  -- fx args -> concrete replay inputs
# ---------------------------------------------------------------------------
def _node_with_val(val):
    """A bare fx.Node carrying ``val`` as meta['val'] (no graph needed for realize)."""
    g = fx.Graph()
    n = g.placeholder("p")
    n.meta["val"] = val
    return n


@requires_cuda
def test_realize_tensor_node_builds_matching_tensor():
    dev = torch.cuda.current_device()
    node = _node_with_val(torch.empty(4, 8, device=dev, dtype=torch.bfloat16))
    out = _realize_arg(node)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (4, 8)
    assert out.dtype == torch.bfloat16
    assert out.device.type == "cuda"


@requires_cuda
def test_realize_float8_tensor_does_not_raise():
    """float8 has no ``randn`` kernel; replay must cast from float32 noise so fp8
    custom-op / SmoothQuant GEMM measurement does not silently fall back to 0."""
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("float8_e4m3fn not available")
    dev = torch.cuda.current_device()
    node = _node_with_val(torch.empty(4, 8, device=dev, dtype=torch.float8_e4m3fn))
    out = _realize_arg(node)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (4, 8)
    assert out.dtype == torch.float8_e4m3fn
    assert out.device.type == "cuda"
    assert torch.isfinite(out.float()).all()


def test_realize_scalar_int_node():
    node = _node_with_val(16)  # a Node carrying a plain int scalar
    assert _realize_arg(node) == 16


def test_realize_immutable_list_of_int_nodes_to_plain_list():
    """The grouped-linear m_splits case: immutable_list of Node(int) -> plain list[int]
    (the strict List[int] C++ check rejects immutable_list / symbolic elements)."""
    elems = immutable_list([_node_with_val(16), _node_with_val(16), _node_with_val(16)])
    out = _realize_arg(elems)
    assert type(out) is list  # NOT immutable_list
    assert out == [16, 16, 16]
    assert all(type(x) is int for x in out)


def test_realize_plain_containers_preserved():
    assert _realize_arg([1, 2, 3]) == [1, 2, 3]
    assert type(_realize_arg((1, 2))) is tuple
    assert _realize_arg({"a": 1}) == {"a": 1}
    # non-node scalars pass through
    assert _realize_arg(3.5) == 3.5


# ---------------------------------------------------------------------------
# _static  -- cache-key shape (symbolic dims stringified, never int()'d)
# ---------------------------------------------------------------------------
def test_static_concrete_dims_are_ints():
    assert _static((4, 8, 16)) == (4, 8, 16)


class _FakeSym:
    """Mimics a torch.SymInt: ``_is_symbolic()`` keys on having a ``.node``."""

    node = object()

    def __init__(self, name="s7", hint=None):
        self._name = name
        self.hint = hint

    def __str__(self):
        return self._name


def test_static_symbolic_dim_keyed_by_hint():
    """The PRODUCTION path (live V.graph): symbolic dims key by their
    rank-identical size hint, NOT the symbol name -- dynamo numbers symbols
    per rank (s82 vs s74 for the same dim), and symbol-name keys broke
    warm_and_sync's cross-rank key-set match on any dynamic graph ->
    analytical-cost fallback -> exposed all-gathers.

    Two symbols sharing a hint sharing one table entry is exact, not
    approximate: ``_measure_extern`` realizes replay inputs at these same
    hints (``_realize_arg`` -> ``_concrete_size``), so the measured ns is a
    pure function of (op, dtype, hint shape) -- re-measuring under a separate
    key would produce the same value."""
    from torch._inductor.virtualized import V

    class _FakeSizevars:
        def size_hint(self, sym, fallback=0):
            return sym.hint

    class _FakeGraph:
        sizevars = _FakeSizevars()

    with V.set_graph_handler(_FakeGraph()):
        seq, tok = _FakeSym("s27", hint=4096), _FakeSym("s82", hint=2048)
        seq_other_rank = _FakeSym("s74", hint=4096)  # same dim, renamed by rank 1

        # hint reaches the key (not the fallback), tagged "~" for dynamic
        assert _static((seq, 3072)) == (("~", 4096), 3072)
        # different hints -> different keys: no false sharing across dims
        assert _static((seq, 3072)) != _static((tok, 3072))
        # same hint, different symbol NAME -> same key (the fix's purpose)
        assert _static((seq, 3072)) == _static((seq_other_rank, 3072))
        # a dynamic dim never collides with a static dim of the same value
        assert _static((seq,)) != _static((4096,))


def test_static_symbolic_dim_fallback_without_graph():
    """DEGRADED path only (no live V.graph, e.g. bare unit tests):
    ``_concrete_size`` falls back to 1 for every symbolic dim.  In production
    ``_structural_key`` always runs inside Inductor scheduling where V.graph
    is set, so this collapse never happens there -- and if it somehow did, the
    measurement itself realizes inputs through the same fallback, so keys and
    values degrade together."""
    out = _static((_FakeSym("s7"), 8))
    # the key SHAPE is what matters: ("~", <hint>) marks the dim dynamic,
    # static dims stay plain ints (never int()'d from a SymInt -> no guard)
    assert out == (("~", 1), 8)
    assert _static((_FakeSym("s99"), 8)) == out


# ---------------------------------------------------------------------------
# ProfilingRuntimeEstimator -- memoization + deepcopy
# ---------------------------------------------------------------------------
def test_table_property_and_initial_state():
    est = ProfilingRuntimeEstimator()
    assert est.table == {}
    assert est.n_measured == 0
    assert est.n_cache_hits == 0
    assert est._sync_across_ranks is False


def test_deepcopy_returns_clean_instance():
    """Inductor deep-copies the pass list into the fx-graph cache key; the estimator's
    __deepcopy__ must return a fresh instance WITHOUT the (FakeTensor-holding) table."""
    est = ProfilingRuntimeEstimator()
    est._sync_across_ranks = True
    est._table[("k",)] = ProfileEntry(ns=123.0, kind="extern", label="x", measured=True)
    est._key_snode[("k",)] = object()

    clone = copy.deepcopy(est)
    assert isinstance(clone, ProfilingRuntimeEstimator)
    assert clone._table == {}  # transient state dropped
    assert clone._key_snode == {}
    assert clone._sync_across_ranks is True  # flag preserved


def test_profile_entry_fields():
    e = ProfileEntry(ns=5.0, kind="collective", label="all_gather(...)", measured=False)
    assert e.ns == 5.0 and e.kind == "collective" and e.measured is False
    assert e.reuse_count == 0


# ---------------------------------------------------------------------------
# _measure_extern  -- eager replay timing of an extern op
# ---------------------------------------------------------------------------
class _FakeIR:
    def __init__(self, fx_node):
        self._fx = fx_node

    def get_origin_node(self):
        return self._fx


class _FakeSnode:
    def __init__(self, fx_node):
        self.node = _FakeIR(fx_node)

    def get_name(self):
        return "op_test"


@requires_cuda
def test_measure_extern_matmul_positive():
    """_measure_extern replays a plain aten.mm on real tensors and returns >0 ns."""
    dev = torch.cuda.current_device()
    g = fx.Graph()
    a = g.placeholder("a")
    b = g.placeholder("b")
    a.meta["val"] = torch.empty(512, 512, device=dev, dtype=torch.bfloat16)
    b.meta["val"] = torch.empty(512, 512, device=dev, dtype=torch.bfloat16)
    mm = g.call_function(torch.ops.aten.mm.default, (a, b))
    mm.meta["val"] = torch.empty(512, 512, device=dev, dtype=torch.bfloat16)

    ns = _measure_extern(_FakeSnode(mm), fixed_iters=False)
    assert ns > 0.0


# ---------------------------------------------------------------------------
# internal-collective extern: __call__ must NOT measure it in sync mode
# ---------------------------------------------------------------------------
def _make_fake_extern_snode():
    """A stub recognized as ExternKernelSchedulerNode by __call__ (init skipped)."""
    from torch._inductor.scheduler import ExternKernelSchedulerNode as _EKSN

    class _FakeExtern(_EKSN):
        def __init__(self, fx_node):
            self.node = _FakeIR(fx_node)

        def get_name(self):
            return "op_fake"

        def get_nodes(self):
            return [self]

        def get_estimated_runtime(self):
            return 123.0

    g = fx.Graph()
    a = g.placeholder("a")
    a.meta["val"] = torch.empty(4, 4)
    mm = g.call_function(torch.ops.aten.mm.default, (a, a))
    mm.meta["val"] = torch.empty(4, 4)
    return _FakeExtern(mm)


@pytest.mark.parametrize("sync", [True, False])
def test_internal_collective_extern_not_measured_in_sync_warmup(monkeypatch, sync):
    """An extern registered via ``register_materialize_inputs`` must NOT be measured
    by __call__ in sync mode (the warm-up runs per-rank without barriers; the
    adaptive benchmarker would issue rank-dependent numbers of the internal NCCL op
    -> hang).  It is seeded analytical + stashed for warm_and_sync.  In non-sync
    mode the normal measurement path still runs."""
    from magi_compiler.profiling import register_materialize_inputs
    from magi_compiler.profiling import runtime_estimator as re_mod
    from magi_compiler.profiling.materialize_inputs import _INTERNAL_COLLECTIVE_OPS, _MATERIALIZE_INPUT_HOOKS

    measured = {"called": False}

    def _boom(*a, **k):
        measured["called"] = True
        raise AssertionError("must not be measured in sync warm-up")

    register_materialize_inputs("aten::mm")
    monkeypatch.setattr(re_mod, "_measure_extern", _boom)
    try:
        est = ProfilingRuntimeEstimator()
        est._sync_across_ranks = sync
        ns = est(_make_fake_extern_snode())
        assert ns == 123.0  # analytical seed (sync) / analytical fallback after _boom (non-sync)
        [entry] = est.table.values()
        assert entry.measured is False and entry.kind == "extern"
        if sync:
            assert measured["called"] is False  # deferred to warm_and_sync
            assert len(est._key_snode) == 1  # stashed for lockstep re-measurement
        else:
            assert measured["called"] is True  # non-sync: measurement path still taken
    finally:
        _MATERIALIZE_INPUT_HOOKS.pop("aten::mm", None)
        _INTERNAL_COLLECTIVE_OPS.discard("aten::mm")


# ---------------------------------------------------------------------------
# window scale: need = comm * scale + margin
# ---------------------------------------------------------------------------
def test_reorder_window_params_survive_deepcopy():
    """Both terms of the window budget must survive the deepcopy Inductor does
    when it folds this pass into the fx-graph cache key.  ``__deepcopy__``
    copies the fields by hand, so a dropped line silently reverts one of them
    to its default instead of raising."""
    import copy

    from magi_compiler.passes.fsdp_overlap import FsdpOverlapReorder

    pass_obj = FsdpOverlapReorder(comm_overlap_window_scale=2.0, comm_overlap_window_margin_ns=1234.0)
    clone = copy.deepcopy(pass_obj)
    for obj in (pass_obj, clone):
        assert obj.comm_overlap_window_scale == pytest.approx(2.0)
        assert obj.comm_overlap_window_margin_ns == pytest.approx(1234.0)


import torch._inductor.config as inductor_config  # noqa: E402

# ===========================================================================
# ACCURACY: estimator vs INDEPENDENT CUDA-event ground truth, for the op
# categories the FSDP-overlap reorder actually meets -- torch-native matmul,
# a fused elementwise chain (FusedSchedulerNode), and a user @triton.jit op.
#
# The estimator's fused-Triton path calls V.graph.scheduler.benchmark_fused_nodes,
# valid only INSIDE Inductor codegen with a live V.graph, so we run the estimator
# from a capture callable installed on the SAME hook the real pass uses
# (reorder_for_compute_comm_overlap_passes) -- it receives the real scheduler nodes.
# The hook calls dist.get_rank(), so a (1-rank) process group is required.
# ===========================================================================
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from torch._inductor.scheduler import ExternKernelSchedulerNode  # noqa: E402
from torch._inductor.utils import contains_collective, contains_wait  # noqa: E402


@pytest.fixture(scope="module")
def pg_1rank():
    """Single-rank process group -- the Inductor reorder hook calls dist.get_rank()."""
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29671")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    created = False
    if not dist.is_initialized():
        dist.init_process_group("gloo")
        created = True
    torch.cuda.set_device(0)
    yield
    if created:
        dist.destroy_process_group()


def _cuda_time_ns(fn, warmup=10, iters=50):
    """Median per-call GPU time (ns) via CUDA events -- independent ground truth
    (NOT the estimator's own benchmark helpers)."""
    import statistics

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e) * 1e6)
    return statistics.median(samples)


class _Capture:
    """Runs a fresh estimator on every snode of one compiled graph; records the
    (kind, est_ns) rows so a test can pick the op it cares about."""

    def __init__(self):
        self.est = ProfilingRuntimeEstimator()
        self.rows = []  # (kind, est_ns)

    def __call__(self, snodes):
        for s in snodes:
            if contains_wait(s) and not contains_collective(s):
                continue
            kind = (
                "collective" if contains_collective(s) else "extern" if isinstance(s, ExternKernelSchedulerNode) else "fused"
            )
            self.rows.append((kind, float(self.est(s))))
        return snodes

    def best(self, kind):
        vals = [ns for k, ns in self.rows if k == kind and ns > 0]
        return max(vals) if vals else None


def _compile_and_capture(fn, *args):
    cap = _Capture()
    prev = (
        inductor_config.reorder_for_compute_comm_overlap,
        inductor_config.reorder_for_compute_comm_overlap_passes,
        inductor_config.force_disable_caches,
    )
    inductor_config.reorder_for_compute_comm_overlap = True
    inductor_config.reorder_for_compute_comm_overlap_passes = [cap]
    inductor_config.force_disable_caches = True  # else a cached graph skips the scheduler
    try:
        torch._dynamo.reset()
        torch.compile(fn, dynamic=False)(*args)
        torch.cuda.synchronize()
    finally:
        (
            inductor_config.reorder_for_compute_comm_overlap,
            inductor_config.reorder_for_compute_comm_overlap_passes,
            inductor_config.force_disable_caches,
        ) = prev
    return cap


# user @triton.jit kernel wrapped as a torch custom op (an extern snode).
@triton.jit
def _add_mul_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * y + x, mask=mask)


@torch.library.custom_op("profiling_test::tri_add_mul", mutates_args=())
def _tri_add_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    _add_mul_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
    return out


@_tri_add_mul.register_fake
def _(x, y):
    return torch.empty_like(x)


def _assert_within(est, truth, lo=0.5, hi=2.0):
    assert est is not None and est > 0, "op was not measured (est is None/0)"
    assert truth > 0
    ratio = est / truth
    assert lo <= ratio <= hi, f"estimate {est/1e3:.1f}us vs truth {truth/1e3:.1f}us = {ratio:.2f}x (want {lo}-{hi}x)"


@requires_cuda
def test_accuracy_matmul(pg_1rank):
    """Extern matmul: estimator (aten replay) within 0.5-2x of CUDA-event truth."""
    dev = torch.cuda.current_device()
    a = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
    cap = _compile_and_capture(lambda a, b: a @ b, a, b)
    _assert_within(cap.best("extern"), _cuda_time_ns(lambda: torch.mm(a, b)))


@requires_cuda
def test_accuracy_elementwise_fusion(pg_1rank):
    """Fused elementwise chain -> ONE FusedSchedulerNode; estimator
    (benchmark_fused_nodes) within 0.5-2x of the compiled kernel's CUDA-event time."""
    dev = torch.cuda.current_device()
    x = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)

    def deep_pointwise(x):
        y = x
        for _ in range(32):
            y = torch.sin(y) * 1.001 + torch.cos(y)
        return y

    cap = _compile_and_capture(deep_pointwise, x)
    est = cap.best("fused")
    # ground truth: the whole compiled graph is ~one fused kernel.
    torch._dynamo.reset()
    cfn = torch.compile(deep_pointwise, dynamic=False)
    cfn(x)
    _assert_within(est, _cuda_time_ns(lambda: cfn(x)))


@requires_cuda
def test_accuracy_custom_triton(pg_1rank):
    """User @triton.jit op (extern): estimator replay within 0.5-2x of truth."""
    dev = torch.cuda.current_device()
    x = torch.randn(8192, 8192, device=dev, dtype=torch.float32)
    y = torch.randn(8192, 8192, device=dev, dtype=torch.float32)
    cap = _compile_and_capture(lambda x, y: _tri_add_mul(x, y) + 1.0, x, y)
    _assert_within(cap.best("extern"), _cuda_time_ns(lambda: _tri_add_mul(x, y)))


# ===========================================================================
# MULTI-CARD: profile_sync collective measurement accuracy.  Driven via a
# torchrun subprocess helper (the collective measurement needs >=2 real ranks).
# The helper measures a real all-gather with _measure_collective_op and compares
# to an independent CUDA-event timing, asserting the estimate is within tolerance,
# and checks warm_and_sync reconciles rank-lockstep without deadlock.
# ===========================================================================
import shutil  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

_COLL_HELPER = Path(__file__).parent / "fsdp_overlap_helper" / "estimator_collective_helper.py"


@requires_cuda
@pytest.mark.skipif(shutil.which("torchrun") is None, reason="requires torchrun")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires >=2 GPUs")
@pytest.mark.skipif(
    TORCH_VERSION >= (2, 12),
    reason="PT 2.12 Inductor generates different scheduler node counts across ranks, "
    "causing cross-rank key-set mismatch and analytical fallback",
)
def test_collective_profile_accuracy_multi_rank():
    env = os.environ.copy()
    env["MAGI_LOGGING_LEVEL"] = "warning"
    p = subprocess.run(
        ["torchrun", "--nproc_per_node=2", "--master_port=29681", str(_COLL_HELPER)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    out = p.stdout + p.stderr
    assert p.returncode == 0, f"helper failed:\n{out[-3000:]}"
    assert "COLL_ACCURATE ok=True" in p.stdout, out[-3000:]
    assert "COLL_WARMSYNC" in p.stdout and "ok=True" in p.stdout, out[-3000:]
    # key-set intersection: shared keys still measured; no hang on mismatch
    assert "COLL_MISMATCH ok=True" in p.stdout, out[-3000:]
    assert "COLL_PASS" in p.stdout, out[-3000:]
