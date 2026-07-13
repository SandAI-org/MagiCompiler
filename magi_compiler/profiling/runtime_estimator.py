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

from __future__ import annotations

"""Profiling-based ``estimate_op_runtime`` replacement.

Inductor's default ``BaseSchedulerNode.get_estimated_runtime`` is a pure
analytical roofline.  It is unreliable for exactly the nodes our FSDP overlap
reorder pass must size:

* fused pointwise/reduction kernels -> ``estimate_flops()`` is None, so the
  estimate degrades to ``bytes / dram_bw`` and IGNORES the compute fused into the
  kernel (measured 60x under-estimate for a 32-deep fusion);
* matmul -> the device TFLOPS table is wrong on this box (0.5 vs ~990), giving a
  ~1500x over-estimate;
* custom ops (Triton / flash-attn) -> ``ExternKernelSchedulerNode`` with
  ``MultiOutputLayout`` -> dtype is None -> the estimate is silently 0.

Full write-up + demos: ``scripts/demo/research_estimate_op_runtime_findings.md``.

Instead we MEASURE each scheduler node's real kernel time:
* Triton (pointwise/reduction/template) snodes -> ``scheduler.benchmark_fused_nodes``,
  which codegens the SAME fused kernel production emits and ``do_bench``es it
  (verified 0.995-1.15x of the compiled-graph kernel);
* extern snodes (matmul / custom op) -> replay the aten/custom op on real inputs
  rebuilt from the fx fake-tensor meta, timed with the Inductor benchmarker.

COLLECTIVES are NOT benchmarked here -- they always use the analytical
``estimate_nccl_collective_runtime``.  Benchmarking a real collective during compile
is not multi-rank-safe: the reorder pass runs per-rank during independent,
non-co-scheduled compilation, so issuing an NCCL op desyncs ranks -> watchdog hang
(seen on 8-GPU gaga4: rank0 raced to FSDP seqNum~194 while peers never matched).
For real measured comm, calibrate OUTSIDE compile at a synchronized point.  How far
the analytical estimate is from reality is measured by
``scripts/demo/verify_collective_estimate.py`` (a standalone, properly-synchronized
harness -- not the compile path).

Op -> time table
----------------
The estimator maintains ``self._table: dict[key -> ProfileEntry]`` -- the
persistent op->time structure for COMPUTE nodes.  The KEY is the op's STRUCTURAL
identity ``(target op, tuple[(input shape, dtype)])`` (``_structural_key``), NOT the
snode's unique name (``buf0``/``op42`` are unique per node and would defeat reuse).
Each distinct key is benchmarked ONCE on first encounter; every later isomorphic
snode (same op + same input shapes, e.g. the same matmul in every layer) reuses the
entry (``reuse_count++``).  A 40-layer model thus measures O(#distinct kernels), not
O(#nodes).  ``ProfileEntry`` carries ``(ns, kind, label, measured, reuse_count)``;
``estimator.summary()`` prints the whole table (see it at DEBUG log level).

Passed to the reorder pass as ``cost_fn`` (a callable ``snode -> nanoseconds``).
The extern measurement path is ShapeEnv-isolated (real tensors from
size_hints, eager call) so they run on the dynamic base compile; only benchmark_fused_nodes
(fused Triton) would specialize the dynamic dim, so that path stays analytical while
free symbols exist.
"""

import dataclasses
from typing import Any

import torch
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.scheduler import BaseSchedulerNode, ExternKernelSchedulerNode, FusedSchedulerNode
from torch._inductor.utils import contains_collective, contains_wait
from torch._inductor.virtualized import V

from magi_compiler.utils import magi_logger

from .benchmark_inputs import get_benchmark_inputs_hook, op_has_internal_collective

# Dedicated GLOO (CPU) group for exchanging profile metadata across ranks (built
# once, cached).  A CPU/gloo group keeps the cost sync off the NCCL process groups
# the forward uses, so it can never interleave with / desync the weight-gather or
# CP collectives.
_COST_SYNC_GROUP = "uninit"


def _get_cost_sync_group():
    global _COST_SYNC_GROUP
    import torch.distributed as dist

    if _COST_SYNC_GROUP != "uninit":
        return _COST_SYNC_GROUP
    try:
        _COST_SYNC_GROUP = dist.new_group(backend="gloo")
    except Exception as exc:  # noqa: BLE001
        magi_logger.warning("cost-sync: gloo group unavailable (%s); using default group", exc)
        _COST_SYNC_GROUP = None
    return _COST_SYNC_GROUP


@dataclasses.dataclass
class ProfileEntry:
    """One row of the op -> time table."""

    ns: float  # measured (or analytical-fallback) runtime, nanoseconds
    kind: str  # "compute" | "extern" | "collective"
    label: str  # human-readable op identity (target + shapes), for logs
    measured: bool  # True if really benchmarked, False if analytical fallback
    reuse_count: int = 0  # how many later snodes reused this entry


def _snode_label(snode: BaseSchedulerNode, max_shapes: int = 3) -> str:
    """Short human-readable identity of an snode for the profile table / logs:
    the op target plus its first few input shapes.  (The snode's own NAME, e.g.
    'op123', is deliberately NOT the cache key -- it is unique per node and would
    defeat cross-layer reuse; the key is the structural identity below.)"""
    node = getattr(snode, "node", None)
    origin = node.get_origin_node() if (node is not None and hasattr(node, "get_origin_node")) else None
    target = str(getattr(origin, "target", type(node).__name__ if node is not None else "?"))
    target = target.split("(")[0].split(" ")[-1][-40:]
    shapes = []
    if origin is not None:
        for a in (*origin.args, *getattr(origin, "kwargs", {}).values()):
            ev = a.meta.get("val") if isinstance(a, torch.fx.Node) else None
            if isinstance(ev, torch.Tensor):
                shapes.append("x".join(str(x) for x in _static(ev.shape)))
                if len(shapes) >= max_shapes:
                    break
    return f"{target}[{','.join(shapes)}]" if shapes else target


def _is_multi_output_unpack(snode: BaseSchedulerNode) -> bool:
    """True for a ``MultiOutput`` snode -- the zero-cost getitem that unpacks one
    output of a multi-output extern (FallbackKernel).  It shares its origin fx
    node with the parent extern, so it must never go through the structural-key
    table (the key collides with the parent's and returns the parent's runtime)."""
    from torch._inductor.ir import MultiOutput

    return type(getattr(snode, "node", None)) is MultiOutput


def _structural_key(snode: BaseSchedulerNode) -> tuple | None:
    """A cache key that is identical for isomorphic kernels (same op set + same
    input shapes/dtypes) so repeated layers share one measurement.  Returns None
    when we can't build a stable key (then we don't cache)."""
    parts: list[Any] = []
    for n in snode.get_nodes():
        node = getattr(n, "node", None)
        if node is None:
            return None
        origin = node.get_origin_node() if hasattr(node, "get_origin_node") else None
        target = str(getattr(origin, "target", type(node).__name__))
        shapes: list[Any] = []
        if origin is not None:
            for a in (*origin.args, *origin.kwargs.values()):
                ev = a.meta.get("val") if isinstance(a, torch.fx.Node) else None
                if isinstance(ev, torch.Tensor):
                    shapes.append((tuple(_static(ev.shape)), str(ev.dtype)))
        parts.append((target, tuple(shapes)))
    return tuple(parts)


def _is_symbolic(s) -> bool:
    return isinstance(s, torch.SymInt) or hasattr(s, "node")


def _static(shape) -> tuple:
    """Cache-key shape.  MUST NOT call ``int()`` on a SymInt -- that adds an
    ``Eq(sym, value)`` guard and specializes the dynamic dim, breaking dynamic
    shape compilation.  Symbolic dims are stringified (stable within a compile)."""
    out = []
    for s in shape:
        if _is_symbolic(s):
            out.append(str(s))
        else:
            out.append(int(s))
    return tuple(out)


def _concrete_size(s, fallback: int = 1) -> int:
    """A concrete size for building real benchmark inputs, WITHOUT specializing:
    use Inductor's size_hint (reads the hint, adds no guard)."""
    if _is_symbolic(s):
        try:
            return int(V.graph.sizevars.size_hint(s, fallback=fallback))
        except Exception:  # noqa: BLE001
            return fallback
    return int(s)


def _realize_arg(v):
    """Turn an fx arg into a concrete replay input (rank-deterministic).

    fx.Node(tensor) -> right-shaped tensor from size-hints; fx.Node/bare SymInt ->
    concrete int; list/tuple/dict -> recursively realized PLAIN container.  The
    container de-immutabilization matters: FX stores list args as
    torch.fx.immutable_collections.immutable_list, and the custom-op C++ arg parser
    requires a plain List[int] for a ``SymInt[]`` arg -- an immutable_list of still-
    symbolic / nested elements is rejected ("Expected List[int] ... found
    immutable_list"), making the op fall back to a 0 cost."""
    if isinstance(v, torch.fx.Node):
        ev = v.meta.get("val")
        if isinstance(ev, torch.Tensor):
            shape = tuple(_concrete_size(s) for s in ev.shape)
            if ev.is_floating_point():
                return torch.randn(shape, device=ev.device, dtype=ev.dtype)
            return torch.zeros(shape, device=ev.device, dtype=ev.dtype)
        if _is_symbolic(ev) or isinstance(ev, int):
            return _concrete_size(ev)  # a Node carrying a scalar -> concrete hint
        return v
    if _is_symbolic(v):
        return _concrete_size(v)
    if isinstance(v, (list, tuple)):
        realized = [_realize_arg(x) for x in v]
        return type(v)(realized) if type(v) in (list, tuple) else list(realized)
    if isinstance(v, dict):
        return {k: _realize_arg(x) for k, x in v.items()}
    return v


def _measure_extern(snode: ExternKernelSchedulerNode, fixed_iters: bool = False) -> float:
    """Time an extern (matmul / custom-op) snode by replaying its aten op.

    ``fixed_iters``: when True, time a CONSTANT number of iterations with CUDA events
    instead of the duration-adaptive ``benchmark_gpu``.  REQUIRED for ops that issue
    an INTERNAL collective (e.g. the CP ``all_to_all`` inside gaga4_fa3_with_sink_cp):
    the adaptive benchmarker runs a rank-dependent iteration count, so different ranks
    would issue different numbers of that internal collective -> NCCL count mismatch ->
    deadlock.  A fixed count keeps every rank in lockstep (mirrors _measure_collective_op).

    Ops whose replay needs VALUE-CONSISTENT metadata (not just right-shaped tensors)
    can register a hook via ``benchmark_inputs.register_benchmark_inputs`` that builds
    a valid ``(args, kwargs)``; otherwise args come from the generic ``_realize_arg``."""
    fx_node = snode.node.get_origin_node()
    if fx_node is None:
        return 0.0
    target = fx_node.target

    hook = get_benchmark_inputs_hook(_op_name(target))
    built = hook(fx_node, _realize_arg) if hook is not None else None
    if built is not None:
        args, kwargs = built
    else:
        args = tuple(_realize_arg(a) for a in fx_node.args)
        kwargs = {k: _realize_arg(v) for k, v in fx_node.kwargs.items()}

    # Replay EAGERLY, decoupled from the enclosing compile:
    # * torch._dynamo.disable(): _measure_extern runs while the OUTER graph is being
    #   compiled (Dynamo/Inductor active).  A custom boundary op whose impl contains
    #   torch.compile'd regions (e.g. gaga4_fa3_with_sink_cp's sink-correction path)
    #   would otherwise RE-ENTER Dynamo on these concrete tensors, re-trace with
    #   DYNAMIC shapes (symbolic s*), and blow up ("Dynamo failed to run FX node ...
    #   broadcast" / "can't pickle cyclic objects") -> the op fell back to a 0 cost.
    #   We want the EAGER kernel time, so disable Dynamo for the replay.
    # * no_grad: the compiled forward runs under inference_mode; some ops branch on
    #   torch.is_grad_enabled() (fa3 takes a training-only flash-attn wrapper path when
    #   grad is on, incompatible with the installed flash-attn).  Match inference.
    @torch._dynamo.disable
    def _call():
        return target(*args, **kwargs)

    def fn():
        with torch.no_grad():
            return _call()

    if not fixed_iters:
        fn()  # warmup / correctness
        return benchmarker.benchmark_gpu(fn) * 1e6  # ms -> ns
    # Fixed-iteration timing (lockstep-safe for internal collectives).
    _WARMUP, _ITERS = 3, 10
    for _ in range(_WARMUP):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(_ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return (start.elapsed_time(end) / _ITERS) * 1e6  # ms/iter -> ns


def _op_name(target) -> str:
    """Overload-qualified op name (e.g. 'athena::gaga4_fa3_with_sink_cp'), or '' for
    a non-op target.  Used to look ops up in the benchmark-input registry."""
    name = getattr(target, "name", None)
    if callable(name):
        try:
            return name()  # OpOverload.name() -> 'ns::op'
        except Exception:  # noqa: BLE001
            return ""
    return ""


def _extern_has_internal_collective(snode: BaseSchedulerNode) -> bool:
    """True for opaque boundary ops that issue collectives internally (CP attention /
    MoE), so we must measure them with fixed iterations under a barrier.  The set of
    such ops is declared by the owning code via ``register_benchmark_inputs(...,
    has_internal_collective=True)`` -- MagiCompiler holds no model-specific op names."""
    node = getattr(snode, "node", None)
    origin = node.get_origin_node() if (node is not None and hasattr(node, "get_origin_node")) else None
    target = getattr(origin, "target", None) if origin is not None else None
    return op_has_internal_collective(_op_name(target)) if target is not None else False


# ---- collective (weight all-gather) benchmarking --------------------------
_AG = torch.ops._c10d_functional.all_gather_into_tensor.default
_AG_COALESCED = torch.ops._c10d_functional.all_gather_into_tensor_coalesced.default
_WAIT = torch.ops._c10d_functional.wait_tensor.default


def _leaf_collective(snode: BaseSchedulerNode):
    """The underlying _CollectiveKernel IR node (unwraps GroupedSchedulerNode)."""
    from torch._inductor.utils import is_collective

    node = getattr(snode, "node", None)
    if node is not None and is_collective(node):
        return node
    for child in getattr(snode, "snodes", []) or []:
        cn = getattr(child, "node", None)
        if cn is not None and is_collective(cn):
            return cn
    return None


def _collective_spec(node):
    """(op_overload, group_name, group_size, [(shape, dtype, device), ...]) for a
    collective IR node, or None if it isn't a benchmarkable all-gather."""
    op = getattr(node, "op_overload", None)
    if op not in (_AG, _AG_COALESCED):
        return None
    group_name = node.constant_args[-1]  # (..., group_size, group_name)
    from torch.distributed.distributed_c10d import _get_group_size_by_name

    group_size = _get_group_size_by_name(group_name)
    specs = []
    for inp in node.inputs:
        shape = tuple(_concrete_size(s) for s in inp.layout.size)
        specs.append((shape, inp.layout.dtype, inp.layout.device))
    return op, group_name, group_size, specs


def _collective_label(snode: BaseSchedulerNode) -> str:
    """Readable identity of a collective: op name, world size, #inputs + first shape."""
    node = _leaf_collective(snode)
    spec = _collective_spec(node) if node is not None else None
    if spec is None:
        return _snode_label(snode)
    _op, _group, group_size, specs = spec
    shape0 = "x".join(str(x) for x in specs[0][0]) if specs else "?"
    return f"all_gather(ws={group_size},n={len(specs)},{shape0})"


def _measure_collective_op(snode: BaseSchedulerNode) -> float:
    """Replay the functional all-gather (+wait) on real tensors and time it."""
    node = _leaf_collective(snode)
    if node is None:
        return 0.0
    spec = _collective_spec(node)
    if spec is None:
        return 0.0
    op, group_name, group_size, specs = spec

    ins = [torch.empty(shape, dtype=dt, device=dev) for shape, dt, dev in specs]
    if op is _AG_COALESCED:

        def fn():
            outs = _AG_COALESCED(ins, group_size, group_name)
            for o in outs:
                _WAIT(o)

    else:

        def fn():
            _WAIT(_AG(ins[0], group_size, group_name))

    # FIXED iteration counts across ALL ranks.  A duration-based benchmarker
    # (e.g. benchmark_gpu shrinks benchmark_iters by per-rank estimated runtime)
    # would issue a DIFFERENT number of collectives on different ranks ->
    # NCCL count mismatch -> deadlock.  A constant loop keeps every rank in lockstep.
    _WARMUP, _ITERS = 3, 10
    for _ in range(_WARMUP):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(_ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return (start.elapsed_time(end) / _ITERS) * 1e6  # ms/iter -> ns


class ProfilingRuntimeEstimator:
    """Callable ``snode -> ns`` for ``config.estimate_op_runtime``.

    Measures compute nodes with real kernels (memoized); defers collectives and
    waits to the analytical ``get_estimated_runtime`` (the reorder pass sizes
    comm separately via ``estimate_nccl_collective_runtime`` or a calibrated
    value).  Never raises -- on any failure returns the analytical estimate (or
    0), so it degrades to today's behaviour rather than breaking compilation.
    """

    def __init__(self) -> None:
        # The op -> time table.  Key is the STRUCTURAL identity of the op
        # (target op + input shapes/dtypes for compute, or (coll, group, group_size,
        # in-shapes) for collectives) -- NOT the snode's unique name, so isomorphic
        # ops across repeated layers share ONE measurement.  Each distinct key is
        # profiled once on first encounter; later encounters reuse the entry.
        self._table: dict[tuple, ProfileEntry] = {}
        self.n_measured = 0
        self.n_cache_hits = 0
        # Set True by MagiBackend for multi-rank runs: the reorder pass then calls
        # warm_and_sync() to reconcile costs across ranks (rank-identical schedule).
        self._sync_across_ranks = False
        # Transient {structural_key -> representative snode}, used ONLY by
        # warm_and_sync() to re-measure in rank-lockstep order.  Kept OFF the
        # ProfileEntry (which Inductor pickles into the fx-graph cache key) because
        # snodes hold FakeTensors that are unpicklable / cyclic.  Never serialized.
        self._key_snode: dict = {}

    def __deepcopy__(self, memo):
        # Shared by reference from the reorder pass; when config serialization
        # deepcopies the pass list, return a clean instance (the memoized
        # measurements are transient and hold no tensors, but avoid copying them).
        new = ProfilingRuntimeEstimator()
        new._sync_across_ranks = self._sync_across_ranks
        memo[id(self)] = new
        return new

    # Backward-compat alias: some callers/tests read `.table`.
    @property
    def table(self) -> "dict[tuple, ProfileEntry]":
        return self._table

    def warm_and_sync(self) -> int:
        """Rank-LOCKSTEP profiling of every distinct op, so multi-rank runs get REAL
        measured costs (more accurate than the analytical roofline) AND a rank-
        identical cost table (required so the FSDP-overlap reorder produces the same
        schedule on every rank -- else weight-PG gathers interleave with the eager CP
        all_to_all in rank-divergent order -> deadlock).

        PRECONDITION (caller guarantees): the graph is structurally IDENTICAL on all
        ranks (the unconditional-pad lowering fix ensures this).  Therefore every rank
        has exactly the same set of ``_structural_key`` keys already populated in the
        table by the reorder pass's warm-up loop, so:
          * the key iteration order (sorted) is identical on every rank;
          * for a collective-containing op (attention/MoE), every rank measures it at
            the same step, so the barrier-wrapped fixed-iteration replay issues the
            op's internal CP all_to_all in lockstep (no NCCL count mismatch);
          * the max-reduce over gloo is symmetric.

        Steps: for each key in sorted order, barrier -> (re)measure locally with FIXED
        iters -> barrier; then all_gather_object the {key: ns} maps and take the MAX.
        Fixed-iteration timing (``_measure_one``) is mandatory -- a duration-adaptive
        benchmark would run a per-rank iteration count and desync the internal
        collective.  Returns the number of table entries whose cost changed."""
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            return 0
        world = dist.get_world_size()
        if world <= 1:
            return 0
        group = _get_cost_sync_group()

        # The reorder warm-up already populated self._table (one entry per distinct
        # structural key) and self._key_snode (a representative snode per key).
        # Re-measure each in a rank-uniform (sorted) order under barriers so any
        # internal collective is issued in lockstep across ranks.
        keys = sorted(self._table.keys(), key=repr)
        local_ns: dict = {}
        for k in keys:
            snode = self._key_snode.get(k)
            dist.barrier(group=group)
            if snode is not None:
                local_ns[k] = self._measure_one(snode)
            else:
                local_ns[k] = self._table[k].ns  # no cached snode -> keep prior measurement
            dist.barrier(group=group)

        # keys re-measured on THIS rank (had a representative snode) -- gather across
        # ranks so a key measured on any rank is flagged measured (the graph is
        # identical, so this is the same set everywhere; the union is just defensive).
        measured_here = set(self._key_snode.keys())

        gathered: list = [None] * world
        dist.all_gather_object(gathered, local_ns, group=group)
        gathered_measured: list = [None] * world
        dist.all_gather_object(gathered_measured, list(measured_here), group=group)
        merged: dict = {}
        for d in gathered:
            for k, ns in (d or {}).items():
                if k not in merged or ns > merged[k]:
                    merged[k] = ns
        measured_keys = set()
        for mk in gathered_measured:
            measured_keys.update(mk or [])
        n = 0
        for k, e in self._table.items():
            if k in measured_keys:
                e.measured = True  # reconciled from a real rank-lockstep measurement
            m = merged.get(k)
            if m is not None and m != e.ns:
                e.ns = m
                n += 1
        self._key_snode.clear()  # drop snode refs (unpicklable) once sync is done
        return n

    def _measure_one(self, snode: BaseSchedulerNode) -> float:
        """Measure a single snode with FIXED iterations (lockstep-safe), never raising
        -- falls back to the analytical estimate.  Collective-containing externs
        (attention / MoE) use fixed-iter replay so every rank issues the same number
        of internal collectives."""
        try:
            if contains_collective(snode):
                return _measure_collective_op(snode)
            if isinstance(snode, ExternKernelSchedulerNode):
                fixed = _extern_has_internal_collective(snode)
                with _shapeenv_sandbox(), _suppress_guards():
                    ns = _measure_extern(snode, fixed_iters=fixed)
                self.n_measured += 1
                return ns
            return self._measure(snode)
        except BaseException as exc:  # noqa: BLE001
            magi_logger.debug("warm/sync measure fell back to analytical for %s: %s", snode.get_name(), exc)
            return _safe_analytical(snode)

    def summary(self) -> str:
        """One line per distinct profiled op: kind, label, per-call time, #calls,
        aggregate time (per-call * #calls).  Each op also gets a machine-parseable
        ``ESTLINE`` tag (kind|label|per_call_us|calls|total_us|measured) so a run's
        estimates can be diffed against the real nsys kernel trace -- see
        scripts/demo/compare_estimate_vs_nsys.py."""
        lines = []
        for e in sorted(self._table.values(), key=lambda e: -e.ns * (e.reuse_count + 1)):
            calls = e.reuse_count + 1  # first encounter + reuses
            per_us = e.ns / 1e3
            total_us = per_us * calls
            meas = "measured" if e.measured else "analytical"
            lines.append(f"  [{e.kind:10}] {e.label:<48} {per_us:9.2f}us/call x{calls:<4} " f"= {total_us:11.2f}us  ({meas})")
            # grep-friendly: ESTLINE|kind|label|per_call_us|calls|total_us|measured
            lines.append(f"  ESTLINE|{e.kind}|{e.label}|{per_us:.3f}|{calls}|{total_us:.3f}|{meas}")
        return (
            f"profile table: {len(self._table)} distinct ops, "
            f"{self.n_measured} measured, {self.n_cache_hits} reuses\n" + "\n".join(lines)
        )

    def __call__(self, snode: BaseSchedulerNode) -> float:
        # A wait_tensor kernel itself takes ~0 time (the collective's cost is
        # attributed to the launch); keep it analytical (returns 0).
        if contains_wait(snode) and not contains_collective(snode):
            return _safe_analytical(snode)

        # A MultiOutput unpack (getitem off a multi-output extern, e.g. attention's
        # (out, lse) FallbackKernel) launches NO kernel -- it is 0-cost.  It MUST be
        # short-circuited BEFORE the op->time table: it shares its origin fx node
        # with the parent extern, so ``_structural_key`` COLLIDES with the parent's
        # key and the table would return the parent's full runtime (measured: the
        # fa3 attention MultiOutput was costed 45.7ms; the FSDP-overlap reorder then
        # believed a weight all-gather placed between the attention kernel and its
        # unpack node was hidden by 45.7ms of "compute" -- one slot short of the
        # real attention window -> the gather ran fully exposed before the MoE).
        if _is_multi_output_unpack(snode):
            return 0.0

        # COLLECTIVES: never benchmark a real collective HERE.  The Inductor reorder
        # pass runs per-rank during independent, non-co-scheduled compilation, so
        # issuing an NCCL op in __call__ desyncs ranks -> watchdog hang (observed on
        # 8-GPU gaga4: rank0 raced to FSDP seqNum~194 while peers never matched).
        # SEED the cost with Inductor's static analytical NCCL estimate (a pure
        # function of shapes+device -> rank-deterministic).  In profile_sync mode we
        # ALSO stash the snode so the rank-lockstep warm_and_sync() later OVERRIDES
        # this seed with a REAL measured time (the only multi-rank-safe place to touch
        # NCCL) -- which is where the accuracy comes from.  Collectives enter the same
        # op->time table as compute so warm_and_sync can find them.
        if contains_collective(snode):
            cnode = _leaf_collective(snode)
            spec = _collective_spec(cnode) if cnode is not None else None
            if spec is None:
                return _safe_analytical(snode)  # non-AG / unparseable -> old behaviour
            op, _group_name, group_size, specs = spec
            ckey = ("collective", str(op), group_size, tuple((tuple(shape), str(dt)) for shape, dt, _dev in specs))
            entry = self._table.get(ckey)
            if entry is not None:
                entry.reuse_count += 1
                self.n_cache_hits += 1
                return entry.ns
            ns = _safe_analytical(snode)  # Inductor static estimate as the seed
            self._table[ckey] = ProfileEntry(ns=ns, kind="collective", label=_collective_label(snode), measured=False)
            if self._sync_across_ranks:
                self._key_snode[ckey] = snode  # warm_and_sync -> real measured override
            return ns

        is_extern = isinstance(snode, ExternKernelSchedulerNode)

        # Benchmark compute nodes on the dynamic graph without specializing the
        # dynamic dim.  Extern (matmul / custom op) is ShapeEnv-isolated (replays the
        # aten op on size-hinted real tensors) so it is safe even with free symbols.
        # Fused Triton pointwise/reduction goes through benchmark_fused_nodes which
        # re-enters Inductor codegen bound to the live ShapeEnv and WOULD specialize,
        # so that path stays analytical while the graph is dynamic.  Matmul/attention/
        # MoE (the dominant costs, and the source of the bogus roofline) are extern.
        if not is_extern and _graph_has_free_symbols():
            return _safe_analytical(snode)

        # --- op -> time table: profile a distinct key ONCE, reuse afterwards ---
        key = _structural_key(snode)
        if key is not None:
            entry = self._table.get(key)
            if entry is not None:
                entry.reuse_count += 1
                self.n_cache_hits += 1
                return entry.ns

        # First encounter of this key -> measure it.  Measuring must NEVER break
        # compilation: any failure (unbenchmarkable extern, FakeTensor deepcopy in
        # the benchmark harness, ...) falls back to the analytical estimate.
        measured = True
        try:
            ns = self._measure_extern_safe(snode) if is_extern else self._measure(snode)
        except BaseException as exc:  # noqa: BLE001
            magi_logger.debug("Profiling estimator fell back to analytical for %s: %s", snode.get_name(), exc)
            ns = _safe_analytical(snode)
            measured = False

        if key is not None:
            kind = "extern" if is_extern else "compute"
            label = _snode_label(snode)
            self._table[key] = ProfileEntry(ns=ns, kind=kind, label=label, measured=measured)
            # Remember a representative snode (transient, unpicklable) so
            # warm_and_sync() can re-measure this key in rank-lockstep order.
            if self._sync_across_ranks:
                self._key_snode[key] = snode
            magi_logger.debug(
                "profile[%s] %s -> %.2fus%s", kind, label, ns / 1e3, "" if measured else " (analytical fallback)"
            )
        return ns

    def _measure_extern_safe(self, snode: BaseSchedulerNode) -> float:
        with _shapeenv_sandbox(), _suppress_guards():
            ns = _measure_extern(snode)
        self.n_measured += 1
        return ns

    def _measure(self, snode: BaseSchedulerNode) -> float:
        # Benchmarking runs real kernels at concrete (hinted) shapes.  Under
        # dynamic-shape compilation that would add ``Eq(sym, hint)`` guards and
        # replacements into the ShapeEnv and SPECIALIZE the dynamic dim (breaking
        # the compile for other seq lens).  Suppress guard creation AND snapshot /
        # restore the ShapeEnv's mutable specialization state so nothing leaks.
        with _shapeenv_sandbox(), _suppress_guards():
            return self._measure_inner(snode)

    def _measure_inner(self, snode: BaseSchedulerNode) -> float:
        try:
            if isinstance(snode, ExternKernelSchedulerNode):
                self.n_measured += 1
                return _measure_extern(snode)
            scheduler = V.graph.scheduler
            nodes = list(snode.get_nodes()) if isinstance(snode, FusedSchedulerNode) else [snode]
            ms, _ = scheduler.benchmark_fused_nodes(nodes)
            self.n_measured += 1
            return ms * 1e6
        except Exception as exc:  # noqa: BLE001
            magi_logger.debug("Profiling estimator fell back to analytical for %s: %s", snode.get_name(), exc)
            return _safe_analytical(snode)


def _safe_analytical(snode: BaseSchedulerNode) -> float:
    try:
        return snode.get_estimated_runtime()
    except Exception:  # noqa: BLE001
        return 0.0


def _graph_has_free_symbols() -> bool:
    """True if the graph being compiled still has symbolic (dynamic) shapes.

    Benchmarking real kernels is only safe when everything is concrete; a
    dynamic-shape compile has free symbols and must use the analytical estimate."""
    try:
        shape_env = V.graph.sizevars.shape_env
    except Exception:  # noqa: BLE001
        return False
    if shape_env is None:
        return False
    try:
        # A fully-static graph has no unbacked/free symbols with a non-singleton
        # range.  If ANY symbol has a range wider than one value and is not yet a
        # constant replacement, the graph is dynamic and must not be benchmarked.
        replacements = getattr(shape_env, "replacements", {})
        for sym, vr in shape_env.var_to_range.items():
            if sym in replacements:
                continue  # already specialized to a constant
            lower, upper = vr.lower, vr.upper
            # int_oo / unbounded upper -> definitely dynamic.  Guard the compare.
            try:
                same = bool(lower == upper)
            except Exception:  # noqa: BLE001
                same = False
            if not same:
                return True
    except Exception:  # noqa: BLE001
        # Cannot prove static -> assume dynamic (safe: fall back to analytical).
        return True
    return False


def _suppress_guards():
    """Context manager that suppresses ShapeEnv guard creation while we run real
    benchmark kernels at hinted shapes, so measurement never specializes a
    dynamic dim.  Falls back to a no-op when no ShapeEnv is active."""
    from contextlib import nullcontext

    try:
        shape_env = V.graph.sizevars.shape_env
        if shape_env is not None:
            return shape_env.suppress_guards()
    except Exception:  # noqa: BLE001
        pass
    return nullcontext()


# ShapeEnv mutable fields that benchmarking could pollute with a `s -> hint`
# specialization; snapshot/restore them so a measurement never persists a guard.
_SHAPEENV_STATE_FIELDS = (
    "guards",
    "axioms",
    "replacements",
    "replacements_slocs",
    "var_to_range",
    "deferred_runtime_asserts",
    "num_deferred_runtime_asserts",
    "specializations",
)


class _shapeenv_sandbox:
    """Snapshot the live ShapeEnv's specialization state on enter, restore it on
    exit.  Shallow-copies the mutable dict/list fields (their *contents* are
    replaced wholesale by the compile, never mutated in place after we restore),
    so benchmarking a kernel at a hinted concrete shape cannot leak an
    ``Eq(sym, hint)`` replacement/guard into the real compile."""

    def __init__(self) -> None:
        self._env = None
        self._saved: dict = {}

    def __enter__(self):
        try:
            self._env = V.graph.sizevars.shape_env
        except Exception:  # noqa: BLE001
            self._env = None
        if self._env is None:
            return self
        import copy

        for f in _SHAPEENV_STATE_FIELDS:
            if hasattr(self._env, f):
                val = getattr(self._env, f)
                try:
                    self._saved[f] = copy.copy(val) if isinstance(val, (dict, list, set)) else val
                except Exception:  # noqa: BLE001
                    pass
        return self

    def __exit__(self, *exc):
        if self._env is None:
            return False
        for f, val in self._saved.items():
            try:
                setattr(self._env, f, val)
            except Exception:  # noqa: BLE001
                pass
        return False
