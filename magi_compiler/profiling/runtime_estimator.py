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

"""Profiling-based ``estimate_op_runtime`` replacement (the reorder pass's cost_fn).

Inductor's analytical roofline is unreliable for exactly the nodes the FSDP
overlap reorder must size: fused pointwise (~60x under), matmul (~1500x over on
this box), custom ops (silently 0).  So we MEASURE:
fused Triton snodes via ``scheduler.benchmark_fused_nodes``, extern snodes
(matmul / custom op) by replaying the aten op on inputs rebuilt from fx meta.

Collectives -- and, in sync mode, externs with an INTERNAL collective (CP
attention / MoE) -- are never benchmarked in ``__call__`` (per-rank compile-time
NCCL desyncs ranks -> hang); they are seeded with the analytical estimate and
re-measured for real in the rank-lockstep ``warm_and_sync``.

The op->time table (``self._table``) is keyed by STRUCTURAL identity
(op + input shapes/dtypes, ``_structural_key``) -- not the per-node name -- so
isomorphic ops across layers share one measurement: O(#distinct kernels), not
O(#nodes).  ``summary()`` dumps the table (DEBUG).

Extern measurement is ShapeEnv-isolated so it is safe on the dynamic base
compile; ``benchmark_fused_nodes`` would specialize the dynamic dim, so fused
Triton stays analytical while free symbols exist.
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

# Dedicated GLOO (CPU) group for the cost sync, built once -- keeps it off the
# NCCL process groups the forward uses (cannot desync weight-gather / CP comms).
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
    """Human-readable identity for the profile table: op target + first few input
    shapes (for logs only; the cache key is ``_structural_key``)."""
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
    """Zero-cost MultiOutput getitem.  Must never hit the structural-key table:
    it shares its origin fx node with the parent extern, so the key collides and
    would return the parent's full runtime."""
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
    """fx arg -> concrete replay input: Node(tensor) -> right-shaped tensor from
    size-hints; SymInt -> concrete int; containers -> recursively realized PLAIN
    list/tuple/dict.  Plain matters: the custom-op C++ parser rejects an fx
    immutable_list where ``SymInt[]`` expects List[int] (op would cost 0)."""
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

    ``fixed_iters=True``: constant iteration count with CUDA events instead of the
    duration-adaptive benchmarker.  Required for ops with an INTERNAL collective
    (CP all_to_all inside attention/MoE): adaptive iteration counts differ per rank
    -> NCCL count mismatch -> deadlock.

    Ops whose replay needs value-consistent metadata register a hook via
    ``benchmark_inputs.register_benchmark_inputs``; otherwise ``_realize_arg``."""
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

    # Replay eagerly, decoupled from the enclosing compile:
    # * dynamo.disable: an op whose impl contains torch.compile'd regions would
    #   re-enter Dynamo mid-compile and blow up; we want the eager kernel time.
    # * no_grad: match the inference forward -- some ops branch on
    #   torch.is_grad_enabled() into incompatible paths.
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
    """Ops that issue collectives internally (declared via
    ``register_benchmark_inputs(..., has_internal_collective=True)``) must be
    measured with fixed iterations under a barrier."""
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

    # Fixed iteration count on all ranks -- an adaptive benchmarker would issue
    # different numbers of collectives per rank -> NCCL count mismatch -> deadlock.
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
    """Callable ``snode -> ns`` (see module docstring).  Never raises -- any
    measurement failure falls back to the analytical estimate."""

    def __init__(self) -> None:
        # op -> time table, keyed by structural identity (see module docstring).
        self._table: dict[tuple, ProfileEntry] = {}
        self.n_measured = 0
        self.n_cache_hits = 0
        # True (profile_sync): the reorder pass calls warm_and_sync() to reconcile
        # costs across ranks.
        self._sync_across_ranks = False
        # Transient {key -> representative snode} for warm_and_sync re-measurement.
        # Kept OFF ProfileEntry: snodes hold unpicklable FakeTensors and the entry
        # is pickled into the fx-graph cache key.
        self._key_snode: dict = {}

    def __deepcopy__(self, memo):
        # Config serialization deepcopies the pass list; return a clean instance.
        new = ProfilingRuntimeEstimator()
        new._sync_across_ranks = self._sync_across_ranks
        memo[id(self)] = new
        return new

    # Backward-compat alias: some callers/tests read `.table`.
    @property
    def table(self) -> "dict[tuple, ProfileEntry]":
        return self._table

    def warm_and_sync(self) -> int:
        """Rank-lockstep re-measurement of every table entry, giving REAL costs
        that are also rank-identical (required for a rank-identical reorder
        schedule).  Steps: verify key sets match across ranks; per sorted key,
        barrier -> measure with FIXED iters -> barrier; finally all_gather the
        {key: ns} maps and take the max.  Returns #entries whose cost changed.

        The key-set check is a hard precondition: with divergent key sequences the
        barrier loop deadlocks (barrier/all_gather count mismatch on gloo, or two
        ranks lockstep-measuring DIFFERENT internal-collective ops -> NCCL
        mismatch).  On mismatch we warn and degrade this compile's entries to the
        analytical estimate -- rank-deterministic, same guarantee as
        fsdp_config.cost_mode=analytical."""
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            return 0
        world = dist.get_world_size()
        if world <= 1:
            return 0
        group = _get_cost_sync_group()

        keys = sorted(self._table.keys(), key=repr)

        # Fail-fast key-set check (see docstring).  all_gather_object is a single
        # symmetric collective and every rank sees the same result, so all ranks
        # take the same branch.
        key_reprs = [repr(k) for k in keys]
        all_key_reprs: list = [None] * world
        dist.all_gather_object(all_key_reprs, key_reprs, group=group)
        if any(kr != all_key_reprs[0] for kr in all_key_reprs[1:]):
            ref = set(all_key_reprs[0] or [])
            mine = set(key_reprs)
            missing = sorted(ref - mine)[:3]
            extra = sorted(mine - ref)[:3]
            magi_logger.warning(
                "warm_and_sync: cross-rank profiling key sets DIFFER (counts per rank: %s; "
                "this rank vs rank0 -- missing %d e.g. %s, extra %d e.g. %s). The per-rank "
                "graphs are not structurally identical, so rank-lockstep measurement would "
                "deadlock. Falling back to the ANALYTICAL cost estimate for this graph "
                "(rank-deterministic, less accurate). Consider fsdp_config.cost_mode=analytical.",
                [len(kr or []) for kr in all_key_reprs],
                len(ref - mine),
                missing,
                len(mine - ref),
                extra,
            )
            n = 0
            for k, snode in self._key_snode.items():
                e = self._table.get(k)
                if e is None:
                    continue
                ns = _safe_analytical(snode)
                if ns != e.ns:
                    n += 1
                e.ns = ns
                e.measured = False
            self._key_snode.clear()
            return n
        local_ns: dict = {}
        for k in keys:
            snode = self._key_snode.get(k)
            dist.barrier(group=group)
            if snode is not None:
                local_ns[k] = self._measure_one(snode)
            else:
                local_ns[k] = self._table[k].ns  # no cached snode -> keep prior measurement
            dist.barrier(group=group)

        # Union of measured keys across ranks -> flag entries measured=True.
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
        """Lockstep-safe single measurement (fixed iters for anything containing a
        collective); never raises -- falls back to the analytical estimate."""
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
        """One line per distinct op + a machine-parseable ``ESTLINE`` tag
        (kind|label|per_call_us|calls|total_us|measured) for diffing against an
        nsys trace."""
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

        if _is_multi_output_unpack(snode):
            return 0.0

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

        # Extern replay is ShapeEnv-isolated -> safe with free symbols; fused
        # Triton (benchmark_fused_nodes) would specialize the dynamic dim, so it
        # stays analytical while the graph is dynamic.
        if not is_extern and _graph_has_free_symbols():
            return _safe_analytical(snode)

        # op -> time table: profile a distinct key once, reuse afterwards.
        key = _structural_key(snode)
        if key is not None:
            entry = self._table.get(key)
            if entry is not None:
                entry.reuse_count += 1
                self.n_cache_hits += 1
                return entry.ns

        # Extern with an INTERNAL collective (CP attention / MoE): in sync mode,
        # never measure it here -- the warm-up runs per-rank WITHOUT barriers, and
        # the adaptive benchmarker would issue rank-dependent numbers of the
        # internal NCCL op -> count mismatch -> hang.  Seed analytical + stash the
        # snode; warm_and_sync re-measures it in rank-lockstep (fixed iters).
        if is_extern and self._sync_across_ranks and _extern_has_internal_collective(snode):
            ns = _safe_analytical(snode)
            if key is not None:
                self._table[key] = ProfileEntry(ns=ns, kind="extern", label=_snode_label(snode), measured=False)
                self._key_snode[key] = snode
            return ns

        # First encounter -> measure; any failure falls back to analytical
        # (measuring must never break compilation).
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
            if self._sync_across_ranks:  # stash a representative snode for warm_and_sync
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
        # Benchmarking at hinted concrete shapes must not leak Eq(sym, hint)
        # guards/replacements into the live ShapeEnv (would specialize the dynamic
        # dim): suppress guards + snapshot/restore the mutable state.
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
    """True if the compile still has dynamic (symbolic) shapes -- any symbol with
    a non-singleton range that is not yet a constant replacement."""
    try:
        shape_env = V.graph.sizevars.shape_env
    except Exception:  # noqa: BLE001
        return False
    if shape_env is None:
        return False
    try:
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
    """Suppress ShapeEnv guard creation during benchmarking (no-op without a
    live ShapeEnv)."""
    from contextlib import nullcontext

    try:
        shape_env = V.graph.sizevars.shape_env
        if shape_env is not None:
            return shape_env.suppress_guards()
    except Exception:  # noqa: BLE001
        pass
    return nullcontext()


# ShapeEnv mutable fields a benchmark could pollute with an `s -> hint`
# specialization; snapshotted/restored by _shapeenv_sandbox.
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
    """Snapshot the live ShapeEnv's specialization state on enter, restore on
    exit, so a benchmark at hinted concrete shapes cannot leak an
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
