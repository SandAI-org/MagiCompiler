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

"""Triton kernel introspection used by ``magi_register_custom_op``.

:func:`introspect_fn` walks ``fn`` + helpers ONCE, returning every kernel /
nested-op / heuristics reference downstream consumers need.
:func:`rewrite_fn_with_wrap_triton` shadows kernel refs in ``fn``'s globals
and closures with ``wrap_triton(k)`` so Inductor can trace through them.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import inspect
import logging
import types
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# Default helper-recursion depth; overridable via
# ``magi_register_custom_op(max_introspect_depth=...)``.
DEFAULT_MAX_INTROSPECT_DEPTH: int = 5


__all__ = ["DEFAULT_MAX_INTROSPECT_DEPTH", "IntrospectionResult", "introspect_fn", "rewrite_fn_with_wrap_triton"]


# ==============================================================================
# SECTION 0 -- Single-pass AST scan (shared primitive)
# ==============================================================================


@dataclasses.dataclass(frozen=True)
class AstScanResult:
    """Raw AST-level findings about ``fn`` (identifiers as written in source;
    caller resolves them via the function's globals/closure).

    Attributes:
        wrapped_kernel_names: identifiers passed to ``wrap_triton`` /
            ``capture_triton`` (already wrapped -- do NOT re-shadow).
        bare_kernel_names: identifiers launched as ``k[grid](...)``,
            ``mod.k[grid](...)``, ``k.run(...)`` or surfaced via bare
            ``return k``; dotted forms recorded as a single string.
        called_helpers: plain function-call identifiers (helpers, recursed).
        nested_op_calls: ``"ns::op"`` strings for ``torch.ops.<ns>.<op>(...)``
            (NOT recursed into -- registered ops stay opaque).
        assignments: ``var -> [RHS expr, ...]`` for alias tracing
            (``k = make_kernel(); k[grid](...)``).
    """

    wrapped_kernel_names: tuple[str, ...]
    bare_kernel_names: tuple[str, ...]
    called_helpers: tuple[str, ...]
    nested_op_calls: tuple[str, ...]
    assignments: dict[str, list[ast.expr]]


def _dotted_attr_name(node: ast.AST) -> Optional[str]:
    """Return ``"a.b.c"`` if ``node`` is an Attribute chain rooted at a
    ``Name``, else ``None`` (forms like ``factory().k`` are not statically
    resolvable and fall back to ``extra_triton_kernels``)."""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def _is_wrap_triton_call(node: ast.AST) -> bool:
    """True if ``node`` is a ``wrap_triton``/``capture_triton`` call (any of
    the forms ``_AstCollector.visit_Call`` recognises)."""
    if not isinstance(node, ast.Call):
        return False
    triton_func_names = ("capture_triton", "wrap_triton")
    triton_wrap_modules = ("_library", "library")  # public + origin
    f = node.func
    if isinstance(f, ast.Name) and f.id in triton_func_names:
        return True
    if (
        isinstance(f, ast.Attribute)
        and f.attr in triton_func_names
        and isinstance(f.value, ast.Attribute)
        and f.value.attr in triton_wrap_modules
        and isinstance(f.value.value, ast.Name)
        and f.value.value.id == "torch"
    ):
        return True
    return False


def _names_outside_wrap_calls(expr: ast.expr) -> list[str]:
    """Collect every ``Name`` in ``expr`` except those inside a
    ``wrap_triton``/``capture_triton`` call (already counted by ``visit_Call``)."""
    names: list[str] = []

    class _Collector(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            if _is_wrap_triton_call(node):
                return
            self.generic_visit(node)

        def visit_Name(self, node: ast.Name) -> None:
            names.append(node.id)

    _Collector().visit(expr)
    return names


class _AstCollector(ast.NodeVisitor):
    """Single AST walker producing the data for :class:`AstScanResult`."""

    _TRITON_FUNC_NAMES = ("capture_triton", "wrap_triton")
    _TRITON_WRAP_MODULES = ("_library", "library")

    def __init__(self) -> None:
        self.wrapped_kernel_names: list[str] = []
        self.bare_kernel_names: list[str] = []
        self.called_helpers: list[str] = []
        self.nested_op_calls: list[str] = []
        self.assignments: dict[str, list[ast.expr]] = {}

    def visit_Return(self, node: ast.Return) -> None:
        # A helper may surface a kernel via ``return k`` for the caller to
        # launch. Names inside ``wrap_triton(...)`` are skipped here and
        # picked up by ``visit_Call`` so we don't double-count them as bare.
        if node.value is not None:
            for name in _names_outside_wrap_calls(node.value):
                self.bare_kernel_names.append(name)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Recognised shapes:
        #   A.1  torch.[_]library.wrap_triton(k)       -> wrapped_kernel_names
        #   A.2  torch.ops.<ns>.<op>(...)              -> nested_op_calls (opaque)
        #   A.3  wrap_triton(k) / capture_triton(k)    -> wrapped_kernel_names
        #   A.4  any other Name(...) call              -> called_helpers (recursed)
        #   A.5  <dotted>.run(*args, grid=...)         -> bare_kernel_names
        #          (Triton's low-level launch API; non-kernel .run filtered at resolve)
        #   Bare k[grid](...) / mod.k[grid](...)       -> Subscript branch below
        if isinstance(node.func, ast.Attribute):
            attr = node.func
            handled = False
            if isinstance(attr.value, ast.Attribute):
                if (
                    isinstance(attr.value.value, ast.Name)
                    and attr.value.value.id == "torch"
                    and attr.value.attr in self._TRITON_WRAP_MODULES
                    and attr.attr in self._TRITON_FUNC_NAMES
                ):
                    # A.1
                    if node.args and isinstance(node.args[0], ast.Name):
                        self.wrapped_kernel_names.append(node.args[0].id)
                    handled = True
                elif (
                    isinstance(attr.value.value, ast.Attribute)
                    and isinstance(attr.value.value.value, ast.Name)
                    and attr.value.value.value.id == "torch"
                    and attr.value.value.attr == "ops"
                ):
                    # A.2
                    self.nested_op_calls.append(f"{attr.value.attr}::{attr.attr}")
                    handled = True
            if not handled and attr.attr == "run":
                # A.5
                dotted = _dotted_attr_name(attr.value)
                if dotted is not None:
                    self.bare_kernel_names.append(dotted)
        elif isinstance(node.func, ast.Name):
            if node.func.id in self._TRITON_FUNC_NAMES:
                # A.3
                if node.args and isinstance(node.args[0], ast.Name):
                    self.wrapped_kernel_names.append(node.args[0].id)
            else:
                # A.4
                self.called_helpers.append(node.func.id)

        # Subscript launch: ``Name[grid](...)`` or attribute-chain rooted at a
        # Name (``mod.k[grid](...)``). ``self.k[grid](...)`` records but can't
        # be resolved statically -- caller must pass ``extra_triton_kernels``.
        if isinstance(node.func, ast.Subscript):
            base = node.func.value
            if isinstance(base, ast.Name):
                self.bare_kernel_names.append(base.id)
            elif isinstance(base, ast.Attribute):
                dotted = _dotted_attr_name(base)
                if dotted is not None:
                    self.bare_kernel_names.append(dotted)

        self.generic_visit(node)


def scan_fn_ast(fn: Callable[..., Any]) -> Optional[AstScanResult]:
    """Parse ``fn``'s source once and return the raw collector data; returns
    ``None`` when source is unavailable (builtins, C-extensions, REPL).
    Only inspects *this* frame -- :func:`introspect_fn` drives the recursion."""
    try:
        fn = inspect.unwrap(fn)
    except ValueError:
        pass

    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        return None

    from torch._inductor.utils import IndentedBuffer

    buffer = IndentedBuffer()
    buffer.splice(source, strip=True)

    try:
        tree = ast.parse(buffer.getrawvalue())
    except SyntaxError:
        return None

    collector = _AstCollector()
    collector.visit(tree)

    return AstScanResult(
        wrapped_kernel_names=tuple(collector.wrapped_kernel_names),
        bare_kernel_names=tuple(collector.bare_kernel_names),
        called_helpers=tuple(collector.called_helpers),
        nested_op_calls=tuple(collector.nested_op_calls),
        assignments=collector.assignments,
    )


def _build_fn_namespace(func_obj: object) -> dict[str, Any]:
    """Combined globals + closures + nonlocals view of ``func_obj``; empty
    dict for non-callables or callables without ``__code__``."""
    if callable(func_obj):
        try:
            func_obj = inspect.unwrap(func_obj)
        except ValueError:
            pass
    if not callable(func_obj) or not hasattr(func_obj, "__code__"):
        return {}
    closure_vars = inspect.getclosurevars(func_obj)
    namespace: dict[str, Any] = {}
    namespace.update(closure_vars.builtins)
    namespace.update(closure_vars.globals)
    namespace.update(closure_vars.nonlocals)
    if hasattr(func_obj, "__globals__"):
        namespace.update(func_obj.__globals__)
    return namespace


# ==============================================================================
# SECTION 1 -- Unified single-pass introspection
# Vendored / extended from torch._library.triton (v2.11.0,
# https://github.com/pytorch/pytorch/blob/v2.11.0/torch/_library/triton.py,
# BSD-licensed; original: ``get_inner_triton_kernels``).
# ==============================================================================


@dataclasses.dataclass(frozen=True)
class IntrospectionResult:
    """Output of :func:`introspect_fn`; all sequences dedup'd (kernels by
    ``id``, op names by string) and ordered by first-discovery in the BFS.

    Attributes:
        bare_triton_kernels: ``JITFunction``/``Autotuner`` launched as
            ``k[grid]``, ``k.run(...)``, or surfaced via ``return k``.
            ``Heuristics`` wrappers are *peeled* (inner kernel recorded);
            the raw wrapper goes to ``referenced_heuristics``.
        user_wrapped_kernels: kernels already passed to
            ``wrap_triton``/``capture_triton`` (rewriter must skip these).
        referenced_heuristics: raw ``Heuristics`` objects -- used to reject
            ``@triton.heuristics``-as-outermost early.
        nested_op_calls: ``"ns::op"`` strings for every reached
            ``torch.ops.<ns>.<op>`` (anywhere in the call tree).
    """

    bare_triton_kernels: tuple[Any, ...]
    user_wrapped_kernels: tuple[Any, ...]
    referenced_heuristics: tuple[Any, ...]
    nested_op_calls: tuple[str, ...]

    @property
    def has_direct_kernel(self) -> bool:
        """True iff a triton kernel that needs the ``triton_op`` path was
        found (heuristics-outermost is already rejected upstream, so by
        the time this runs ``referenced_heuristics`` is guaranteed empty)."""
        return bool(self.bare_triton_kernels)

    @property
    def user_wrapped_kernel_ids(self) -> frozenset[int]:
        """``id``-set of already-wrapped kernels for the rewriter's
        ``excluded_kernel_ids`` argument."""
        return frozenset(id(k) for k in self.user_wrapped_kernels)


def _extract_names_from_expr(expr: ast.expr) -> list[str]:
    """Pull every ``ast.Name`` reachable from ``expr`` (descends into nested
    calls so ``k = factory(arg).method`` surfaces ``factory``)."""
    names: list[str] = []

    class _NameExtractor(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            names.append(node.id)

        def visit_Call(self, node: ast.Call) -> None:
            self.generic_visit(node)

    _NameExtractor().visit(expr)
    return names


def introspect_fn(
    fn: Callable[..., Any],
    *,
    extra_triton_kernels: list[Any] | tuple[Any, ...] | None = None,
    max_depth: int = DEFAULT_MAX_INTROSPECT_DEPTH,
) -> IntrospectionResult:
    """Walk ``fn`` + helpers once (BFS, depth-capped) and return everything
    needed downstream; recurses into helper calls but NOT ``torch.ops.*``,
    traces ``k = make_kernel()`` aliases, peels ``Heuristics``. Never raises."""
    try:
        from triton.runtime.autotuner import Autotuner
        from triton.runtime.jit import JITFunction
    except ImportError:
        logger.warning("Triton not available, introspect_fn returns empty result")
        return IntrospectionResult((), (), (), ())
    try:
        from triton.runtime.autotuner import Heuristics
    except ImportError:
        Heuristics = None  # older triton -- heuristics rejection no-ops

    kernel_types: tuple[type, ...] = (JITFunction, Autotuner)

    # Dedup-on-insert ordered bucket keyed by ``id(obj)`` (or value, for strings).
    class _Bucket:
        __slots__ = ("items", "_seen", "_key")

        def __init__(self, key=id):
            self.items: list[Any] = []
            self._seen: set[Any] = set()
            self._key = key

        def add(self, obj: Any) -> None:
            k = self._key(obj)
            if k not in self._seen:
                self._seen.add(k)
                self.items.append(obj)

    bare = _Bucket()
    user_wrapped = _Bucket()
    heuristics = _Bucket()
    nested_ops = _Bucket(key=lambda s: s)

    for k in extra_triton_kernels or ():
        bare.add(k)

    visited_fns: set[int] = set()

    def _walk(func: Callable[..., Any], depth: int) -> None:
        try:
            f = inspect.unwrap(func)
        except ValueError:
            f = func
        if id(f) in visited_fns:
            return
        if depth > max_depth:
            logger.debug("reached max introspect depth (%s) in introspect_fn", max_depth)
            return
        visited_fns.add(id(f))

        scan = scan_fn_ast(f)
        if scan is None:
            return

        for op_name in scan.nested_op_calls:
            nested_ops.add(op_name)

        namespace = _build_fn_namespace(f)

        def _lookup_dotted(name: str) -> object | None:
            """Resolve ``"a.b.c"`` via successive ``getattr`` on ``namespace[a]``."""
            if "." not in name:
                return namespace.get(name)
            root, *rest = name.split(".")
            if root not in namespace:
                return None
            obj: object = namespace[root]
            for attr in rest:
                try:
                    obj = getattr(obj, attr)
                except AttributeError:
                    return None
            return obj

        def _classify_name(name: str, *, as_bare: bool, visited_names: set[str]) -> None:
            """Classify ``name`` and route it to the right bucket: a kernel
            bucket (``bare`` vs ``user_wrapped`` per ``as_bare``), the
            ``heuristics`` bucket if the resolved object is a Triton
            ``Heuristics``; otherwise recurse into a user helper via
            ``_walk`` or follow ``k = make_kernel()`` assignment chains."""
            if name in visited_names:
                return
            visited_names.add(name)

            obj = _lookup_dotted(name)
            if obj is not None:
                # Raw-object check first so Heuristics is captured before peel.
                if Heuristics is not None and isinstance(obj, Heuristics):
                    heuristics.add(obj)
                kernel = _resolve_kernel(obj, kernel_types)
                if kernel is not None:
                    (bare if as_bare else user_wrapped).add(kernel)
                    return
                if callable(obj):
                    try:
                        unwrapped = inspect.unwrap(obj)
                    except ValueError:
                        unwrapped = obj
                    if hasattr(unwrapped, "__code__"):
                        _walk(unwrapped, depth + 1)
                        return
                logger.debug("failed to resolve %s to a triton kernel", name)
                return

            # Trace local aliases like ``k = make_kernel()``.
            if name in scan.assignments:
                for rhs_expr in scan.assignments[name]:
                    for sub in _extract_names_from_expr(rhs_expr):
                        _classify_name(sub, as_bare=as_bare, visited_names=visited_names)
            else:
                logger.debug("%s not found in namespace or assignments", name)

        # Per-frame visited set: an alias chain shouldn't escape its frame.
        bare_visited: set[str] = set()
        for n in scan.bare_kernel_names:
            _classify_name(n, as_bare=True, visited_names=bare_visited)

        wrapped_visited: set[str] = set()
        for n in scan.wrapped_kernel_names:
            _classify_name(n, as_bare=False, visited_names=wrapped_visited)

        for helper_name in scan.called_helpers:
            helper_obj = namespace.get(helper_name)
            if helper_obj is None or not callable(helper_obj):
                continue
            # ``Heuristics`` is callable (``h(args)`` launches inner kernel);
            # record before the ``__code__`` bail so bare ``h(args)`` form is caught.
            if Heuristics is not None and isinstance(helper_obj, Heuristics):
                heuristics.add(helper_obj)
            if not hasattr(helper_obj, "__code__"):
                continue
            try:
                _walk(helper_obj, depth + 1)
            except Exception:
                logger.debug("failed to analyze called helper %s", helper_name, exc_info=True)

    _walk(fn, 0)

    return IntrospectionResult(
        bare_triton_kernels=tuple(bare.items),
        user_wrapped_kernels=tuple(user_wrapped.items),
        referenced_heuristics=tuple(heuristics.items),
        nested_op_calls=tuple(nested_ops.items),
    )


# ==============================================================================
# SECTION 2 -- Runtime ``wrap_triton`` shadow rewriter
# Inductor needs ``wrap_triton(k)[grid]``; this clones ``fn`` with globals/
# closures rewritten so bare ``k`` resolves to the wrapped version.
# ==============================================================================


def _resolve_kernel(obj: object, kernel_types: tuple[type, ...]) -> Optional[object]:
    """Peel ``obj`` to the underlying ``JITFunction``/``Autotuner`` (returns
    ``obj`` if it already is one, unwrapped ``obj.fn`` for thin wrappers like
    ``Heuristics``, else ``None``). The result feeds ``wrap_triton``."""
    if isinstance(obj, kernel_types):
        return obj
    if callable(obj) and hasattr(obj, "fn"):
        try:
            inner = obj.fn
        except Exception:
            return None
        if isinstance(inner, kernel_types):
            return inner
    return None


def _is_user_helper(obj: object) -> bool:
    """True if ``obj`` is a plain Python function we can recursively rebuild
    (excludes triton kernels, builtins, and torch/triton internals)."""
    if not isinstance(obj, types.FunctionType):
        return False
    code = getattr(obj, "__code__", None)
    if code is None:
        return False
    mod = getattr(obj, "__module__", "") or ""
    if mod.startswith(("torch._library", "triton.")):
        return False
    return True


def rewrite_fn_with_wrap_triton(
    fn: Callable[..., Any], kernels: list[object], excluded_kernel_ids: Optional[set[int]] = None
) -> Callable[..., Any]:
    """Return a clone of ``fn`` whose globals/closures shadow each ``k`` in
    ``kernels`` with ``wrap_triton(k)``; helper functions called from ``fn``
    are rebuilt the same way. Original objects are not modified."""
    if not kernels:
        return fn

    # Triton must be importable here: callers only reach this point after
    # ``introspect_fn`` produced non-empty kernels (which requires triton).
    from triton.runtime.autotuner import Autotuner
    from triton.runtime.jit import JITFunction

    kernel_types: tuple[type, ...] = (JITFunction, Autotuner)

    try:
        from torch.library import wrap_triton
    except ImportError:
        try:
            from torch._library.triton import wrap_triton  # type: ignore
        except ImportError:
            logger.debug("wrap_triton unavailable; skipping rewrite")
            return fn

    # ``id(kernel) -> wrap_triton(kernel)`` (cache so identical kernels share
    # one wrapper; ``wrapped_value_ids`` is the O(1) inverse for ``_maybe_wrap``).
    wrapped_cache: dict[int, Any] = {}
    wrapped_value_ids: set[int] = set()

    def _wrap_once(k: object) -> Any:
        kid = id(k)
        if kid not in wrapped_cache:
            wrapper = wrap_triton(k)
            wrapped_cache[kid] = wrapper
            wrapped_value_ids.add(id(wrapper))
        return wrapped_cache[kid]

    # Pre-populate cache with explicitly detected kernels so identical objects
    # encountered later resolve to the same wrapper.
    target_ids: set[int] = set()
    for k in kernels:
        if isinstance(k, kernel_types):
            _wrap_once(k)
            target_ids.add(id(k))
        else:
            resolved = _resolve_kernel(k, kernel_types)
            if resolved is not None:
                _wrap_once(resolved)
                target_ids.add(id(resolved))

    excluded_kernel_ids = set(excluded_kernel_ids or set())

    def _maybe_wrap(obj: object) -> Optional[Any]:
        """Return ``wrap_triton(obj)`` if it's a target kernel; ``None`` if
        ``obj`` should be left alone (already-wrapped, excluded, or non-kernel)."""
        if id(obj) in wrapped_value_ids:  # already a wrap_triton wrapper
            return None

        resolved = _resolve_kernel(obj, kernel_types)
        if resolved is None:
            return None
        # Caller flagged this kernel as already user-wrapped in source; don't
        # shadow its module-globals ref or ``wrap_triton(wrap_triton(k))`` results.
        if id(resolved) in excluded_kernel_ids:
            return None
        if id(resolved) in target_ids or isinstance(resolved, kernel_types):
            # Wrap any encountered kernel (not just initially-detected ones) so
            # dynamically-resolved kernels in helper globals are also captured.
            return _wrap_once(resolved)
        return None

    rebuilt_fns: dict[int, Callable[..., Any]] = {}

    # All functions in a module share one ``__globals__`` dict; rewrite it once
    # per module (else O(N_helpers * N_globals_per_module) blows up).
    rebuilt_globals: dict[int, dict[str, Any]] = {}

    def _build_new_globals(old_globals: dict[str, Any]) -> dict[str, Any]:
        gid = id(old_globals)
        if gid in rebuilt_globals:
            return rebuilt_globals[gid]
        new_globals: dict[str, Any] = dict(old_globals)
        # Pre-register so reentrant _rebuild (helper back-refs module) terminates.
        rebuilt_globals[gid] = new_globals

        for name, obj in list(old_globals.items()):
            wrapped = _maybe_wrap(obj)
            if wrapped is not None:
                new_globals[name] = wrapped
                continue
            if _is_user_helper(obj):
                try:
                    new_globals[name] = _rebuild(obj)
                except Exception:
                    logger.debug("failed to rebuild helper %s", name, exc_info=True)
        return new_globals

    def _rebuild(f: Callable[..., Any]) -> Callable[..., Any]:
        if not isinstance(f, types.FunctionType):
            return f
        if id(f) in rebuilt_fns:
            return rebuilt_fns[id(f)]

        # Pre-register a placeholder so back-references through globals/closures
        # don't recurse forever; the real new_fn replaces it at the bottom.
        rebuilt_fns[id(f)] = f

        new_globals = _build_new_globals(f.__globals__)

        new_closure: Optional[tuple] = None
        if f.__closure__ is not None:
            new_cells = []
            for cell in f.__closure__:
                try:
                    contents = cell.cell_contents
                except ValueError:
                    new_cells.append(cell)  # empty cell
                    continue

                wrapped = _maybe_wrap(contents)
                if wrapped is not None:
                    new_cells.append(types.CellType(wrapped))
                    continue
                if _is_user_helper(contents) and id(contents) != id(f):
                    try:
                        new_cells.append(types.CellType(_rebuild(contents)))
                        continue
                    except Exception:
                        logger.debug("failed to rebuild closure helper %s", getattr(contents, "__name__", "?"), exc_info=True)
                new_cells.append(cell)
            new_closure = tuple(new_cells)

        new_fn = types.FunctionType(f.__code__, new_globals, f.__name__, f.__defaults__, new_closure)
        # Preserve metadata for infer_schema / register_fake.
        try:
            functools.update_wrapper(new_fn, f, updated=())
        except Exception:
            pass
        new_fn.__kwdefaults__ = f.__kwdefaults__
        new_fn.__module__ = f.__module__
        new_fn.__qualname__ = f.__qualname__
        # Drop __wrapped__: ``inspect.unwrap`` must stop at the rewritten fn,
        # otherwise it walks back to ``f`` whose globals lack wrap_triton.
        try:
            del new_fn.__wrapped__
        except AttributeError:
            pass

        rebuilt_fns[id(f)] = new_fn
        return new_fn

    return _rebuild(fn)
