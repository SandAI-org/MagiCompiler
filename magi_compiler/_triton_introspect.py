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

"""
Triton kernel introspection utilities used by ``magi_register_custom_op``.

This module vendors the (more capable) v2.11.0 implementation of
``torch._library.triton.get_inner_triton_kernels`` and adds a runtime
"globals/closure shadow rewrite" pass (``rewrite_fn_with_wrap_triton``) that
replaces every reference to a detected triton kernel inside a function (and
any helper functions it calls) with ``torch.library.wrap_triton(kernel)``,
without touching the source code.

Only ``get_inner_triton_kernels`` and ``rewrite_fn_with_wrap_triton`` are
intended to be public (used by ``_magi_register_custom_op``).
"""

from __future__ import annotations

import ast
import functools
import inspect
import logging
import textwrap
import types
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


__all__ = [
    "get_inner_triton_kernels",
    "get_bare_triton_kernels",
    "get_referenced_heuristics_kernels",
    "rewrite_fn_with_wrap_triton",
]


# ==============================================================================
# SECTION 1: Triton Kernel AST Introspection
# ------------------------------------------------------------------------------
# Vendored from torch._library.triton (v2.11.0 / pytorch main) and extended.
# Local copy so that ``magi_register_custom_op`` works even if the user is on
# an older PyTorch whose helper is much weaker (e.g. 2.9.x).
#
# These functions parse the AST of the decorated function (and its helpers)
# to discover `_cos_kernel[...]` usage, tracing globals and closure cells.
# ==============================================================================


def _find_triton_kernels_impl(
    fn: Callable[..., Any], only_bare: bool = False
) -> list[object]:
    """Shared driver for :func:`get_inner_triton_kernels` and
    :func:`get_bare_triton_kernels`.

    When ``only_bare`` is True, only kernels invoked via the bare
    ``kernel[grid](...)`` pattern (i.e. without an explicit ``wrap_triton`` /
    ``capture_triton`` call) are returned. Those are the ones we MUST shadow
    in :func:`rewrite_fn_with_wrap_triton`; kernels the user already wrapped
    explicitly should be left alone to avoid double wrapping.
    """

    # prevent infinite recursion
    MAX_RECURSION_DEPTH = 5

    def find_triton_kernels(
        fn: Callable[..., Any],
        visited_fns: set[int] | None = None,
        depth: int = 0,
    ) -> list[object]:
        try:
            from triton.runtime.autotuner import Autotuner
            from triton.runtime.jit import JITFunction
        except ImportError:
            logger.warning("Triton not available, find_triton_kernels = []")
            return []

        # unwrap decorated fn's (e.g., @lru_cache) to get the original
        fn = inspect.unwrap(fn)

        # init visited set and check for cycles/depth limit
        if visited_fns is None:
            visited_fns = set()

        fn_id = id(fn)
        if fn_id in visited_fns:
            return []
        if depth > MAX_RECURSION_DEPTH:
            logger.debug(
                "reached max recursion depth (%s) in find_triton_kernels",
                MAX_RECURSION_DEPTH,
            )
            return []

        visited_fns.add(fn_id)

        try:
            source = inspect.getsource(fn)
        except (OSError, TypeError):
            return []  # Source code not available

        from torch._inductor.utils import IndentedBuffer

        buffer = IndentedBuffer()
        buffer.splice(source, strip=True)
        tree = ast.parse(buffer.getrawvalue())

        # Visitor to collect function calls, assignments, and triton kernels
        class Visitor(ast.NodeVisitor):
            def __init__(self) -> None:
                # Names referenced via wrap_triton(name) / capture_triton(name).
                # The user has *already* wrapped these, so the rewrite pass
                # must NOT shadow them (doing so would produce
                # wrap_triton(wrap_triton(kernel)) at runtime).
                self.wrapped_kernel_names: list[Any] = []
                # Names invoked via bare ``kernel[grid](...)`` syntax.
                # These are the ones we need to wrap_triton-shadow at runtime
                # so the resulting triton_op is traceable.
                self.bare_kernel_names: list[Any] = []
                # track local variable assignments: var_name -> list of RHS expressions
                self.assignments: dict[str, list[ast.expr]] = {}
                # track function calls
                self.called_functions: list[str] = []
                # track return statement expressions
                self.return_exprs: list[ast.expr] = []

            def visit_Assign(self, node: ast.Assign) -> None:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.assignments.setdefault(target.id, []).append(node.value)
                self.generic_visit(node)

            def visit_Return(self, node: ast.Return) -> None:
                if node.value is not None:
                    self.return_exprs.append(node.value)
                self.generic_visit(node)

            def visit_Call(self, node: ast.Call) -> None:
                triton_func_names = ("capture_triton", "wrap_triton")
                if isinstance(node.func, ast.Attribute):
                    attr = node.func
                    if isinstance(attr.value, ast.Attribute):
                        if (
                            isinstance(attr.value.value, ast.Name)
                            and attr.value.value.id == "torch"
                            and attr.value.attr == "_library"
                            and attr.attr in triton_func_names
                        ):
                            if node.args and isinstance(node.args[0], ast.Name):
                                self.wrapped_kernel_names.append(node.args[0].id)
                        elif (
                            isinstance(attr.value.value, ast.Attribute)
                            and isinstance(attr.value.value.value, ast.Name)
                            and attr.value.value.value.id == "torch"
                            and attr.value.value.attr == "ops"
                        ):
                            self.called_functions.append(
                                f"{attr.value.attr}::{attr.attr}"
                            )
                # Catch capture_triton, wrap_triton that's been
                # imported directly
                elif isinstance(node.func, ast.Name):
                    if node.func.id in triton_func_names:
                        if node.args and isinstance(node.args[0], ast.Name):
                            self.wrapped_kernel_names.append(node.args[0].id)
                    else:
                        # track regular function calls for recursive analysis
                        self.called_functions.append(node.func.id)

                # Also detect bare triton-style launches: ``kernel[grid](args)``.
                # The decorated function is allowed to invoke the kernel without
                # explicit wrap_triton(...); we pick up the kernel name here so
                # downstream code can wrap it. We only look at Subscript whose
                # value is a plain Name (the most common pattern); subscripted
                # attributes (e.g. ``self.kernel[grid](...)``) need the
                # ``extra_triton_kernels`` escape hatch.
                if isinstance(node.func, ast.Subscript) and isinstance(
                    node.func.value, ast.Name
                ):
                    self.bare_kernel_names.append(node.func.value.id)

                self.generic_visit(node)

        collector = Visitor()
        collector.visit(tree)

        def extract_names_from_expr(expr: ast.expr) -> list[str]:
            """Extract all Name references from an AST expression."""
            names: list[str] = []

            class NameExtractor(ast.NodeVisitor):
                def visit_Name(self, node: ast.Name) -> None:
                    names.append(node.id)

                def visit_Call(self, node: ast.Call) -> None:
                    # for function calls, visit the function and all args
                    self.generic_visit(node)

            NameExtractor().visit(expr)
            return names

        def resolve_to_kernel(obj: object) -> object | None:
            """Check if obj is a triton kernel or wrapper and return the kernel."""
            if isinstance(obj, (JITFunction, Autotuner)):
                return obj
            # handle wrappers that have a .fn attribute pointing to JITFunction
            if callable(obj) and hasattr(obj, "fn"):
                inner = obj.fn
                if isinstance(inner, JITFunction):
                    return inner
            return None

        def build_namespace(func_obj: object) -> dict[str, Any]:
            """Build a combined namespace from a function's globals and closures."""
            # unwrap decorated fns (e.g., @lru_cache)
            if callable(func_obj):
                try:
                    func_obj = inspect.unwrap(func_obj)
                except ValueError:
                    pass
            if not callable(func_obj) or not hasattr(func_obj, "__code__"):
                return {}
            try:
                func_closure_vars = inspect.getclosurevars(func_obj)
            except Exception:
                func_closure_vars = None
            namespace: dict[str, Any] = {}
            if func_closure_vars is not None:
                namespace.update(func_closure_vars.builtins)
                namespace.update(func_closure_vars.globals)
                namespace.update(func_closure_vars.nonlocals)
            if hasattr(func_obj, "__globals__"):
                namespace.update(func_obj.__globals__)
            return namespace

        all_names = build_namespace(fn)

        def resolve_names_to_kernels(
            names: list[str],
            namespace: dict[str, Any],
            assignments: dict[str, list[ast.expr]] | None = None,
            visited: set[str] | None = None,
        ) -> list[object]:
            """
            Resolve a list of names to triton kernels using the given namespace.
            """
            if visited is None:
                visited = set()

            results: list[object] = []
            for name in names:
                if name in visited:
                    continue
                visited.add(name)

                if name in namespace:
                    obj = namespace[name]
                    kernel = resolve_to_kernel(obj)
                    if kernel is not None:
                        results.append(kernel)
                        continue
                    # recurse into callable objects (factory fn's),
                    # unwrapping decorators if applicable
                    if callable(obj):
                        try:
                            unwrapped = inspect.unwrap(obj)
                        except ValueError:
                            unwrapped = obj
                        if hasattr(unwrapped, "__code__"):
                            nested = find_triton_kernels(
                                unwrapped,
                                visited_fns,
                                depth + 1,
                            )
                            if nested:
                                results.extend(nested)
                                continue
                    logger.debug("failed to resolve %s to a triton kernel", name)
                elif assignments is not None and name in assignments:
                    # trace through local assignments
                    for rhs_expr in assignments[name]:
                        referenced = extract_names_from_expr(rhs_expr)
                        traced = resolve_names_to_kernels(
                            referenced, namespace, assignments, visited
                        )
                        results.extend(traced)
                else:
                    logger.debug("%s not found in namespace or assignments", name)

            return results

        # resolve kernel names, tracing through local variables if needed
        resolved: list[object] = []
        seen_ids: set[int] = set()

        if only_bare:
            names_to_resolve: list[str] = list(collector.bare_kernel_names)
        else:
            names_to_resolve = list(collector.bare_kernel_names) + list(
                collector.wrapped_kernel_names
            )
            for expr in collector.return_exprs:
                names_to_resolve.extend(extract_names_from_expr(expr))

        for name in names_to_resolve:
            traced_objects = resolve_names_to_kernels(
                [name], all_names, collector.assignments
            )
            for obj in traced_objects:
                obj_id = id(obj)
                if obj_id not in seen_ids:
                    seen_ids.add(obj_id)
                    resolved.append(obj)

        for func_name in collector.called_functions:
            func_obj = all_names.get(func_name)

            if func_obj is None:
                try:
                    from torch._library.custom_ops import OPDEFS

                    if func_name in OPDEFS:
                        func_obj = OPDEFS[func_name]._abstract_fn
                except Exception:
                    pass

            # skip if not a callable or if it's a triton kernel itself
            if func_obj is None or not callable(func_obj):
                continue

            # skip built-in functions and C extensions (they can't contain triton kernels)
            if not hasattr(func_obj, "__code__"):
                continue

            try:
                nested_kernels = find_triton_kernels(func_obj, visited_fns, depth + 1)
                for kernel in nested_kernels:
                    kernel_id = id(kernel)
                    if kernel_id not in seen_ids:
                        seen_ids.add(kernel_id)
                        resolved.append(kernel)
            except Exception:
                logger.debug(
                    "failed to analyze called function %s", func_name, exc_info=True
                )

        return resolved

    return find_triton_kernels(fn)


def get_user_wrapped_triton_kernels(fn: Callable[..., Any]) -> list[object]:
    """Return triton kernels that ``fn``'s source explicitly wraps in a
    ``wrap_triton(kernel)`` / ``capture_triton(kernel)`` call.

    These are the kernels the user has *already* taken responsibility for
    wrapping; :func:`rewrite_fn_with_wrap_triton` must not rewrite their
    module-globals references (doing so would produce
    ``wrap_triton(wrap_triton(kernel))`` at runtime). Exposed so the caller
    can build an ``excluded_kernel_ids`` set to pass into the rewriter.
    """
    return _find_user_wrapped_kernels_impl(fn)


def _find_user_wrapped_kernels_impl(fn: Callable[..., Any]) -> list[object]:
    triton_types_pair = _try_import_triton_types()
    if triton_types_pair is None:
        return []
    kernel_types: tuple[type, ...] = triton_types_pair

    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        return []
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return []

    wrapped_names: list[str] = []
    triton_func_names = ("capture_triton", "wrap_triton")

    class _Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Name) and node.func.id in triton_func_names:
                if node.args and isinstance(node.args[0], ast.Name):
                    wrapped_names.append(node.args[0].id)
            elif isinstance(node.func, ast.Attribute):
                attr = node.func
                if (
                    isinstance(attr.value, ast.Attribute)
                    and isinstance(attr.value.value, ast.Name)
                    and attr.value.value.id == "torch"
                    and attr.value.attr == "_library"
                    and attr.attr in triton_func_names
                ):
                    if node.args and isinstance(node.args[0], ast.Name):
                        wrapped_names.append(node.args[0].id)
            self.generic_visit(node)

    _Visitor().visit(tree)
    if not wrapped_names:
        return []

    namespace: dict[str, Any] = {}
    namespace.update(getattr(fn, "__globals__", {}) or {})
    if fn.__closure__ is not None:
        try:
            for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
                try:
                    namespace[name] = cell.cell_contents
                except ValueError:
                    pass
        except Exception:
            pass

    out: list[object] = []
    seen: set[int] = set()
    for n in wrapped_names:
        obj = namespace.get(n)
        if obj is None:
            continue
        resolved = (
            obj if isinstance(obj, kernel_types) else _resolve_kernel(obj, kernel_types)
        )
        if resolved is None:
            continue
        if id(resolved) in seen:
            continue
        seen.add(id(resolved))
        out.append(resolved)
    return out


def get_inner_triton_kernels(fn: Callable[..., Any]) -> list[object]:
    """
    Inspect the source of an arbitrary callable, and grab all of the triton
    kernels that are wrapped inside of it.

    Traces local variable assignments, follows ``return`` expressions, and
    recursively descends into helper functions called from ``fn`` so that
    kernels hidden behind launcher wrappers are still detected.

    Returns an empty list if triton is not installed or no kernels are found.
    Best-effort: deeply recursive call graphs (>5 levels) are not followed.
    """
    return _find_triton_kernels_impl(fn, only_bare=False)


def get_referenced_heuristics_kernels(fn: Callable[..., Any]) -> list[object]:
    """Return ``triton.runtime.autotuner.Heuristics`` instances that ``fn``
    (or any helper function it transitively calls) references via a name in
    its globals/closure.

    Designed specifically to surface ``@triton.heuristics`` placed at the
    *top* of the decorator stack, which :func:`get_inner_triton_kernels`
    deliberately peels through to expose the inner ``JITFunction`` (the only
    type that ``wrap_triton`` accepts together with ``Autotuner``). Without
    this helper the top-level ``Heuristics`` would be silently dropped
    (registration falls back to plain custom_op, no Inductor speedup) or
    explode later in ``wrap_triton`` with an opaque error.

    Returns ``[]`` if triton is not installed or no such kernels are found.
    """
    try:
        from triton.runtime.autotuner import Heuristics
    except ImportError:
        return []

    MAX_DEPTH = 5
    found: list[object] = []
    seen_objs: set[int] = set()
    visited_fns: set[int] = set()

    def _maybe_add(obj: Any) -> None:
        if isinstance(obj, Heuristics) and id(obj) not in seen_objs:
            seen_objs.add(id(obj))
            found.append(obj)

    def _walk(func: Callable[..., Any], depth: int) -> None:
        try:
            f = inspect.unwrap(func)
        except ValueError:
            f = func
        if id(f) in visited_fns or depth > MAX_DEPTH:
            return
        visited_fns.add(id(f))

        try:
            source = inspect.getsource(f)
        except (OSError, TypeError):
            return
        try:
            tree = ast.parse(inspect.cleandoc(source))
        except SyntaxError:
            return

        names: set[str] = set()

        class NameCollector(ast.NodeVisitor):
            def visit_Subscript(self, node: ast.Subscript) -> None:
                if isinstance(node.value, ast.Name):
                    names.add(node.value.id)
                self.generic_visit(node)

            def visit_Call(self, node: ast.Call) -> None:
                if isinstance(node.func, ast.Name):
                    names.add(node.func.id)
                self.generic_visit(node)

        NameCollector().visit(tree)

        # Build the same combined namespace used by find_triton_kernels.
        ns: dict[str, Any] = {}
        try:
            cv = inspect.getclosurevars(f)
            ns.update(cv.builtins)
            ns.update(cv.globals)
            ns.update(cv.nonlocals)
        except Exception:
            pass
        if hasattr(f, "__globals__"):
            ns.update(f.__globals__)

        for n in names:
            obj = ns.get(n)
            if obj is None:
                continue
            _maybe_add(obj)
            # Recurse into helper python functions so launchers nested one
            # level deep also get inspected.
            if isinstance(obj, types.FunctionType):
                _walk(obj, depth + 1)

    _walk(fn, 0)
    return found


def get_bare_triton_kernels(fn: Callable[..., Any]) -> list[object]:
    """
    Like :func:`get_inner_triton_kernels`, but only returns kernels invoked
    via the bare ``kernel[grid](...)`` syntax (NOT via an explicit
    ``wrap_triton`` / ``capture_triton`` call).

    These are the kernels that :func:`rewrite_fn_with_wrap_triton` actually
    needs to shadow at runtime. Skipping the already-wrapped ones avoids
    producing a ``wrap_triton(wrap_triton(kernel))`` at runtime, which raises
    ``RuntimeError`` from ``torch.library.wrap_triton``.
    """
    return _find_triton_kernels_impl(fn, only_bare=True)


# ==============================================================================
# SECTION 2: Runtime "wrap_triton" Shadow Rewriter
# ------------------------------------------------------------------------------
# In order for Inductor to trace into bare `kernel[grid]` calls, they must be
# `torch.library.wrap_triton(kernel)[grid]`.
# Instead of forcing the user to rewrite their code, this pass builds a clone of
# the user's function where `__globals__` maps the kernel name to a wrapped
# version.
# ==============================================================================


def _try_import_triton_types() -> Optional[tuple[type, type]]:
    try:
        from triton.runtime.autotuner import Autotuner
        from triton.runtime.jit import JITFunction

        return (JITFunction, Autotuner)
    except ImportError:
        return None


def _resolve_kernel(obj: object, kernel_types: tuple[type, ...]) -> Optional[object]:
    """Return the underlying triton kernel object if ``obj`` is one (or wraps one)."""
    if isinstance(obj, kernel_types):
        return obj
    if callable(obj) and hasattr(obj, "fn"):
        try:
            inner = obj.fn
        except Exception:
            return None
        if isinstance(inner, kernel_types):
            return obj
    return None


def _is_user_helper(obj: object) -> bool:
    """True if ``obj`` is a plain Python function we can recursively rebuild.

    Excludes triton kernels (JITFunction / Autotuner) themselves, builtins,
    and C-extension callables.
    """
    if not isinstance(obj, types.FunctionType):
        return False
    code = getattr(obj, "__code__", None)
    if code is None:
        return False
    # skip torch.library helpers and triton internals to avoid surprises
    mod = getattr(obj, "__module__", "") or ""
    if mod.startswith(("torch._library", "triton.")):
        return False
    return True


def rewrite_fn_with_wrap_triton(
    fn: Callable[..., Any],
    kernels: list[object],
    excluded_kernel_ids: Optional[set[int]] = None,
) -> Callable[..., Any]:
    """
    Return a copy of ``fn`` whose globals / closures are shadowed so that every
    reference to a kernel in ``kernels`` resolves to
    ``torch.library.wrap_triton(kernel)``. Helper functions called from ``fn``
    are also rebuilt the same way, so kernels referenced from launcher helpers
    are wrapped too.

    The original ``fn`` (and the kernel objects) are not modified. This works
    for kernels referenced via globals, closures, or factory functions, as
    long as the reference can be reached through the function's
    ``__globals__`` / ``__closure__``.

    If ``kernels`` is empty or triton is not installed, returns ``fn`` unchanged.
    """
    if not kernels:
        return fn

    triton_types_pair = _try_import_triton_types()
    if triton_types_pair is None:
        return fn
    kernel_types: tuple[type, ...] = triton_types_pair

    try:
        from torch.library import wrap_triton
    except ImportError:
        try:
            from torch._library.triton import wrap_triton  # type: ignore
        except ImportError:
            logger.debug("wrap_triton unavailable; skipping rewrite")
            return fn

    # Map id(kernel_object) -> wrap_triton(kernel_object). Cached so the same
    # kernel passed through multiple references shares one wrapper, and so we
    # never wrap_triton(wrap_triton(k)).
    wrapped_cache: dict[int, Any] = {}

    def _wrap_once(k: object) -> Any:
        kid = id(k)
        if kid not in wrapped_cache:
            wrapped_cache[kid] = wrap_triton(k)
        return wrapped_cache[kid]

    # Pre-populate cache with the explicitly detected kernels so identical
    # kernel objects encountered later resolve to the same wrapper.
    target_ids: set[int] = set()
    for k in kernels:
        if isinstance(k, kernel_types):
            _wrap_once(k)
            target_ids.add(id(k))
        else:
            # Allow callers to pass wrappers (e.g. objects with .fn) too.
            resolved = _resolve_kernel(k, kernel_types)
            if resolved is not None:
                _wrap_once(resolved)
                target_ids.add(id(resolved))

    excluded_kernel_ids = set(excluded_kernel_ids or set())

    def _maybe_wrap(obj: object) -> Optional[Any]:
        """If ``obj`` is one of our target kernels, return the wrap_triton wrapper.

        Returns ``None`` if ``obj`` should be left alone.
        """
        # Don't double-wrap something that already came out of wrap_triton.
        # ``wrap_triton`` returns a TraceableTritonKernelWrapper; importing
        # that class is brittle across torch versions, so we identity-check
        # against the cache values instead.
        if id(obj) in {id(v) for v in wrapped_cache.values()}:
            return None

        resolved = _resolve_kernel(obj, kernel_types)
        if resolved is None:
            return None
        # Caller has explicitly told us this kernel is already user-wrapped
        # (e.g. via ``wrap_triton(kernel)`` in the op body); don't shadow its
        # module-globals reference, otherwise the explicit ``wrap_triton(k)``
        # in the source becomes ``wrap_triton(wrap_triton(k))`` at runtime.
        if id(resolved) in excluded_kernel_ids:
            return None
        if id(resolved) in target_ids or isinstance(resolved, kernel_types):
            # Always wrap any encountered triton kernel (not just the
            # initially-detected ones) so dynamically-resolved kernels in
            # helper globals are also captured.
            return _wrap_once(resolved)
        return None

    rebuilt_fns: dict[int, Callable[..., Any]] = {}

    # Per-module globals_dict cache: every function defined in the same module
    # shares the same ``__globals__`` dict, so we only need to walk + rewrite
    # that dict ONCE per module instead of once per helper. Without this cache
    # the rebuilder is O(N_helpers * N_globals_per_module), which becomes
    # catastrophic (multi-second per registration) for any module with many
    # top-level functions / fixtures.
    rebuilt_globals: dict[int, dict[str, Any]] = {}

    def _build_new_globals(old_globals: dict[str, Any]) -> dict[str, Any]:
        gid = id(old_globals)
        if gid in rebuilt_globals:
            return rebuilt_globals[gid]
        new_globals: dict[str, Any] = dict(old_globals)
        # Register the partially-populated dict immediately so any reentrant
        # _rebuild call (e.g. helper -> back-references the module) finds it
        # and short-circuits without infinite recursion.
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

        # Pre-register a placeholder so cycles (helper that references back
        # into ``f`` through globals or closures) don't recurse forever.
        # We swap in the real new_fn at the bottom of this function.
        rebuilt_fns[id(f)] = f

        new_globals = _build_new_globals(f.__globals__)

        # Rebuild closure cells.
        new_closure: Optional[tuple] = None
        if f.__closure__ is not None:
            new_cells = []
            for cell in f.__closure__:
                try:
                    contents = cell.cell_contents
                except ValueError:
                    # empty cell
                    new_cells.append(cell)
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
                        logger.debug(
                            "failed to rebuild closure helper %s",
                            getattr(contents, "__name__", "?"),
                            exc_info=True,
                        )
                new_cells.append(cell)
            new_closure = tuple(new_cells)

        new_fn = types.FunctionType(
            f.__code__,
            new_globals,
            f.__name__,
            f.__defaults__,
            new_closure,
        )
        # Preserve introspectable metadata so that downstream tooling
        # (infer_schema, register_fake, etc.) continues to work.
        try:
            functools.update_wrapper(new_fn, f, updated=())
        except Exception:
            pass
        new_fn.__kwdefaults__ = f.__kwdefaults__
        new_fn.__module__ = f.__module__
        new_fn.__qualname__ = f.__qualname__
        # update_wrapper sets __wrapped__ which makes inspect.unwrap follow
        # back to the original; that's actually undesirable here because the
        # original function's globals do NOT have wrap_triton applied. Strip
        # it so inspect.signature / unwrap stop at the rewritten function.
        try:
            del new_fn.__wrapped__
        except AttributeError:
            pass

        rebuilt_fns[id(f)] = new_fn
        return new_fn

    return _rebuild(fn)
