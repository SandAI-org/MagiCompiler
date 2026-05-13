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

"""Magi custom-op registration: dataclass-aware wrapper around ``torch.library``.

This module implements ``magi_register_custom_op`` -- a decorator that takes a
plain Python function (possibly with frozen-dataclass parameters,
``Literal[str, ...]`` / string-Enum annotations, or other Python-rich
signatures that ``torch.library.infer_schema`` cannot consume) and registers
it as a real custom op while letting the user keep calling it with their
original, ergonomic signature.


Part A. Registration-time pipeline -- the four slots
====================================================

When ``@magi_register_custom_op(...)`` is applied to a user function, up to
four named slots are produced. Each slot is a concrete callable object.

    slot 0 -- fn
        The user's original function. Always present.

    slot 1 -- lowered_fn
        A thin wrapper around ``fn`` whose ``__signature__`` /
        ``__annotations__`` have been *lowered* (Literal/Enum -> str,
        unsupported defaults scrubbed, dataclasses flattened into primitive
        leaves) so that ``torch.library.infer_schema`` accepts it.
        Skipped when ``fn``'s signature is already schema-compatible.

    slot 2 -- torch_registered_op
        The ``OpOverload`` returned by ``torch.library.custom_op`` /
        ``register_fake`` after registering whichever of ``fn`` /
        ``lowered_fn`` reached this point. Always present.

    slot 3 -- magi_exposed_op
        A magi-level Python wrapper around ``torch_registered_op`` that
        preserves the user's ORIGINAL (dataclass-bearing) calling
        convention. At call time it flattens incoming args via the static
        ``param_mapping_tree`` and dispatches into slot 2. Only created
        on the dataclass-flatten path.

The naming is a deliberate dual: ``torch_registered_op`` is *registered
into* torch.library's dispatcher; ``magi_exposed_op`` is *exposed out of*
magi to user code.


Part B. Runtime paths -- the three pipelines
============================================

Three pipelines are possible; the decorator returns whichever object sits
at the end of the path:

    1. simple                  fn -> torch_registered_op
       Returned: ``torch._ops.OpOverload`` (slot 2).
       Runtime: zero magi-level overhead -- straight into torch.library's
       dispatcher.

    2. sig-only-rewrite        fn -> lowered_fn -> torch_registered_op
       Returned: ``torch._ops.OpOverload`` (slot 2).
       Runtime: same as simple -- ``lowered_fn`` is a transparent
       forwarding shim (the rewrite is registration-time only).

    3. dataclass-flatten       fn -> lowered_fn -> torch_registered_op
                                                -> magi_exposed_op
       Returned: a Python callable carrying the
       ``_magi_torch_registered_op`` attribute (slot 3).
       Runtime forward (per call):
         user code calls magi_exposed_op(x, cfg=...)
           -> _flatten_call_args                (original kwargs -> flat tuple)
              -> _flatten_value_into            (DFS over param_mapping_tree)
           -> torch_registered_op(*flat)        (slot 2 -- enters dispatcher)
              -> lowered_fn(*flat)              (slot 1 -- still in lowered shape)
                 -> _reassemble_kwargs          (flat tuple -> original kwargs)
                    -> _build_value_from_node   (rebuilds dataclass instances)
                 -> fn(**original_kwargs)       (slot 0 -- user code finally sees
                                                 its original dataclass-bearing
                                                 signature)
       Runtime backward (when backward_fn is supplied):
         autograd calls _bridged_backward(ctx, *grads)
           -> user_backward(ctx, *grads)        (returns one grad per ORIGINAL
                                                 input, possibly a dataclass-shaped
                                                 grad object)
           -> _flatten_grads                    (original grads -> flat grads)
              -> _flatten_grad_into             (DFS over param_mapping_tree)

You can tell at runtime which pipeline an op went through by inspecting
the decorator's return value: an ``OpOverload`` means simple/sig-rewrite;
a Python callable carrying ``_magi_torch_registered_op`` means
dataclass-flatten.


File layout
===========

    -- registration-time helpers (executed once) --
    1. Validate the user's fn signature
    2. Resolve types & sanitise defaults for infer_schema
    3. Build & query the param mapping tree                (used by sec 4 and sec 7)
    4. Lower fn's signature                                (produces slot 1)
    5. Synthesise the meta/fake function                   (input to slot 2)
    6. Register the op                                     (produces slot 2)

    -- runtime helpers (executed on every call) --
    7. Runtime bridge: flatten / unflatten on every call

    -- main pipeline --
    8. The decorator: orchestrates sec 1-6 and builds the runtime
       closures from sec 7 (produces slot 3 on the flatten path)
"""

import dataclasses
import functools
import inspect
from typing import Any, Callable, get_args, get_origin

import torch
import torch.utils._pytree as pytree

from .config import get_compile_config
from .utils.logger import magi_logger

# ==============================================================================
# 1. Validate the user's fn signature
# ------------------------------------------------------------------------------
# Predicate + assert helpers that reject `fn` signatures torch.library cannot
# consume, each raising a clear `TypeError` instead of the opaque error that
# would otherwise surface deep inside `infer_schema`. Called from
# `_lower_op_signature` (sec 4) and `_build_dataclass_sub_mapping_tree` (sec 3).
# ==============================================================================


def _is_frozen_dataclass(tp) -> bool:
    """Return True if ``tp`` is a frozen dataclass type."""
    return (
        isinstance(tp, type)
        and dataclasses.is_dataclass(tp)
        and getattr(tp, "__dataclass_params__", None) is not None
        and tp.__dataclass_params__.frozen
    )


def _assert_not_unsupported_container(tp, *, where: str) -> None:
    """Reject ``tuple[...]`` / ``dict[...]`` annotations (schema only models ``list``)."""
    origin = get_origin(tp)
    if origin is tuple:
        raise TypeError(
            f"@magi_register_custom_op: {where} has tuple annotation {tp!r}; "
            f"use ``list[...]`` or split into separate fields."
        )
    if origin is dict:
        raise TypeError(
            f"@magi_register_custom_op: {where} has dict-typed annotation {tp!r}; " f"promote the values to explicit fields."
        )


def _assert_not_dataclass_return(tp, *, fn_name: str) -> None:
    """Reject dataclass return annotations (schema only returns Tensor / tuple / list / None)."""
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        raise TypeError(
            f"@magi_register_custom_op: {fn_name!r} returns dataclass "
            f"{tp.__name__!r}; only Tensor / tuple[Tensor, ...] / list[Tensor] "
            f"are supported -- destructure into a tuple at the op boundary."
        )


def _assert_not_mutable_dataclass(tp, *, where: str) -> None:
    """Reject non-frozen dataclasses (schema needs hashable, stable inputs)."""
    if (
        isinstance(tp, type)
        and dataclasses.is_dataclass(tp)
        and getattr(tp, "__dataclass_params__", None) is not None
        and not tp.__dataclass_params__.frozen
    ):
        raise TypeError(
            f"@magi_register_custom_op: {where} has mutable dataclass "
            f"{tp.__name__!r}; add ``frozen=True`` to {tp.__name__}."
        )


def _assert_has_annotation(annotation, *, where: str) -> None:
    """Require an annotation on every parameter / field / return value (needed
    to recognise dataclasses and to feed ``infer_schema``)."""
    if annotation is inspect.Parameter.empty or annotation is inspect.Signature.empty:
        raise TypeError(
            f"@magi_register_custom_op: {where} has no type annotation " f"(e.g. ``x: torch.Tensor`` or ``cfg: MyFrozenCfg``)."
        )


def _assert_no_var_args(param: inspect.Parameter, *, fn_name: str) -> None:
    """Reject ``*args`` / ``**kwargs`` (op schemas are positional-or-keyword only)."""
    if param.kind is inspect.Parameter.VAR_POSITIONAL:
        raise TypeError(
            f"@magi_register_custom_op: {fn_name!r} declares ``*{param.name}``; "
            f"variadics aren't supported -- replace with explicit annotated parameters."
        )
    if param.kind is inspect.Parameter.VAR_KEYWORD:
        raise TypeError(
            f"@magi_register_custom_op: {fn_name!r} declares ``**{param.name}``; "
            f"variadics aren't supported -- replace with explicit annotated parameters."
        )


def _assert_resolved_field_type(f_type, *, where: str) -> None:
    """Reject unresolved string annotations -- typically a local class combined
    with stringified annotations that ``get_type_hints`` could not eval."""
    if isinstance(f_type, str):
        raise TypeError(
            f"@magi_register_custom_op: {where} has unresolved string "
            f"annotation {f_type!r}; move the type to module scope so "
            f"``get_type_hints`` can resolve it."
        )


# ==============================================================================
# 2. Resolve types & sanitise defaults for infer_schema
# ------------------------------------------------------------------------------
# Resolve stringified annotations to real types, downgrade Literal/string-Enum
# to `str`, and scrub defaults that `infer_schema` cannot render. Called by
# `_lower_op_signature` (sec 4) and `_build_dataclass_sub_mapping_tree` (sec 3).
# ==============================================================================


def _resolve_annotations(fn: Callable) -> dict[str, Any]:
    """Return ``fn``'s annotations as real types, resolving stringified ones.

    Falls back to per-annotation eval against ``globals + closure nonlocals``
    when ``get_type_hints`` can't resolve atomically (typical for functions
    defined inside another function whose annotations reference enclosing
    names).
    """
    import typing

    try:
        return typing.get_type_hints(fn)
    except Exception:
        pass

    # Build an eval namespace from module globals + closure nonlocals.
    # ``__globals__`` covers the common case; closure vars from
    # ``getclosurevars`` cover annotations that name enclosing locals.
    fn_globals = getattr(fn, "__globals__", {}) or {}
    namespace: dict[str, Any] = dict(fn_globals)
    try:
        cv = inspect.getclosurevars(fn)
        namespace.update(cv.builtins)
        namespace.update(cv.globals)
        namespace.update(cv.nonlocals)
    except Exception as e:
        magi_logger.debug(
            "inspect.getclosurevars(%s) failed: %s; falling back to module globals only",
            getattr(fn, "__qualname__", fn),
            e,
            rank="all",
        )

    anns: dict[str, Any] = {}
    raw = getattr(fn, "__annotations__", {}) or {}
    for k, v in raw.items():
        if isinstance(v, str):
            try:
                anns[k] = eval(v, namespace, None)
            except Exception:
                anns[k] = v
        else:
            anns[k] = v
    return anns


def _resolve_dataclass_field_types(cls: type) -> dict[str, Any]:
    """Return ``cls``'s field name -> resolved type (best-effort)."""
    import typing as _typing

    try:
        return _typing.get_type_hints(cls)
    except Exception:
        return {f.name: f.type for f in dataclasses.fields(cls)}


def _maybe_downgrade_literal_or_enum(annotation, *, where: str):
    """Collapse ``Literal[str, ...]`` and string-Enum annotations to plain ``str``.

    Lossless because the op body still receives the original string value.
    Mixed/numeric Literals and non-string Enums raise (no safe downgrade).
    """
    import enum
    import typing

    _LITERAL_STRING_DOWNGRADE_HINT = (
        "Use ``str`` and validate the value inside the op body, e.g. " "``assert mode in ('a', 'b')``."
    )
    origin = get_origin(annotation)
    if origin is typing.Literal:
        choices = get_args(annotation)
        if choices and all(isinstance(c, str) for c in choices):
            return str
        raise TypeError(
            f"@magi_register_custom_op: {where} has Literal {annotation!r}; "
            f"only ``Literal[str, ...]`` is auto-downgraded. "
            f"{_LITERAL_STRING_DOWNGRADE_HINT}"
        )
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        members = list(annotation)
        if members and all(isinstance(m.value, str) for m in members):
            return str
        raise TypeError(
            f"@magi_register_custom_op: {where} has non-string Enum "
            f"{annotation.__name__!r}. {_LITERAL_STRING_DOWNGRADE_HINT}"
        )
    return annotation


_SCHEMA_DEFAULT_TYPES: tuple[type, ...] = (int, float, bool, str, torch.device, torch.dtype)


def _schema_compatible_param_default(default: Any) -> Any:
    """Scrub a top-level parameter default that ``infer_schema`` cannot render.

    Same rules as :func:`_schema_compatible_default`, but for raw values
    rather than ``dataclasses.Field``.
    """
    if default is inspect.Parameter.empty:
        return inspect.Parameter.empty
    if default is None or isinstance(default, _SCHEMA_DEFAULT_TYPES):
        return default
    return inspect.Parameter.empty


def _schema_compatible_default(f: "dataclasses.Field") -> Any:
    """Lowered default for dataclass field ``f``: keep ``None`` / int / float /
    bool / str / torch.device / torch.dtype; drop everything else (the user-
    constructed dataclass instance still carries the real default at runtime)."""
    if f.default is not dataclasses.MISSING:
        d = f.default
        if d is None or isinstance(d, _SCHEMA_DEFAULT_TYPES):
            return d
        return inspect.Parameter.empty
    if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        try:
            d = f.default_factory()
        except Exception:
            return inspect.Parameter.empty
        if d is None or isinstance(d, _SCHEMA_DEFAULT_TYPES):
            return d
        return inspect.Parameter.empty
    return inspect.Parameter.empty


# ==============================================================================
# 3. Build & query the param mapping tree
# ------------------------------------------------------------------------------
# The `param_mapping_tree` is the single source of truth bridging the user's
# (possibly nested-dataclass) signature and the lowered primitive signature.
# Built once at registration time and consumed twice afterwards: by
# `_expand_mutates_args` (statically, sec 6) and by the runtime bridge (sec 7).
# ==============================================================================

_DATACLASS_PYTREE_REGISTERED: set[type] = set()


def _register_dataclass_pytree(cls: type) -> None:
    """Register ``cls`` as a pytree node (idempotent) so Dynamo / AOTAutograd
    can flatten and unflatten dataclass instances during tracing."""
    if cls in _DATACLASS_PYTREE_REGISTERED:
        return

    field_names = tuple(f.name for f in dataclasses.fields(cls))

    def _flatten(obj):
        return [getattr(obj, n) for n in field_names], field_names

    def _unflatten(values, ctx):
        return cls(**dict(zip(ctx, values)))

    try:
        pytree.register_pytree_node(cls, _flatten, _unflatten)
    except ValueError:
        # Already registered elsewhere (e.g. user code).
        pass
    _DATACLASS_PYTREE_REGISTERED.add(cls)


def _build_dataclass_sub_mapping_tree(cls: type, attr_name: str, flat_prefix: str) -> tuple[tuple, list[inspect.Parameter]]:
    """Recursively expand a frozen-dataclass type into one ``param_mapping_tree``
    subtree plus its flat list of leaf ``inspect.Parameter`` objects (DFS order).

    ``attr_name`` is the field name on the parent dataclass (or the parameter
    name on ``fn`` for a top-level dataclass arg). ``flat_prefix`` builds the
    leaf parameter name; e.g. ``cfg: OuterCfg(inner: InnerCfg(val: float))``
    becomes a lowered leaf parameter ``cfg__inner__val``.
    """
    _register_dataclass_pytree(cls)

    field_types = _resolve_dataclass_field_types(cls)
    children: list[tuple] = []
    lowered_params: list[inspect.Parameter] = []

    for f in dataclasses.fields(cls):
        f_type = field_types.get(f.name, f.type)
        child_flat_name = f"{flat_prefix}__{f.name}"
        _assert_has_annotation(f_type, where=f"field {cls.__name__}.{f.name}")
        _assert_resolved_field_type(f_type, where=f"field {cls.__name__}.{f.name}")
        _assert_not_mutable_dataclass(f_type, where=f"field {cls.__name__}.{f.name}")
        if _is_frozen_dataclass(f_type):
            sub_node, sub_params = _build_dataclass_sub_mapping_tree(f_type, attr_name=f.name, flat_prefix=child_flat_name)
            children.append(sub_node)
            lowered_params.extend(sub_params)
        else:
            _assert_not_unsupported_container(f_type, where=f"field {cls.__name__}.{f.name}")
            f_type = _maybe_downgrade_literal_or_enum(f_type, where=f"field {cls.__name__}.{f.name}")
            children.append(("primitive", f.name, child_flat_name, None))
            lowered_params.append(
                inspect.Parameter(
                    child_flat_name,
                    # POSITIONAL_OR_KEYWORD: torch.library.custom_op does not yet
                    # support kwarg-only Tensor arguments.
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=_schema_compatible_default(f),
                    annotation=f_type,
                )
            )

    return ("dataclass", attr_name, cls, children), lowered_params


def _count_leaves(node: tuple) -> int:
    """Number of lowered parameter slots a ``param_mapping_tree`` ``node`` occupies."""
    if node[0] == "primitive":
        return 1
    return sum(_count_leaves(c) for c in node[3])


def _collect_tensor_leaf_lowered_names(node: tuple) -> list[str]:
    """Lowered names of every leaf under ``node``. Used to expand a dataclass
    parameter referenced in ``mutates_args`` (torch.library does its own
    Tensor-type validation, so over-specifying non-Tensor leaves is fine)."""
    if node[0] == "primitive":
        _, _attr, lowered_name, _ = node
        return [lowered_name]
    out: list[str] = []
    for child in node[3]:
        out.extend(_collect_tensor_leaf_lowered_names(child))
    return out


def _expand_mutates_args(mutates_args: tuple[str, ...] | list[str], param_mapping_tree: list[tuple]) -> tuple[str, ...]:
    """Translate ``mutates_args`` from the original parameter space to the
    lowered space: top-level dataclass names expand to all their leaves;
    primitive top-level names and already-lowered names pass through; unknown
    names raise ``ValueError`` listing valid choices."""
    if not mutates_args:
        return tuple(mutates_args)
    by_attr: dict[str, tuple] = {node[1]: node for node in param_mapping_tree}
    valid_lowered: set[str] = set()
    for node in param_mapping_tree:
        valid_lowered.update(_collect_tensor_leaf_lowered_names(node))
    out: list[str] = []
    for name in mutates_args:
        if name in by_attr:
            node = by_attr[name]
            if node[0] == "primitive":
                out.append(node[2])
            else:
                out.extend(_collect_tensor_leaf_lowered_names(node))
        elif name in valid_lowered:
            out.append(name)
        else:
            raise ValueError(
                f"@magi_register_custom_op: mutates_args entry {name!r} does "
                f"not match any parameter. Valid: {sorted(by_attr.keys())} "
                f"(or lowered: {sorted(valid_lowered)})."
            )
    seen: set[str] = set()
    deduped: list[str] = []
    for n in out:
        if n not in seen:
            seen.add(n)
            deduped.append(n)
    return tuple(deduped)


# ==============================================================================
# 4. Lower fn's signature                                  (produces slot 1)
# ------------------------------------------------------------------------------
# Produces slot 1 (`lowered_fn`) in two stages:
#   data:   `_lower_op_signature` walks `fn`'s parameters once and emits
#           `(original_sig, lowered_sig, param_mapping_tree)`, calling into
#           sec 1 (validate), sec 2 (resolve/sanitise), sec 3 (tree build).
#   object: `_make_lowered_signature_wrapper` stamps the lowered signature
#           onto a forwarding wrapper of `fn`.
# `_signatures_differ` lets the decorator (sec 6) skip the wrapper entirely
# when lowering was a no-op (zero-overhead path).
# ==============================================================================


def _signatures_differ(original: inspect.Signature, lowered: inspect.Signature) -> bool:
    """True iff ``lowered`` differs from ``original`` on parameter names,
    annotations, defaults, kinds, or return annotation. The decorator uses
    this to skip slot 1 entirely when lowering was a no-op (zero-overhead path)."""
    return original != lowered


def _apply_lowered_signature_metadata(wrapper: Callable, lowered_sig: inspect.Signature) -> None:
    """In-place: stamp ``wrapper`` with ``lowered_sig`` as its ``__signature__``
    / ``__annotations__``, and strip ``__wrapped__`` so ``inspect.signature``
    cannot fall back to the original (un-lowered) signature on ``fn``."""
    wrapper.__signature__ = lowered_sig
    lowered_annotations = {
        p.name: p.annotation for p in lowered_sig.parameters.values() if p.annotation is not inspect.Parameter.empty
    }
    if lowered_sig.return_annotation is not inspect.Signature.empty:
        lowered_annotations["return"] = lowered_sig.return_annotation
    wrapper.__annotations__ = lowered_annotations
    # ``functools.wraps`` sets ``__wrapped__`` -> ``fn``; strip it so
    # introspection cannot bypass our ``__signature__``.
    try:
        del wrapper.__wrapped__
    except AttributeError:
        pass


def _make_lowered_signature_wrapper(fn: Callable, lowered_sig: inspect.Signature) -> Callable:
    """Forwarding wrapper around ``fn`` carrying ``lowered_sig`` as metadata.
    Used on the no-flattening path so ``infer_schema`` sees the cleaned-up
    signature instead of ``fn``'s original annotations."""

    @functools.wraps(fn)
    def _wrapped(*args, **kwargs):
        return fn(*args, **kwargs)

    _apply_lowered_signature_metadata(_wrapped, lowered_sig)
    return _wrapped


def _lower_op_signature(fn: Callable):
    """Lower ``fn``'s signature into a form ``torch.library.infer_schema`` accepts.

    "Lower" is used in the compiler sense (high-level -> low-level): we walk
    ``fn``'s parameters once and do six things at the same time -- they all
    need the same resolved annotations and the same iteration:

    1. VALIDATE  -- reject variadics, missing annotations, mutable dataclasses,
                    unsupported containers, dataclass returns (sec 1).
    2. RESOLVE   -- turn stringified annotations into real types via
                    ``_resolve_annotations``, so dataclass detection works.
    3. NORMALIZE -- collapse parameter kinds to POSITIONAL_OR_KEYWORD,
                    downgrade Literal/Enum to ``str``, scrub unsupported defaults.
    4. FLATTEN   -- expand each frozen-dataclass parameter (recursively) into
                    its primitive leaves via ``_build_dataclass_sub_mapping_tree``.
    5. PYTREE    -- side effect of step 4: register every dataclass as a pytree
                    node so Dynamo / AOTAutograd can trace through it.
    6. EMIT      -- assemble ``(original_sig, lowered_sig, param_mapping_tree)``.

    A single pass is intentional: splitting concerns would force re-resolving
    annotations and threading accumulator state. When the input is already
    schema-compatible the lowered signature is bit-identical to the original,
    and the caller's ``_signatures_differ`` check restores the zero-overhead path.

    Returns:
        original_sig (inspect.Signature): the user's un-flattened signature.
        lowered_sig  (inspect.Signature): what ``infer_schema`` will see.
        param_mapping_tree (list[tuple]): the bridge between the two; a list
            of root nodes (one per original parameter), each of which is:
              * ``("primitive", attr_name, lowered_name, None)``, or
              * ``("dataclass", attr_name, cls, [child_nodes...])``.
            ``attr_name`` is the parameter name at top level / field name
            deeper down. The same tree drives both runtime translation
            directions (sec 7).
    """
    original_sig = inspect.signature(fn)
    resolved = _resolve_annotations(fn)
    lowered_params: list[inspect.Parameter] = []
    param_mapping_tree: list[tuple] = []

    for name, param in original_sig.parameters.items():
        _assert_no_var_args(param, fn_name=fn.__name__)
        annotation = resolved.get(name, param.annotation)
        _assert_has_annotation(annotation, where=f"parameter {name!r} of {fn.__name__!r}")
        _assert_not_mutable_dataclass(annotation, where=f"parameter {name!r}")
        if _is_frozen_dataclass(annotation):
            node, sub_params = _build_dataclass_sub_mapping_tree(annotation, attr_name=name, flat_prefix=name)
            param_mapping_tree.append(node)
            lowered_params.extend(sub_params)
        else:
            _assert_not_unsupported_container(annotation, where=f"parameter {name!r}")
            annotation = _maybe_downgrade_literal_or_enum(annotation, where=f"parameter {name!r}")
            new_param = param.replace(
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=annotation,
                default=_schema_compatible_param_default(param.default),
            )
            lowered_params.append(new_param)
            param_mapping_tree.append(("primitive", name, name, None))

    return_annotation = resolved.get("return", original_sig.return_annotation)
    _assert_has_annotation(return_annotation, where=f"return value of {fn.__name__!r}")
    _assert_not_dataclass_return(return_annotation, fn_name=fn.__name__)
    lowered_sig = inspect.Signature(lowered_params, return_annotation=return_annotation)
    return original_sig, lowered_sig, param_mapping_tree


# ==============================================================================
# 5. Synthesise the meta/fake function                     (input to slot 2)
# ------------------------------------------------------------------------------
# Constructors for the meta ("fake") function torch.library uses for shape
# propagation during tracing. Fallbacks: identity meta when the user passes
# no `infer_output_meta_fn`; param-name-echoing meta when they pass a
# `list[str]`. The result is handed to `register_fake` by sec 6.
# ==============================================================================


def _get_num_outputs_from_return_annotation(fn: Callable) -> int:
    """Output count from ``fn``'s return annotation: ``N`` for
    ``tuple[T1, ..., TN]``, else ``1`` (default / unrecognized)."""
    sig = inspect.signature(fn)
    return_annotation = sig.return_annotation

    if return_annotation is inspect.Parameter.empty:
        return 1

    origin = get_origin(return_annotation)
    if origin is tuple:
        args = get_args(return_annotation)
        # tuple[T, ...] (variable-length) collapses to a single output.
        if args and args[-1] is not ...:
            return len(args)
        return 1

    return 1


def _create_identity_meta_fn(fn: Callable) -> Callable:
    """Default meta/fake: copy shape/dtype/device of the first N tensor inputs
    to N outputs (N from the return annotation)."""
    num_outputs = _get_num_outputs_from_return_annotation(fn)
    sig = inspect.signature(fn)
    param_names = [name for name in sig.parameters.keys() if name != "self"]

    def identity_meta_fn(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        tensor_args = []
        for name in param_names:
            arg = bound.arguments.get(name)
            if isinstance(arg, torch.Tensor):
                tensor_args.append(arg)
                if len(tensor_args) >= num_outputs:
                    break

        if len(tensor_args) < num_outputs:
            raise ValueError(
                f"@magi_register_custom_op: identity_meta_fn needs {num_outputs} "
                f"tensor input(s) but found {len(tensor_args)}; provide a custom "
                f"infer_output_meta_fn."
            )

        if num_outputs == 1:
            return torch.empty_like(tensor_args[0])
        return tuple(torch.empty_like(t) for t in tensor_args[:num_outputs])

    return identity_meta_fn


def _create_meta_fn_from_param_names(fn: Callable, param_names: list[str]) -> Callable:
    """Meta/fake that echoes the listed tensor parameters as outputs
    (``torch.empty_like`` each). Raises ``ValueError`` for unknown or
    non-Tensor names."""
    sig = inspect.signature(fn)

    def meta_fn(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        tensor_outputs = []
        for name in param_names:
            if name not in bound.arguments:
                raise ValueError(
                    f"@magi_register_custom_op: infer_output_meta_fn references "
                    f"unknown parameter {name!r}; available: "
                    f"{list(bound.arguments.keys())}."
                )
            arg = bound.arguments[name]
            if not isinstance(arg, torch.Tensor):
                raise ValueError(
                    f"@magi_register_custom_op: infer_output_meta_fn entry "
                    f"{name!r} is not a Tensor (got {type(arg).__name__}); "
                    f"list must contain only tensor parameter names."
                )
            tensor_outputs.append(torch.empty_like(arg))

        if len(tensor_outputs) == 1:
            return tensor_outputs[0]
        return tuple(tensor_outputs)

    return meta_fn


# ==============================================================================
# 6. Register the op                                       (produces slot 2)
# ------------------------------------------------------------------------------
# `_register_torch_op` calls `custom_op` + `register_fake`, yielding slot 2
# (`torch_registered_op`). `_generate_op_name` derives a default op name
# from the user's `fn` when one isn't supplied. The orchestrator that
# stitches these together with sec 1-5 (and the runtime closures from
# sec 7) lives in sec 8.
# ==============================================================================


def _generate_op_name(fn: Callable) -> str:
    """Op name ``{filename_stem}::{fn.__name__}``, falling back to
    ``magi_custom::`` if the source file isn't available."""
    import re
    from pathlib import Path

    func_name = fn.__name__
    try:
        source_file = inspect.getfile(fn)
        namespace = Path(source_file).stem
        namespace = re.sub(r"[^a-zA-Z0-9_]", "_", namespace)
    except (TypeError, OSError):
        namespace = "magi_custom"

    return f"{namespace}::{func_name}"


def _register_torch_op(op_name: str, fn: Callable, mutates_args: tuple[str, ...], meta_fn: Callable):
    """``custom_op`` + ``register_fake``; returns slot 2 (``torch_registered_op``)."""
    torch_registered_op = torch.library.custom_op(op_name, mutates_args=mutates_args)(fn)
    torch.library.register_fake(op_name)(meta_fn)
    return torch_registered_op


# ==============================================================================
# 7. Runtime bridge: flatten / unflatten on every call
# ------------------------------------------------------------------------------
# Executed on every call to slot 1 (`lowered_fn`) or slot 3 (`magi_exposed_op`),
# consuming the static `param_mapping_tree` from sec 3 to translate between
# the original (dataclass) and lowered (primitive) call shapes. See Part B
# of the module docstring for the full call-stack picture.
#   original -> lowered: `_flatten_value_into`, `_flatten_call_args`
#   lowered  -> original: `_build_value_from_node`, `_reassemble_kwargs`
#   grad bridge:          `_flatten_grad_into`, `_flatten_grads`
# ==============================================================================


def _build_value_from_node(node: tuple, lowered_kwargs: dict):
    """``lowered_kwargs`` -> one original-shaped value (recursive)."""
    kind = node[0]
    if kind == "primitive":
        _, _attr, lowered_name, _ = node
        return lowered_kwargs[lowered_name]
    _, _attr, cls, children = node
    init_kwargs: dict[str, Any] = {}
    for child in children:
        field_name = child[1]
        init_kwargs[field_name] = _build_value_from_node(child, lowered_kwargs)
    return cls(**init_kwargs)


def _reassemble_kwargs(param_mapping_tree: list[tuple], lowered_kwargs: dict) -> dict:
    """``lowered_kwargs`` -> original kwargs (the ``lowered -> original`` walk)."""
    out: dict[str, Any] = {}
    for node in param_mapping_tree:
        out[node[1]] = _build_value_from_node(node, lowered_kwargs)
    return out


def _flatten_value_into(node: tuple, value: Any, out: list) -> None:
    """Append leaves of ``value`` to ``out`` in DFS order (no isinstance check
    on ``cls`` -- duck-typed via ``getattr`` so mocks / SimpleNamespace work)."""
    kind = node[0]
    if kind == "primitive":
        out.append(value)
        return
    _, _attr, cls, children = node
    for child in children:
        field_name = child[1]
        _flatten_value_into(child, getattr(value, field_name), out)


def _flatten_call_args(param_mapping_tree: list[tuple], original_sig: inspect.Signature, args: tuple, kwargs: dict) -> list:
    """User-side call -> flat positional list matching the lowered signature
    (the ``original -> lowered`` walk)."""
    bound = original_sig.bind(*args, **kwargs)
    bound.apply_defaults()
    flat: list = []
    for node in param_mapping_tree:
        _flatten_value_into(node, bound.arguments[node[1]], flat)
    return flat


def _flatten_grad_into(node: tuple, grad: Any, out: list) -> None:
    """Spread a user-returned grad across the lowered slots of one original input.

    ``primitive`` -> append ``grad`` as-is. ``dataclass`` -> if ``grad`` is
    ``None`` fill every leaf with ``None`` (the common whole-dataclass-not-
    differentiable case); otherwise descend with ``dict``-aware lookup so
    users can return dict / SimpleNamespace / dataclass-shaped objects.
    """
    if node[0] == "primitive":
        out.append(grad)
        return
    _, _attr, _cls, children = node
    if grad is None:
        for child in children:
            for _ in range(_count_leaves(child)):
                out.append(None)
        return
    is_mapping = isinstance(grad, dict)
    for child in children:
        field_name = child[1]
        if is_mapping:
            sub = grad.get(field_name)
        else:
            sub = getattr(grad, field_name, None)
        _flatten_grad_into(child, sub, out)


def _flatten_grads(param_mapping_tree: list[tuple], original_grads: tuple | list) -> list:
    """Grads keyed by original-parameter order -> grads keyed by lowered order."""
    if len(original_grads) != len(param_mapping_tree):
        raise ValueError(
            f"@magi_register_custom_op: backward_fn returned {len(original_grads)} "
            f"grad(s) but the function has {len(param_mapping_tree)} input(s); "
            f"return one grad per ORIGINAL parameter (``None`` for non-differentiable "
            f"or whole-dataclass args)."
        )
    flat: list = []
    for node, g in zip(param_mapping_tree, original_grads):
        _flatten_grad_into(node, g, flat)
    return flat


# ==============================================================================
# 8. The decorator: main pipeline                          (produces slot 3)
# ------------------------------------------------------------------------------
# The single public entry point. Its inner `decorator` closure orchestrates the
# full 4-slot pipeline (see module docstring): it calls sec 4 to lower the user's
# signature (slot 1), sec 5 to synthesise the meta function, sec 6 to register
# the op with torch.library (slot 2), and -- on the dataclass-flatten path --
# additionally builds the user-facing wrapper (slot 3) plus the runtime closures
# that drive sec 7 on each call.
# ==============================================================================


def _magi_register_custom_op_impl(
    name: str | None = None,
    mutates_args: tuple[str, ...] = (),
    infer_output_meta_fn: Callable | list[str] | None = None,
    setup_context_fn: Callable | None = None,
    backward_fn: Callable | None = None,
    is_compute_sensitive: bool = False,
    is_subgraph_boundary: bool = False,
):
    def decorator(fn: Callable) -> Callable:
        # See the module docstring for the 4-slot pipeline / 3-runtime-path
        # picture; the body below just walks slot 0 -> 1 -> 2 (-> 3 if needed).

        op_name = name if name is not None else _generate_op_name(fn)
        if is_compute_sensitive:
            get_compile_config().recompute_config.custom_compute_sensitive_ops.append(op_name)
        if is_subgraph_boundary:
            get_compile_config().splitting_ops.append(op_name)

        # Dataclass parameters are the only thing forcing slot 3; other lowering
        # (Literal/Enum/default scrub) is handled by slot 1 alone at zero
        # per-call cost.
        original_sig, lowered_sig, param_mapping_tree = _lower_op_signature(fn)
        needs_flattening = any(kind == "dataclass" for kind, *_ in param_mapping_tree)

        if not needs_flattening:
            # ----- No-flattening path: fn -> [lowered_fn?] -> torch_registered_op -----
            # Step 1 (slot 1): only when the lowering actually rewrote the
            # signature -- otherwise register ``fn`` directly (zero-overhead).
            if _signatures_differ(original_sig, lowered_sig):
                lowered_fn = _make_lowered_signature_wrapper(fn, lowered_sig)
                fn_to_register = lowered_fn
            else:
                fn_to_register = fn

            # Step 2: meta/fake function.
            if infer_output_meta_fn is None:
                meta_fn = _create_identity_meta_fn(fn_to_register)
            elif isinstance(infer_output_meta_fn, list):
                meta_fn = _create_meta_fn_from_param_names(fn_to_register, infer_output_meta_fn)
            else:
                meta_fn = infer_output_meta_fn

            # Step 3 (slot 2): custom_op + register_fake.
            torch_registered_op = _register_torch_op(
                op_name=op_name, fn=fn_to_register, mutates_args=mutates_args, meta_fn=meta_fn
            )

            # Step 4: autograd.
            if backward_fn is not None:
                torch_registered_op.register_autograd(backward_fn, setup_context=setup_context_fn)

            # No slot 3 needed: the user's calling convention already matches
            # the lowered one, so ``torch_registered_op`` is itself returned.
            return torch_registered_op

        else:
            # ----- Flattening path: fn -> lowered_fn -> torch_registered_op -> magi_exposed_op -----
            # Step 1 (slot 1): build the lowered-signature bridge. ``lowered_fn``
            # speaks the flat primitive signature; it rebinds args, reassembles
            # dataclasses, then dispatches to the user's ``fn``.
            def _bind_to_original_kwargs(args, kwargs):
                bound = lowered_sig.bind(*args, **kwargs)
                bound.apply_defaults()
                return _reassemble_kwargs(param_mapping_tree, bound.arguments)

            @functools.wraps(fn)
            def lowered_fn(*args, **kwargs):
                return fn(**_bind_to_original_kwargs(args, kwargs))

            _apply_lowered_signature_metadata(lowered_fn, lowered_sig)

            # Step 2: meta/fake function. User-supplied meta_fn is bridged so
            # it sees the original (dataclass-bearing) signature it was
            # written against.
            if infer_output_meta_fn is None:
                meta_fn = _create_identity_meta_fn(lowered_fn)
            elif isinstance(infer_output_meta_fn, list):
                meta_fn = _create_meta_fn_from_param_names(lowered_fn, infer_output_meta_fn)
            else:
                user_meta = infer_output_meta_fn

                def _bridged_meta_fn(*args, **kwargs):
                    return user_meta(**_bind_to_original_kwargs(args, kwargs))

                _bridged_meta_fn.__signature__ = lowered_sig
                meta_fn = _bridged_meta_fn

            # Step 3 (slot 2): custom_op + register_fake. ``mutates_args`` is
            # expanded from original-space to lowered-space so torch.library
            # sees the leaf parameter names it actually owns.
            flat_mutates_args = _expand_mutates_args(mutates_args, param_mapping_tree)
            torch_registered_op = _register_torch_op(
                op_name=op_name, fn=lowered_fn, mutates_args=flat_mutates_args, meta_fn=meta_fn
            )

            # Step 4: autograd. The user's hooks speak the ORIGINAL signature,
            # but torch.library passes/expects LOWERED inputs and grads, so we
            # wrap both ends.
            if backward_fn is not None:
                user_setup = setup_context_fn
                user_backward = backward_fn

                def _bridged_setup_context(ctx, inputs, output):
                    if user_setup is None:
                        return None
                    # Reassemble the lowered positional tuple into the user's
                    # original (possibly nested-dataclass) shape, preserving
                    # original positional order so ``x, cfg = inputs`` works.
                    lowered_kwargs = {p.name: v for p, v in zip(lowered_sig.parameters.values(), inputs)}
                    original_kwargs = _reassemble_kwargs(param_mapping_tree, lowered_kwargs)
                    original_inputs = tuple(original_kwargs[p] for p in original_sig.parameters)
                    return user_setup(ctx, original_inputs, output)

                def _bridged_backward(ctx, *grads):
                    original_grads = user_backward(ctx, *grads)
                    if not isinstance(original_grads, tuple):
                        # Single-input convenience: PyTorch allows a bare grad
                        # when the op has one input.
                        original_grads = (original_grads,)
                    return tuple(_flatten_grads(param_mapping_tree, original_grads))

                torch_registered_op.register_autograd(_bridged_backward, setup_context=_bridged_setup_context)

            # Step 5 (slot 3, flattening-only): the user-facing op that
            # preserves the original signature, flattens at entry, and
            # dispatches to ``torch_registered_op``.
            @functools.wraps(fn)
            def magi_exposed_op(*args, **kwargs):
                flat = _flatten_call_args(param_mapping_tree, original_sig, args, kwargs)
                return torch_registered_op(*flat)

            # Internal handles so downstream tooling can drop one slot lower
            # (e.g. dispatch the OpOverload directly with pre-flattened args).
            magi_exposed_op._magi_torch_registered_op = torch_registered_op
            magi_exposed_op._magi_param_mapping_tree = param_mapping_tree
            return magi_exposed_op

    return decorator
