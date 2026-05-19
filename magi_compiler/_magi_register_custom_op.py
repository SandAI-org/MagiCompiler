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
Magi custom-op registration: dataclass-aware wrapper around ``torch.library``.

This module implements ``magi_register_custom_op`` -- a decorator that takes
a plain Python function and registers it as a real custom op while letting
the user keep calling it with their original signature.


File layout
===========

The file has five blocks. Each block groups its own helpers (private,
above) with the one core piece it exists to support (below). Block
boundaries follow the 5-stage pipeline.

    Block 0 -- VALIDATE op signature constraints (registration-time)
        helpers:  assertion predicates + validation primitives
        core:     _validate_op_signature_constraints

    Block 1 -- LOWER                   (registration-time)
        helpers:  type resolution, default scrubbing, param-mapping-tree construction
        core:     _lower_op_signature                       (produces slot 1)

    Block 2 -- REGISTER                (registration-time)
        helpers:  op-name generation, meta/fake-fn synthesis
        core:     _register_torch_op                        (produces slot 2)

    Block 3 -- RUNTIME ADAPTER         (runtime)
        helpers:  flatten / unflatten primitives + signature-bound wrappers
        core:     _DataclassRuntimeAdapter                  (used by slot 3)

    Block 4 -- MAIN PIPELINE
        core:     _magi_register_custom_op_impl             (the decorator;
                  orchestrates blocks 0-3, produces slot 3 on the flatten path)
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
from typing import Any, Callable, get_args, get_origin

import torch
import torch.utils._pytree as pytree

from .config import get_compile_config
from .utils.logger import magi_logger

# ==============================================================================
# BLOCK 0 -- VALIDATE op signature constraints
#
# Helpers:
#   - type predicates:
#       _is_frozen_dataclass
#   - assertion primitives:
#       _assert_not_unsupported_container, _assert_not_dataclass_return,
#       _assert_not_mutable_dataclass, _assert_has_annotation,
#       _assert_no_var_args, _assert_resolved_field_type
# Core: _validate_op_signature_constraints
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


def _validate_op_signature_constraints(fn: Callable) -> None:
    """Validate fn parameters/return and recursively validate frozen dataclass subtrees."""
    original_sig = inspect.signature(fn)
    resolved = _resolve_annotations(fn)

    def _validate_through(
        params_or_fields, *, owner_name: str, is_fn_params: bool, field_types: dict[str, Any] | None = None
    ) -> None:
        for item in params_or_fields:
            if is_fn_params:
                name, param = item
                _assert_no_var_args(param, fn_name=fn.__name__)
                annotation = resolved.get(name, param.annotation)
                where = f"parameter {name!r} of {owner_name}"
            else:
                field = item
                name = field.name
                annotation = (field_types or {}).get(name, field.type)
                where = f"field {owner_name}.{name}"
                _assert_resolved_field_type(annotation, where=where)

            _assert_has_annotation(annotation, where=where)
            _assert_not_mutable_dataclass(annotation, where=where)

            if _is_frozen_dataclass(annotation):
                _validate_through(
                    dataclasses.fields(annotation),
                    owner_name=annotation.__name__,
                    is_fn_params=False,
                    field_types=_resolve_dataclass_field_types(annotation),
                )
            else:
                _assert_not_unsupported_container(annotation, where=where)

    _validate_through(original_sig.parameters.items(), owner_name=f"{fn.__name__!r}", is_fn_params=True)

    return_annotation = resolved.get("return", original_sig.return_annotation)
    _assert_has_annotation(return_annotation, where=f"return value of {fn.__name__!r}")
    _assert_not_dataclass_return(return_annotation, fn_name=fn.__name__)


# ==============================================================================
# BLOCK 1 -- LOWER fn signature
#
# Helpers:
#   - type resolution & default scrubbing:
#       _resolve_annotations, _resolve_dataclass_field_types,
#       _maybe_downgrade_literal_or_enum,
#       _schema_compatible_field_default, _schema_compatible_param_default
#   - param-mapping-tree construction:
#       _register_dataclass_pytree,
#       _expand_mutates_args
#   - lowered-signature utilities:
#       _apply_lowered_signature,
#       _make_lowered_signature_wrapper
# Core: _lower_op_signature
# ==============================================================================


# ------------------------------------------------------------------------------
# helpers: resolve types & sanitise defaults for infer_schema
# ------------------------------------------------------------------------------


def _resolve_annotations(fn: Callable) -> dict[str, Any]:
    """Return ``fn`` annotations as real types, with globals+closure eval fallback."""
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
    import sys
    import typing as _typing

    try:
        return _typing.get_type_hints(cls)
    except Exception:
        pass

    # ``get_type_hints(cls)`` is all-or-nothing; fall back to per-field eval so
    # one unresolved annotation does not poison the whole dataclass.
    namespace: dict[str, Any] = {}
    module = sys.modules.get(getattr(cls, "__module__", ""))
    if module is not None:
        namespace.update(vars(module))
    namespace.update(getattr(cls, "__dict__", {}))
    namespace.setdefault(cls.__name__, cls)

    resolved: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        tp = f.type
        if isinstance(tp, str):
            try:
                resolved[f.name] = eval(tp, namespace, None)
            except Exception:
                resolved[f.name] = tp
        else:
            resolved[f.name] = tp
    return resolved


def _maybe_downgrade_literal_or_enum(annotation, *, where: str):
    """Downgrade ``Literal[str,...]`` and string-valued Enums to ``str`` or raise."""
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


def _schema_compatible_field_default(f: "dataclasses.Field") -> Any:
    """Return schema-safe field default (including resolved ``default_factory``) or empty."""
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


def _schema_compatible_param_default(default: Any) -> Any:
    """Return schema-safe parameter default or ``inspect.Parameter.empty``."""
    if default is inspect.Parameter.empty:
        return inspect.Parameter.empty
    if default is None or isinstance(default, _SCHEMA_DEFAULT_TYPES):
        return default
    return inspect.Parameter.empty


# ------------------------------------------------------------------------------
# helpers: build & query the param mapping tree
# ------------------------------------------------------------------------------

_DATACLASS_PYTREE_REGISTERED: set[type] = set()


def _register_dataclass_pytree(cls: type) -> None:
    """Idempotently register dataclass ``cls`` as a pytree node for tracing."""
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


def _expand_mutates_args(param_mapping_tree: list[tuple], mutates_args: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Expand/validate ``mutates_args`` from original names into lowered leaf names."""

    def _collect_tensor_leaf_lowered_attr_names(node: tuple) -> list[str]:
        if node[0] == "primitive":
            _, _attr, lowered_attr_name, _ = node
            return [lowered_attr_name]
        out: list[str] = []
        for child in node[3]:
            out.extend(_collect_tensor_leaf_lowered_attr_names(child))
        return out

    if not mutates_args:
        return tuple(mutates_args)
    by_attr: dict[str, tuple] = {node[1]: node for node in param_mapping_tree}
    valid_lowered: set[str] = set()
    for node in param_mapping_tree:
        valid_lowered.update(_collect_tensor_leaf_lowered_attr_names(node))
    out: list[str] = []
    for name in mutates_args:
        if name in by_attr:
            node = by_attr[name]
            if node[0] == "primitive":
                out.append(node[2])
            else:
                out.extend(_collect_tensor_leaf_lowered_attr_names(node))
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


# ------------------------------------------------------------------------------
# core: _lower_op_signature  (and its lowered-signature wrapper utilities)
# ------------------------------------------------------------------------------


def _apply_lowered_signature(lowered_sig: inspect.Signature, wrapper: Callable) -> None:
    """Stamp wrapper signature/annotations with ``lowered_sig`` and clear ``__wrapped__``."""
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


def _lower_op_signature(fn: Callable):
    """Lower ``fn``'s signature into a form ``torch.library.infer_schema`` accepts.

    Pipeline:
    1. RESOLVE   -- turn stringified annotations/dataclass fields into real types (best-effort).
                    If failed, trying globals+closure evalining per annotation/dataclass field.
    2. FLATTEN   -- recursively flatten frozen dataclass parameters into primitive leaves
                    and register the dataclass types as pytree nodes for Dynamo/AOTAutograd tracing.
    3. NORMALIZE -- collapse parameter kinds to POSITIONAL_OR_KEYWORD,
                    downgrade Literal/Enum to ``str``, scrub unsupported defaults.
    4. EMIT      -- assemble ``(original_sig, lowered_sig, param_mapping_tree)``.

    Returns:
        original_sig (inspect.Signature): the user's original input signature.
        lowered_sig (inspect.Signature): schema-compatible signature for ``infer_schema``.
        param_mapping_tree (list[tuple]): the bridge between the two; a list
            of root nodes (one per original parameter), each of which is:
              * ``("primitive", attr_name, lowered_attr_name, None)``, or
              * ``("dataclass", attr_name, dataclass_cls_type, [child_nodes...])``.

    Example:
        ``fn(x: Tensor, cfg: Outer(inner: Inner(scale: float, bias: Tensor), mode: str))``
        -> ``original_sig(x: Tensor, cfg: Outer)``
        -> ``lowered_sig(x: Tensor, cfg__inner__scale: float, cfg__inner__bias: Tensor, cfg__mode: str)``
        -> ``param_mapping_tree = [("primitive", "x", "x", None),
                                   ("dataclass", "cfg", Outer, [
                                       ("dataclass", "inner", Inner, [
                                           ("primitive", "scale", "cfg__inner__scale", None),
                                           ("primitive", "bias", "cfg__inner__bias", None),
                                       ]),
                                       ("primitive", "mode", "cfg__mode", None),
                                   ])]``
    """

    original_sig = inspect.signature(fn)
    resolved = _resolve_annotations(fn)

    def _lower_through(
        params_or_fields,
        *,
        is_fn_params: bool,
        owner_name: str,
        flat_prefix: str | None = None,
        field_types: dict[str, Any] | None = None,
    ) -> tuple[list[tuple], list[inspect.Parameter]]:
        nodes: list[tuple] = []
        lowered: list[inspect.Parameter] = []

        if is_fn_params:
            iterator = ((name, resolved.get(name, param.annotation), param) for name, param in params_or_fields)
        else:
            resolved_fields = field_types or {}
            iterator = ((field.name, resolved_fields.get(field.name, field.type), field) for field in params_or_fields)

        for name, annotation, source in iterator:
            leaf_flat_name = name if is_fn_params else f"{flat_prefix}__{name}"
            where = f"parameter {name!r}" if is_fn_params else f"field {owner_name}.{name}"

            if _is_frozen_dataclass(annotation):
                _register_dataclass_pytree(annotation)
                child_nodes, child_params = _lower_through(
                    dataclasses.fields(annotation),
                    is_fn_params=False,
                    owner_name=annotation.__name__,
                    flat_prefix=leaf_flat_name,
                    field_types=_resolve_dataclass_field_types(annotation),
                )
                nodes.append(("dataclass", name, annotation, child_nodes))
                lowered.extend(child_params)
            else:
                annotation = _maybe_downgrade_literal_or_enum(annotation, where=where)
                if is_fn_params:
                    param = source
                    lowered.append(
                        param.replace(
                            kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            annotation=annotation,
                            default=_schema_compatible_param_default(param.default),
                        )
                    )
                else:
                    field = source
                    lowered.append(
                        inspect.Parameter(
                            leaf_flat_name,
                            # POSITIONAL_OR_KEYWORD: torch.library.custom_op does not yet
                            # support kwarg-only Tensor arguments.
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            annotation=annotation,
                            default=_schema_compatible_field_default(field),
                        )
                    )
                nodes.append(("primitive", name, leaf_flat_name, None))

        return nodes, lowered

    param_mapping_tree, lowered_params = _lower_through(
        original_sig.parameters.items(), is_fn_params=True, owner_name=f"{fn.__name__!r}"
    )

    return_annotation = resolved.get("return", original_sig.return_annotation)
    lowered_sig = inspect.Signature(lowered_params, return_annotation=return_annotation)
    return original_sig, lowered_sig, param_mapping_tree


# ==============================================================================
# BLOCK 2 -- REGISTER torch op
#
# Helpers:
#   - meta/fake-fn synthesis:
#       _get_num_outputs_from_return_annotation,
#       _create_identity_meta_fn, _create_meta_fn_from_param_names
#   - op-name generation:
#       _generate_op_name
# Core: _register_torch_op
# ==============================================================================


# ------------------------------------------------------------------------------
# helpers: synthesise the meta/fake function
# ------------------------------------------------------------------------------


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


# ------------------------------------------------------------------------------
# helpers: generate op name
# ------------------------------------------------------------------------------


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


# ------------------------------------------------------------------------------
# core: _register_torch_op
#
# Forward reference: ``_DataclassRuntimeAdapter`` using ``from __future__ import annotations``.
# ------------------------------------------------------------------------------


def _register_torch_op(
    op_name: str,
    fn: Callable,
    mutates_args: tuple[str, ...],
    infer_output_meta_fn: Callable | list[str] | None,
    setup_context_fn: Callable | None,
    backward_fn: Callable | None,
    dataclass_runtime_adapter: _DataclassRuntimeAdapter | None = None,
):
    """Register the op in torch.library.custom_op."""
    effective_mutates_args = (
        dataclass_runtime_adapter.expand_mutates_args(mutates_args) if dataclass_runtime_adapter is not None else mutates_args
    )
    torch_registered_op = torch.library.custom_op(op_name, mutates_args=effective_mutates_args)(fn)

    # Build & register the meta/fake function.
    if infer_output_meta_fn is None:
        meta_fn = _create_identity_meta_fn(fn)
    elif isinstance(infer_output_meta_fn, list):
        meta_fn = _create_meta_fn_from_param_names(fn, infer_output_meta_fn)
    elif dataclass_runtime_adapter is None:  # No flattening scenario
        meta_fn = infer_output_meta_fn
    else:  # Flattening scenario
        user_meta = infer_output_meta_fn

        def _bridged_meta_fn(*args, **kwargs):
            return user_meta(**dataclass_runtime_adapter.unflatten_call_args(args, kwargs))

        _bridged_meta_fn.__signature__ = inspect.signature(fn)
        meta_fn = _bridged_meta_fn
    torch.library.register_fake(op_name)(meta_fn)

    # Register autograd.
    if backward_fn is not None:
        if dataclass_runtime_adapter is None:  # No flattening scenario
            torch_registered_op.register_autograd(backward_fn, setup_context=setup_context_fn)
        else:  # Flattening scenario

            def _bridged_setup_context(ctx, inputs, output):
                if setup_context_fn is None:
                    return None
                original_inputs = dataclass_runtime_adapter.unflatten_setup_ctx_inputs(inputs)
                return setup_context_fn(ctx, original_inputs, output)

            def _bridged_backward(ctx, *grads):
                original_grads = backward_fn(ctx, *grads)
                if not isinstance(original_grads, tuple):
                    original_grads = (original_grads,)
                return dataclass_runtime_adapter.flatten_input_grads(original_grads)

            torch_registered_op.register_autograd(_bridged_backward, setup_context=_bridged_setup_context)

    return torch_registered_op


# ==============================================================================
# BLOCK 3 -- RUNTIME ADAPTER
#
# Helpers (adapter field <- bound function):
#   original -> lowered:  flatten_call_args          <- _flatten_call_args
#                         flatten_input_grads        <- _flatten_input_grads
#   lowered  -> original: unflatten_call_args        <- _unflatten_call_args
#                         unflatten_setup_ctx_inputs <- _unflatten_setup_ctx_inputs
#                         _reassemble_kwargs         (internal primitive)
#   mutates_args expand:  expand_mutates_args        <- _expand_mutates_args
#   signature stamping:   apply_lowered_signature    <- _apply_lowered_signature
# Core: _DataclassRuntimeAdapter
# ==============================================================================


# ---- flatten_call_args ----


def _flatten_call_args(param_mapping_tree: list[tuple], original_sig: inspect.Signature, args: tuple, kwargs: dict) -> list:
    """User-side call -> flat positional list matching the lowered signature
    (the ``original -> lowered`` walk)."""

    def _flatten_value_into(node: tuple, value: Any, out: list) -> None:
        """Append leaves of ``value`` to ``out`` in DFS order (no isinstance check
        on ``cls`` -- duck-typed via ``getattr`` so mocks / SimpleNamespace work)."""
        kind = node[0]
        if kind == "primitive":
            out.append(value)
            return
        _, _attr, _cls, children = node
        for child in children:
            field_name = child[1]
            _flatten_value_into(child, getattr(value, field_name), out)

    bound = original_sig.bind(*args, **kwargs)
    bound.apply_defaults()
    flat: list = []
    for node in param_mapping_tree:
        _flatten_value_into(node, bound.arguments[node[1]], flat)
    return flat


# ---- flatten_input_grads ----


def _flatten_input_grads(param_mapping_tree: list[tuple], original_grads: tuple) -> tuple:
    """Original-space input grads -> lowered-space input grads."""

    def _count_leaves(node: tuple) -> int:
        if node[0] == "primitive":
            return 1
        return sum(_count_leaves(c) for c in node[3])

    def _flatten_grad_into(node: tuple, grad: Any, out: list) -> None:
        """Spread a user-returned grad across lowered slots for one original input."""
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
    return tuple(flat)


# ---- unflatten_call_args / unflatten_setup_ctx_inputs ----


def _reassemble_kwargs(param_mapping_tree: list[tuple], lowered_kwargs: dict) -> dict:
    """``lowered_kwargs`` -> original kwargs (the ``lowered -> original`` walk)."""

    def _build_value_from_node(node: tuple):
        kind = node[0]
        if kind == "primitive":
            _, _attr, lowered_attr_name, _ = node
            return lowered_kwargs[lowered_attr_name]
        _, _attr, cls, children = node
        init_kwargs: dict[str, Any] = {}
        for child in children:
            field_name = child[1]
            init_kwargs[field_name] = _build_value_from_node(child)
        return cls(**init_kwargs)

    out: dict[str, Any] = {}
    for node in param_mapping_tree:
        out[node[1]] = _build_value_from_node(node)
    return out


def _unflatten_call_args(lowered_sig: inspect.Signature, param_mapping_tree: list[tuple], args: tuple, kwargs: dict) -> dict:
    """Lowered call args/kwargs -> original kwargs (dict for ``fn(**dict)``)."""
    bound = lowered_sig.bind(*args, **kwargs)
    bound.apply_defaults()
    return _reassemble_kwargs(param_mapping_tree, bound.arguments)


def _unflatten_setup_ctx_inputs(
    lowered_sig: inspect.Signature, original_sig: inspect.Signature, param_mapping_tree: list[tuple], inputs: tuple
) -> tuple:
    """Lowered positional inputs tuple -> original positional inputs tuple
    (for ``setup_context_fn(ctx, inputs, output)``)."""
    lowered_kwargs = {p.name: v for p, v in zip(lowered_sig.parameters.values(), inputs)}
    original_kwargs = _reassemble_kwargs(param_mapping_tree, lowered_kwargs)
    return tuple(original_kwargs[p] for p in original_sig.parameters)


# ---- core ----


@dataclasses.dataclass(frozen=True)
class _DataclassRuntimeAdapter:
    """Runtime conversion adapter."""

    flatten_call_args: Callable[[tuple, dict], list]
    flatten_input_grads: Callable[[tuple], tuple]
    unflatten_call_args: Callable[[tuple, dict], dict]
    unflatten_setup_ctx_inputs: Callable[[tuple], tuple]
    expand_mutates_args: Callable[[tuple[str, ...]], tuple[str, ...]]
    apply_lowered_signature: Callable[[Callable], None]


# ==============================================================================
# BLOCK 4 -- MAIN PIPELINE
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
        # A 4-slot pipeline.

        op_name = name if name is not None else _generate_op_name(fn)
        if is_compute_sensitive:
            get_compile_config().recompute_config.custom_compute_sensitive_ops.append(op_name)
        if is_subgraph_boundary:
            get_compile_config().splitting_ops.append(op_name)

        _validate_op_signature_constraints(fn)
        original_sig, lowered_sig, param_mapping_tree = _lower_op_signature(fn)
        needs_flattening = any(kind == "dataclass" for kind, *_ in param_mapping_tree)

        if not needs_flattening:
            # ----- No-flattening scenario -----
            # Path: fn -> [lowered_fn ->] torch_registered_op

            # Step 1: Build ``lowered_fn`` iff the signature was rewritten.
            if original_sig == lowered_sig:
                fn_to_register = fn
            else:  # Signatures differ, need to wrap the function

                @functools.wraps(fn)
                def lowered_fn(*args, **kwargs):
                    return fn(*args, **kwargs)

                _apply_lowered_signature(lowered_sig, lowered_fn)
                fn_to_register = lowered_fn

            # Step 2: Register the op in torch and get ``torch_registered_op``.
            torch_registered_op = _register_torch_op(
                op_name=op_name,
                fn=fn_to_register,
                mutates_args=mutates_args,
                infer_output_meta_fn=infer_output_meta_fn,
                setup_context_fn=setup_context_fn,
                backward_fn=backward_fn,
                dataclass_runtime_adapter=None,
            )

            # Return bare torch-level op (slot 2).
            return torch_registered_op

        else:
            # ----- Flattening scenario -----
            # Path: fn -> lowered_fn -> torch_registered_op -> magi_exposed_op

            # Step 0 (only in the flattening scenario): Build the scenario-wide adapter.
            dataclass_runtime_adapter = _DataclassRuntimeAdapter(
                flatten_call_args=functools.partial(_flatten_call_args, param_mapping_tree, original_sig),
                flatten_input_grads=functools.partial(_flatten_input_grads, param_mapping_tree),
                unflatten_call_args=functools.partial(_unflatten_call_args, lowered_sig, param_mapping_tree),
                unflatten_setup_ctx_inputs=functools.partial(
                    _unflatten_setup_ctx_inputs, lowered_sig, original_sig, param_mapping_tree
                ),
                expand_mutates_args=functools.partial(_expand_mutates_args, param_mapping_tree),
                apply_lowered_signature=functools.partial(_apply_lowered_signature, lowered_sig),
            )

            # Step 1: Build ``lowered_fn``.
            @functools.wraps(fn)
            def lowered_fn(*args, **kwargs):
                return fn(**dataclass_runtime_adapter.unflatten_call_args(args, kwargs))

            dataclass_runtime_adapter.apply_lowered_signature(lowered_fn)

            # Step 2: Register the op in torch and get ``torch_registered_op``.
            torch_registered_op = _register_torch_op(
                op_name=op_name,
                fn=lowered_fn,
                mutates_args=mutates_args,
                infer_output_meta_fn=infer_output_meta_fn,
                setup_context_fn=setup_context_fn,
                backward_fn=backward_fn,
                dataclass_runtime_adapter=dataclass_runtime_adapter,
            )

            # Step 3 (only in the flattening scenario): Wrap the torch-level op and get ``magi_exposed_op``.
            @functools.wraps(fn)
            def magi_exposed_op(*args, **kwargs):
                flat = dataclass_runtime_adapter.flatten_call_args(args, kwargs)
                return torch_registered_op(*flat)

            magi_exposed_op._magi_torch_registered_op = torch_registered_op
            magi_exposed_op._magi_param_mapping_tree = param_mapping_tree

            # Return magi-level op.
            return magi_exposed_op

    return decorator
