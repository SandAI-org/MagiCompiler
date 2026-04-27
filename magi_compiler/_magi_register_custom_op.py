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

import dataclasses
import functools
import inspect
import logging
from typing import Any, Callable, get_args, get_origin

import torch
import torch.utils._pytree as pytree

from ._triton_introspect import (
    get_bare_triton_kernels,
    get_inner_triton_kernels,
    get_referenced_heuristics_kernels,
    get_user_wrapped_triton_kernels,
    rewrite_fn_with_wrap_triton,
)
from .config import get_compile_config

logger = logging.getLogger(__name__)

_DATACLASS_PYTREE_REGISTERED: set[type] = set()

# ==============================================================================
# SECTION 1: Type Validation & Schema Error Prevention
# ------------------------------------------------------------------------------
# These helpers intercept various unsupported edge cases (like tuples in
# dataclasses, Literal/Enum annotations, or returning a Dataclass) and replace
# opaque torch.library internal errors setup with clear, actionable messages.
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
    """Reject container annotations that ``torch.library.infer_schema`` cannot
    consume in dataclass fields, with an actionable hint.

    Specifically:

    - ``tuple[T, ...]`` / ``Tuple[T, T]``: schema only models ``list``;
      suggest splitting into independent fields or switching to ``list``.
    - ``dict[K, V]`` / ``Dict[...]``: not supported by the schema at all.
    """
    origin = get_origin(tp)
    if origin is tuple:
        raise TypeError(
            f"@magi_register_custom_op: {where} is annotated with "
            f"{tp!r}. torch.library does not accept tuple-typed parameters "
            f"in op schemas. Either change the annotation to ``list[...]`` "
            f"or split the tuple into separate dataclass fields / op args."
        )
    if origin is dict:
        raise TypeError(
            f"@magi_register_custom_op: {where} is annotated with "
            f"{tp!r}. torch.library does not accept dict-typed parameters "
            f"in op schemas. Promote the values to explicit fields or "
            f"split them into separate op arguments."
        )


def _assert_op_name_namespaced(op_name: str) -> None:
    """Reject ``name=`` values that don't follow the ``namespace::op_name``
    convention.

    ``torch.library.custom_op("my_op", ...)`` (no ``::``) raises a relatively
    opaque error that doesn't suggest the fix to a first-time user. We catch
    it up front with a clear message that mirrors PyTorch's docs.
    """
    if "::" not in op_name:
        raise ValueError(
            f"@magi_register_custom_op: op name {op_name!r} is missing a "
            "namespace. Use ``namespace::op_name`` (e.g. "
            "``my_lib::my_op``). Pick a unique namespace for your project to "
            "avoid clashing with other libraries."
        )


_LITERAL_STRING_DOWNGRADE_HINT = (
    "Use ``str`` and validate the value inside the op body, e.g. "
    "``assert mode in ('a', 'b')``."
)


def _maybe_downgrade_literal_or_enum(annotation, *, where: str):
    """Return a schema-compatible annotation for ``Literal[str, ...]`` /
    ``Enum``-of-str inputs, by collapsing them to plain ``str``.

    ``torch.library.infer_schema`` does not understand ``Literal`` or ``Enum``
    annotations, but the underlying op only ever receives the *value* — and
    for the common "tag string" case (e.g. ``Literal["fp16", "bf16", "fp8"]``
    or ``class Quant(Enum): FP8 = "fp8"``) collapsing the annotation to
    ``str`` is both safe and lossless. The op body still gets the original
    string value at runtime.

    Numeric Literals / heterogeneous Literals / Enums whose values aren't all
    strings raise a clear ``TypeError`` instead, since those don't have an
    obvious safe downgrade.
    """
    import enum
    import typing

    origin = get_origin(annotation)
    # Handle ``Literal["a", "b"]``.
    if origin is typing.Literal:
        choices = get_args(annotation)
        if choices and all(isinstance(c, str) for c in choices):
            return str
        raise TypeError(
            f"@magi_register_custom_op: {where} is annotated with "
            f"{annotation!r}. Only ``Literal[str, ...]`` is auto-downgraded "
            f"to ``str``; mixed / numeric Literals are not supported by "
            f"torch.library schemas. {_LITERAL_STRING_DOWNGRADE_HINT}"
        )
    # Handle ``MyEnum`` whose members are all strings.
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        members = list(annotation)
        if members and all(isinstance(m.value, str) for m in members):
            return str
        raise TypeError(
            f"@magi_register_custom_op: {where} is annotated with Enum "
            f"{annotation.__name__!r} whose values are not all strings. "
            f"torch.library schemas don't support Enum directly. "
            f"{_LITERAL_STRING_DOWNGRADE_HINT}"
        )
    return annotation


def _assert_not_dataclass_return(tp, *, fn_name: str) -> None:
    """Reject return-type annotations that ``torch.library.infer_schema`` cannot
    consume (notably dataclasses), with an actionable hint.

    Returning a dataclass is a common mistake when users start grouping op
    inputs into a config dataclass and want symmetric outputs, but the schema
    layer only models ``Tensor`` / ``tuple[Tensor, ...]`` / ``list[Tensor]`` /
    ``None``. Without this guard users get a cryptic
    ``ValueError: Return has unsupported type`` deep inside ``infer_schema``.
    """
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        raise TypeError(
            f"@magi_register_custom_op: function {fn_name!r} is annotated to "
            f"return dataclass {tp.__name__!r}. torch.library only supports "
            "returning Tensor / tuple[Tensor, ...] / list[Tensor]. Either "
            "destructure the dataclass into a tuple at the op boundary, or "
            "wrap the dataclass-returning logic in a thin Python helper that "
            "calls the registered op."
        )


def _assert_not_mutable_dataclass(tp, *, where: str) -> None:
    """Raise a clear error if ``tp`` is a *non-frozen* dataclass type.

    ``magi_register_custom_op`` only supports ``frozen=True`` dataclasses as
    op inputs (and as nested fields). Without this guard the user gets a
    confusing ``ValueError: Unsupported type annotation X. It is not a type``
    from ``torch.library.infer_schema`` deep inside the registration call.

    Frozenness is required because ``torch.library`` / Inductor assume the
    flattened scalar inputs are stable for the duration of a tracing call;
    mutable dataclass instances would also break the pytree node hashing
    used by AOTAutograd.
    """
    if (
        isinstance(tp, type)
        and dataclasses.is_dataclass(tp)
        and getattr(tp, "__dataclass_params__", None) is not None
        and not tp.__dataclass_params__.frozen
    ):
        raise TypeError(
            f"@magi_register_custom_op: {where} is annotated with mutable "
            f"dataclass {tp.__name__!r}. Only @dataclass(frozen=True) is "
            f"supported (the schema needs a stable, hashable value). "
            f"Add ``frozen=True`` to {tp.__name__}."
        )


def _register_dataclass_pytree(cls: type) -> None:
    """
    Idempotently register ``cls`` as a pytree node so that TorchDynamo /
    AOTAutograd can flatten/unflatten dataclass instances when tracing.
    """
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
        # Already registered elsewhere (e.g. user code). Treat as success.
        pass
    _DATACLASS_PYTREE_REGISTERED.add(cls)


# ==============================================================================
# SECTION 2: Meta Function Auto-Generation
# ------------------------------------------------------------------------------
# Helpers to generate the meta/fake implementation required by `torch.library`.
# Fallbacks to identity_meta_fn (copying input properties to outputs) when
# the user does not provide `infer_output_meta_fn`.
# ==============================================================================


def _get_num_outputs_from_return_annotation(fn: Callable) -> int:
    """
    Get the number of outputs from the function's return type annotation.

    Returns:
    - 1 if the return type is a single Tensor
    - N if the return type is tuple[Tensor, Tensor, ...] with N elements
    - 1 if no annotation or unrecognized annotation (default to single output)
    """
    sig = inspect.signature(fn)
    return_annotation = sig.return_annotation

    if return_annotation is inspect.Parameter.empty:
        return 1

    # Check if it's a tuple type (e.g., tuple[Tensor, Tensor])
    origin = get_origin(return_annotation)
    if origin is tuple:
        args = get_args(return_annotation)
        # Filter out ellipsis (for variable-length tuples like tuple[Tensor, ...])
        if args and args[-1] is not ...:
            return len(args)
        return 1

    return 1


def _generate_op_name(fn: Callable) -> str:
    """
    Generate a unique operator name from function's name and source file.

    Format: {filename_stem}::{function_name}
    Example: my_module.py with function `my_op` -> "my_module::my_op"

    Falls back to "magi_custom::{function_name}" if source file cannot be determined.
    """
    import re
    from pathlib import Path

    func_name = fn.__name__

    # Get the source file path
    try:
        source_file = inspect.getfile(fn)
        # Extract the file stem (without extension) as namespace
        namespace = Path(source_file).stem
        # Clean up namespace: replace invalid characters with underscores
        namespace = re.sub(r"[^a-zA-Z0-9_]", "_", namespace)
    except (TypeError, OSError):
        # If we can't get the source file, use a default namespace
        namespace = "magi_custom"

    return f"{namespace}::{func_name}"


def _create_identity_meta_fn(fn: Callable) -> Callable:
    """
    Create a default identity meta function for the given function.

    The generated meta function:
    - Determines number of outputs from return type annotation
    - Uses first N tensor inputs to infer output metadata
    - Returns torch.empty_like() tensors with matching shape/dtype/device

    Raises ValueError if not enough tensor inputs are provided.
    """
    num_outputs = _get_num_outputs_from_return_annotation(fn)
    sig = inspect.signature(fn)
    # Get parameter names, excluding 'self' if present
    param_names = [name for name in sig.parameters.keys() if name != "self"]

    def identity_meta_fn(*args, **kwargs):
        # Bind arguments to get a mapping of param_name -> value
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        # Collect the first `num_outputs` tensor arguments
        tensor_args = []
        for name in param_names:
            arg = bound.arguments.get(name)
            if isinstance(arg, torch.Tensor):
                tensor_args.append(arg)
                if len(tensor_args) >= num_outputs:
                    break

        if len(tensor_args) < num_outputs:
            raise ValueError(
                f"identity_meta_fn requires at least {num_outputs} tensor inputs to match "
                f"{num_outputs} outputs, but only found {len(tensor_args)} tensor inputs. "
                f"Please provide a custom infer_output_meta_fn."
            )

        # Return outputs with same metadata as the first N inputs
        if num_outputs == 1:
            return torch.empty_like(tensor_args[0])
        return tuple(torch.empty_like(t) for t in tensor_args[:num_outputs])

    return identity_meta_fn


def _create_meta_fn_from_param_names(fn: Callable, param_names: list[str]) -> Callable:
    """
    Create a meta function that returns torch.empty_like() for each specified parameter.

    Args:
        fn: Target function to inspect
        param_names: List of parameter names to use as output templates

    Returns:
        Meta function that maps specified input params to output tensors

    Raises:
        ValueError: If parameter name doesn't exist or isn't a Tensor
    """
    sig = inspect.signature(fn)

    def meta_fn(*args, **kwargs):
        # Bind arguments to get a mapping of param_name -> value
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        # Collect tensors for each specified parameter name
        tensor_outputs = []
        for name in param_names:
            if name not in bound.arguments:
                raise ValueError(
                    f"Parameter '{name}' not found in function signature. "
                    f"Available parameters: {list(bound.arguments.keys())}"
                )
            arg = bound.arguments[name]
            if not isinstance(arg, torch.Tensor):
                raise ValueError(
                    f"Parameter '{name}' is not a Tensor (got {type(arg).__name__}). "
                    f"infer_output_meta_fn list should only contain tensor parameter names."
                )
            tensor_outputs.append(torch.empty_like(arg))

        # Return single tensor or tuple based on number of outputs
        if len(tensor_outputs) == 1:
            return tensor_outputs[0]
        return tuple(tensor_outputs)

    return meta_fn


# ==============================================================================
# SECTION 3: Dataclass Flattening & Schema Bridging
# ------------------------------------------------------------------------------
# These functions implement the core "dataclass-aware" mapping: recursively
# unpacking `@dataclass(frozen=True)` inputs into primitive scalars/tensors that
# `torch.library` can trace, and reassembling them into Python objects before
# handing them to the user's `backward_fn` or inner forward implementation.
# ==============================================================================


def _resolve_annotations(fn: Callable) -> dict[str, Any]:
    """Return ``fn``'s annotations, resolving any stringified ones (e.g. when
    ``from __future__ import annotations`` is in effect) into real types.

    Falls back to per-annotation resolution if ``get_type_hints`` cannot
    resolve every name atomically (which happens when the function is defined
    in a local scope, e.g. inside a test method, so its annotations reference
    names that live only in the enclosing closure).
    """
    import typing

    try:
        return typing.get_type_hints(fn)
    except Exception:
        pass

    # Build a best-effort namespace that combines globals + nonlocal closure
    # variables, so we can eval ``cfg: '_LocalDataclass'`` annotations from
    # functions defined inside other functions.
    fn_globals = getattr(fn, "__globals__", {}) or {}
    namespace: dict[str, Any] = dict(fn_globals)
    try:
        cv = inspect.getclosurevars(fn)
        namespace.update(cv.builtins)
        namespace.update(cv.nonlocals)
        namespace.update(cv.globals)
    except Exception:
        pass

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
    """Return ``cls``'s field name -> resolved type, with PEP 563 strings
    resolved best-effort (so nested dataclass types are real classes).
    """
    import typing as _typing

    try:
        return _typing.get_type_hints(cls)
    except Exception:
        # Fallback: take whatever ``dataclasses.fields`` exposes (which may be
        # a string under ``from __future__ import annotations``). Best-effort.
        return {f.name: f.type for f in dataclasses.fields(cls)}


_SCHEMA_DEFAULT_TYPES: tuple[type, ...] = (
    int,
    float,
    bool,
    str,
    torch.device,
    torch.dtype,
)


def _schema_compatible_param_default(default: Any) -> Any:
    """Return a default value safe to attach to an ``inspect.Parameter`` that
    will be handed to ``torch.library.infer_schema``.

    Same rules as :func:`_schema_compatible_default` but for raw values (used
    on the top-level parameter path, where defaults come from the user's
    function signature directly rather than from a ``dataclasses.Field``).
    """
    if default is inspect.Parameter.empty:
        return inspect.Parameter.empty
    if default is None or isinstance(default, _SCHEMA_DEFAULT_TYPES):
        return default
    return inspect.Parameter.empty


def _schema_compatible_default(f: "dataclasses.Field") -> Any:
    """Return a value safe to attach as ``inspect.Parameter.default`` for the
    flat parameter representing dataclass field ``f``.

    ``torch.library.infer_schema`` only renders defaults of ``None`` /
    ``int`` / ``float`` / ``bool`` / ``str`` / ``torch.device`` /
    ``torch.dtype``; anything else (e.g. a ``list`` from ``default_factory``)
    triggers ``"unsupported default value type"``. We therefore drop unsupported
    defaults — the outer dataclass instance still carries the real default for
    the user, so behaviour is preserved.
    """
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


def _build_dataclass_subplan(
    cls: type, attr_name: str, flat_prefix: str
) -> tuple[tuple, list[inspect.Parameter]]:
    """Recursively build a (sub-)plan and the corresponding flat parameters
    for one frozen-dataclass-typed value.

    ``attr_name`` is the attribute name on the parent dataclass (or the
    parameter name on ``fn`` when called for a top-level argument).

    ``flat_prefix`` is the dot-replaced prefix used to build leaf parameter
    names. For a top-level dataclass parameter ``cfg`` of type ``OuterCfg``
    with a nested ``inner: InnerCfg(val: float)``, the flat parameter name
    for the leaf is ``cfg__inner__val``.

    Returns ``(node, flat_params)`` where ``node`` is the recursive plan node
    and ``flat_params`` is the list of leaf ``inspect.Parameter`` objects in
    DFS order.
    """
    _register_dataclass_pytree(cls)

    field_types = _resolve_dataclass_field_types(cls)
    children: list[tuple] = []
    flat_params: list[inspect.Parameter] = []

    for f in dataclasses.fields(cls):
        f_type = field_types.get(f.name, f.type)
        child_flat_name = f"{flat_prefix}__{f.name}"
        if isinstance(f_type, str):
            raise TypeError(
                f"@magi_register_custom_op: field {cls.__name__}.{f.name} has "
                f"an unresolved string annotation {f_type!r}. This usually "
                "happens when the field's type is defined inside a function "
                "body (a 'local class') combined with "
                "``from __future__ import annotations``. Move the type to "
                "module scope, or import it at module scope, so "
                "``typing.get_type_hints`` can resolve it."
            )
        _assert_not_mutable_dataclass(f_type, where=f"field {cls.__name__}.{f.name}")
        if _is_frozen_dataclass(f_type):
            sub_node, sub_params = _build_dataclass_subplan(
                f_type, attr_name=f.name, flat_prefix=child_flat_name
            )
            children.append(sub_node)
            flat_params.extend(sub_params)
        else:
            _assert_not_unsupported_container(
                f_type, where=f"field {cls.__name__}.{f.name}"
            )
            f_type = _maybe_downgrade_literal_or_enum(
                f_type, where=f"field {cls.__name__}.{f.name}"
            )
            children.append(("primitive", f.name, child_flat_name, None))
            # Carry the dataclass field's default (or default_factory product)
            # over to the flat parameter so torch.library.infer_schema records
            # it as optional. ``infer_schema`` only accepts defaults of types
            # ``None``, ``int``, ``float``, ``bool``, ``str``, ``torch.device``,
            # ``torch.dtype``; other defaults (notably ``list``/``dict`` from
            # ``default_factory``) are left as "required" on the flat param.
            # Either way the outer wrapper still gets the real default value
            # via ``cls(**user_kwargs)`` since users construct the dataclass
            # instance themselves.
            flat_params.append(
                inspect.Parameter(
                    child_flat_name,
                    # NOTE: POSITIONAL_OR_KEYWORD because torch.library.custom_op
                    # does not yet support kwarg-only Tensor arguments.
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=_schema_compatible_default(f),
                    annotation=f_type,
                )
            )

    return ("dataclass", attr_name, cls, children), flat_params


def _build_flat_signature(fn: Callable):
    """
    Build a "flat" signature that only contains primitive-typed parameters by
    recursively expanding any frozen-dataclass-typed parameter into its
    individual leaf fields. Nested dataclasses (dataclass-of-dataclass) are
    fully unwrapped: ``cfg: OuterCfg`` with ``inner: InnerCfg(val: float)``
    becomes a flat parameter ``cfg__inner__val: float``.

    Returns:
        flat_sig (inspect.Signature): signature with all dataclass params
            recursively expanded.
        plan (list[tuple]): per-original-parameter plan describing how to
            reassemble values from a flat kwargs dict. Each entry is one of:
              * ``("primitive", attr_name, flat_name, None)`` for a leaf
                whose runtime value is read from / written to the flat
                ``flat_name`` slot;
              * ``("dataclass", attr_name, cls, [sub_plan_nodes...])`` for a
                dataclass node whose children follow the same recursive
                structure.
            ``attr_name`` is the parameter name on ``fn`` at the top level
            and the field name on the parent dataclass deeper in the tree.
        user_sig (inspect.Signature): the original (un-flattened) signature
            of ``fn`` for binding user calls.
    """
    user_sig = inspect.signature(fn)
    # Resolve stringified annotations (PEP 563 / ``from __future__ import
    # annotations``) so ``_is_frozen_dataclass`` can recognise dataclass-typed
    # parameters and dataclass field types are real ``type`` objects.
    resolved = _resolve_annotations(fn)
    flat_params: list[inspect.Parameter] = []
    plan: list[tuple] = []

    for name, param in user_sig.parameters.items():
        annotation = resolved.get(name, param.annotation)
        _assert_not_mutable_dataclass(annotation, where=f"parameter {name!r}")
        if _is_frozen_dataclass(annotation):
            node, sub_params = _build_dataclass_subplan(
                annotation, attr_name=name, flat_prefix=name
            )
            plan.append(node)
            flat_params.extend(sub_params)
        else:
            _assert_not_unsupported_container(annotation, where=f"parameter {name!r}")
            annotation = _maybe_downgrade_literal_or_enum(
                annotation, where=f"parameter {name!r}"
            )
            new_param = param.replace(
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=annotation,
                default=_schema_compatible_param_default(param.default),
            )
            flat_params.append(new_param)
            plan.append(("primitive", name, name, None))

    return_annotation = resolved.get("return", user_sig.return_annotation)
    _assert_not_dataclass_return(return_annotation, fn_name=fn.__name__)
    flat_sig = inspect.Signature(flat_params, return_annotation=return_annotation)
    return flat_sig, plan, user_sig


def _signatures_differ(a: inspect.Signature, b: inspect.Signature) -> bool:
    """Return True iff two signatures disagree on at least one of: parameter
    names, annotations, defaults, kinds, or return annotation.

    We use this to decide whether the flat signature meaningfully differs
    from the user signature (and thus needs an ``inner_fn`` wrapper) on the
    no-dataclass path. ``inspect.Signature.__eq__`` is conservative enough
    for our purposes; we wrap it for clarity at the call site.
    """
    return a != b


def _make_flat_signature_wrapper(fn: Callable, flat_sig: inspect.Signature) -> Callable:
    """Return a thin wrapper around ``fn`` whose ``__signature__`` /
    ``__annotations__`` reflect ``flat_sig`` (annotations downgraded /
    defaults scrubbed). The body simply forwards every argument through.

    The wrapper is required because ``torch.library.infer_schema`` reads
    ``inspect.signature(fn)`` and would otherwise see the user's original
    (un-scrubbed) annotations. We also strip ``__wrapped__`` so
    ``inspect.signature`` doesn't unwrap back to the original.
    """

    @functools.wraps(fn)
    def _wrapped(*args, **kwargs):
        return fn(*args, **kwargs)

    _wrapped.__signature__ = flat_sig
    flat_annotations = {
        p.name: p.annotation
        for p in flat_sig.parameters.values()
        if p.annotation is not inspect.Parameter.empty
    }
    if flat_sig.return_annotation is not inspect.Signature.empty:
        flat_annotations["return"] = flat_sig.return_annotation
    _wrapped.__annotations__ = flat_annotations
    try:
        del _wrapped.__wrapped__
    except AttributeError:
        pass
    return _wrapped


def _build_value_from_node(node: tuple, flat_kwargs: dict):
    """Recursively materialise the value described by a plan ``node`` from
    the flat kwargs dict produced by ``_build_flat_signature``.

    Used by both the meta-side reassembly (when torch.library hands us back
    flat kwargs) and any other site that needs the original-shaped value.
    """
    kind = node[0]
    if kind == "primitive":
        _, _attr, flat_name, _ = node
        return flat_kwargs[flat_name]
    # ``("dataclass", attr, cls, children)``
    _, _attr, cls, children = node
    init_kwargs: dict[str, Any] = {}
    for child in children:
        # ``child[1]`` is the field name on ``cls`` regardless of node kind.
        field_name = child[1]
        init_kwargs[field_name] = _build_value_from_node(child, flat_kwargs)
    return cls(**init_kwargs)


def _reassemble_user_kwargs(plan: list[tuple], flat_kwargs: dict) -> dict:
    """Reconstruct the original (possibly nested-dataclass-bearing) kwargs
    from flat kwargs. Mirrors :func:`_build_flat_signature`.
    """
    out: dict[str, Any] = {}
    for node in plan:
        # Top-level node: ``node[1]`` is the original parameter name on ``fn``.
        out[node[1]] = _build_value_from_node(node, flat_kwargs)
    return out


def _flatten_value_into(node: tuple, value: Any, out: list) -> None:
    """Recursively flatten ``value`` according to plan ``node``, appending
    leaf primitives to ``out`` in DFS order.
    """
    kind = node[0]
    if kind == "primitive":
        out.append(value)
        return
    _, _attr, cls, children = node
    # We don't isinstance-check ``cls`` here on purpose: users may pass
    # arbitrary objects that quack like the dataclass (e.g. mocks). We just
    # rely on getattr for each declared field.
    for child in children:
        field_name = child[1]
        _flatten_value_into(child, getattr(value, field_name), out)


def _flatten_call_args(
    plan: list[tuple], user_sig: inspect.Signature, args: tuple, kwargs: dict
) -> list:
    """
    Flatten a user-side call (which may pass nested dataclass instances) into
    a positional list. The order matches the flat signature produced by
    :func:`_build_flat_signature`.
    """
    bound = user_sig.bind(*args, **kwargs)
    bound.apply_defaults()
    flat: list = []
    for node in plan:
        # ``node[1]`` is the top-level parameter name; primitives are passed
        # through unchanged, dataclasses are recursively unwrapped.
        _flatten_value_into(node, bound.arguments[node[1]], flat)
    return flat


def _count_leaves(node: tuple) -> int:
    """Number of flat parameter slots a plan ``node`` occupies."""
    if node[0] == "primitive":
        return 1
    return sum(_count_leaves(c) for c in node[3])


def _flatten_grad_into(node: tuple, grad: Any, out: list) -> None:
    """Spread a user-returned grad for one original-signature input across
    the flat parameter slots described by ``node``.

    Rules:
    * ``primitive`` node: append ``grad`` (whatever the user returned) as-is.
    * ``dataclass`` node:
        - If the user returned ``None``: every leaf slot under this node
          gets ``None`` (whole-dataclass-not-differentiable, the common case).
        - Otherwise: the user is returning a dataclass-shaped grad object.
          We descend recursively, reading each child via
          ``getattr(grad, field_name)``. Missing fields are treated as
          ``None``. This mirrors :func:`_flatten_value_into` but is more
          forgiving so users can return a plain ``SimpleNamespace`` /
          ``dict``-like object too (we accept ``dict`` via ``__getitem__``).
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


def _collect_tensor_leaf_flat_names(node: tuple) -> list[str]:
    """Return the flat parameter names of every Tensor leaf under ``node``.

    Used to expand a top-level dataclass parameter referenced in
    ``mutates_args`` into the actual flat parameter names that
    ``torch.library`` sees.
    """
    if node[0] == "primitive":
        _, _attr, flat_name, _ = node
        anno = node[3]  # currently always ``None`` for primitive nodes
        # We don't actually have the annotation cached here -- callers expand
        # all leaves under a dataclass and torch.library will validate Tensor
        # types itself, so over-specifying is acceptable.
        return [flat_name]
    out: list[str] = []
    for child in node[3]:
        out.extend(_collect_tensor_leaf_flat_names(child))
    return out


def _expand_mutates_args(
    mutates_args: tuple[str, ...] | list[str],
    plan: list[tuple],
) -> tuple[str, ...]:
    """Translate ``mutates_args`` from the *original* parameter space to the
    *flat* parameter space.

    * Names that already match a primitive top-level parameter pass through.
    * A name that matches a top-level dataclass parameter expands into every
      flat leaf under it (``cfg`` -> ``cfg__a``, ``cfg__b__x``, ...).
    * Names of the form ``cfg__inner__x`` (already in flat space) pass through
      unchanged so users can be precise if they want to.
    * Unknown names raise ``ValueError`` with the list of valid choices.
    """
    if not mutates_args:
        return tuple(mutates_args)
    by_attr: dict[str, tuple] = {node[1]: node for node in plan}
    valid_flat: set[str] = set()
    for node in plan:
        valid_flat.update(_collect_tensor_leaf_flat_names(node))
    out: list[str] = []
    for name in mutates_args:
        if name in by_attr:
            node = by_attr[name]
            if node[0] == "primitive":
                out.append(node[2])
            else:
                out.extend(_collect_tensor_leaf_flat_names(node))
        elif name in valid_flat:
            out.append(name)
        else:
            raise ValueError(
                f"@magi_register_custom_op: mutates_args entry {name!r} does "
                f"not match any parameter. Valid original-space names: "
                f"{sorted(by_attr.keys())}; valid flat-space names: "
                f"{sorted(valid_flat)}."
            )
    seen: set[str] = set()
    deduped: list[str] = []
    for n in out:
        if n not in seen:
            seen.add(n)
            deduped.append(n)
    return tuple(deduped)


def _flatten_user_grads(plan: list[tuple], user_grads: tuple | list) -> list:
    """Convert a tuple of grads keyed by *original* parameter order into the
    flat tuple keyed by the flat-op parameter order.

    Length of ``user_grads`` MUST equal ``len(plan)`` (one grad per original
    input). Raises ``ValueError`` otherwise so users get a clear message
    instead of an opaque autograd shape error.
    """
    if len(user_grads) != len(plan):
        raise ValueError(
            f"backward_fn returned {len(user_grads)} grad(s) but the original "
            f"function had {len(plan)} input(s). When using a frozen-dataclass "
            f"input, return one grad per ORIGINAL parameter (use ``None`` for "
            f"non-differentiable ones, including whole dataclass arguments)."
        )
    flat: list = []
    for node, g in zip(plan, user_grads):
        _flatten_grad_into(node, g, flat)
    return flat


# ==============================================================================
# SECTION 4: Triton Introspection & Wrapping Intercept
# ------------------------------------------------------------------------------
# These helpers interface with the `torch.library` triton registry and our custom
# introspector. They identify which kernels the op uses, detect unsupported
# heuristics, and optionally rebuild the function to shadow module-global kernel
# references with `wrap_triton(...)` so Inductor can trace through the op.
# ==============================================================================


def _assert_wrap_triton_compatible(kernels: list[Any]) -> None:
    """Reject kernels whose outermost decorator is ``@triton.heuristics``.

    ``torch.library.wrap_triton`` (the only entry point into ``triton_op`` /
    Inductor's traceable HOP path) hard-codes::

        isinstance(triton_kernel, (JITFunction, Autotuner))

    A bare ``@triton.heuristics`` (or ``@triton.heuristics`` wrapping
    ``@triton.autotune``) produces a ``Heuristics`` object that fails this
    check at runtime with a confusing
    ``"wrap_triton only works on functions annotated with triton.jit or
    triton.autotune"``. We surface a clearer error here pointing at the fix
    (``@triton.autotune → @triton.heuristics → @triton.jit``).
    """
    if not kernels:
        return
    try:
        from triton.runtime.autotuner import Autotuner, Heuristics
        from triton.runtime.jit import JITFunction
    except ImportError:
        return
    for k in kernels:
        if isinstance(k, (JITFunction, Autotuner)):
            continue
        if isinstance(k, Heuristics):
            name = getattr(getattr(k, "fn", None), "__name__", repr(k))
            raise RuntimeError(
                f"@magi_register_custom_op: triton kernel {name!r} has "
                "@triton.heuristics as its outermost decorator. "
                "torch.library.wrap_triton (and therefore triton_op / Inductor) "
                "only accepts triton.jit or triton.autotune at the top level. "
                "Either remove @triton.heuristics, or place @triton.autotune "
                "outside it: @triton.autotune -> @triton.heuristics -> @triton.jit."
            )


def _resolve_triton_kernels(
    fn: Callable,
    extra_triton_kernels: list[Any] | tuple[Any, ...] | None,
) -> tuple[list[Any], list[Any], set[int]]:
    """Best-effort: collect triton kernels referenced inside ``fn``.

    Returns ``(all_kernels, bare_kernels, user_wrapped_ids)``:

    * ``all_kernels`` is the union of user-supplied ``extra_triton_kernels``
      and *every* kernel discovered by source introspection. This is the list
      we use to decide whether to take the ``triton_op`` registration path.
    * ``bare_kernels`` is the subset that is invoked via the bare
      ``kernel[grid](...)`` pattern (the user did NOT wrap them themselves).
      Only these need to be shadowed by ``rewrite_fn_with_wrap_triton``;
      shadowing kernels the user already wrapped would yield
      ``wrap_triton(wrap_triton(kernel))`` and crash at runtime.
    * ``user_wrapped_ids`` is the set of kernel object ids the user has
      already wrapped explicitly via ``wrap_triton(k)`` in the source. We
      forward this to the rewriter as an exclusion list so its blanket
      "wrap every JITFunction in fn.__globals__" pass does not double-wrap
      a kernel that's also referenced from a manual ``wrap_triton(k)`` call
      in the same body.

    User-supplied ``extra_triton_kernels`` are treated as bare (the user is
    asking us to wrap them on their behalf). Both lists are deduplicated by
    ``id(kernel)`` and preserve user-supplied order first.
    """
    seen_all: set[int] = set()
    all_kernels: list[Any] = []
    seen_bare: set[int] = set()
    bare_kernels: list[Any] = []
    for k in extra_triton_kernels or ():
        kid = id(k)
        if kid not in seen_all:
            seen_all.add(kid)
            all_kernels.append(k)
        if kid not in seen_bare:
            seen_bare.add(kid)
            bare_kernels.append(k)
    try:
        detected_all = get_inner_triton_kernels(fn)
    except Exception:
        logger.debug("get_inner_triton_kernels(%r) failed", fn, exc_info=True)
        detected_all = []
    try:
        detected_bare = get_bare_triton_kernels(fn)
    except Exception:
        logger.debug("get_bare_triton_kernels(%r) failed", fn, exc_info=True)
        detected_bare = []
    for k in detected_all:
        if id(k) not in seen_all:
            seen_all.add(id(k))
            all_kernels.append(k)
    for k in detected_bare:
        if id(k) not in seen_bare:
            seen_bare.add(id(k))
            bare_kernels.append(k)

    # Reject kernels whose outermost decorator is @triton.heuristics with a
    # readable error before the user hits either (a) wrap_triton's opaque
    # RuntimeError or (b) a silent fallback to plain custom_op (because the
    # introspector unwraps Heuristics via ``obj.fn`` and may even drop it
    # entirely when callable(Heuristics_instance) is False, hiding the
    # outer layer from ``all_kernels``).
    try:
        referenced_heuristics = get_referenced_heuristics_kernels(fn)
    except Exception:
        logger.debug("get_referenced_heuristics_kernels(%r) failed", fn, exc_info=True)
        referenced_heuristics = []
    _assert_wrap_triton_compatible(
        list(extra_triton_kernels or ()) + list(referenced_heuristics)
    )
    try:
        user_wrapped = get_user_wrapped_triton_kernels(fn)
    except Exception:
        logger.debug("get_user_wrapped_triton_kernels(%r) failed", fn, exc_info=True)
        user_wrapped = []
    user_wrapped_ids = {id(k) for k in user_wrapped}
    return all_kernels, bare_kernels, user_wrapped_ids


# ==============================================================================
# SECTION 5: Core Registration & Main Decorator
# ------------------------------------------------------------------------------
# These functions implement the actual dispatch layer: registering the op name
# avoiding duplicates, handling the fallback from `<triton_op>` to `<custom_op>`,
# and the outermost `magi_register_custom_op_impl` orchestration.
# ==============================================================================

_REGISTERED_OP_NAMES: set[str] = set()


def _assert_op_name_unused(op_name: str) -> None:
    """Raise a clear error if ``op_name`` was already registered through this
    decorator (or already exists on ``torch.ops``).

    Without this guard, ``torch.library.custom_op`` raises a low-level
    ``RuntimeError`` referring to schema fingerprints that is hard to map back
    to "you have two ``@magi_register_custom_op`` calls with the same name".
    """
    if op_name in _REGISTERED_OP_NAMES:
        raise RuntimeError(
            f"@magi_register_custom_op: op name {op_name!r} is already "
            "registered. Each magi op must use a unique "
            "``namespace::op_name``. If you really want to override, delete "
            "the previous registration with "
            "``torch.library._del_library_impl`` first, or pass an explicit "
            "``name=`` to disambiguate."
        )
    ns, _, opname = op_name.partition("::")
    if ns and opname:
        ns_obj = getattr(torch.ops, ns, None)
        if ns_obj is not None and hasattr(ns_obj, opname):
            raise RuntimeError(
                f"@magi_register_custom_op: op name {op_name!r} is already "
                f"defined on torch.ops.{ns}. Use a different name (or pass an "
                "explicit ``name=`` to your decorator) to avoid clashing with "
                "an existing operator."
            )


def _register_op(
    op_name: str,
    fn: Callable,
    mutates_args: tuple[str, ...],
    meta_fn: Callable,
    user_supplied_meta: bool,
    triton_kernels: list[Any],
    bare_triton_kernels: list[Any] | None = None,
    signature_override: inspect.Signature | None = None,
    excluded_kernel_ids: set[int] | None = None,
):
    """Register ``fn`` either as a triton_op (when triton kernels are present)
    or as a plain custom_op, with sensible fallback if triton_op registration
    fails.

    ``bare_triton_kernels`` is the subset of ``triton_kernels`` that the user
    did NOT already wrap in ``wrap_triton(...)`` themselves. Only those are
    fed to :func:`rewrite_fn_with_wrap_triton` so we never produce a
    ``wrap_triton(wrap_triton(kernel))``. Defaults to ``triton_kernels`` for
    backwards compatibility.

    Returns the resulting ``CustomOpDef`` instance.
    """
    if bare_triton_kernels is None:
        bare_triton_kernels = triton_kernels
    if triton_kernels:
        try:
            from torch.library import triton_op
        except ImportError:
            triton_op = None  # type: ignore[assignment]
            logger.warning(
                "torch.library.triton_op not available; falling back to "
                "torch.library.custom_op for op %s",
                op_name,
            )

        if triton_op is not None:
            try:
                fn_for_register = rewrite_fn_with_wrap_triton(
                    fn, bare_triton_kernels, excluded_kernel_ids=excluded_kernel_ids
                )
                # ``rewrite_fn_with_wrap_triton`` builds a fresh
                # ``types.FunctionType`` from ``fn.__code__``; if ``fn`` is a
                # thin signature-rewriting wrapper (e.g. for Literal /
                # default-list scrubbing), the freshly built function has the
                # wrapper's ``(*args, **kwargs)`` code object, so we need to
                # re-attach the cleaned signature for ``infer_schema``.
                if signature_override is not None:
                    fn_for_register.__signature__ = signature_override
                    fn_for_register.__annotations__ = {
                        p.name: p.annotation
                        for p in signature_override.parameters.values()
                        if p.annotation is not inspect.Parameter.empty
                    }
                    if (
                        signature_override.return_annotation
                        is not inspect.Signature.empty
                    ):
                        fn_for_register.__annotations__["return"] = (
                            signature_override.return_annotation
                        )
                registered_op = triton_op(op_name, mutates_args=mutates_args)(
                    fn_for_register
                )
                # ``triton_op`` already registers ``fn`` as the fake/meta
                # implementation. Only override when the user explicitly
                # supplied an ``infer_output_meta_fn``.
                if user_supplied_meta:
                    registered_op.register_fake(meta_fn)
                return registered_op
            except Exception:
                logger.warning(
                    "triton_op registration failed for %s; falling back to "
                    "custom_op + register_fake. Inductor will not be able to "
                    "see through the op.",
                    op_name,
                    exc_info=True,
                )

    registered_op = torch.library.custom_op(op_name, mutates_args=mutates_args)(fn)
    torch.library.register_fake(op_name)(meta_fn)
    return registered_op


def _magi_register_custom_op_impl(
    name: str | None = None,
    mutates_args: tuple[str, ...] = (),
    infer_output_meta_fn: Callable | list[str] | None = None,
    setup_context_fn: Callable | None = None,
    backward_fn: Callable | None = None,
    is_compute_sensitive: bool = False,
    is_subgraph_boundary: bool = False,
    extra_triton_kernels: list[Any] | tuple[Any, ...] | None = None,
):
    def decorator(fn: Callable) -> Callable:
        # Auto-generate name if not provided
        op_name = name if name is not None else _generate_op_name(fn)
        _assert_op_name_namespaced(op_name)
        _assert_op_name_unused(op_name)
        if is_compute_sensitive:
            get_compile_config().recompute_config.custom_compute_sensitive_ops.append(
                op_name
            )
        if is_subgraph_boundary:
            get_compile_config().splitting_ops.append(op_name)

        # Detect whether any input is a frozen dataclass; if not, fall through
        # to the original (zero-overhead) registration path.
        flat_sig, plan, user_sig = _build_flat_signature(fn)
        has_dataclass = any(kind == "dataclass" for kind, *_ in plan)

        if not has_dataclass:
            # The flat signature may differ from the user's signature even
            # without any dataclass input: we may have downgraded a Literal /
            # Enum annotation to ``str`` or scrubbed a list/dict default that
            # ``infer_schema`` cannot consume. In those cases we route through
            # a thin wrapper whose ``__signature__`` is ``flat_sig`` so the
            # schema sees the cleaned-up version. Otherwise we register ``fn``
            # directly to preserve the original zero-overhead path.
            sig_was_rewritten = _signatures_differ(flat_sig, user_sig)
            fn_for_register = (
                _make_flat_signature_wrapper(fn, flat_sig) if sig_was_rewritten else fn
            )

            # Step 1: Build the meta/fake function (used either as a
            # register_fake override on the triton path, or as the regular
            # fake implementation on the plain custom_op path).
            meta_target = fn_for_register
            if infer_output_meta_fn is None:
                meta_fn = _create_identity_meta_fn(meta_target)
                user_supplied_meta = False
            elif isinstance(infer_output_meta_fn, list):
                meta_fn = _create_meta_fn_from_param_names(
                    meta_target, infer_output_meta_fn
                )
                user_supplied_meta = True
            else:
                meta_fn = infer_output_meta_fn
                user_supplied_meta = True

            # Step 2: Detect inner triton kernels and register the op via
            # triton_op (if any kernels are present) or custom_op (otherwise).
            triton_kernels, bare_triton_kernels, user_wrapped_ids = (
                _resolve_triton_kernels(fn, extra_triton_kernels)
            )
            registered_op = _register_op(
                op_name=op_name,
                fn=fn_for_register,
                mutates_args=mutates_args,
                meta_fn=meta_fn,
                user_supplied_meta=user_supplied_meta,
                triton_kernels=triton_kernels,
                bare_triton_kernels=bare_triton_kernels,
                signature_override=flat_sig if sig_was_rewritten else None,
                excluded_kernel_ids=user_wrapped_ids,
            )

            # Step 3: Register autograd if backward_fn is provided
            if backward_fn is not None:
                registered_op.register_autograd(
                    backward_fn, setup_context=setup_context_fn
                )

            _REGISTERED_OP_NAMES.add(op_name)
            return registered_op

        # ----- Dataclass-aware path -----
        # Build inner_fn whose signature contains only primitive types so that
        # torch.library.custom_op's schema validator is happy. The inner_fn
        # accepts positional/keyword args following the flat signature; we
        # bind them, reassemble dataclasses, then call the original ``fn``.
        def _bind_to_user_kwargs(args, kwargs):
            bound = flat_sig.bind(*args, **kwargs)
            bound.apply_defaults()
            return _reassemble_user_kwargs(plan, bound.arguments)

        # Detect triton kernels referenced from the original (dataclass-typed)
        # fn. If any are present, route ``inner_fn`` through a wrap_triton-aware
        # copy of ``fn`` so the eventual triton_op registration captures them.
        triton_kernels, bare_triton_kernels, user_wrapped_ids = _resolve_triton_kernels(
            fn, extra_triton_kernels
        )
        fn_for_inner = (
            rewrite_fn_with_wrap_triton(
                fn, bare_triton_kernels, excluded_kernel_ids=user_wrapped_ids
            )
            if bare_triton_kernels
            else fn
        )

        @functools.wraps(fn)
        def inner_fn(*args, **kwargs):
            return fn_for_inner(**_bind_to_user_kwargs(args, kwargs))

        inner_fn.__signature__ = flat_sig
        # ``functools.wraps`` set ``__wrapped__`` to ``fn``; that makes
        # ``inspect.signature(inner_fn)`` follow back to ``fn`` (which still
        # carries the dataclass-typed annotations) and bypass our flat
        # ``__signature__`` override. ``triton_op`` / ``infer_schema`` rely on
        # ``inspect.signature`` and would then choke on the dataclass type
        # annotation. Strip the wrapper marker so the flat signature wins.
        try:
            del inner_fn.__wrapped__
        except AttributeError:
            pass
        # Replace the dataclass-typed annotations copied over by
        # ``functools.wraps`` with the flat-signature annotations so that any
        # tool reading ``__annotations__`` directly (e.g. ``get_type_hints``)
        # also sees the primitive types torch.library expects.
        flat_annotations = {
            p.name: p.annotation
            for p in flat_sig.parameters.values()
            if p.annotation is not inspect.Parameter.empty
        }
        if flat_sig.return_annotation is not inspect.Signature.empty:
            flat_annotations["return"] = flat_sig.return_annotation
        inner_fn.__annotations__ = flat_annotations

        # Build the meta function based on the flat signature.
        if infer_output_meta_fn is None:
            meta_fn = _create_identity_meta_fn(inner_fn)
            user_supplied_meta = False
        elif isinstance(infer_output_meta_fn, list):
            meta_fn = _create_meta_fn_from_param_names(inner_fn, infer_output_meta_fn)
            user_supplied_meta = True
        else:
            user_meta = infer_output_meta_fn

            def meta_fn(*args, **kwargs):
                return user_meta(**_bind_to_user_kwargs(args, kwargs))

            meta_fn.__signature__ = flat_sig
            user_supplied_meta = True

        flat_mutates_args = _expand_mutates_args(mutates_args, plan)
        registered_op = _register_op(
            op_name=op_name,
            fn=inner_fn,
            mutates_args=flat_mutates_args,
            meta_fn=meta_fn,
            signature_override=flat_sig,
            user_supplied_meta=user_supplied_meta,
            triton_kernels=triton_kernels,
            # ``inner_fn`` already wraps a rewritten copy of ``fn``, so we do
            # NOT want _register_op to rewrite a second time (that would
            # introspect ``inner_fn`` and re-wrap kernels referenced via the
            # original ``fn`` closure). Pass an empty list to short-circuit.
            bare_triton_kernels=[],
        )

        # Bridge user-supplied autograd hooks (which speak the ORIGINAL
        # dataclass signature) into the FLAT signature actually registered
        # with torch.library.
        #
        # On the forward pass torch.library calls
        #   setup_context(ctx, inputs=<flat tuple>, output=...)
        # On the backward pass it expects the user's ``backward`` to return
        # one grad per FLAT input. Users naturally want to write both in
        # terms of the original (dataclass-bearing) signature, so we wrap
        # both ends.
        if backward_fn is not None:
            user_setup = setup_context_fn
            user_backward = backward_fn

            def _bridged_setup_context(ctx, inputs, output):
                if user_setup is None:
                    return None
                # ``inputs`` is the flat positional tuple in the order of
                # ``flat_sig``. Reassemble it into the user's original
                # (possibly nested-dataclass-bearing) shape.
                flat_kwargs = {
                    p.name: v for p, v in zip(flat_sig.parameters.values(), inputs)
                }
                user_kwargs = _reassemble_user_kwargs(plan, flat_kwargs)
                # Preserve original positional order so users can do
                # ``x, cfg = inputs`` exactly like in the no-dataclass case.
                user_inputs = tuple(user_kwargs[p] for p in user_sig.parameters)
                return user_setup(ctx, user_inputs, output)

            def _bridged_backward(ctx, *grads):
                user_grads = user_backward(ctx, *grads)
                if not isinstance(user_grads, tuple):
                    # Single-input convenience: PyTorch allows returning a
                    # bare grad if the op has a single input. Mirror that.
                    user_grads = (user_grads,)
                return tuple(_flatten_user_grads(plan, user_grads))

            registered_op.register_autograd(
                _bridged_backward, setup_context=_bridged_setup_context
            )

        # Outer wrapper preserves the original (dataclass-aware) signature for
        # users while routing through the registered (flat) op underneath.
        @functools.wraps(fn)
        def outer_wrapper(*args, **kwargs):
            flat = _flatten_call_args(plan, user_sig, args, kwargs)
            return registered_op(*flat)

        outer_wrapper._magi_inner_op = registered_op
        outer_wrapper._magi_flat_plan = plan
        _REGISTERED_OP_NAMES.add(op_name)
        return outer_wrapper

    return decorator
