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

import copy
import functools
import inspect
from typing import Any, Callable, TypeVar

from ._api import (
    _check_dynamic_arg_dims,
    _infer_dynamic_arg_dims,
    _magi_compile_bound_method,
    _magi_compile_class,
    _magi_compile_function,
)
from ._magi_register_custom_op import _magi_register_custom_op_impl
from .config import CompileConfig, CompileMode, get_compile_config

_T = TypeVar("_T", bound=type)
_F = TypeVar("_F", bound=Callable)
_O = TypeVar("_O", bound=object)


def magi_compile(
    obj: _T | _O | _F | None = None,
    *,
    model_tag: str | None = None,
    dynamic_arg_dims: dict[str, int | list[int]] | None = None,
    enable_if: Callable[[], bool] | None = None,
    config_patch: Callable[[CompileConfig], CompileConfig] | None = None,
    method_name: str | None = None,
) -> _T | _O | _F | Callable[[_T | _O | _F], _T | _O | _F]:
    """
    Compile classes, instances, standalone functions, or bound methods.

    Default compile target when no explicit method is passed:
    - ``nn.Module``: compile ``forward``.
    - Non-module callable class/instance: compile ``forward`` by default;
      if missing, users must pass ``method_name`` explicitly.

    Supported target types
    ----------------------
    1) Class:
        - Hooks ``__init__`` so every new instance gets the default method compiled (same mechanism for
          ``nn.Module`` and non-module callable classes).
        - Example:
            @magi_compile
            class MyModel(nn.Module):
                def forward(self, x): return x

    2) Function (Standalone):
        - Wraps a callable with MagiCompiler's dispatch logic.
        - Useful for non-member functions or general callables.
        - Example:
            @magi_compile
            def my_func(x): return x

    3) Instance:
        - Compiles only that object’s default method (``forward`` by default, or
          explicit ``method_name`` for non-module targets).
        - Example:
            model = MyModel()
            model = magi_compile(model)

    4) Bound method:
        - Compiles that method on its ``__self__`` (works for ``nn.Module`` and plain objects).
        - Example:
            model = MyModel()
            model.forward = magi_compile(model.forward)

    Usage Styles
    ------------
    The compiler supports both declarative (decorator) and imperative (function call) styles.

    A) Decorator Style:
        - Example:
            @magi_compile(dynamic_arg_dims={"x": 0})
            class MyModel(nn.Module): ...

            class MyModel(nn.Module):
                @magi_compile
                def forward(self, x): ...

    B) Imperative Style:
        - Apply directly to an existing object:
            model = magi_compile(model, dynamic_arg_dims={"x": 0})

    C) Factory Style:
        - Configure a compiler first, then apply to multiple objects:
            compiler = magi_compile(dynamic_arg_dims={"x": 0})
            model = compiler(model)
            cls = compiler(MyModel)

    Arguments
    ---------
    - dynamic_arg_dims: Dictionary mapping argument names to dynamic dimensions (int or list[int]).
    - model_tag: Optional tag for caching path (defaults to class/function name).
    - enable_if: Callable returning bool; compilation happens only if this returns True.
    - method_name: Optional explicit method for class/instance targets. If omitted,
      ``forward`` is used by default; for non-module targets without ``forward``,
      this argument is required.

    Notes
    -----
    - If `dynamic_arg_dims` is omitted, it is inferred from type annotations:
      `torch.Tensor` arguments default to dynamic dimension 0.
    - Consistency: For graph stability, maintain consistent input types (e.g., avoid switching between Tensor and None).
    """
    if obj is None:
        return functools.partial(
            magi_compile,
            model_tag=model_tag,
            dynamic_arg_dims=dynamic_arg_dims,
            enable_if=enable_if,
            config_patch=config_patch,
            method_name=method_name,
        )

    config_patch = config_patch or (lambda x: x)
    conf = config_patch(copy.deepcopy(get_compile_config()))
    enable = enable_if is None or enable_if()
    if not enable or conf.compile_mode == CompileMode.NONE:
        return obj

    is_bound_method = inspect.ismethod(obj)
    is_function = inspect.isfunction(obj)
    is_class = inspect.isclass(obj)
    is_instance = callable(obj) and not any((is_class, is_function, is_bound_method))
    if not any((is_class, is_instance, is_bound_method, is_function)):
        raise TypeError(f"Unsupported type for magi_compile: {type(obj)}")

    if method_name is not None and (is_bound_method or is_function):
        entry_name = "bound method" if is_bound_method else "function"
        raise ValueError(f"method_name cannot be used when compiling a {entry_name} directly")

    # 1. Determine target function for dynamic dim inference
    owner_instance = obj.__self__ if is_bound_method else obj if is_instance else None
    owner_class = obj if is_class else owner_instance.__class__ if is_bound_method else obj.__class__ if is_instance else None

    if is_class or is_instance:
        method_name = method_name or "forward"
        target_func = getattr(owner_class, method_name, None)
        context_name = f"{'class' if is_class else 'instance'} {owner_class.__name__}.{method_name}"
    elif is_bound_method:
        method_name = method_name or obj.__name__
        target_func = obj
        context_name = f"bound method {method_name}"
    else:
        method_name = None
        target_func = obj
        context_name = f"function {obj.__name__}"

    if not callable(target_func):
        if is_class and not method_name:
            raise AssertionError(f"Class '{owner_class.__name__}' must have forward method or pass method_name explicitly.")
        if is_instance and not method_name:
            raise AssertionError(f"Instance '{owner_class.__name__}' must have forward method or pass method_name explicitly.")
        raise TypeError(f"Target '{target_func.__name__}' is not callable for {type(obj)}")

    # 2. Infer dynamic dims
    inferred_dims = dynamic_arg_dims or _infer_dynamic_arg_dims(target_func, context_name)
    assert (
        len(inferred_dims) > 0
    ), f"No dynamic dimensions found in {context_name}. Please provide dynamic_arg_dims explicitly."

    _check_dynamic_arg_dims(inferred_dims, target_func)

    if model_tag is None:
        model_tag = getattr(obj, "__name__", obj.__class__.__name__)

    # 3. Dispatch by entry kind (class / instance / bound method / bare function)

    if is_class:
        return _magi_compile_class(obj, inferred_dims, conf, model_tag, method_name)
    elif is_instance:
        return _magi_compile_bound_method(obj, inferred_dims, conf, model_tag, method_name)
    elif is_bound_method:
        _magi_compile_bound_method(owner_instance, inferred_dims, conf, model_tag, method_name)
        return getattr(owner_instance, method_name)
    elif is_function:
        return _magi_compile_function(obj, inferred_dims, conf, model_tag)

    raise TypeError(f"Unsupported type for magi_compile: {type(obj)}")


def magi_register_custom_op(
    name: str | None = None,
    mutates_args: tuple[str, ...] = (),
    infer_output_meta_fn: Callable | list[str] | None = None,
    setup_context_fn: Callable | None = None,
    backward_fn: Callable | None = None,
    is_compute_sensitive: bool = False,
    is_subgraph_boundary: bool = False,
    extra_triton_kernels: list[Any] | tuple[Any, ...] | None = None,
):
    """
    A unified decorator to register a custom operator with PyTorch's library.

    This decorator combines the functionality of:
    - @torch.library.custom_op
    - @torch.library.register_fake
    - fn.register_autograd
    - @torch.library.triton_op (auto-detected: see "Triton kernels" below)

    plus two convenience layers on top:
    - frozen-dataclass inputs (including arbitrarily nested dataclasses)
      are transparently flattened into primitive parameters before being
      handed to ``torch.library``, then reassembled at runtime so user code
      sees the original signature; see "Frozen-dataclass inputs" below.
    - autograd hooks expressed against the original signature keep
      working when dataclass inputs are present: ``setup_context_fn`` and
      ``backward_fn`` are bridged in/out of the flat parameter space
      automatically; see "Autograd with dataclass inputs" below.

    Triton kernels
    --------------
    If the decorated function (or any helper it calls) launches one or more
    ``triton.jit`` kernels (with or without an explicit ``wrap_triton`` call),
    the function is automatically registered via ``torch.library.triton_op``
    instead of ``torch.library.custom_op``. Detected kernel references are
    transparently rewritten to go through ``torch.library.wrap_triton`` at
    runtime, so the user does not need to add ``wrap_triton(...)`` manually.
    Already-wrapped kernels are detected and not re-wrapped (the rewrite is
    idempotent), and the same op may launch any number of triton kernels:
    Inductor will see all of them and may inline / fuse them.

    This makes the kernels visible to ``torch.compile`` / Inductor (instead of
    keeping the op opaque), enabling kernel inlining and fusion in the
    generated graph. Falls back to plain ``custom_op`` if no kernels are
    detected or if ``triton_op`` registration fails.

    Detection is best-effort source introspection and recurses through helper
    functions called from the decorated function (including helpers in other
    modules and helpers reached via ``torch.ops.<ns>.<op>(...)`` calls). For
    pathological cases (kernels stored on instance attributes, kernels behind
    user-defined conditional wrappers, kernels constructed only at runtime,
    etc.), pass ``extra_triton_kernels`` to provide the list explicitly.

    Frozen-dataclass inputs
    -----------------------
    Any parameter typed as a ``@dataclass(frozen=True)`` is recursively
    flattened into its individual leaf fields before being registered with
    ``torch.library``. Nested dataclasses (dataclass-of-dataclass, to any
    depth) are fully unwrapped using ``__`` as the join separator, e.g. an
    outer ``cfg: OuterCfg`` whose ``OuterCfg.inner: InnerCfg(val: float)``
    becomes a flat parameter named ``cfg__inner__val: float``. At call time
    the user passes (and the body sees) the original dataclass instance; the
    decorator handles the conversion in both directions.

    Requirements:
    - Each dataclass MUST be ``frozen=True``; the leaf field types must be
      types accepted by ``torch.library.custom_op``. Supported field types
      include:
        * ``torch.Tensor``
        * Scalars: ``int``, ``float``, ``bool``, ``str``
        * Structured scalars: ``torch.dtype``, ``torch.device``
        * Optional variants: ``Optional[Tensor]``, ``Optional[int]``, etc.,
          as well as PEP 604 syntax (e.g. ``Tensor | None``).
        * Lists of scalars/tensors: ``list[int]``, ``list[float]``,
          ``list[bool]``, ``list[Tensor]``, ``list[Optional[Tensor]]``.
          Note that PyTorch does **not** support ``list[Optional[int]]``
          but it does support ``list[Optional[Tensor]]``. Similarly,
          ``Optional[list[Tensor]]`` is **not** supported (use
          ``list[Optional[Tensor]]`` instead).
        * ``Literal[str, ...]`` and ``Enum`` containing only string values
          are automatically supported by downgrading them to ``str`` at the
          schema boundary (but your op body still receives the original string).
    - Returning a dataclass from the op body is not supported; only
      ``torch.Tensor`` / ``tuple[torch.Tensor, ...]`` returns are allowed.

    Autograd with dataclass inputs
    ------------------------------
    ``setup_context_fn`` and ``backward_fn`` are written against the
    *original* (dataclass-bearing) signature, not the flat one:
    - ``setup_context_fn(ctx, inputs, output)`` receives ``inputs`` in the
      same positional order as ``fn``'s signature, with each dataclass
      argument reassembled back into its original instance.
    - ``backward_fn(ctx, *grad_outputs)`` must return one grad per
      *original* input (dataclass arguments count as one slot). For a
      dataclass slot the user may return any of:
        * ``None``                 -> equivalent to "no grad for any field"
                                      (the bridge fills ``None`` into every
                                      flat slot under that dataclass).
        * a same-shape dataclass / namedtuple instance -> per-field grad,
          ``None`` leaves are allowed and are spread to the corresponding
          flat slots.
        * a ``dict`` keyed by field name -> same as above but without
          having to construct a new dataclass instance.
      Returning the wrong number of top-level grads raises ``ValueError``.

    Limitations and known caveats
    -----------------------------
    - Return type: only ``torch.Tensor`` / ``tuple[torch.Tensor, ...]`` /
      ``list[torch.Tensor]`` / ``None`` are accepted by the underlying
      ``torch.library`` schema. Returning a dataclass raises a clear
      ``TypeError`` at registration time -- destructure the dataclass into a
      tuple at the op boundary instead.
    - Top-level tuple/dict: parameters typed as ``tuple[...]`` or
      ``dict[...]`` are not supported by the schema and will raise a
      ``TypeError``. Wrap them in a ``@dataclass(frozen=True)`` instead.
    - Local nested types: a dataclass field annotated with a class defined
      inside another function body, combined with
      ``from __future__ import annotations``, cannot be resolved by
      ``typing.get_type_hints`` and produces a clear ``TypeError`` pointing at
      the offending field. Move the type to module scope to fix.
    - Double backward: not supported automatically. ``backward_fn`` runs
      under autograd but does not get its own backward registered. If you need
      higher-order derivatives, either compute them manually inside
      ``backward_fn`` (using ``torch.autograd.grad(..., create_graph=True)``
      against differentiable building blocks), or split the op so the second
      derivative comes from a separately registered op.
    - vmap / functorch: there is no automatic ``vmap`` rule. Calling a
      registered op under ``torch.vmap`` falls back to the default per-sample
      loop. If you need a real batched implementation, register one with
      ``torch.library.register_vmap`` against the *flat* inner op
      (``op._magi_inner_op`` when dataclass inputs are present).
    - Triton kernel imported inside the op body: ``import`` statements
      executed at call time are not visible to source introspection. Either
      hoist the import to module scope, or pass the kernel object explicitly
      via ``extra_triton_kernels=``.
    - Mixed wrapped/bare Triton kernels: If your op body uses both
      ``wrap_triton(kernel)[grid]`` and bare ``kernel[grid]`` calls, avoid
      using the same kernel function for both styles, or the automated wrapper
      might double-wrap it. Standardize on one style (bare is recommended).
    - dataclass field of type ``list[Dataclass]``: not supported. The flat
      schema requires a static, finite leaf count; a runtime-sized list of
      dataclass instances has no fixed shape. Restructure into parallel
      ``list[Tensor]`` / ``list[int]`` fields, or split into per-element op
      calls.
    - Mixed-type tuple returns (e.g. ``tuple[Tensor, int]``): not
      supported by the schema (only homogeneous ``tuple[Tensor, ...]`` /
      ``list[Tensor]`` are accepted). Either return only the tensors, or
      stash the scalar on ``ctx`` and recover it from the call site.
    - Custom CUDA streams inside the op body (``with torch.cuda.stream(s):``):
      not analysed. Inductor will treat the op as opaque w.r.t. the
      alternate stream; do stream-overlap orchestration above the op
      boundary, not inside it.
    - 0-dim Tensor used as a scalar: works but goes through a Tensor
      schema slot (not a ``Scalar`` slot), so the value enters the FX graph
      as a tensor input and won't constant-fold. Pass an actual
      ``int``/``float``/``bool`` if you want scalar semantics.
    - CPU-only execution on the Triton path: a Triton-backed op only
      registers a ``cuda`` kernel. Calling it on CPU tensors raises
      ``"no kernel registered"`` from PyTorch; do CPU dispatch above the op
      boundary.
    - Decorating a function twice with magi_register_custom_op: the
      second decoration receives the wrapper from the first, not the user's
      original function, and produces a confusing schema error. Decorate at
      most once per function object.

    Arguments:
        name: The fully qualified name of the operator (e.g., "namespace::op_name").
              If None, auto-generated from the function name and source file.
        mutates_args: Tuple of argument names that are mutated by the operator.
        infer_output_meta_fn: Specifies output tensor metadata (shape, dtype, device) for tracing.
            - None (default): Assumes each output has the same metadata as the corresponding
              input tensor (1st output matches 1st tensor input, 2nd matches 2nd, etc.).
              On the triton path, when None is passed the decorated function itself is used as
              the fake/meta implementation (must be make_fx-traceable, which it is once kernel
              calls go through ``wrap_triton``).
            - list[str]: Parameter names whose metadata to use for outputs.
              E.g., ["weight", "bias"] means output[0] has same shape as `weight`,
              output[1] has same shape as `bias`.
            - Callable: Custom function with same signature as the op (in the
              *original* signature space, including dataclass arguments
              -- the bridge handles flattening for you), returns
              torch.empty_like() tensors matching the expected output shapes.
        setup_context_fn: Function to save tensors/values for backward.
            Signature: ``setup_context_fn(ctx, inputs, output)``. ``inputs``
            mirrors the *original* signature: dataclass arguments are
            reassembled into their original instances rather than exposed as
            flat fields. Safe to use both with and without dataclass inputs.
        backward_fn: Function to compute gradients.
            Signature: ``backward_fn(ctx, *grad_outputs) -> tuple of grads``.
            Return one grad per *original* parameter (use ``None`` for
            non-differentiable / non-tensor parameters). For dataclass
            parameters see "Autograd with dataclass inputs" above.
        is_compute_sensitive: If True, marks this operator as compute-intensive (e.g., MatMul,
            Attention). During activation recomputation (rematerialization), outputs of
            compute-sensitive ops are prioritized for saving rather than recomputing,
            since recomputing them would be expensive.
        is_subgraph_boundary: If True, the FX graph will be split at this operator during
            compilation. Each sub-graph between boundary operators is compiled independently
            by Inductor, enabling piecewise compilation and more flexible scheduling
            (e.g., for CPU offloading or overlapping computation with data transfer).
        extra_triton_kernels: Optional explicit list of triton kernels (``triton.jit`` /
            ``triton.autotune`` objects) referenced inside the decorated function. Use this
            when automatic source-based detection fails to discover a kernel
            (e.g., kernel stored on ``self``, kernel selected by a user-defined ``maybe_capture``
            wrapper, etc.). Kernels listed here are merged with the auto-detected
            ones and deduplicated by object identity, so it is safe (and harmless)
            to also list a kernel that is statically detectable.

    Returns:
        The registered custom operator function.

    Examples:
        1. Basic usage (forward only, auto-generated name and meta function):

        >>> @magi_register_custom_op()
        ... def my_relu(x: torch.Tensor) -> torch.Tensor:
        ...     return torch.maximum(x, torch.zeros_like(x))

        2. Multiple outputs with explicit output metadata via parameter names:

        >>> @magi_register_custom_op(
        ...     infer_output_meta_fn=["weight", "bias"],  # output shapes match weight and bias
        ... )
        ... def compute_gradients(
        ...     grad_output: torch.Tensor,
        ...     weight: torch.Tensor,
        ...     bias: torch.Tensor,
        ... ) -> tuple[torch.Tensor, torch.Tensor]:
        ...     grad_weight = grad_output.sum(dim=0).view_as(weight)
        ...     grad_bias = grad_output.sum(dim=0).view_as(bias)
        ...     return grad_weight, grad_bias

        3. Full custom op with autograd support:

        >>> def _square_meta(x: torch.Tensor) -> torch.Tensor:
        ...     return torch.empty_like(x)
        ...
        >>> def _square_setup_context(ctx, inputs, output):
        ...     (x,) = inputs
        ...     ctx.save_for_backward(x)
        ...
        >>> def _square_backward(ctx, grad_output):
        ...     (x,) = ctx.saved_tensors
        ...     return grad_output * 2 * x
        ...
        >>> @magi_register_custom_op(
        ...     name="my_ops::square",
        ...     infer_output_meta_fn=_square_meta,
        ...     setup_context_fn=_square_setup_context,
        ...     backward_fn=_square_backward,
        ... )
        ... def square(x: torch.Tensor) -> torch.Tensor:
        ...     return x * x

        4. With a (nested) frozen dataclass argument (auto pytree-flattened):

        >>> from dataclasses import dataclass
        >>>
        >>> @dataclass(frozen=True)
        ... class NormCfg:
        ...     eps: float
        ...     affine: bool
        ...
        >>> @dataclass(frozen=True)
        ... class AttnCfg:
        ...     scale: float
        ...     norm: NormCfg            # nested dataclass field
        ...
        >>> @magi_register_custom_op()
        ... def my_attn(q: torch.Tensor, k: torch.Tensor, cfg: AttnCfg) -> torch.Tensor:
        ...     out = (q @ k.transpose(-1, -2)) * cfg.scale
        ...     return out / (out.std() + cfg.norm.eps)

        Internally the registered op has flat parameters
        ``q, k, cfg__scale, cfg__norm__eps, cfg__norm__affine``; users still
        call ``my_attn(q, k, AttnCfg(scale=..., norm=NormCfg(...)))``.

        5. Dataclass input + custom backward (signature is the original one):

        >>> @dataclass(frozen=True)
        ... class ScaleCfg:
        ...     scale: float
        ...
        >>> def _setup(ctx, inputs, output):
        ...     x, cfg = inputs                 # original signature view
        ...     ctx.save_for_backward(x)
        ...     ctx.scale = cfg.scale
        ...
        >>> def _bwd(ctx, grad_out):
        ...     # one grad per ORIGINAL input; dataclass slot -> ``None``.
        ...     return grad_out * ctx.scale, None
        ...
        >>> @magi_register_custom_op(
        ...     setup_context_fn=_setup,
        ...     backward_fn=_bwd,
        ... )
        ... def scale_op(x: torch.Tensor, cfg: ScaleCfg) -> torch.Tensor:
        ...     return x * cfg.scale

        6. Triton kernel inside the body, no manual ``wrap_triton`` needed:

        >>> import triton
        >>> import triton.language as tl
        >>>
        >>> @triton.jit
        ... def cos_kernel(in_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        ...     pid = tl.program_id(axis=0)
        ...     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        ...     mask = offsets < n
        ...     x = tl.load(in_ptr + offsets, mask=mask)
        ...     tl.store(out_ptr + offsets, tl.cos(x), mask=mask)
        ...
        >>> @magi_register_custom_op()
        ... def my_cos(x: torch.Tensor) -> torch.Tensor:
        ...     out = torch.empty_like(x)
        ...     n = x.numel()
        ...     # Plain ``kernel[grid](...)`` -- the decorator detects this and
        ...     # registers ``my_cos`` as a triton_op so torch.compile can
        ...     # inline ``cos_kernel``.
        ...     cos_kernel[((n + 127) // 128,)](x, out, n, BLOCK_SIZE=128)
        ...     return out
    """
    return _magi_register_custom_op_impl(
        name=name,
        mutates_args=mutates_args,
        infer_output_meta_fn=infer_output_meta_fn,
        setup_context_fn=setup_context_fn,
        backward_fn=backward_fn,
        is_compute_sensitive=is_compute_sensitive,
        is_subgraph_boundary=is_subgraph_boundary,
        extra_triton_kernels=extra_triton_kernels,
    )
