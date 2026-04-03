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

import copy
import functools
import gc
import inspect
import os
from contextlib import contextmanager
from typing import Callable
from unittest.mock import patch

import torch
from torch import distributed as dist
from torch import nn
from torch._dynamo.symbolic_convert import InliningInstructionTranslator

from magi_compiler.config import cache_dump_path, debug_dump_path
from magi_compiler.cuda.cudart import pin_memory_in_place
from magi_compiler.magi_backend.magi_compiler_base import MagiCompileState
from magi_compiler.utils import compilation_counter, envs, magi_logger
from magi_compiler.utils.compile_time_monitor import CompileMonitor

from .config import CompileConfig, CompileMode, get_compile_config


# =============================================================================
# Workaround: TorchInductor autotune get_raw_stream
# =============================================================================
# TorchInductor autotune code blocks may reference get_raw_stream() without
# defining it, causing "name 'get_raw_stream' is not defined" at runtime.
# Register it as a builtin so the exec'd autotune snippets can always find it.
def _patch_get_raw_stream():
    try:
        import builtins

        from torch._C import _cuda_getCurrentRawStream as _get_raw_stream
    except Exception:
        return
    if not hasattr(builtins, "get_raw_stream"):
        builtins.get_raw_stream = _get_raw_stream


_patch_get_raw_stream()


# =============================================================================
# Dynamo Config Isolation
# =============================================================================
_DEFAULT_DYNAMO_CONFIG: dict = torch._dynamo.config.get_config_copy()


@contextmanager
def _isolated_dynamo_config():
    """
    Context manager that provides an isolated dynamo config environment.

    Ensures that any changes made to torch._dynamo.config within this block
    do not leak out to the global state.
    """
    with torch._dynamo.config.patch(**_DEFAULT_DYNAMO_CONFIG):
        yield


def get_attr_name_for_wrapper_installed_flag() -> str:
    return "_magi_wrapper_installed"


def get_attr_name_for_default_state() -> str:
    return "_magi"


def get_attr_name_for_method_state(method_name: str) -> str:
    return f"_magi_state_{method_name}"


def _run_orchestration(state: MagiCompileState, args, kwargs):
    """
    Central orchestration logic for magi_compile.

    Handles the logic for:
    1. JIT Fast Path: If bytecode is already captured, swap and run.
    2. AOT Fast Path: If AOT artifacts exist, load, swap, and run.
    3. First-time Compilation:
       - Run Dynamo tracing/compilation.
       - Capture compiled bytecode (for future JIT fast path).
       - (Optional) Perform AOT compilation and save artifacts.
    """
    # JIT Fast Path
    if state.jit_compiled_code is not None:
        with state.dispatch_to_compiled_fwd(mode="jit") as compiled_runtime_invoker:
            return compiled_runtime_invoker(*args, **kwargs)

    # AOT Fast Path
    if state.compile_config.aot:
        if state.aot_compiled_fn or state.load_aot_compile_artifacts():
            with state.dispatch_to_compiled_fwd(mode="aot") as compiled_runtime_invoker:
                return compiled_runtime_invoker(*args, **kwargs)

    # First compilation
    state._ensure_compiled()

    # Mark dynamic and static shapes
    _apply_shape_marks(state, args, kwargs)

    magi_logger.info(f"Start compiling function {state.original_code_for_hook}")
    torch._dynamo.eval_frame.remove_from_cache(state.original_code_for_hook)
    CompileMonitor().start()

    try:
        if state.compile_config.aot:
            with _compilation_context(state):
                state.aot_compile(*args, **kwargs)
            state.save_aot_compile_artifacts()
            with state.dispatch_to_compiled_fwd(mode="aot") as compiled_runtime_invoker:
                return compiled_runtime_invoker(*args, **kwargs)
        else:
            with _compilation_context(state):
                # For JIT, we need to capture bytecode.
                with state._jit_capture_compiled_bytecode():
                    return state.compiled_entry(*args, **kwargs)
    finally:
        CompileMonitor().end()
        state.traced_files.clear()


def _lazy_init_magi_state(
    state_holder: object,
    compile_obj: object,
    dynamic_arg_dims: dict[str, int | list[int]] | None,
    enable_if: Callable[[], bool] | None,
    config_patch: Callable[[CompileConfig], CompileConfig],
    model_tag: str | None,
    *,
    target_method_name: str | None = None,
    state_attr: str | None = None,
):
    """Lazily initialize MagiCompileState and attach it on ``state_attr``."""
    state_attr = state_attr or get_attr_name_for_default_state()
    if getattr(state_holder, state_attr, None) is not None:
        return

    conf = config_patch(copy.deepcopy(get_compile_config()))
    enable = enable_if is None or enable_if()
    if conf.compile_mode == CompileMode.NONE or not enable:
        return

    compilation_counter.num_models_seen += 1

    # Infer default model tag if not provided
    if model_tag is None:
        model_tag = getattr(compile_obj, "__name__", compile_obj.__class__.__name__)

    setattr(
        state_holder,
        state_attr,
        MagiCompileState(
            compile_obj,
            conf,
            model_idx=compilation_counter.num_models_seen,
            model_tag=model_tag,
            dynamic_arg_dims=dynamic_arg_dims,
            target_method_name=target_method_name,
        ),
    )


def _magi_compile_class(
    cls: type,
    dynamic_arg_dims: dict[str, int | list[int]],
    enable_if: Callable[[], bool] | None,
    config_patch: Callable[[CompileConfig], CompileConfig],
    model_tag: str | None,
    method_name: str,
):
    """Install class-level lazy compilation for ``method_name``.

    This wraps ``cls.__init__`` so every new instance is patched by
    ``_magi_compile_bound_method`` after initialization.
    """
    compile_flag_attr = get_attr_name_for_wrapper_installed_flag()
    if getattr(cls, compile_flag_attr, False):
        return cls

    if not callable(getattr(cls, method_name, None)):
        raise AttributeError(f"{cls.__name__} has no callable method '{method_name}'")

    if issubclass(cls, nn.Module) and config_patch(copy.deepcopy(get_compile_config())).offload_config.model_cpu_offload:
        _patch_cpu_offload_apply(cls)

    old_init = cls.__init__

    @functools.wraps(old_init)
    def wrapped_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        _magi_compile_bound_method(self, dynamic_arg_dims, enable_if, config_patch, model_tag, method_name=method_name)

    cls.__init__ = wrapped_init
    setattr(cls, compile_flag_attr, True)
    return cls


def _magi_compile_bound_method(
    instance: object,
    dynamic_arg_dims: dict[str, int | list[int]],
    enable_if: Callable[[], bool] | None,
    config_patch: Callable[[CompileConfig], CompileConfig],
    model_tag: str | None,
    method_name: str,
):
    """Patch one instance method with lazy state initialization and compiled routing."""
    if not callable(getattr(instance, method_name, None)):
        raise AttributeError(f"{instance.__class__.__name__} instance has no callable method '{method_name}'")

    state_attr = get_attr_name_for_method_state(method_name)
    if getattr(instance, state_attr, None) is not None:
        return instance

    old_method = getattr(instance, method_name)

    @torch.compiler.disable()
    def new_call(*args, **kwargs):
        state = getattr(instance, state_attr, None)

        if state is None:
            _lazy_init_magi_state(
                instance,
                instance,
                dynamic_arg_dims,
                enable_if,
                config_patch,
                model_tag,
                target_method_name=method_name,
                state_attr=state_attr,
            )
            state = getattr(instance, state_attr, None)

        if state is None or torch.compiler.is_compiling():
            return old_method(*args, **kwargs)

        with _isolated_dynamo_config():
            return _run_orchestration(state, args, kwargs)

    setattr(instance, method_name, new_call)
    setattr(instance, get_attr_name_for_wrapper_installed_flag(), True)
    return instance


def _magi_compile_function(
    func: Callable,
    dynamic_arg_dims: dict[str, int | list[int]],
    enable_if: Callable[[], bool] | None,
    config_patch: Callable[[CompileConfig], CompileConfig] | None,
    model_tag: str | None,
):
    """Wrap a function entry with lazy ``MagiCompileState`` and compiled routing.

    The returned wrapper initializes state on first invocation and then dispatches
    through ``_run_orchestration``.
    """
    state_attr = get_attr_name_for_default_state()
    if getattr(func, state_attr, None) is not None:
        return func

    config_patch = config_patch or (lambda x: x)

    @torch.compiler.disable()
    @functools.wraps(func)  # for the original function name and docstring
    def wrapper(*args, **kwargs):
        state = getattr(wrapper, state_attr, None)
        if state is None:
            _lazy_init_magi_state(wrapper, func, dynamic_arg_dims, enable_if, config_patch, model_tag, state_attr=state_attr)
            state = getattr(wrapper, state_attr, None)

        if state is None or torch.compiler.is_compiling():
            return func(*args, **kwargs)

        with _isolated_dynamo_config():
            return _run_orchestration(state, args, kwargs)

    return wrapper


def _resolve_nested_arg(bound_args: inspect.BoundArguments, key: str):
    """
    resolve the actual argument value from the key in dynamic_arg_dims.
    support nested arguments, e.g. "arg.attr"
    """
    if "." in key:
        base_k, *path = key.split(".")
    else:
        base_k, path = key, []

    arg = bound_args.arguments.get(base_k)
    if arg is None:
        return None

    for field in path:
        if arg is None:
            break
        if isinstance(arg, dict):
            arg = arg[field]
        else:
            arg = getattr(arg, field)
    return arg


def _apply_shape_marks(state: MagiCompileState, args, kwargs):
    """
    Main entry point for applying dynamic and static shape marks.

    This is called just before Dynamo tracing to ensure dimensions are
    correctly generalized in the captured graph.
    """
    sig = inspect.signature(state.original_entry)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()

    dynamic_records = _mark_dynamic_shapes(state, bound)

    _mark_static_shapes(bound, dynamic_records)


def _mark_dynamic_shapes(state: MagiCompileState, bound):
    """
    Manually mark dynamic dimensions for arguments specified in dynamic_arg_dims.
    """
    dynamic_records = {}

    for k, dims in state.dynamic_arg_dims.items():
        arg = _resolve_nested_arg(bound, k)
        if arg is None:
            continue

        dims = [dims] if isinstance(dims, int) else dims
        assert isinstance(arg, torch.Tensor), f"Expected tensor for {k}, got {type(arg)}"

        final_dims = [arg.ndim + d if d < 0 else d for d in dims]

        torch._dynamo.mark_dynamic(arg, final_dims)

        dynamic_records[id(arg)] = set(final_dims)

    return dynamic_records


def _mark_static_shapes(bound, dynamic_records):
    """
    Mark static dimensions for tensors that are not marked as dynamic,
    dynamic_records is a dictionary that maps the id of the tensor to the set of dynamic dimensions.
    """
    visited = set()

    def traverse_and_mark(obj):
        obj_id = id(obj)
        if obj_id in visited or isinstance(obj, (int, float, str, bool, type(None))):
            return
        visited.add(obj_id)

        if isinstance(obj, torch.Tensor):
            dyn_dims = dynamic_records.get(obj_id, set())
            for dim_idx in range(obj.ndim):
                if dim_idx not in dyn_dims:
                    torch._dynamo.mark_static(obj, dim_idx)
            return

        if isinstance(obj, (list, tuple, set)):
            for item in obj:
                traverse_and_mark(item)

        elif isinstance(obj, dict):
            for val in obj.values():
                traverse_and_mark(val)

        elif hasattr(obj, '__dict__'):
            for val in vars(obj).values():
                traverse_and_mark(val)

        elif hasattr(obj, '__slots__'):
            for slot in obj.__slots__:
                if hasattr(obj, slot):
                    traverse_and_mark(getattr(obj, slot))

    for arg_val in bound.arguments.values():
        traverse_and_mark(arg_val)


@contextmanager
def _compilation_context(state: MagiCompileState):
    """Active only during Dynamo tracing + inductor compilation.

    Dynamo config:
    - assume_static_by_default=False: Python int attrs (e.g. group_size_cpu)
      become SymInt graph inputs instead of specialized constants.
    - enable_cpp_symbolic_shape_guards=False: C++ guards do not support
      the symbolic shape patterns produced by our dynamic setup.
    - force_nn_module_property_static_shapes=False: allow nn.Module tensor
      properties (e.g. registered buffers) to keep dynamic shapes.

    Tracing hooks:
    - _hijack_inline_call: collect traced Python source files for
      compilation cache invalidation.

    Inductor env:
    - TORCHINDUCTOR_CACHE_DIR: redirect inductor cache into magi's
      managed cache tree.
    - explain_compilation: capture compilation debug artifacts.
    """
    from .magi_depyf.inspect import explain_compilation

    _debug_dump_path = debug_dump_path(state.compile_config.cache_root_dir, state.model_idx, state.model_tag)
    _cache_dump_path = cache_dump_path(state.compile_config.cache_root_dir, state.model_idx, state.model_tag)

    with (
        patch.object(torch._dynamo.config, "assume_static_by_default", False),
        patch.object(torch._dynamo.config, "enable_cpp_symbolic_shape_guards", False),
        patch.object(torch._dynamo.config, "force_nn_module_property_static_shapes", False),
        _hijack_inline_call_to_collect_traced_files(state),
        patch.dict(os.environ, {"TORCHINDUCTOR_CACHE_DIR": (_cache_dump_path / "inductor_cache").as_posix()}),
        explain_compilation(_debug_dump_path.as_posix()),
    ):
        yield


# Collect all relevant files traced by Dynamo, re-compile the model when any of these files change.
# 1. the file containing the top-level forward function
# 2. hijack function to know all the functions called during Dynamo tracing, every time Dynamo sees a function call, it will inline
# the function by calling InliningInstructionTranslator.inline_call_
def _hijack_inline_call_to_collect_traced_files(state: MagiCompileState):
    state.traced_files.add(state.original_code_for_hook.co_filename)
    inline_call = InliningInstructionTranslator.inline_call_

    def patched(self_):
        state.traced_files.add(self_.f_code.co_filename)
        return inline_call(self_)

    return patch.object(InliningInstructionTranslator, "inline_call_", patched)


def _infer_dynamic_arg_dims(fn: Callable, context_name: str) -> dict[str, int | list[int]]:
    sig = inspect.signature(fn)
    dims = {}
    for k, v in sig.parameters.items():
        if k == "self":
            continue
        if v.annotation in [torch.Tensor, torch.Tensor | None]:
            dims[k] = 0
    magi_logger.info(f"Inferred dynamic dims for {context_name}: {list(dims.keys())}")
    return dims


def _check_dynamic_arg_dims(inferred_dims: dict[str, int | list[int]], target_func: Callable):
    for k in inferred_dims:
        base_k = k.split(".")[0]
        # Skip "self" parameter check for bound methods
        if base_k == "self" and inspect.ismethod(target_func):
            continue
        # Also need to consider that `target_func` might be an unbound method (e.g. MyModel.forward)
        # However, for signature, `self` is typically included.
        assert base_k in inspect.signature(target_func).parameters, f"Argument {base_k} (from {k}) not found in {target_func}"


def _patch_cpu_offload_apply(cls: type[nn.Module]):
    magi_logger.info(f"Enabling CPU offload for {cls}")
    _orig_apply = cls._apply

    def _cpu_apply(self, fn):
        is_cuda_lambda = getattr(fn, "__qualname__", "") == "Module.cuda.<locals>.<lambda>"
        id_cpu_lambda = getattr(fn, "__qualname__", "") == "Module.cpu.<locals>.<lambda>"
        is_to_lambda = getattr(fn, "__qualname__", "") == "Module.to.<locals>.convert"

        # after first time to call _apply(cuda), skip "Module.to" and "Module.cpu" and "Module.cuda"
        if getattr(self, "_magi_offloaded_once", False):
            if is_cuda_lambda or id_cpu_lambda or is_to_lambda:
                return self
            else:
                return _orig_apply(self, fn)
        else:
            # first time to call _apply(cuda), move all parameters/buffers to CPU
            if not is_cuda_lambda:
                return _orig_apply(self, fn)

        # move all parameters/buffers to CPU
        def _force_cpu(t):
            return fn(t).cpu()

        _orig_apply(self, _force_cpu)

        # create shared memory tensors for all parameters/buffers on CPU
        if dist.is_initialized():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            full_state_dict = self.state_dict()

            grouped_params: dict[torch.dtype, list[tuple[str, torch.Tensor]]] = {}
            for name, tensor in full_state_dict.items():
                if tensor.device.type == "cpu":
                    dt = tensor.dtype
                    if dt not in grouped_params:
                        grouped_params[dt] = []
                    grouped_params[dt].append((name, tensor))

            shared_state_dict = {}
            self._magi_giant_buffers = []

            dist.barrier()

            for dtype, param_list in grouped_params.items():
                dtype_str = str(dtype).split(".")[-1]
                shared_bin_path = f"{envs.MAGI_SHARED_BIN_PATH}/magi_model_shared_{dtype_str}_{self.__class__.__name__}.bin"

                total_numel = sum(t.numel() for _, t in param_list)

                if local_rank == 0:
                    flat_buffer = torch.zeros(total_numel, dtype=dtype)
                    offset = 0
                    for _, tensor in param_list:
                        numel = tensor.numel()
                        flat_buffer[offset : offset + numel].copy_(tensor.view(-1))
                        offset += numel

                    if dtype == torch.bfloat16:
                        flat_buffer.view(torch.int16).numpy().tofile(shared_bin_path)
                    elif dtype.itemsize == 1 and dtype.is_floating_point:
                        # fp8
                        flat_buffer.view(torch.uint8).numpy().tofile(shared_bin_path)
                    else:
                        flat_buffer.numpy().tofile(shared_bin_path)

                    del flat_buffer
                    gc.collect()

                dist.barrier()

                giant_shared_tensor = torch.from_file(
                    shared_bin_path, shared=True, size=total_numel, dtype=dtype, device="cpu"
                )
                self._magi_giant_buffers.append(giant_shared_tensor)

                pin_memory_in_place(giant_shared_tensor)

                offset = 0
                for name, original_tensor in param_list:
                    numel = original_tensor.numel()
                    shared_param = giant_shared_tensor[offset : offset + numel].view(original_tensor.shape)

                    if original_tensor.requires_grad:
                        shared_param.requires_grad_(True)

                    shared_state_dict[name] = shared_param
                    offset += numel

                dist.barrier()
                if local_rank == 0 and os.path.exists(shared_bin_path):
                    os.remove(shared_bin_path)

            self.load_state_dict(shared_state_dict, assign=True)

        else:

            def _pinner(t):
                return t.pin_memory()

            _orig_apply(self, _pinner)

        self._magi_offloaded_once = True
        return self

    cls._apply = _cpu_apply


def offload(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    if isinstance(obj, dict):
        return {k: offload(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(offload(i) for i in obj)
    return obj
