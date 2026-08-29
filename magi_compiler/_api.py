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

from magi_compiler.config import debug_dump_path, inductor_cache_dump_path, triton_cache_dump_path
from magi_compiler.cuda.cudart import pin_memory_in_place
from magi_compiler.magi_backend.magi_compiler_base import MagiCompileState
from magi_compiler.utils import compilation_counter, envs, magi_logger
from magi_compiler.utils.compile_time_monitor import CompileMonitor

from .config import CompileConfig, CompileMode


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


def get_attr_name_for_bound_wrapper_flag(entry_name: str) -> str:
    """Name the marker saying *entry_name* has already been patched on this instance.

    Deliberately not the class-level flag: the class decorator sets that one and then
    patches each new instance from __init__, and an instance inherits it, so reading it
    here would skip the patching it is supposed to guard. Per method, so an instance can
    carry more than one entry point.
    """
    return f"_magi_wrapper_installed_for_{entry_name}"


def get_attr_name_for_state(entry_name: str) -> str:
    """Name the attribute holding an entry point's compile state, one per topology.

    The state owns the captured bytecode and AOT artifacts that _run_orchestration's fast
    paths replay directly, without going through dynamo's guards. Those artifacts have the
    ProcessGroup they traced with baked in, because a ProcessGroup is not something dynamo
    can turn into a graph input. So a runtime that changes CP/DP between calls -- what
    adaptive DP does between requests -- needs one state per topology; a shared one would
    replay the first topology's graph under every later one. Keyed this way each topology
    compiles once and is reused whenever the runtime returns to it.

    MAGI_COMPILE_TOPOLOGY_KEY is set by the runtime on every mesh change; empty means a
    single fixed topology, i.e. the plain name.
    """
    topology = os.environ.get("MAGI_COMPILE_TOPOLOGY_KEY", "")
    suffix = f"__{topology}" if topology else ""
    return f"_magi_state_for_{entry_name}{suffix}"


def _run_orchestration(state: MagiCompileState, args, kwargs):
    """
    Central orchestration logic for magi_compile.

    Dispatch order:
    0. TORCH_COMPILE — short-circuit before MAGI-specific logic.
    1. JIT Fast Path: If bytecode is already captured, swap and run.
    2. AOT Fast Path: If AOT artifacts exist, load, swap, and run.
    3. First-time Compilation (MAGI_COMPILE only):
       - Run Dynamo tracing/compilation.
       - Capture compiled bytecode (for future JIT fast path).
       - (Optional) Perform AOT compilation and save artifacts.
    """
    compile_mode = state.compile_config.compile_mode

    if compile_mode == CompileMode.TORCH_COMPILE:
        if state.compiled_entry is None:
            state._ensure_compiled()
            # First invocation triggers lazy Dynamo tracing; apply the
            # isolated config so _DEFAULT_DYNAMO_CONFIG overrides take effect.
            with _isolated_dynamo_config():
                return state.compiled_entry(*args, **kwargs)
        return state.compiled_entry(*args, **kwargs)

    # --- MAGI_COMPILE path below ---

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
    _apply_shape_marks(state, args, kwargs)

    magi_logger.info(f"Start compiling function {state.original_code_for_hook}")
    torch._dynamo.eval_frame.remove_from_cache(state.original_code_for_hook)
    CompileMonitor().start()

    try:
        with _compilation_context(state):
            state._ensure_compiled()

            if state.compile_config.aot:
                state.aot_compile(*args, **kwargs)
            else:
                with state._jit_capture_compiled_bytecode():
                    return state.compiled_entry(*args, **kwargs)

        if state.compile_config.aot:
            state.save_aot_compile_artifacts()
            with state.dispatch_to_compiled_fwd(mode="aot") as compiled_runtime_invoker:
                return compiled_runtime_invoker(*args, **kwargs)
    finally:
        CompileMonitor().end()
        state.traced_files.clear()


def _lazy_init_magi_state(
    state_holder: object,
    compile_obj: object,
    dynamic_arg_dims: dict[str, int | list[int]] | None,
    conf: CompileConfig,
    model_tag: str,
    target_method_name: str | None,
    state_attr: str,
):
    """Lazily initialize MagiCompileState and attach it on ``state_attr``."""
    if getattr(state_holder, state_attr, None) is not None:
        return

    compilation_counter.num_models_seen += 1

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
    cls: type, dynamic_arg_dims: dict[str, int | list[int]], conf: CompileConfig, model_tag: str, method_name: str
):
    """Install class-level compilation for ``method_name``.

    This wraps ``cls.__init__`` so every new instance is patched by
    ``_magi_compile_bound_method`` after initialization.
    """
    compile_flag_attr = get_attr_name_for_wrapper_installed_flag()
    if getattr(cls, compile_flag_attr, False):
        return cls

    if not callable(getattr(cls, method_name, None)):
        raise AttributeError(f"{cls.__name__} has no callable method '{method_name}'")

    if issubclass(cls, nn.Module) and conf.offload_config.model_cpu_offload:
        _patch_cpu_offload_apply(cls)

    old_init = cls.__init__

    @functools.wraps(old_init)
    def wrapped_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        _magi_compile_bound_method(self, dynamic_arg_dims, conf, model_tag, method_name=method_name)

    cls.__init__ = wrapped_init
    setattr(cls, compile_flag_attr, True)
    return cls


def _magi_compile_bound_method(
    instance: object, dynamic_arg_dims: dict[str, int | list[int]], conf: CompileConfig, model_tag: str, method_name: str
):
    """Patch one instance method with compiled routing."""
    if not callable(getattr(instance, method_name, None)):
        raise AttributeError(f"{instance.__class__.__name__} instance has no callable method '{method_name}'")

    installed_attr = get_attr_name_for_bound_wrapper_flag(method_name)
    if getattr(instance, installed_attr, False):
        return instance

    old_method = getattr(instance, method_name)

    @torch.compiler.disable()
    def new_call(*args, **kwargs):
        # Per call, not per wrap: the name carries the topology, and adaptive DP changes
        # topology between calls. Binding it at construction pinned every later call to
        # the topology that happened to be live back then.
        state_attr = get_attr_name_for_state(method_name)
        state = getattr(instance, state_attr, None)
        if state is None:
            _lazy_init_magi_state(instance, instance, dynamic_arg_dims, conf, model_tag, method_name, state_attr)
            state = getattr(instance, state_attr)

        if state.compile_config.offload_config.model_cpu_offload and state.jit_compiled_code is None:
            args = offload(args)
            kwargs = offload(kwargs)

        if torch.compiler.is_compiling():
            return old_method(*args, **kwargs)

        return _run_orchestration(state, args, kwargs)

    setattr(instance, method_name, new_call)
    setattr(instance, installed_attr, True)
    return instance


def _magi_compile_function(func: Callable, dynamic_arg_dims: dict[str, int | list[int]], conf: CompileConfig, model_tag: str):
    """Wrap a function entry with compiled routing."""
    if getattr(func, get_attr_name_for_wrapper_installed_flag(), False):
        return func

    @torch.compiler.disable()
    @functools.wraps(func)  # for the original function name and docstring
    def wrapper(*args, **kwargs):
        state_attr = get_attr_name_for_state("function")  # per call: see _magi_compile_bound_method
        state = getattr(wrapper, state_attr, None)
        if state is None:
            _lazy_init_magi_state(wrapper, func, dynamic_arg_dims, conf, model_tag, None, state_attr)
            state = getattr(wrapper, state_attr)

        if torch.compiler.is_compiling():
            return func(*args, **kwargs)

        return _run_orchestration(state, args, kwargs)

    setattr(wrapper, get_attr_name_for_wrapper_installed_flag(), True)
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

    _mark_static_shapes(bound, dynamic_records, owner=state.obj if state.target_method_name else None)


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

        for d in final_dims:
            dim_size = arg.shape[d]
            if dim_size <= 1:
                raise ValueError(
                    f"Argument '{k}' has size {dim_size} on dynamic dim {d}. "
                    "PyTorch Dynamo specializes on 0/1 sizes (see "
                    "https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/"
                    "dynamic_shapes_zero_one_specialization.html), "
                    "so this dimension will NOT be treated as dynamic. "
                    "Use an initial input with size >= 2 on dynamic dims to enable shape generalization."
                )

        torch._dynamo.mark_dynamic(arg, final_dims)

        dynamic_records[id(arg)] = set(final_dims)

    return dynamic_records


def _mark_static_shapes(bound, dynamic_records, owner=None):
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

    if owner is not None:
        traverse_and_mark(owner)


@contextmanager
def _compilation_context(state: MagiCompileState):
    """Active only during first-time Dynamo tracing + inductor compilation.

    Isolates all dynamo config changes so they do not leak to the caller.

    Dynamo config patches:
    - assume_static_by_default=False: Python int attrs (e.g. group_size_cpu)
      become SymInt graph inputs instead of specialized constants.
    - enable_cpp_symbolic_shape_guards=False: C++ guards do not support
      the symbolic shape patterns produced by our dynamic setup.
    - force_nn_module_property_static_shapes=False: allow nn.Module tensor
      properties (e.g. registered buffers) to keep dynamic shapes.
    - enable_aot_compile=True: so that torch.compile produces an
      .aot_compile entry-point (harmless for JIT path).

    All dynamo config is restored on exit via ``_isolated_dynamo_config``
    (a full snapshot-restore), which also catches any implicit config
    mutations made by Dynamo internals during compilation.

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
    _inductor_cache_dump_path = inductor_cache_dump_path(state.compile_config.cache_root_dir)
    _triton_cache_dump_path = triton_cache_dump_path(state.compile_config.cache_root_dir)

    with (
        _isolated_dynamo_config(),
        patch.object(torch._dynamo.config, "assume_static_by_default", False),
        patch.object(torch._dynamo.config, "enable_cpp_symbolic_shape_guards", False),
        patch.object(torch._dynamo.config, "force_nn_module_property_static_shapes", False),
        patch.object(torch._dynamo.config, "enable_aot_compile", True),
        _hijack_inline_call_to_collect_traced_files(state),
        patch.dict(
            os.environ,
            {
                "TORCHINDUCTOR_CACHE_DIR": (_inductor_cache_dump_path).as_posix(),
                "TRITON_CACHE_DIR": (_triton_cache_dump_path).as_posix(),
            },
        ),
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



def _shm_path(cls_name: str, dtype: torch.dtype, rank: int | None = None) -> str:
    """Build the /dev/shm path for a shared weight file."""
    dtype_str = str(dtype).split(".")[-1]
    suffix = f"_rank{rank}" if rank is not None else ""
    return f"{envs.MAGI_SHARED_BIN_PATH}/magi_model_shared_{dtype_str}_{cls_name}{suffix}.bin"


def _pack_params_flat(flat: torch.Tensor, param_list: list[tuple[str, torch.Tensor]]) -> None:
    """Copy a list of named tensors into a contiguous flat buffer."""
    offset = 0
    for _, tensor in param_list:
        numel = tensor.numel()
        flat[offset : offset + numel].copy_(tensor.view(-1))
        offset += numel


def _split_flat_to_params(
    flat: torch.Tensor,
    param_list: list[tuple[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Return views into *flat* shaped like the original parameters."""
    out: dict[str, torch.Tensor] = {}
    offset = 0
    for name, orig in param_list:
        numel = orig.numel()
        view = flat[offset : offset + numel].view(orig.shape)
        if orig.requires_grad:
            view.requires_grad_(True)
        out[name] = view
        offset += numel
    return out


def _create_shm_tensor(
    shm_path: str,
    param_list: list[tuple[str, torch.Tensor]],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create a shared-memory mmap file, pack *param_list* into it, return the giant tensor."""
    total_numel = sum(t.numel() for _, t in param_list)
    elem_size = torch.empty(0, dtype=dtype).element_size()
    with open(shm_path, "wb") as f:
        f.truncate(total_numel * elem_size)
    giant = torch.from_file(shm_path, shared=True, size=total_numel, dtype=dtype, device="cpu")
    _pack_params_flat(giant, param_list)
    return giant



def _staggered_pin_memory(
    full_state_dict: dict[str, torch.Tensor],
    local_rank: int,
    pin_budget_gb: float,
):
    """Pin CPU tensors in coordinated waves to avoid host OOM.

    Automatically determines how many ranks can pin concurrently based on
    host RAM and per-rank parameter size.  Override with env var
    ``MAGI_OFFLOAD_PIN_CONCURRENCY``.
    """
    import resource
    import time

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)

    per_rank_bytes = sum(
        t.numel() * t.element_size()
        for t in full_state_dict.values()
        if t.device.type == "cpu"
    )
    per_rank_gb = per_rank_bytes / (1024 ** 3)

    concurrency_env = os.environ.get("MAGI_OFFLOAD_PIN_CONCURRENCY", "")
    if concurrency_env:
        pin_concurrency = int(concurrency_env)
    else:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total_ram_kb = int(line.split()[1])
                        break
                else:
                    total_ram_kb = 512 * 1024 * 1024
            total_ram_gb = total_ram_kb / (1024 * 1024)
        except Exception:
            total_ram_gb = 512.0
        safe_ram_gb = total_ram_gb / 2
        pin_concurrency = max(1, int(safe_ram_gb / per_rank_gb)) if per_rank_gb > 0 else world_size
    pin_concurrency = min(pin_concurrency, world_size)

    num_waves = (world_size + pin_concurrency - 1) // pin_concurrency
    magi_logger.info(
        "[Rank %d] pin_memory: world=%d per_rank=%.2f GB, "
        "concurrency=%d (waves=%d), RLIMIT_MEMLOCK soft=%s hard=%s",
        local_rank, world_size, per_rank_gb,
        pin_concurrency, num_waves,
        "unlimited" if soft == resource.RLIM_INFINITY else f"{soft / (1024 ** 3):.1f}GB",
        "unlimited" if hard == resource.RLIM_INFINITY else f"{hard / (1024 ** 3):.1f}GB",
    )

    limit_bytes = int(pin_budget_gb * (1024 ** 3))
    my_wave = local_rank // pin_concurrency
    for wave in range(num_waves):
        if wave == my_wave:
            t0 = time.perf_counter()
            params = [
                (name, tensor, tensor.numel() * tensor.element_size())
                for name, tensor in full_state_dict.items()
                if tensor.device.type == "cpu"
            ]
            params.sort(key=lambda x: x[2], reverse=True)

            pinned_bytes = 0
            pinned_count = 0
            for name, tensor, size in params:
                if pinned_bytes + size > limit_bytes:
                    continue
                try:
                    pin_memory_in_place(tensor)
                    pinned_bytes += size
                    pinned_count += 1
                except RuntimeError as e:
                    magi_logger.warning(
                        "[Rank %d] pin failed at %.2f GB: %s",
                        local_rank, pinned_bytes / (1024 ** 3), e,
                    )
                    break

            total_bytes = sum(s for _, _, s in params)
            magi_logger.info(
                "[Rank %d] pin_memory DONE (wave %d/%d) %.1fs: "
                "%d params (%.2f / %.2f GB, budget=%.1f GB)",
                local_rank, wave + 1, num_waves,
                time.perf_counter() - t0,
                pinned_count, pinned_bytes / (1024 ** 3),
                total_bytes / (1024 ** 3), pin_budget_gb,
            )
        if dist.is_initialized():
            dist.barrier()


def _patch_cpu_offload_apply(cls: type[nn.Module]):
    magi_logger.info(f"Enabling CPU offload for {cls}")
    _orig_apply = cls._apply

    def _cpu_apply(self, fn):
        is_cuda_lambda = getattr(fn, "__qualname__", "") == "Module.cuda.<locals>.<lambda>"
        id_cpu_lambda = getattr(fn, "__qualname__", "") == "Module.cpu.<locals>.<lambda>"
        is_to_lambda = getattr(fn, "__qualname__", "") == "Module.to.<locals>.convert"

        # Detect .to("cuda:X") by probing the lambda on a small CPU tensor.
        # .to("cpu") also produces is_to_lambda=True but should not trigger offload.
        is_to_cuda = False
        if is_to_lambda and not getattr(self, "_magi_offloaded_once", False):
            try:
                is_to_cuda = fn(torch.empty(0, device="cpu")).is_cuda
            except Exception:
                pass

        is_moving_to_gpu = is_cuda_lambda or is_to_cuda

        # after first time to call _apply(cuda), skip "Module.to" and "Module.cpu" and "Module.cuda"
        if getattr(self, "_magi_offloaded_once", False):
            if is_cuda_lambda or id_cpu_lambda or is_to_lambda:
                return self
            else:
                return _orig_apply(self, fn)
        else:
            # first time to call _apply(cuda) or _apply(to_cuda), move all parameters/buffers to CPU
            if not is_moving_to_gpu:
                return _orig_apply(self, fn)

        # move all parameters/buffers to CPU
        # Optimized: skip GPU roundtrip when tensor is already on CPU and fn
        # only changes device (not dtype). The roundtrip was originally needed
        # for cases where fn includes dtype conversion (e.g. model.to(dtype=fp16)),
        # but the common offload path is just model.cuda() with no dtype change.
        _dtype_target_cache: dict = {}

        def _force_cpu(t):
            if t.device.type == "cpu":
                dt = t.dtype
                if dt not in _dtype_target_cache:
                    probe = torch.empty(0, dtype=dt, device="cpu")
                    _dtype_target_cache[dt] = fn(probe).dtype
                target_dt = _dtype_target_cache[dt]
                if target_dt == dt:
                    return t
                return t.to(dtype=target_dt)
            return fn(t).cpu()

        _orig_apply(self, _force_cpu)

        # create shared memory tensors for all parameters/buffers on CPU
        ep_size = int(os.environ.get("ENGINE_CONFIG__EP_SIZE", os.environ.get("EP_SIZE", "1")))
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

            per_rank_shm = ep_size > 1

            skip_shm = os.environ.get("MAGI_OFFLOAD_SKIP_SHM", "0") == "1"
            if skip_shm:
                pin_budget_gb = float(os.environ.get("MAGI_OFFLOAD_PIN_BUDGET_GB", "0"))
                if pin_budget_gb > 0:
                    _staggered_pin_memory(full_state_dict, local_rank, pin_budget_gb)
                else:
                    magi_logger.info(
                        "[Rank %d] MAGI_OFFLOAD_SKIP_SHM=1, PIN_BUDGET=0: "
                        "params remain as unpinned CPU tensors.",
                        local_rank,
                    )
                del full_state_dict, grouped_params
                gc.collect()
            elif per_rank_shm:
                # EP > 1: each rank writes its own mmap (staggered to cap peak host memory).
                cls_name = self.__class__.__name__
                world_size = dist.get_world_size()
                for turn in range(world_size):
                    if local_rank == turn:
                        for dtype, param_list in grouped_params.items():
                            path = _shm_path(cls_name, dtype, rank=local_rank)
                            giant = _create_shm_tensor(path, param_list, dtype)
                            pin_memory_in_place(giant)
                            self._magi_giant_buffers.append(giant)
                            shared_state_dict.update(_split_flat_to_params(giant, param_list))
                            if os.path.exists(path):
                                os.remove(path)
                        self.load_state_dict(shared_state_dict, assign=True)
                        del full_state_dict, grouped_params
                        gc.collect()
                    dist.barrier()
            else:
                # EP <= 1: rank 0 writes mmap, all ranks share the same pages.
                cls_name = self.__class__.__name__
                dist.barrier()
                for dtype, param_list in grouped_params.items():
                    path = _shm_path(cls_name, dtype)
                    total_numel = sum(t.numel() for _, t in param_list)
                    if local_rank == 0:
                        _create_shm_tensor(path, param_list, dtype)
                    dist.barrier()
                    giant = torch.from_file(path, shared=True, size=total_numel, dtype=dtype, device="cpu")
                    pin_memory_in_place(giant)
                    self._magi_giant_buffers.append(giant)
                    shared_state_dict.update(_split_flat_to_params(giant, param_list))
                    dist.barrier()
                    if local_rank == 0 and os.path.exists(path):
                        os.remove(path)
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
