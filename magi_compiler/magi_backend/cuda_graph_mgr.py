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

from dataclasses import dataclass, fields, is_dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from magi_compiler.magi_depyf.timeline import observe_lifecycle
from magi_compiler.utils import magi_logger, nvtx
from magi_compiler.utils.device import accelerator_graph_cls, accelerator_module, accelerator_type


class InplaceSubstituteFakeClass:
    """
    The class which inherits from this class will not be replaced with a new instance,
    but the attributes will be updated in-place.
    For example, InferenceParams.
    """

    pass


@dataclass
class FakeTensor:
    shape: Tuple[int, ...] = None
    dtype: str = None
    device: str = None


@dataclass
class HashableDataclass:
    _cached_hash: Optional[int] = None

    @nvtx.instrument_nvtx
    def _get_hashable_fields(self) -> Tuple[Any, ...]:
        hashable_values = []
        for f in fields(self):
            if f.name == "_cached_hash":
                continue
            value = getattr(self, f.name)
            if value is None:
                continue
            if isinstance(value, HashableDataclass):
                hashable_values.append(value._get_cached_hash())
            elif isinstance(value, tuple):
                tuple_vals = []
                for item in value:
                    if isinstance(item, (HashableDataclass, str, int, float, bool)):
                        if isinstance(item, HashableDataclass):
                            tuple_vals.append(item._get_cached_hash())
                        else:
                            tuple_vals.append(item)
                if tuple_vals:
                    hashable_values.append(tuple(tuple_vals))
            elif isinstance(value, (str, int, float, bool)):
                hashable_values.append(value)
        return tuple(hashable_values)

    @nvtx.instrument_nvtx
    def _compute_hash(self) -> int:
        """Computes a hash value based on the dataclass's hashable fields."""
        hashable_fields = self._get_hashable_fields()
        return hash(hashable_fields) % (1 << 64)

    @nvtx.instrument_nvtx
    def _get_cached_hash(self) -> int:
        if self._cached_hash is None:
            self._cached_hash = self._compute_hash()
        return self._cached_hash

    @nvtx.instrument_nvtx
    def __hash__(self) -> int:
        return self._get_cached_hash()

    @nvtx.instrument_nvtx
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if self._get_cached_hash() != other._get_cached_hash():
            return False
        return True


@dataclass(unsafe_hash=True)
class LiteralsInfo(HashableDataclass):
    literals: Tuple[Any, ...] = tuple()


@dataclass(unsafe_hash=True)
class TensorStaticInfo(HashableDataclass):
    name: str = ""
    shapes: Tuple[int, ...] = tuple()
    dtype: str = ""


@dataclass(unsafe_hash=True)
class TensorDynamicInfo(HashableDataclass):
    name: str = ""
    shapes: Tuple[int, ...] = tuple()


@dataclass(unsafe_hash=True)
class StaticSignature(HashableDataclass):
    func_name: str = ""
    tensor_static_infos: Tuple[TensorStaticInfo, ...] = tuple()


@dataclass(unsafe_hash=True)
class DynamicSignature(HashableDataclass):
    tensor_dynamic_infos: Tuple[TensorDynamicInfo, ...] = tuple()
    literals_info: LiteralsInfo = None


@dataclass
class GraphEntry:
    graph: Optional[torch.cuda.CUDAGraph] = None
    inconsistent: bool = False
    invalid: bool = False


@dataclass
class OutputTemplateEntry:
    graph_entry_dict: Dict[int, GraphEntry] = None  # key = layer_number
    output_template: Any = None  # Template for reconstructing output objects with literal fields


@dataclass
class StaticTensorEntry:
    input_tensors: Optional[List[torch.Tensor]] = None
    output_tensors: Optional[List[torch.Tensor]] = None
    template_entry_dict: Dict[DynamicSignature, OutputTemplateEntry] = None


class ArgsUtils:
    @staticmethod
    @nvtx.instrument_nvtx
    def generate_both_signatures_from_tensors(
        func_name: str, tensors: List[torch.Tensor], names: List[str], literals: List[Any]
    ) -> Tuple[StaticSignature, DynamicSignature]:
        num_tensors = len(tensors)
        tensor_static_infos = [TensorStaticInfo() for _ in range(num_tensors)]
        tensor_dynamic_infos = [TensorDynamicInfo() for _ in range(num_tensors)]

        # Local references for performance
        TensorStaticInfo_setattr = TensorStaticInfo.__setattr__
        TensorDynamicInfo_setattr = TensorDynamicInfo.__setattr__
        _tuple = tuple

        for i in range(num_tensors):
            t = tensors[i]
            t_dim = t.dim()
            t_shape = t.shape
            t_dtype_str = str(t.dtype)
            # Last dimension is static, others are dynamic (except for 1D tensor)
            static_shapes = (
                _tuple(-1 if idx != t_dim - 1 else dim_size for idx, dim_size in enumerate(t_shape)) if t_dim > 1 else (-1,)
            )
            static_info = tensor_static_infos[i]
            TensorStaticInfo_setattr(static_info, "shapes", static_shapes)
            TensorStaticInfo_setattr(static_info, "dtype", t_dtype_str)

            dynamic_shapes = static_shapes = (
                _tuple(-1 if idx == t_dim - 1 else dim_size for idx, dim_size in enumerate(t_shape))
                if t_dim > 1
                else _tuple(t_shape)
            )
            dynamic_info = tensor_dynamic_infos[i]
            TensorDynamicInfo_setattr(dynamic_info, "shapes", dynamic_shapes)

        literals_info = LiteralsInfo(literals=_tuple(literals))
        static_sig = StaticSignature(func_name=func_name, tensor_static_infos=_tuple(tensor_static_infos))
        dynamic_sig = DynamicSignature(tensor_dynamic_infos=_tuple(tensor_dynamic_infos), literals_info=literals_info)
        return static_sig, dynamic_sig

    @staticmethod
    @nvtx.instrument_nvtx
    def replace_sliced_with_static(obj: Any, static_tensors: List[torch.Tensor]) -> Any:
        tensor_idx = 0

        def recursive_replace(o: Any) -> Any:
            nonlocal tensor_idx
            if isinstance(o, torch.Tensor) and not isinstance(o, torch.nn.Parameter):
                # Copy data to the corresponding static tensor slice
                static_tensor = static_tensors[tensor_idx]
                slices = [slice(None)] * static_tensor.ndim
                for i in range(min(o.ndim, static_tensor.ndim)):
                    slices[i] = slice(0, o.shape[i])
                # Only copy if the data_ptrs are different
                if not o.data_ptr() == static_tensor[tuple(slices)].data_ptr():
                    static_tensor[tuple(slices)].copy_(o)
                tensor_idx += 1
                return static_tensor[tuple(slices)]

            elif isinstance(o, dict):
                return {k: recursive_replace(v) for k, v in o.items()}
            elif isinstance(o, (list, tuple)):
                return type(o)(recursive_replace(item) for item in o)
            elif is_dataclass(o):
                field_values = {f.name: recursive_replace(getattr(o, f.name)) for f in fields(o)}
                return type(o)(**field_values)
            elif issubclass(o.__class__, InplaceSubstituteFakeClass):
                # Do not create a new instance, but modify attributes in place (to keep original initialization logic)
                for k, v in o.__dict__.items():
                    if not callable(v):
                        o.__dict__[k] = recursive_replace(v)
                return o
            elif o is None or isinstance(o, (int, float, str, bool)):
                return o  # Keep None and basic types
            else:
                return o

        return recursive_replace(obj)

    @staticmethod
    @nvtx.instrument_nvtx
    def replace_sliced_with_static_simple(
        sliced_tensors: List[torch.Tensor], static_tensors: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        for sliced_tensor, static_tensor in zip(sliced_tensors, static_tensors):
            if not sliced_tensor.data_ptr() == static_tensor.data_ptr():
                slices = [slice(None)] * static_tensor.ndim
                for i in range(sliced_tensor.ndim):
                    slices[i] = slice(0, sliced_tensor.shape[i])
                static_tensor[tuple(slices)].copy_(sliced_tensor)

    @staticmethod
    @nvtx.instrument_nvtx
    def replace_static_with_sliced(obj: Any, static_tensors: List[torch.Tensor]) -> Any:
        tensor_idx = 0

        def recursive_replace(o: Any) -> Any:
            nonlocal tensor_idx
            if (isinstance(o, torch.Tensor) and not isinstance(o, torch.nn.Parameter)) or isinstance(o, FakeTensor):
                # Replace with the corresponding sliced tensor
                static_tensor = static_tensors[tensor_idx]
                shape_to_slice = o.shape
                slices = [slice(0, dim_size) for dim_size in shape_to_slice]
                result_tensor = static_tensor[tuple(slices)]
                tensor_idx += 1
                return result_tensor

            elif isinstance(o, dict):
                return {k: recursive_replace(v) for k, v in o.items()}
            elif isinstance(o, (list, tuple)):
                return type(o)(recursive_replace(item) for item in o)
            elif is_dataclass(o):
                field_values = {f.name: recursive_replace(getattr(o, f.name)) for f in fields(o)}
                return type(o)(**field_values)
            elif issubclass(o.__class__, InplaceSubstituteFakeClass):
                # Do not create a new instance, but modify attributes in place (to keep original initialization logic)
                for k, v in o.__dict__.items():
                    if not callable(v):
                        o.__dict__[k] = recursive_replace(v)
                return o
            elif o is None or isinstance(o, (int, float, str, bool)):
                return o  # Keep None and basic types
            else:
                return o

        return recursive_replace(obj)

    @staticmethod
    @nvtx.instrument_nvtx
    def try_fx_extract_core(
        obj: Any, extract_tensors: bool = True, extract_literals: bool = True, with_names: bool = False
    ) -> Tuple[List[torch.Tensor], List[str], List[Any]]:
        failed_tuple = None, None, None
        tensors = []
        names = []
        literals = []

        if not isinstance(obj, dict) or "args" not in obj or "kwargs" not in obj:
            return failed_tuple
        args, kwargs = obj["args"], obj["kwargs"]
        if kwargs:
            return failed_tuple
        if not isinstance(args, (list, tuple)):
            return failed_tuple

        for idx, item in enumerate(args):
            if extract_tensors and isinstance(item, torch.Tensor) and not isinstance(item, torch.nn.Parameter):
                tensors.append(item)
            elif extract_literals and isinstance(item, (int, float, str, bool)):
                literals.append(item)

        names = [""] * len(tensors)
        return tensors, names, literals

    @staticmethod
    @nvtx.instrument_nvtx
    def recursive_extract_core(
        obj: Any, extract_tensors: bool = True, extract_literals: bool = True, with_names: bool = False
    ) -> Tuple[List[torch.Tensor], List[str], List[Any]]:
        tensors = []
        names = []
        literals = []

        def recursive_traverse(o: Any, prefix: str = ""):
            # 1. Extract tensors (if enabled)
            if extract_tensors and isinstance(o, torch.Tensor) and not isinstance(o, torch.nn.Parameter):
                tensors.append(o)
                names.append(prefix) if with_names else None
            elif extract_literals and isinstance(o, (int, float, str, bool)):
                literals.append(o) if extract_literals else None
            elif isinstance(o, dict):
                for k, v in o.items():
                    new_prefix = f"{prefix}.{k}" if (with_names and extract_tensors) else prefix
                    recursive_traverse(v, new_prefix)
            elif isinstance(o, (list, tuple)):
                for idx, item in enumerate(o):
                    new_prefix = f"{prefix}[{idx}]" if (with_names and extract_tensors) else prefix
                    recursive_traverse(item, new_prefix)
            elif is_dataclass(o):
                for f in fields(o):
                    new_prefix = f"{prefix}.{f.name}" if (with_names and extract_tensors) else prefix
                    recursive_traverse(getattr(o, f.name), new_prefix)
            elif issubclass(o.__class__, InplaceSubstituteFakeClass):
                for k, v in o.__dict__.items():
                    if not callable(v):
                        new_prefix = f"{prefix}.{k}" if (with_names and extract_tensors) else prefix
                        recursive_traverse(v, new_prefix)
            elif o is None:
                pass
            else:
                pass

        recursive_traverse(obj)
        return tensors, names if with_names else [""] * len(tensors), literals if extract_literals else None

    @staticmethod
    @nvtx.instrument_nvtx
    def extract_output_template(obj: Any) -> Any:
        def recursive_template(o: Any) -> Any:
            if isinstance(o, torch.Tensor) and not isinstance(o, torch.nn.Parameter):
                return FakeTensor(shape=list(o.shape), dtype=str(o.dtype), device=str(o.device))
            elif isinstance(o, dict):
                return {k: recursive_template(v) for k, v in o.items()}
            elif isinstance(o, (list, tuple)):
                return type(o)(recursive_template(item) for item in o)
            elif is_dataclass(o):
                field_values = {f.name: recursive_template(getattr(o, f.name)) for f in fields(o)}
                return type(o)(**field_values)
            elif issubclass(o.__class__, InplaceSubstituteFakeClass):
                # Mutate attributes in-place instead of recreating the instance
                for k, v in o.__dict__.items():
                    if not callable(v):
                        o.__dict__[k] = recursive_template(v)
                return o
            elif o is None or isinstance(o, (int, float, str, bool)):
                return o
            else:
                return o

        return recursive_template(obj)


class CudaGraphMgr:
    """CUDA Graph Manager for caching and managing CUDA Graphs and static tensors."""

    def __init__(self):
        self.cache: Dict[StaticSignature, StaticTensorEntry] = dict()
        self.graph_mem_pool: Optional[torch.cuda.graph_pool_handle] = None
        self.check_output_inconsistency = False  # Not enabled by default

    @property
    def graph_count(self) -> int:
        count = 0
        for tensor_entry in self.cache.values():
            if tensor_entry.template_entry_dict is not None:
                for template_entry in tensor_entry.template_entry_dict.values():
                    for graph_entry in template_entry.graph_entry_dict.values():
                        if graph_entry.graph is not None and not graph_entry.inconsistent and not graph_entry.invalid:
                            count += 1
        return count

    @property
    def tensor_entry_count(self) -> int:
        count = 0
        for tensor_entry in self.cache.values():
            if tensor_entry.input_tensors is not None and tensor_entry.output_tensors is not None:
                count += 1
        return count

    @property
    def graph_mem_pool_size(self) -> float:
        if not hasattr(self, "graph_mem_pool") or self.graph_mem_pool is None:
            return 0.0
        pool_stats = accelerator_module().memory.memory_stats(self.graph_mem_pool)
        used_mem = pool_stats.get("allocated_bytes.all.current", 0)
        return used_mem / (1024 * 1024)

    @property
    def tensor_mem_size(self) -> float:
        total_size = 0
        for tensor_entry in self.cache.values():
            if tensor_entry.input_tensors is not None:
                for t in tensor_entry.input_tensors:
                    total_size += t.element_size() * t.nelement()
            if tensor_entry.output_tensors is not None:
                for t in tensor_entry.output_tensors:
                    total_size += t.element_size() * t.nelement()
        return total_size / (1024 * 1024)

    @nvtx.instrument_nvtx
    def formatted_cache_str(self) -> str:
        """Format the cache content as a string for debugging."""
        lines = []
        for static_sig, tensor_entry in self.cache.items():
            lines.append(f"StaticSignature: {static_sig}")
            s = "  Input Static Tensors: "
            for it in tensor_entry.input_tensors:
                s += f"[shape={list(it.shape)},dtype={str(it.dtype)}] "
            lines.append(s)
            s = "  Output Static Tensors: "
            for ot in tensor_entry.output_tensors:
                s += f"[shape={list(ot.shape)},dtype={str(ot.dtype)}] "
            lines.append(s)
            if tensor_entry.template_entry_dict is not None:
                for dynamic_sig, template_entry in tensor_entry.template_entry_dict.items():
                    lines.append(f"  DynamicSignature: {dynamic_sig}")
                    lines.append(f"    Output Template: {template_entry.output_template}")
                    for layer_number, graph_entry in template_entry.graph_entry_dict.items():
                        status = "Valid"
                        if graph_entry.inconsistent:
                            status = "Inconsistent"
                        elif graph_entry.invalid:
                            status = "Invalid"
                        lines.append(f"    Layer {layer_number}: Graph Status: {status}")
        return "\n".join(lines)

    @nvtx.instrument_nvtx
    def try_get_cuda_graph(
        self, static_sig: StaticSignature, dynamic_sig: DynamicSignature, layer_number: int
    ) -> Optional[torch.cuda.CUDAGraph]:
        graph_entry = self.try_get_graph_entry(static_sig, dynamic_sig, layer_number)
        if (
            graph_entry is not None
            and graph_entry.graph is not None
            and not graph_entry.inconsistent
            and not graph_entry.invalid
        ):
            return graph_entry.graph
        return None

    @nvtx.instrument_nvtx
    def get_static_tensors(self, input_static_sig: StaticSignature) -> Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        if input_static_sig in self.cache:
            cached_entry = self.cache[input_static_sig]
            return cached_entry.input_tensors, cached_entry.output_tensors
        raise ValueError("Cached input/output tensors not found for the given static signature.")

    @nvtx.instrument_nvtx
    def warmup_run(self, func: Callable, *args, **kwargs) -> Union[torch.Tensor, List[torch.Tensor]]:
        warmup_outputs = None
        accel = accelerator_module()
        s = accel.Stream()
        s.wait_stream(accel.current_stream())
        with accel.stream(s), torch.no_grad():
            for _ in range(1):
                warmup_outputs = func(*args, **kwargs)
        accel.current_stream().wait_stream(s)
        return warmup_outputs

    @nvtx.instrument_nvtx
    def add_static_entry(
        self,
        static_sig: StaticSignature,
        input_tensors: Optional[List[torch.Tensor]] = None,
        output_tensors: Optional[List[torch.Tensor]] = None,
    ) -> None:
        assert static_sig not in self.cache
        self.cache[static_sig] = StaticTensorEntry(
            input_tensors=input_tensors, output_tensors=output_tensors, template_entry_dict=dict()
        )

    @nvtx.instrument_nvtx
    def add_template_entry(
        self, input_static_sig: StaticSignature, input_dynamic_sig: DynamicSignature, output_obj: Any = None
    ) -> None:
        try:
            output_template = ArgsUtils.extract_output_template(output_obj)
            self.cache[input_static_sig].template_entry_dict[input_dynamic_sig] = OutputTemplateEntry(
                graph_entry_dict=dict(), output_template=output_template
            )
        except KeyError:
            raise ValueError("StaticSignature not found in cache when adding template entry.")

    @nvtx.instrument_nvtx
    def add_graph_entry(
        self,
        input_static_sig: StaticSignature,
        input_dynamic_sig: DynamicSignature,
        layer_number: int,
        graph: torch.cuda.CUDAGraph,
    ) -> None:
        try:
            self.cache[input_static_sig].template_entry_dict[input_dynamic_sig].graph_entry_dict[layer_number] = GraphEntry(
                graph=graph, inconsistent=False, invalid=False
            )
        except KeyError:
            raise ValueError("StaticSignature or DynamicSignature not found in cache when adding graph entry.")

    @nvtx.instrument_nvtx
    def try_get_graph_entry(
        self, input_static_sig: StaticSignature, input_dynamic_sig: DynamicSignature, layer_number: int
    ) -> Optional[GraphEntry]:
        try:
            return self.cache[input_static_sig].template_entry_dict[input_dynamic_sig].graph_entry_dict[layer_number]
        except KeyError:
            pass
        return None

    @nvtx.instrument_nvtx
    def batch_set_graph_invalid(self, static_sig: StaticSignature) -> None:
        if static_sig in self.cache:
            static_tensor_entry = self.cache[static_sig]
            if static_tensor_entry.template_entry_dict is not None:
                for template_entry in static_tensor_entry.template_entry_dict.values():
                    for graph_entry in template_entry.graph_entry_dict.values():
                        graph_entry.invalid = True

    @nvtx.instrument_nvtx
    def set_graph_inconsistent(
        self, input_static_sig: StaticSignature, input_dynamic_sig: DynamicSignature, layer_number: int
    ) -> None:
        if input_static_sig not in self.cache:
            self.add_static_entry(input_static_sig, None, None)
        if input_dynamic_sig not in self.cache[input_static_sig].template_entry_dict:
            self.add_template_entry(input_static_sig, input_dynamic_sig, None)
        if layer_number not in self.cache[input_static_sig].template_entry_dict[input_dynamic_sig].graph_entry_dict:
            self.cache[input_static_sig].template_entry_dict[input_dynamic_sig].graph_entry_dict[layer_number] = GraphEntry(
                graph=None, inconsistent=True, invalid=False
            )
        self.cache[input_static_sig].template_entry_dict[input_dynamic_sig].graph_entry_dict[layer_number].inconsistent = True

    @nvtx.instrument_nvtx
    def wrapped_graph_capture(
        self,
        func: Callable,
        input_obj: Any,
        static_input_tensors: List[torch.Tensor],
        static_output_tensors: List[torch.Tensor],
    ) -> torch.cuda.CUDAGraph:
        init_cudagraph_global_pool()
        _set_capture_start()
        try:
            graph = accelerator_graph_cls()()
            _static_input_obj = ArgsUtils.replace_sliced_with_static(input_obj, static_input_tensors)
            s = None
            # Under the default "global" mode torch_musa fails every capture, even a bare ``x * 2``.
            mode = "global" if accelerator_type() == "cuda" else "relaxed"
            with accelerator_module().graph(
                graph, pool=self.graph_mem_pool, stream=s, capture_error_mode=mode
            ), torch.no_grad():
                _sliced_output_obj = func(*_static_input_obj["args"], **_static_input_obj["kwargs"])
                _static_output_obj = ArgsUtils.replace_sliced_with_static(_sliced_output_obj, static_output_tensors)
        except Exception as e:
            accelerator_module().synchronize()
            _set_capture_end()
            raise e
        _set_capture_end()
        return graph

    @nvtx.instrument_nvtx
    def wrapped_graph_replay(
        self,
        graph: torch.cuda.CUDAGraph,
        static_input_tensors: List[torch.Tensor],
        static_output_tensors: List[torch.Tensor],
        input_obj: Any,
        output_template: Any,
    ) -> Any:
        _static_input_obj = ArgsUtils.replace_sliced_with_static(input_obj, static_input_tensors)
        graph.replay()
        output_obj = ArgsUtils.replace_static_with_sliced(output_template, static_output_tensors)
        return output_obj

    @nvtx.instrument_nvtx
    def replay_graph(
        self, input_static_sig: StaticSignature, input_dynamic_sig: DynamicSignature, input_obj: Any, layer_number: int
    ) -> Any:
        output_template = self.cache[input_static_sig].template_entry_dict[input_dynamic_sig].output_template
        static_input_tensors = self.cache[input_static_sig].input_tensors
        static_output_tensors = self.cache[input_static_sig].output_tensors
        graph = self.try_get_cuda_graph(input_static_sig, input_dynamic_sig, layer_number=layer_number)
        assert graph is not None, "CUDA Graph not found for replay."
        output_obj = self.wrapped_graph_replay(
            graph=graph,
            static_input_tensors=static_input_tensors,
            static_output_tensors=static_output_tensors,
            input_obj=input_obj,
            output_template=output_template,
        )
        return output_obj

    @nvtx.instrument_nvtx
    def capture_and_cache(
        self,
        func: Callable,
        input_obj: Any,
        layer_number: int,
        input_static_sig: StaticSignature,
        input_dynamic_sig: DynamicSignature,
    ) -> Any:
        """Capture a new CUDA Graph and cache it."""
        # Access static tensors from cache
        static_tensor_entry = self.cache[input_static_sig]
        assert static_tensor_entry.input_tensors is not None
        assert static_tensor_entry.output_tensors is not None
        static_input_tensors = static_tensor_entry.input_tensors
        static_output_tensors = static_tensor_entry.output_tensors

        # Capture CUDA Graph
        graph = self.wrapped_graph_capture(
            func=func,
            input_obj=input_obj,
            static_input_tensors=static_input_tensors,
            static_output_tensors=static_output_tensors,
        )

        # Cache the captured graph
        graph_entry = self.try_get_graph_entry(input_static_sig, input_dynamic_sig, layer_number)
        if graph_entry:
            graph_entry.graph = graph
            graph_entry.inconsistent = False
            graph_entry.invalid = False
        else:
            self.add_graph_entry(
                input_static_sig=input_static_sig, input_dynamic_sig=input_dynamic_sig, layer_number=layer_number, graph=graph
            )

    @nvtx.instrument_nvtx
    def if_need_expand_static_tensors(
        self, static_tensors: List[torch.Tensor], new_tensors: List[torch.Tensor], input_static_sig: StaticSignature
    ) -> bool:
        """Judge whether static tensors need to be expanded based on new tensors."""
        res = False
        static_infos = input_static_sig.tensor_static_infos

        if len(static_tensors) != len(new_tensors) or len(static_tensors) != len(static_infos):
            raise AssertionError(
                f"[CUDA Graph] Tensor count mismatch. {len(static_tensors)=}, {len(new_tensors)=}, {len(static_infos)=}"
            )

        for static_t, new_t, static_info in zip(static_tensors, new_tensors, static_infos):
            if static_t.ndim != new_t.ndim:
                raise AssertionError(f"[CUDA Graph] Rank mismatch. {static_t.shape=}, {new_t.shape=}")
            if static_t.dtype != new_t.dtype:
                raise AssertionError(f"[CUDA Graph] Dtype mismatch. {static_t.dtype=}, {new_t.dtype=}")
            for i in range(static_t.ndim):
                if static_info.shapes[i] != -1 and static_info.shapes[i] != new_t.shape[i]:
                    raise AssertionError(
                        f"[CUDA Graph] Static dimension mismatch. {static_t.shape=}, {new_t.shape=}, {static_info.shapes=}, dim={i}"
                    )
                if static_t.shape[i] < new_t.shape[i]:
                    res = True
        return res

    @nvtx.instrument_nvtx
    def get_expanded_static_tensors(
        self, static_tensors: List[torch.Tensor], new_tensors: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Get expanded static tensors based on new tensors. Reuses existing tensors when possible."""
        expanded_tensors = []
        for static_t, new_t in zip(static_tensors, new_tensors):
            if static_t.ndim != new_t.ndim:
                raise AssertionError(
                    f"[CUDA Graph] Rank mismatch during expansion. Static: {static_t.shape}, New: {new_t.shape}"
                )
            new_shape = tuple(max(s, n) for s, n in zip(static_t.shape, new_t.shape))

            if static_t.shape == new_shape:
                expanded_tensors.append(static_t)
            elif new_shape == new_t.shape:
                expanded_tensors.append(new_t)
            else:
                expanded_tensor = torch.empty(new_shape, dtype=static_t.dtype, device=static_t.device)
                expanded_tensors.append(expanded_tensor)
        return expanded_tensors

    @nvtx.instrument_nvtx
    def try_replay_graph_inline(
        self, func: Callable, args: Tuple, kwargs: Dict, layer_number: int
    ) -> Tuple[bool, Optional[Union[torch.Tensor, List[torch.Tensor]]]]:
        """Try to replay the CUDA Graph inline for fast execution."""
        try:
            func_name = func.__qualname__
            input_obj = {"args": args, "kwargs": kwargs}

            input_tensors, input_tensor_names, literals = ArgsUtils.try_fx_extract_core(input_obj)
            if None in (input_tensors, input_tensor_names, literals):
                input_tensors, input_tensor_names, literals = ArgsUtils.recursive_extract_core(input_obj)
            input_static_sig, input_dynamic_sig = ArgsUtils.generate_both_signatures_from_tensors(
                func_name, input_tensors, input_tensor_names, literals
            )
            static_tensor_entry = self.cache[input_static_sig]
            static_input_tensors = static_tensor_entry.input_tensors
            static_output_tensors = static_tensor_entry.output_tensors

            template_entry = static_tensor_entry.template_entry_dict[input_dynamic_sig]
            output_template = template_entry.output_template

            graph_entry = template_entry.graph_entry_dict[layer_number]
            graph = graph_entry.graph

            assert graph is not None, "CUDA Graph not found for inline replay."
            assert graph_entry.inconsistent is False, "CUDA Graph marked as inconsistent for inline replay."
            assert graph_entry.invalid is False, "CUDA Graph marked as invalid for inline replay."

            ArgsUtils.replace_sliced_with_static_simple(input_tensors, static_input_tensors)
            graph.replay()
            output_obj = ArgsUtils.replace_static_with_sliced(output_template, static_output_tensors)

            if self.check_output_inconsistency:
                cur_output_tensors, cur_output_tensor_names, cur_output_literals = ArgsUtils.recursive_extract_core(output_obj)
                cur_output_static_sig, cur_output_dynamic_sig = ArgsUtils.generate_both_signatures_from_tensors(
                    func.__qualname__, cur_output_tensors, cur_output_tensor_names, cur_output_literals
                )
                output_tensors, output_tensor_names, output_literals = ArgsUtils.recursive_extract_core(output_obj)
                cached_output_static_sig, cached_output_dynamic_sig = ArgsUtils.generate_both_signatures_from_tensors(
                    func.__qualname__, output_tensors, output_tensor_names, output_literals
                )
                if cur_output_static_sig != cached_output_static_sig or cur_output_dynamic_sig != cached_output_dynamic_sig:
                    magi_logger.warning(
                        f"[CUDA Graph] Warning: Output signature changed during inline replay. {func.__qualname__=}, {layer_number=}"
                    )
                    self.set_graph_inconsistent(input_static_sig, input_dynamic_sig, layer_number)
                    return False, None
            return True, output_obj
        except KeyError:
            return False, None
        except AssertionError:
            return False, None
        except Exception as e:
            magi_logger.info(
                f"[CUDA Graph] Exception during inline replay: {e=}, {func.__qualname__=}, {layer_number=}", rank="all"
            )
            raise e

    @nvtx.instrument_nvtx
    def run(self, func: Callable, *args, layer_number: Optional[int], **kwargs) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Run the function with CUDA Graph optimization if possible."""

        # Try inline replay first
        success, output_obj = self.try_replay_graph_inline(func=func, args=args, kwargs=kwargs, layer_number=layer_number)
        if success:
            return output_obj

        # Extract input signatures
        func_name = func.__qualname__
        input_obj = {"args": args, "kwargs": kwargs}
        input_tensors, input_tensor_names, literals = ArgsUtils.recursive_extract_core(input_obj)
        input_static_sig, input_dynamic_sig = ArgsUtils.generate_both_signatures_from_tensors(
            func_name, input_tensors, input_tensor_names, literals
        )

        # Judge if the graph is marked as inconsistent
        graph_entry = self.try_get_graph_entry(input_static_sig, input_dynamic_sig, layer_number)
        if graph_entry is not None and graph_entry.inconsistent:
            return func(*args, **kwargs)

        # Judge if need to expand static tensors
        if_need_expand_static_tensors = False
        if_cached_tensor_entry = input_static_sig in self.cache
        if if_cached_tensor_entry:
            static_input_tensors, static_output_tensors = self.get_static_tensors(input_static_sig)
            if_need_expand_static_tensors = self.if_need_expand_static_tensors(
                static_input_tensors, input_tensors, input_static_sig
            )

        # Warmup run
        warmup_output_obj = self.warmup_run(func, *args, **kwargs)

        # Check input signature consistency after warmup
        warmup_input_tensors, warmup_input_tensor_names, warmup_literals = ArgsUtils.recursive_extract_core(input_obj)
        warmup_input_static_sig, warmup_input_dynamic_sig = ArgsUtils.generate_both_signatures_from_tensors(
            func_name, warmup_input_tensors, warmup_input_tensor_names, warmup_literals
        )
        if warmup_input_static_sig != input_static_sig or warmup_input_dynamic_sig != input_dynamic_sig:
            magi_logger.warning(
                f"[CUDA Graph] Warning: Input signature changed during warmup run. {func_name=}, {layer_number=}"
            )
            self.set_graph_inconsistent(input_static_sig, input_dynamic_sig, layer_number)
            return warmup_output_obj

        # Update cache entries
        if if_cached_tensor_entry:
            if if_need_expand_static_tensors:
                output_tensors, _, _ = ArgsUtils.recursive_extract_core(warmup_output_obj, extract_literals=False)
                # Need to expand static tensors
                new_static_input_tensors = self.get_expanded_static_tensors(static_input_tensors, input_tensors)
                new_static_output_tensors = self.get_expanded_static_tensors(static_output_tensors, output_tensors)
                # Register as new cache entries
                self.batch_set_graph_invalid(input_static_sig)
                self.cache[input_static_sig].input_tensors = new_static_input_tensors
                self.cache[input_static_sig].output_tensors = new_static_output_tensors

                self.add_template_entry(input_static_sig, input_dynamic_sig, warmup_output_obj)
            else:
                # Simply reuse existing static tensor entry
                static_tensor_entry = self.cache[input_static_sig]
                if input_dynamic_sig not in static_tensor_entry.template_entry_dict:
                    self.add_template_entry(input_static_sig, input_dynamic_sig, warmup_output_obj)

        else:
            # Create new static tensor entry
            output_tensors, _, _ = ArgsUtils.recursive_extract_core(warmup_output_obj, extract_literals=False)
            self.add_static_entry(input_static_sig, input_tensors, output_tensors)
            self.add_template_entry(input_static_sig, input_dynamic_sig, warmup_output_obj)

        # Capture and cache new CUDA Graph
        self.capture_and_cache(
            func=func,
            input_obj=input_obj,
            layer_number=layer_number,
            input_static_sig=input_static_sig,
            input_dynamic_sig=input_dynamic_sig,
        )

        magi_logger.info(
            f"[CUDA Graph] Current cache stats: {self.tensor_entry_count=}, {self.graph_count=}, {self.tensor_mem_size=:.2f} MB, {self.graph_mem_pool_size=:.2f} MB"
        )
        return warmup_output_obj


_IS_GRAPH_CAPTURING = False


def _is_graph_capturing():
    """Query if currently capturing."""
    global _IS_GRAPH_CAPTURING
    return _IS_GRAPH_CAPTURING


def _set_capture_start():
    """Set graph capture has started."""
    global _IS_GRAPH_CAPTURING
    _IS_GRAPH_CAPTURING = True


def _set_capture_end():
    """Set graph capture has ended."""
    global _IS_GRAPH_CAPTURING
    _IS_GRAPH_CAPTURING = False


# Singleton instance of CudaGraphMgr
_CUDA_GRAPH_MGR = CudaGraphMgr()


def cuda_graph_mgr() -> CudaGraphMgr:
    """
    Get the current CudaGraphMgr instance.
    Returns:
        CudaGraphMgr: The current CudaGraphMgr instance.
    Raises:
        AssertionError: If the CudaGraphMgr has not been initialized.
    """
    assert _CUDA_GRAPH_MGR is not None, "cuda graph manager is not initialized"
    return _CUDA_GRAPH_MGR


def cuda_graph_enable_if(condition: Callable):
    def decorator(func):
        """
        Decorator to enable CUDA graph option for a function. The function will be executed using CUDA Graph if the condition func provided outputs True.
        Args:
        condition (Callable): A callable that returns a bool indicating whether enable CUDA Graph.
        """

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            enable_cuda_graph = condition()
            if not enable_cuda_graph or _is_graph_capturing():
                return func(*args, **kwargs)

            layer_number = getattr(args[0], "layer_number", None) if args else None

            return cuda_graph_mgr().run(func, *args, layer_number=layer_number, **kwargs)

        return wrapped_func

    return decorator


@observe_lifecycle("cudagraph_wrap")
def gen_wrap_func_for_cudagraph(func: Callable, mode_prefix: str, target_prefix=None) -> Callable:
    """
    Wrap the given function for CUDA Graph:
    1. Generate a unique __qualname__ for caching
    2. Built-in call to cuda_graph_mgr().run
    """
    # Generate a unique identifier to avoid cache conflicts
    func_id = id(func) if not hasattr(func, "__name__") else func.__name__
    if mode_prefix == "full":
        wrapped_func_name = f"Athena_CUDAGraph_{mode_prefix}_{func_id}"
    else:  # piecewise
        wrapped_func_name = f"Athena_CUDAGraph_{mode_prefix}_{target_prefix}_{func_id}"

    @nvtx.instrument_nvtx
    def wrapped_func(*args, **kwargs):
        layer_number = kwargs.pop("layer_number", None)
        res = cuda_graph_mgr().run(func, *args, layer_number=layer_number, **kwargs)
        return res

    func.__qualname__ = wrapped_func_name
    magi_logger.info(f"Set original function qualname to {wrapped_func_name} for CUDA Graph caching.")

    # Copy attributes from the original function to the wrapped function
    wrapped_func.__dict__.update(func.__dict__)
    wrapped_func.__qualname__ = wrapped_func_name
    for attr in ["__is_first_graph", "__is_last_graph", "__sym_shape_indices"]:
        if hasattr(func, attr):
            setattr(wrapped_func, attr, getattr(func, attr))

    return wrapped_func


def init_cudagraph_global_pool():
    """Initialize the global CUDA graph memory pool if not already initialized."""
    from magi_compiler.magi_backend.cuda_graph_mgr import cuda_graph_mgr

    if cuda_graph_mgr().graph_mem_pool is None:
        cuda_graph_mgr().graph_mem_pool = accelerator_module().graph_pool_handle()
        magi_logger.info("Initialized the global graph memory pool.")
