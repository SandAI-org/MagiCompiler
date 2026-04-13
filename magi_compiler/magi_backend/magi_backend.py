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

import ast
import dataclasses
import pprint
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from magi_compiler.utils import OrderedSet

import torch
import torch.fx as fx
from torch._dispatch.python import enable_python_dispatcher
from torch._guards import detect_fake_mode

import magi_compiler.utils.envs as envs
from magi_compiler.config import CompileConfig, CompileMode, CudaGraphMode, cache_dump_path, inductor_compile_config_hash
from magi_compiler.magi_depyf.timeline import observe_lifecycle, observe_lifecycle_context
from magi_compiler.offload.offload_warpper import OffloadWrapper
from magi_compiler.passes import CustomJointGraphPartitionFn, FullGraphPassManager, PostGradPassManager, pass_context
from magi_compiler.utils import compilation_counter, compute_code_hash, compute_hash, magi_logger
from magi_compiler.utils.visualize import save_fx_graph_visualization

from ._cache_data_cls import CacheEntry, CacheHandle
from .compile_artifacts import MagiSerializableFunction
from .cuda_graph_mgr import gen_wrap_func_for_cudagraph
from .partition_rules import resolve_defined_ops
from .piecewise_backend import PiecewiseBackend
from .piecewise_compiler import CompilerInterface, EagerAdaptor, InductorStandaloneAdaptor

compilation_start_time: float = 0.0


def _print_with_shape_and_time(runtime_shape: int | None, prefix: str = ""):
    elapsed = time.time() - compilation_start_time
    if runtime_shape is None:
        magi_logger.info("%s for dynamic shape, took %.3f s", prefix, elapsed)
    else:
        magi_logger.info("%s for shape %s, took %.3f s", prefix, str(runtime_shape), elapsed)


@dataclasses.dataclass
class SplitItem:
    submod_name: str
    graph_id: int
    is_splitting_graph: bool
    graph: fx.GraphModule


def make_compiler(compile_config: CompileConfig) -> CompilerInterface:
    if compile_config.backend == "inductor":
        # Use standalone_compile with PyTorch 2.8+
        assert hasattr(torch._inductor, "standalone_compile"), "standalone_compile not found in PyTorch Inductor"
        magi_logger.info("Using InductorStandaloneAdaptor")
        return InductorStandaloneAdaptor(compile_config)
    else:
        assert compile_config.backend == "eager", f"Invalid backend for MagiCompiler: {compile_config.backend}"
        magi_logger.info("Using EagerAdaptor")
        return EagerAdaptor()


class CompilerManager:
    """
    Manage the compilation process, including graph compilation, compile artifacts caching and loading.

    The cache is a dict mapping `(runtime_shape, graph_index, backend_name)` to `any_data` returned from the compiler.

    When serializing the cache, we save it to a Python file for readability. We don't use json here because json doesn't support int as key.
    """

    def __init__(self, compile_config: CompileConfig):
        self.cache: dict[CacheEntry, CacheHandle] = dict()
        self._remaining_restart_skips: dict[int, int] = {}
        self.compile_config = compile_config
        self.compiler = make_compiler(compile_config)
        self.disable_cache = compile_config.disable_cache

    @property
    def hash(self) -> str:
        return self.compiler.hash

    @contextmanager
    def compile_context(self, runtime_shape: int | None = None, graph_index: int | None = None):
        """Provide compilation context for the duration of compilation to set
        any torch global properties we want to scope to a single Inductor
        compilation (e.g. pass context)."""
        with observe_lifecycle_context("pass_context", runtime_shape=runtime_shape, subgraph_index=graph_index):
            with pass_context(runtime_shape, graph_index):
                yield

    def initialize_cache(self, cache_dir: Path, prefix: str | None = None):
        """
        Initialize the cache directory for the compiler.

        The organization of the cache directory is as follows:
        cache_dir=/path/to/magi_cache/model_{idx}[_{tag}]_rank_{rank}/hash_str/[prefix/]
        inside cache_dir, there will be:
        - magi_cache_indices.py
        - computation_graph.py

        for multiple prefixes, they can share the same base cache dir of
        /path/to/magi_cache/model_{idx}[_{tag}]_rank_{rank}/hash_str/ to store some
        common compilation artifacts.
        """

        self.cache_dir: Path = cache_dir
        self.cache_file_path: Path = cache_dir / "magi_cache_indices.py"

        if self.disable_cache:
            magi_logger.info("MagiCompiler's cache is disabled.")
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        magi_logger.info("Using cache directory: %s for MagiCompiler", cache_dir)
        if self.cache_file_path.exists():
            # load the cache from the file
            with self.cache_file_path.open() as f:
                # Parse Python literals using ast.literal_eval, which is a safe alternative to eval().
                raw = ast.literal_eval(f.read())
                self.cache = {}
                for entry, handle in raw.items():
                    cache_entry = CacheEntry(*entry)
                    cache_handle = CacheHandle(*handle)
                    self.cache[cache_entry] = cache_handle

        self.compiler.initialize_cache(cache_dir=self.cache_dir, prefix=prefix)

    def save_to_file(self):
        if self.disable_cache:
            return
        # serialize to a literal-friendly dict
        serializable = {
            (e.runtime_shape, e.graph_index, e.backend_name): (h.key, h.path, h.restart_analysis_count)
            for e, h in self.cache.items()
        }
        printer = pprint.PrettyPrinter(indent=4)
        data = printer.pformat(serializable)
        with self.cache_file_path.open("w") as f:
            f.write(data)

    @observe_lifecycle("compiler_manager_load")
    def load(self, graph: fx.GraphModule, example_inputs: list[Any], cache_entry: CacheEntry) -> Callable | None:
        if cache_entry not in self.cache:
            return None

        cache_handle = self.cache[cache_entry]
        if cache_entry.graph_index not in self._remaining_restart_skips:
            self._remaining_restart_skips[cache_entry.graph_index] = cache_handle.restart_analysis_count
        remaining = self._remaining_restart_skips[cache_entry.graph_index]
        if remaining > 0:
            remaining_after = remaining - 1
            self._remaining_restart_skips[cache_entry.graph_index] = remaining_after
            magi_logger.info(
                "skip artifact load due to prior RestartAnalysis: "
                f"{cache_handle.key=} {cache_entry.runtime_shape=} {cache_entry.graph_index=} {remaining_after=}"
            )
            return None

        _print_with_shape_and_time(
            cache_entry.runtime_shape,
            f"Directly load the {cache_entry.graph_index}-th graph from {cache_entry.backend_name} via handle {cache_handle}",
        )
        return self.compiler.load(graph, example_inputs, cache_entry, cache_handle)

    @observe_lifecycle("compiler_manager_compile")
    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: tuple[torch.fx.node.Argument, ...],
        inductor_compile_config: dict[str, Any],
        graph_index: int = 0,
        num_graphs: int = 1,
        runtime_shape: int | None = None,
    ) -> Callable:
        import time

        # Step0: update some global metrics
        compilation_counter.num_backend_compilations += 1
        if graph_index == 0:
            global compilation_start_time
            compilation_start_time = time.time()

        # Step1: Try loading from the cache
        cache_entry = CacheEntry(runtime_shape, graph_index, self.compiler.name)
        compiled_graph = self.load(graph, example_inputs, cache_entry)
        if compiled_graph is not None:
            return compiled_graph

        # Step2: Compile the graph
        key = f"artifact_shape_{runtime_shape}_subgraph_{graph_index}"

        with self.compile_context(runtime_shape, graph_index):
            compiled_graph, cache_handle = self.compiler.compile(
                graph, example_inputs, inductor_compile_config, runtime_shape, key
            )
            assert compiled_graph is not None, "Failed to compile the graph"

        # Step3: Store the artifact in the cache
        self._maybe_store_cache_entry(cache_entry, cache_handle, runtime_shape, key)
        _print_with_shape_and_time(runtime_shape, f"Compile the {graph_index}/{num_graphs} graph")

        return compiled_graph

    @observe_lifecycle("compiler_manager_cache_store")
    def _maybe_store_cache_entry(
        self, cache_entry: CacheEntry, cache_handle: CacheHandle | None, runtime_shape: int | None, key: str
    ) -> bool:
        if self.disable_cache:
            return False
        if cache_handle is None:
            return False

        prev_handle = self.cache.get(cache_entry)
        if prev_handle is None:
            compilation_counter.num_cache_entries += 1
        self.cache[cache_entry] = cache_handle
        return True


class PiecewiseCompileInterpreter(torch.fx.Interpreter):
    """
    Code adapted from `torch.fx.passes.shape_prop.ShapeProp`.
    It runs the given graph with fake inputs, and compile some submodules specified by `compile_submod_names` with compilation configs.

    NOTE: the order in `compile_submod_names` matters, because it will be used to determine the order of the compiled piecewise graphs.
    The first graph will handle logging, and the last graph has some special cudagraph output handling.
    """

    def __init__(
        self,
        module: torch.fx.GraphModule,
        compiler_manager: CompilerManager,
        compile_submod_names: list[str],
        compile_config: CompileConfig,
        inductor_config: dict[str, Any],
    ):
        super().__init__(module)

        self.fake_mode = detect_fake_mode()
        self.compiler_manager = compiler_manager
        self.compile_submod_names = compile_submod_names
        self.compile_config = compile_config
        self.inductor_config = inductor_config
        # extra_traceback is attribute of torch.fx.Interpreter, when it is True, it annoyingly dumps the torch.fx.Graph on errors.
        self.extra_traceback = False

    def _fix_graph_device_placement(self, module: torch.nn.Module):
        for name, child in module.named_children():
            self._fix_graph_device_placement(child)

        if isinstance(module, torch.fx.GraphModule):
            needs_recompile = False
            target_device = torch.cuda.current_device()

            factory_functions = [
                torch.empty,
                torch.zeros,
                torch.ones,
                torch.full,
                torch.rand,
                torch.randn,
                torch.arange,
                torch.tensor,
                torch.ops.aten.empty.memory_format,
            ]

            for node in module.graph.nodes:
                if node.op == 'call_function':
                    is_factory = node.target in factory_functions or (
                        hasattr(node.target, '__name__') and node.target.__name__ in ['empty', 'zeros', 'ones', 'full']
                    )

                    if is_factory:
                        if 'device' in node.kwargs:
                            current_dev = node.kwargs['device']
                            if str(current_dev) == 'cpu' or current_dev == torch.device('cpu'):
                                node.update_kwarg('device', target_device)
                                needs_recompile = True

            if needs_recompile:
                module.recompile()

    @observe_lifecycle("piecewise_compile")
    def run(self, *args):
        fake_args = self._build_fake_args(args)
        if self.compile_config.offload_config.model_cpu_offload:
            self._fix_graph_device_placement(self.module)
            for i, arg in enumerate(fake_args):
                if isinstance(arg, torch.Tensor):
                    fake_args[i] = arg.cuda()

        with self.fake_mode, enable_python_dispatcher():
            return super().run(*fake_args)

    def _build_fake_args(self, args: tuple) -> list:
        """Convert real tensor args to FakeTensors.

        When ``TracingContext.tensor_to_context`` is populated (JIT mode),
        ``from_tensor()`` can look up the correct ``SymbolicContext`` for each
        tensor and only symbolise the dimensions that Dynamo marked dynamic.

        In AOT-compile mode the ``TracingContext`` created around the backend
        call is **empty** (see ``aot_compile_fullgraph``), so ``from_tensor()``
        falls back to making *every* non-0/1 dimension symbolic.  This produces
        unexpected derived expressions (e.g. ``(s49 + 5) // 6``) and triggers
        Inductor codegen ordering errors.

        Fix: prefer the ``example_value`` FakeTensors that Dynamo already
        attached to the graph's placeholder nodes – they carry exactly the
        right mix of concrete and symbolic dimensions.
        """
        from torch._guards import TracingContext
        from torch._subclasses.fake_tensor import FakeTensor

        tc = TracingContext.try_get()
        has_tensor_context = tc is not None and hasattr(tc, "tensor_to_context") and len(tc.tensor_to_context) > 0

        if has_tensor_context:
            # JIT path: TracingContext has the full symbolic context mapping.
            # from_tensor() will look it up automatically.
            return [self.fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t for t in args]

        # AOT path: TracingContext.tensor_to_context is empty (aot_compile_fullgraph
        # wraps the backend in a fresh TracingContext that lacks the mapping).
        # Without this mapping, from_tensor() symbolises ALL non-0/1 dims.
        # Fix: extract DimDynamic info from graph placeholder example_values
        # (which Dynamo populated correctly) and pass it explicitly.
        from torch._dynamo.source import ConstantSource
        from torch.fx.experimental.symbolic_shapes import DimDynamic, StatefulSymbolicContext

        placeholder_example_values: list = []
        for node in self.module.graph.nodes:
            if node.op == "placeholder":
                placeholder_example_values.append(node.meta.get("example_value"))

        fake_args = []
        for i, t in enumerate(args):
            if isinstance(t, FakeTensor):
                fake_args.append(t)
            elif isinstance(t, torch.Tensor):
                ev = placeholder_example_values[i] if i < len(placeholder_example_values) else None
                if isinstance(ev, FakeTensor):
                    dynamic_sizes = [
                        DimDynamic.DYNAMIC if isinstance(s, torch.SymInt) else DimDynamic.STATIC for s in ev.shape
                    ]
                    source = ConstantSource(f"ph_{i}")
                    sym_ctx = StatefulSymbolicContext(dynamic_sizes=dynamic_sizes, tensor_source=source)
                    fake_args.append(self.fake_mode.from_tensor(t, source=source, symbolic_context=sym_ctx))
                else:
                    fake_args.append(self.fake_mode.from_tensor(t))
            else:
                fake_args.append(t)
        return fake_args

    def call_module(
        self, target: torch.fx.node.Target, args: tuple[torch.fx.node.Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        assert isinstance(target, str)
        output = super().call_module(target, args, kwargs)
        if target not in self.compile_submod_names:
            return output

        index = self.compile_submod_names.index(target)
        submod = self.fetch_attr(target)
        sym_shape_indices = [i for i, x in enumerate(args) if isinstance(x, torch.SymInt)]
        magi_logger.info(f"Compiling {target=}, {sym_shape_indices=}, {args=}")

        compiled_graph_for_dynamic_shape = self.compiler_manager.compile(
            submod,
            args,
            self.inductor_config,
            graph_index=index,
            num_graphs=len(self.compile_submod_names),
            runtime_shape=None,
        )

        piecewise_backend = PiecewiseBackend(
            submod,
            compiled_graph_for_dynamic_shape,
            self.compile_config,
            self.inductor_config,
            index,
            len(self.compile_submod_names),
            sym_shape_indices,
            self.compiler_manager,
        )

        if self.compile_config.cudagraph_mode != CudaGraphMode.PIECEWISE:
            self.module.__dict__[target] = piecewise_backend
        else:
            wrapped_backend = gen_wrap_func_for_cudagraph(
                func=piecewise_backend, mode_prefix=CudaGraphMode.PIECEWISE.name.lower(), target_prefix=target
            )

            self.module.__dict__[target] = wrapped_backend
            magi_logger.info(
                f"Wrapped piecewise submodule {target} (index {index}) with CUDA Graph "
                f"[PIECEWISE mode, first_graph={piecewise_backend.is_first_graph}, last_graph={piecewise_backend.is_last_graph}]"
            )

        return output


class MagiBackend:
    """
    The compilation backend for `torch.compile` with MagiCompiler.
    It is used for compilation mode of `CompileMode.MAGI_COMPILE`,
    where we customize the compilation.

    The major work of this backend is to split the graph into
    piecewise graphs, and pass them to the piecewise backend.

    This backend also adds the PostGradPassManager to Inductor config,
    which handles the post-grad passes.
    """

    def __init__(
        self,
        compile_config: CompileConfig,
        model_idx: int,
        model_tag: str,
        traced_files: "OrderedSet",
        inductor_compile_config: dict[str, Any],
    ):
        self.compile_config = compile_config
        self.model_idx = model_idx
        self.model_tag = model_tag
        self.traced_files = traced_files
        self.inductor_compile_config = inductor_compile_config
        self._configure_custom_passes()
        self.compiler_manager: CompilerManager = CompilerManager(self.compile_config)
        self._called_once = False

    def _configure_custom_passes(self):
        # Custom pass 1: full graph passes between Dynamo and AOTAutograd
        self.full_graph_pass_manager = FullGraphPassManager(self.compile_config.pass_config)

        # Custom pass 2: custom partitioner function
        custom_partitioner_fn = CustomJointGraphPartitionFn()
        partitioner_key = self.compile_config.custom_partitioner_fn
        if partitioner_key in self.inductor_compile_config:
            existing_fn = self.inductor_compile_config[partitioner_key]
            assert isinstance(existing_fn, CustomJointGraphPartitionFn)
            assert existing_fn.uuid() == custom_partitioner_fn.uuid()
        self.inductor_compile_config[partitioner_key] = custom_partitioner_fn

        # Custom pass 3: post-grad passes after AOTAutograd
        post_grad_pass_manager = PostGradPassManager()
        post_grad_pass_manager.configure(self.compile_config.pass_config)

        post_grad_key = self.compile_config.post_grad_pass
        if post_grad_key in self.inductor_compile_config:
            existing_pass = self.inductor_compile_config[post_grad_key]
            assert isinstance(existing_pass, PostGradPassManager)
            assert existing_pass.uuid() == post_grad_pass_manager.uuid()

        self.inductor_compile_config[post_grad_key] = post_grad_pass_manager

    def _init_cache(self) -> str:
        hash_key = compute_hash(
            [
                self.compile_config.hash,
                inductor_compile_config_hash(self.inductor_compile_config),
                self.compiler_manager.hash,
                compute_code_hash(self.traced_files),
            ]
        )

        # Path: .../model_{idx}_{model_tag}_rank_{rank}/{hash}/
        self.local_magi_cache_path: Path = (
            cache_dump_path(self.compile_config.cache_root_dir, self.model_idx, self.model_tag) / hash_key
        )
        self.local_magi_cache_path.mkdir(parents=True, exist_ok=True)
        self.compiler_manager.initialize_cache(self.local_magi_cache_path)

    @observe_lifecycle("graph_split")
    def _split_graph(self, graph: fx.GraphModule) -> tuple[fx.GraphModule, list[SplitItem]]:
        # Step 1: resolve the splitting ops
        fx_split_ops = self.compile_config.splitting_ops or []
        resolved_ops: list[torch._ops.OpOverload] = resolve_defined_ops(fx_split_ops)
        magi_logger.info(f"Setting up FX-level graph split with ops: {fx_split_ops=}")
        magi_logger.info(f"Resolved splitting ops for FX-level graph split: {resolved_ops=}")

        # Step 2: split graph by ops, we split graph based on resolved_ops, which becomes the partitioned single graph.
        subgraph_id = 0
        node_to_subgraph_id = {}
        split_op_graphs = []
        for node in graph.graph.nodes:
            if node.op in ("output", "placeholder"):
                continue
            # Match node.target against resolved_ops, node.target can be OpOverloadPacket, need to check .default
            if node.op == "call_function" and (
                node.target in resolved_ops or (hasattr(node.target, "default") and node.target.default in resolved_ops)
            ):
                magi_logger.info(f"Splitting graph at {node=} with {node.target=}")
                subgraph_id += 1
                node_to_subgraph_id[node] = subgraph_id
                split_op_graphs.append(subgraph_id)
                subgraph_id += 1
            else:
                node_to_subgraph_id[node] = subgraph_id

        # Step 3: split the graph based on node_to_subgraph_id
        # pytorch might reorder the nodes and the semantics of the graph will change when we have mutations in the graph, if we don't set keep_original_order=True
        split_gm = torch.fx.passes.split_module.split_module(
            graph, None, lambda node: node_to_subgraph_id[node], keep_original_order=True
        )

        # Step 4: fetch all the submodules
        piecewise_graphs = []
        names = [name for (name, module) in split_gm.named_modules()]
        for name in names:
            # Only keep the top-level modules, skip recursive child modules or the root module
            if "." in name or name == "":
                continue

            module = getattr(split_gm, name)
            assert isinstance(module, fx.GraphModule), f"Expected fx.GraphModule, got {type(module)}"

            graph_id = int(name.replace("submod_", ""))
            piecewise_graphs.append(SplitItem(name, graph_id, (graph_id in split_op_graphs), module))
        # sort by integer graph_id, rather than string name
        piecewise_graphs.sort(key=lambda x: x.graph_id)

        # Step 5: visualize the split graph
        if envs.MAGI_ENABLE_FX_GRAPH_VIZ:
            save_fx_graph_visualization(split_gm.graph, sub_dir="after_split", filename="split_gm_root")
            for item in piecewise_graphs:
                save_fx_graph_visualization(item.graph.graph, sub_dir="after_split", filename=item.submod_name)

        return split_gm, piecewise_graphs

    @observe_lifecycle("magi_backend_call")
    def __call__(self, graph: fx.GraphModule, example_inputs) -> MagiSerializableFunction:
        assert not self._called_once, "MagiBackend can only be called once cause compilation is a one-time process"
        magi_logger.info("Dynamo traced files (for compilation cache):\n%s", "\n".join(self.traced_files))
        compilation_counter.num_graphs_seen += 1

        self._init_cache()

        self.full_graph_pass_manager(graph)

        split_gm, piecewise_graphs = self._split_graph(graph)

        submod_names_to_compile = [item.submod_name for item in piecewise_graphs if not item.is_splitting_graph]
        compilation_counter.num_piecewise_graphs_seen += len(piecewise_graphs)
        compilation_counter.num_piecewise_capturable_graphs_seen += len(submod_names_to_compile)
        magi_logger.info(f"Piecewise modules waiting for compilation: {submod_names_to_compile}")

        # Compile piecewise submodules with symbolic shapes
        # NOTE: `tensorify_python_scalars` pass triggers dynamo recapture by raising `TensorifyScalarRestartAnalysis` error.
        # So that we need to update `_called_once` after all compilation is done.

        PiecewiseCompileInterpreter(
            split_gm, self.compiler_manager, submod_names_to_compile, self.compile_config, self.inductor_compile_config
        ).run(*example_inputs)

        self._called_once = True

        # TODO: Support DBO (Dynamic Batching Orchestration) and NAT here.
        # TODO: Support TokenFlow graph forking here.

        if self.compile_config.offload_config.model_cpu_offload:
            split_gm = OffloadWrapper(split_gm, self.compile_config)

        runnable_gm = split_gm
        if self.compile_config.cudagraph_mode == CudaGraphMode.FULL:
            runnable_gm = gen_wrap_func_for_cudagraph(func=split_gm, mode_prefix=CudaGraphMode.FULL.name.lower())

        return MagiSerializableFunction(
            graph,
            example_inputs,
            self.model_tag,
            runnable_gm,
            model_idx=self.model_idx,
            traced_files=list(self.traced_files),
            compile_config=self.compile_config,
        )


def init_backend(
    compile_config: CompileConfig, model_idx: int, model_tag: str, traced_files: "OrderedSet", inductor_config: dict[str, Any]
) -> str | Callable:
    """
    Initialize the backend based on CompileConfig.
    """
    if compile_config.compile_mode is None or compile_config.compile_mode == CompileMode.NONE:
        raise ValueError("No compilation mode is set.")

    from torch._dynamo.backends.registry import list_backends

    torch_backends = list_backends(exclude_tags=tuple())
    magi_logger.info("Supported torch backends: %s", torch_backends)
    if compile_config.compile_mode == CompileMode.TORCH_COMPILE:
        assert compile_config.backend in torch_backends, f"Invalid backend for torch compilation: {compile_config.backend}"
        return compile_config.backend
    elif compile_config.compile_mode == CompileMode.MAGI_COMPILE:
        assert compile_config.backend in ["eager", "inductor"], f"Invalid backend for MagiCompiler: {compile_config.backend}"
        return MagiBackend(compile_config, model_idx, model_tag, traced_files, inductor_config)
    else:
        raise ValueError(f"Invalid compile mode: {compile_config.compile_mode}")
