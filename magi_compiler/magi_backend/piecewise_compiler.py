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

import re
from abc import abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.fx as fx

from magi_compiler.magi_depyf.timeline import observe_lifecycle
from magi_compiler.utils import compilation_counter, compute_hash, magi_logger

from ._cache_data_cls import CacheEntry, CacheHandle


def _placeholder_names(graph: fx.GraphModule) -> list[str]:
    return [str(node.target) for node in graph.graph.nodes if node.op == "placeholder"]


def _summarize_compile_input(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return f"Tensor(shape={tuple(value.shape)},dtype={value.dtype},device={value.device})"
    if isinstance(value, torch.SymInt):
        return f"SymInt({value})"
    return type(value).__name__


def _read_generated_code_expected_arity(path: str) -> int | None:
    try:
        py_files = list(Path(path).rglob("*.py"))
        if not py_files:
            return None
        text = py_files[0].read_text()
        m = re.search(r"def call\(self, args\):\n\s+([^\n]+)= args", text)
        if m is None:
            return None
        lhs = m.group(1)
        return len([x for x in lhs.split(",") if x.strip()])
    except Exception:
        return None


class CompilerInterface:
    """
    The interface for a compiler that can be used by MagiCompiler.
    """

    # The name of the compiler, e.g. inductor. This is a class-level attribute.
    name: str

    @abstractmethod
    def initialize_cache(self, cache_dir: Path):
        """
        when the MagiCompiler process uses `cache_dir` as the cache directory,
        the compiler should initialize itself with the cache directory,
        e.g. by re-directing its own cache directory to a sub-directory.

        prefix can be used in combination with cache_dir to figure out the base
        cache directory, e.g. there're multiple parts of model being compiled,
        but we want to share the same cache directory for all of them.

        e.g.
        cache_dir = "/path/to/dir/backbone", prefix = "backbone"
        cache_dir = "/path/to/dir/eagle_head", prefix = "eagle_head"
        """
        pass

    @property
    @abstractmethod
    def hash(self) -> str:
        """
        Gather all the relevant information from the config, to compute a hash so that we can cache the compiled model.

        This function should only consider the information that is specific to the compiler.
        """
        return ""

    @abstractmethod
    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        inductor_compile_config: dict[str, Any],
        runtime_shape: int | None = None,
        key: str | None = None,
    ) -> tuple[Callable | None, Any | None]:
        """
        Compile the graph with the given example inputs and compiler config,
        with a runtime shape. If the `runtime_shape` is None, it means
        the `example_inputs` have a dynamic shape. Otherwise, the
        `runtime_shape` specifies the shape of the inputs.
        Right now we only support one variable shape for all inputs, which is the sequence length
        (number of tokens) during inference.

        Dynamo will make sure `graph(*example_inputs)` is valid.

        The function should return a compiled callable function, as well as
        a handle that can be used to directly load the compiled function.

        The handle should be a plain Python object, preferably a string or a
        file path for readability.

        If the compiler doesn't support caching, it should return None for the
        handle. If the compiler fails to compile the graph, it should return
        None for the compiled function as well.

        `key` is required for StandaloneInductorAdapter, it specifies where to
        save the compiled artifact. The compiled artifact gets saved to
        `cache_dir/key`.
        """
        return None, None

    @abstractmethod
    def load(
        self, graph: fx.GraphModule, example_inputs: list[Any], cache_entry: CacheEntry, cache_handle: CacheHandle
    ) -> Callable:
        """
        Load the compiled function from the handle. Raises an error if the handle is invalid.

        The handle is the second return value of the `compile` function.
        """
        raise NotImplementedError("caching is not supported")


class InductorStandaloneAdaptor(CompilerInterface):
    """
    The adaptor for the Inductor compiler, which requires PyTorch 2.8+.

    Mainly reuses the standalone_compile function from PyTorch Inductor.
    """

    name = "inductor_standalone"

    def __init__(self, compile_config):
        from magi_compiler.config import CompileConfig

        self.compile_config: CompileConfig = compile_config
        self._restart_analysis_counts: dict[str, int] = {}

    @property
    def hash(self) -> str:
        # summarize system state and pytorch state
        from torch._inductor.codecache import CacheBase, torch_key

        factors: list[Any] = [CacheBase.get_system(), torch_key()]
        return compute_hash(factors)

    def initialize_cache(self, cache_dir: Path):
        self.cache_dir: Path = cache_dir

    @observe_lifecycle("compiler_compile")
    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        inductor_compile_config: dict[str, Any],
        runtime_shape: int | None = None,
        key: str | None = None,
    ) -> tuple[Callable | None, CacheHandle | None]:
        # Step1: Update compile settings
        compilation_counter.num_inductor_compiles += 1
        current_config = {
            # standalone_compile hardcodes autotune_at_compile_time=True, but
            # Triton autotune benchmarks with unbacked SymInt dimensions cause
            # CUDA illegal-memory-access errors.  Disable compile-time autotune
            # so that tuning happens at first runtime invocation instead (same
            # kernel quality, avoids the crash, and tunes on real shapes).
            "triton.autotune_at_compile_time": False
        }
        if inductor_compile_config is not None:
            current_config.update(inductor_compile_config)
        if isinstance(runtime_shape, int):
            # for a specific sequence length, tuning triton kernel parameters can be beneficial
            current_config.update(
                {
                    "max_autotune": self.compile_config.enable_inductor_max_autotune,
                    "coordinate_descent_tuning": self.compile_config.enable_inductor_coordinate_descent_tuning,
                }
            )
            dynamic_shapes = "from_example_inputs"
        else:
            dynamic_shapes = "from_tracing_context"

        # Step2: Compile the graph
        import torch._functorch.config as functorch_config
        from torch._inductor import standalone_compile

        try:
            with functorch_config.patch(autograd_cache_allow_custom_autograd_functions=True):
                compiled_graph = standalone_compile(
                    graph, example_inputs, dynamic_shapes=dynamic_shapes, options={"config_patches": current_config}
                )
        except torch._dynamo.exc.RestartAnalysis as e:
            if key is not None:
                self._restart_analysis_counts[key] = self._restart_analysis_counts.get(key, 0) + 1
            magi_logger.info(
                "standalone_compile raised RestartAnalysis: type=%s key=%s runtime_shape=%s "
                "restart_reason=%s placeholder_count=%s example_input_count=%s",
                type(e).__name__,
                key,
                runtime_shape,
                getattr(e, "restart_reason", None),
                len(_placeholder_names(graph)),
                len(example_inputs),
            )
            raise

        # Step3: Save the compiled artifact
        # autograd_cache_allow_custom_autograd_functions=True is required above so that
        # autograd_function_apply (a HigherOrderOperator) does not bypass AOTAutograd cache
        # key computation, which would leave aot_autograd_artifacts empty and cause save() to fail.
        assert key is not None
        restart_analysis_count = self._restart_analysis_counts.get(key, 0)
        if hasattr(self, "cache_dir") and self.cache_dir is not None:
            try:
                path: Path = self.cache_dir / key
                compiled_graph.save(path=path.as_posix(), format="unpacked")
                compilation_counter.num_compiled_artifacts_saved += 1
                return compiled_graph, CacheHandle(key, path.as_posix(), restart_analysis_count)
            except Exception as e:
                magi_logger.warning("Failed to save compiled artifact for key '%s', skipping cache: %s", key, e)

        return compiled_graph, None

    def load(
        self, graph: fx.GraphModule, example_inputs: list[Any], cache_entry: CacheEntry, cache_handle: CacheHandle
    ) -> Callable | None:
        assert isinstance(cache_handle.key, str) and cache_handle.key is not None
        assert isinstance(cache_handle.path, str) and cache_handle.path is not None

        expected_arity = _read_generated_code_expected_arity(cache_handle.path)
        actual_arity = len(example_inputs)
        summarized_inputs = [_summarize_compile_input(x) for x in example_inputs[:8]]
        magi_logger.info(
            "artifact load ABI observe: key=%s runtime_shape=%s graph_index=%s "
            "placeholder_count=%s example_input_count=%s expected_input_count=%s sample_inputs=%s",
            cache_handle.key,
            cache_entry.runtime_shape,
            cache_entry.graph_index,
            len(_placeholder_names(graph)),
            actual_arity,
            expected_arity,
            summarized_inputs,
        )
        if expected_arity is not None and expected_arity != actual_arity:
            magi_logger.info(
                "skip artifact load because ABI arity mismatch: key=%s runtime_shape=%s graph_index=%s "
                "expected_input_count=%s actual_input_count=%s",
                cache_handle.key,
                cache_entry.runtime_shape,
                cache_entry.graph_index,
                expected_arity,
                actual_arity,
            )
            return None

        inductor_compiled_graph = torch._inductor.CompiledArtifact.load(path=cache_handle.path, format="unpacked")

        from torch._inductor.compile_fx import graph_returns_tuple

        is_return_tuple = graph_returns_tuple(graph)

        def compiled_graph_wrapper(*args):
            graph_output = inductor_compiled_graph(*args)
            if is_return_tuple:
                return graph_output
            else:
                return graph_output[0]

        return compiled_graph_wrapper


class EagerAdaptor(CompilerInterface):
    name = "eager"

    @observe_lifecycle("compiler_compile")
    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        inductor_compile_config: dict[str, Any],
        runtime_shape: int | None = None,
        key: str | None = None,
    ) -> tuple[Callable | None, CacheHandle | None]:
        compilation_counter.num_eager_compiles += 1
        # we don't need to compile the graph, just return the graph itself.
        # It does not support caching, return None for the handle.
        return graph, None
