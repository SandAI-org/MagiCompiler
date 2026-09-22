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
Tests for @magi_compile decorator functionality.
"""

import shutil
import tempfile
from typing import Tuple
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from torch.testing import assert_close

from magi_compiler.api import magi_compile
from magi_compiler.config import CompileConfig
from tests.perf_tests import cuda_benchmark


@pytest.fixture(autouse=True)
def compile_config_fixture():
    """Fixture to set up a clean compile configuration for each test."""
    cache_dir = tempfile.mkdtemp()
    compile_config = CompileConfig(cache_root_dir=cache_dir)

    with patch("magi_compiler.config.get_compile_config") as mock_get_config, patch("torch.distributed.get_rank") as mock_rank:
        mock_get_config.return_value = compile_config
        mock_rank.return_value = 0
        yield compile_config

    shutil.rmtree(cache_dir, ignore_errors=True)


class TestDynamicArgDims:
    """Tests for dynamic_arg_dims inference and handling."""

    def test_automatic_inference(self):
        """Test that dynamic_arg_dims is automatically inferred."""

        # Base class
        class BaseModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return self.linear(x + y)

        # 1. Class level
        @magi_compile
        class ClsBaseModel(BaseModel):
            pass

        cls_model = ClsBaseModel()

        # 2. Function level
        class FuncBaseModel(BaseModel):
            @magi_compile
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return super().forward(x, y)

        func_model = FuncBaseModel()

        # 3. Instance level
        inst_model = BaseModel()
        inst_model = magi_compile(inst_model)

        # 4. Method level
        mtd_model = BaseModel()
        mtd_model.forward = magi_compile(mtd_model.forward)

        models = [cls_model, func_model, inst_model, mtd_model]

        for model in models:
            # Test with different batch sizes to ensure dynamic shape works
            x1 = torch.randn(4, 10)
            y1 = torch.randn(4, 10)
            output1 = model(x1, y1)
            assert output1.shape == (4, 5)

            x2 = torch.randn(8, 10)
            y2 = torch.randn(8, 10)
            output2 = model(x2, y2)
            assert output2.shape == (8, 5)

    def test_negative_index(self):
        """Test that negative indices in dynamic_arg_dims are handled correctly."""

        class BaseDynamicLastDimModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.out_features = 5
                self._weight = None
                self._bias = None

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                in_features = x.size(-1)
                if self._weight is None or self._weight.size(1) != in_features:
                    self._weight = torch.randn(self.out_features, in_features, device=x.device, dtype=x.dtype)
                    self._bias = torch.randn(self.out_features, device=x.device, dtype=x.dtype)
                return torch.matmul(x, self._weight.t()) + self._bias

        # 1. Class level
        @magi_compile(dynamic_arg_dims={"x": -1})
        class ClsBaseDynamicLastDimModel(BaseDynamicLastDimModel):
            pass

        cls_model = ClsBaseDynamicLastDimModel()

        # 2. Function level
        class FuncBaseDynamicLastDimModel(BaseDynamicLastDimModel):
            @magi_compile(dynamic_arg_dims={"x": -1})
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return super().forward(x)

        func_model = FuncBaseDynamicLastDimModel()

        # 3. Instance level
        inst_model = BaseDynamicLastDimModel()
        inst_model = magi_compile(inst_model, dynamic_arg_dims={"x": -1})

        # 4. Method level
        mtd_model = BaseDynamicLastDimModel()
        mtd_model.forward = magi_compile(mtd_model.forward, dynamic_arg_dims={"x": -1})

        models = [cls_model, func_model, inst_model, mtd_model]

        for model in models:
            x1 = torch.randn(4, 10)
            x2 = torch.randn(4, 15)

            output1 = model(x1)
            output2 = model(x2)

            assert output1.shape == (4, 5)
            assert output2.shape == (4, 5)
            # Accessing weight depends on which level we patched
            m = model.obj if hasattr(model, "obj") else model
            if isinstance(m, nn.Module):
                assert m._weight.size(1) == 15

    def test_nested_dataclass(self):
        """Test that dynamic_arg_dims can handle nested dataclasses correctly."""
        from dataclasses import dataclass

        @dataclass
        class InnerConfig:
            tensor_val: torch.Tensor
            other_val: int = 1

        @dataclass
        class OuterConfig:
            inner: InnerConfig
            flag: bool = True

        class DataclassModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, config: OuterConfig) -> torch.Tensor:
                x = config.inner.tensor_val
                return self.linear(x) * config.inner.other_val

        @magi_compile(dynamic_arg_dims={"config.inner.tensor_val": 0})
        class CompiledDataclassModel(DataclassModel):
            pass

        model = CompiledDataclassModel()

        # Test with batch size 4
        x1 = torch.randn(4, 10)
        config1 = OuterConfig(inner=InnerConfig(tensor_val=x1))
        out1 = model(config1)
        assert out1.shape == (4, 5)

        # Test with batch size 8
        x2 = torch.randn(8, 10)
        config2 = OuterConfig(inner=InnerConfig(tensor_val=x2))
        out2 = model(config2)
        assert out2.shape == (8, 5)


class TestCompilationCorrectness:
    """Tests verifying that compiled models produce correct outputs."""

    def test_compiled_matches_native_forward_complex(self):
        """Test the correctness of ComplexModel across all 12 configuration combinations.

        12 combinations = 4 objects (Class, Function, Instance, Method) * 3 ways (Decorator, Imperative, Factory)
        """

        size = 32
        x = torch.randn(2, size)

        # Define base structure for testing different styles
        class ComplexBlock(nn.Module):
            def __init__(self, size):
                super().__init__()
                self.ln = nn.LayerNorm(size)
                self.linear = nn.Linear(size, size)
                self.act = nn.GELU()

            def forward(self, x):
                return self.act(self.linear(self.ln(x)))

        class ComplexModel(nn.Module):
            def __init__(self, size):
                super().__init__()
                self.blocks = nn.Sequential(*[ComplexBlock(size) for _ in range(3)])

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for block in self.blocks:
                    x = x + block(x)
                return x

        # Prepare native model
        native_model = ComplexModel(size)
        native_output = native_model(x)

        # Prepare factory compiler
        compiler = magi_compile(dynamic_arg_dims={"x": 0})

        # --- Test Matrix ---
        # 1. Class
        @magi_compile(dynamic_arg_dims={"x": 0})
        class ClsDeco(ComplexModel):
            pass

        class ClsImp(ComplexModel):
            pass

        ClsImp = magi_compile(ClsImp, dynamic_arg_dims={"x": 0})

        @compiler
        class ClsFact(ComplexModel):
            pass

        # 2. Instance
        inst_deco = ComplexModel(size)  # N/A: Instance cannot be decorated directly
        inst_imp = magi_compile(ComplexModel(size), dynamic_arg_dims={"x": 0})
        inst_fact = compiler(ComplexModel(size))

        # 3. Function (Using model forward)
        @magi_compile(dynamic_arg_dims={"x": 0})
        def func_deco(x: torch.Tensor):
            return native_model(x)

        def func_imp_inner(x: torch.Tensor):
            return native_model(x)

        func_imp = magi_compile(func_imp_inner, dynamic_arg_dims={"x": 0})

        func_fact_inner = lambda x: native_model(x)
        func_fact = compiler(func_fact_inner)

        # 4. Method
        class MtdDeco(ComplexModel):
            @magi_compile(dynamic_arg_dims={"x": 0})
            def forward(self, x):
                return super().forward(x)

        mtd_deco = MtdDeco(size)

        mtd_imp = ComplexModel(size)
        mtd_imp.forward = magi_compile(mtd_imp.forward, dynamic_arg_dims={"x": 0})

        mtd_fact = ComplexModel(size)
        mtd_fact.forward = compiler(mtd_fact.forward)

        # 3. Unified verification function
        def _check(m_or_f, name):
            out = m_or_f(x)
            assert_close(out, native_output, rtol=1e-3, atol=1e-3, msg=f"Failed at {name}")

        # Helper function: synchronize state
        def _sync(model):
            if hasattr(model, "load_state_dict"):
                model.load_state_dict(native_model.state_dict())
            return model

        _check(_sync(ClsDeco(size)), "ClsDeco")
        _check(_sync(ClsImp(size)), "ClsImp")
        _check(_sync(ClsFact(size)), "ClsFact")

        # For instances, they are new instances upon construction and require weight synchronization
        _check(_sync(inst_imp), "inst_imp")
        _check(_sync(inst_fact), "inst_fact")

        _check(func_deco, "func_deco")
        _check(func_imp, "func_imp")
        _check(func_fact, "func_fact")

        # For methods, mtd_... also requires synchronization
        _check(_sync(mtd_deco), "mtd_deco")
        _check(_sync(mtd_imp), "mtd_imp")
        _check(_sync(mtd_fact), "mtd_fact")

        # 5. Non-nn.Module callable class / instance / method
        class CallableModel:
            def __init__(self, dim: int):
                self.dim = dim
                self.ln_weight = [torch.randn(dim) for _ in range(3)]
                self.ln_bias = [torch.randn(dim) for _ in range(3)]
                self.linear_weight = [torch.randn(dim, dim) for _ in range(3)]
                self.linear_bias = [torch.randn(dim) for _ in range(3)]

            def copy_from(self, other: "CallableModel"):
                self.ln_weight = [w.clone() for w in other.ln_weight]
                self.ln_bias = [b.clone() for b in other.ln_bias]
                self.linear_weight = [w.clone() for w in other.linear_weight]
                self.linear_bias = [b.clone() for b in other.linear_bias]

            def _run_blocks(self, x: torch.Tensor) -> torch.Tensor:
                for i in range(3):
                    res = x
                    x = torch.nn.functional.layer_norm(x, (self.dim,), self.ln_weight[i], self.ln_bias[i])
                    x = torch.nn.functional.linear(x, self.linear_weight[i], self.linear_bias[i])
                    x = torch.nn.functional.gelu(x)
                    x = x + res
                return x

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self._run_blocks(x)

            def __call__(self, x: torch.Tensor) -> torch.Tensor:
                return self.forward(x)

            def step(self, x: torch.Tensor) -> torch.Tensor:
                return self._run_blocks(x)

        x_non_module = torch.randn(2, size)
        native_callable = CallableModel(size)
        native_callable_out = native_callable(x_non_module)

        # class path (factory style)
        CallableClsFact = compiler(CallableModel)
        callable_cls_inst = CallableClsFact(size)
        callable_cls_inst.copy_from(native_callable)

        # instance path (factory style, returns compiled callable entry)
        callable_inst_obj = CallableModel(size)
        callable_inst_obj.copy_from(native_callable)
        callable_inst_entry = compiler(callable_inst_obj)

        # method path (bound method; dispatch decided by self type)
        callable_mtd_inst = CallableModel(size)
        callable_mtd_inst.copy_from(native_callable)
        callable_mtd_inst.step = compiler(callable_mtd_inst.step)

        assert_close(callable_cls_inst(x_non_module), native_callable_out, rtol=1e-3, atol=1e-3)
        assert_close(callable_inst_entry(x_non_module), native_callable_out, rtol=1e-3, atol=1e-3)
        assert_close(callable_mtd_inst.step(x_non_module), native_callable_out, rtol=1e-3, atol=1e-3)

        x_non_module_2 = torch.randn(5, size)
        native_callable_out_2 = native_callable(x_non_module_2)
        assert_close(callable_cls_inst(x_non_module_2), native_callable_out_2, rtol=1e-3, atol=1e-3)
        assert_close(callable_inst_entry(x_non_module_2), native_callable_out_2, rtol=1e-3, atol=1e-3)
        assert_close(callable_mtd_inst.step(x_non_module_2), native_callable_out_2, rtol=1e-3, atol=1e-3)

    def test_nested_function_calls(self):
        """Test compilation of model with nested function calls."""

        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 8, kernel_size=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self._preprocess(x)
                return self.conv(x)

            def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2 + 1

        x = torch.randn(2, 3, 8, 8)
        native_model = NestedModel()
        native_output = native_model.forward(x)

        # Class level
        @magi_compile(dynamic_arg_dims={"x": 0})
        class ClsLevelCompiledNestedModel(NestedModel):
            pass

        cls_model = ClsLevelCompiledNestedModel()
        cls_model.load_state_dict(native_model.state_dict())
        cls_level_output = cls_model.forward(x)

        # Instance level
        inst_model = NestedModel()
        inst_model.load_state_dict(native_model.state_dict())
        inst_model = magi_compile(inst_model, dynamic_arg_dims={"x": 0})
        inst_level_output = inst_model.forward(x)

        assert_close(cls_level_output, native_output, rtol=1e-3, atol=1e-3)
        assert_close(inst_level_output, native_output, rtol=1e-3, atol=1e-3)

    def test_multiple_outputs(self):
        """Test compilation of model with multiple outputs."""

        class MultiOutputModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 5)
                self.linear2 = nn.Linear(10, 5)

            def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                return self.linear1(x), self.linear2(x)

        x = torch.randn(4, 10)
        native_model = MultiOutputModel()
        native_output1, native_output2 = native_model(x)

        # Class level
        @magi_compile(dynamic_arg_dims={"x": 0})
        class ClsLevelCompiledMultiOutputModel(MultiOutputModel):
            pass

        cls_model = ClsLevelCompiledMultiOutputModel()
        cls_model.load_state_dict(native_model.state_dict())
        cls_level_output1, cls_level_output2 = cls_model(x)

        assert_close(cls_level_output1, native_output1, rtol=1e-3, atol=1e-3)
        assert_close(cls_level_output2, native_output2, rtol=1e-3, atol=1e-3)


class TestRecompilation:
    """Tests for source code change detection and recompilation."""

    def test_source_change_triggers_recompile(self, tmpdir):
        """Test that modifying source code triggers recompilation."""
        import importlib
        import sys

        model_config = MagicMock()

        # Create a temporary source file for the model
        src_file = tmpdir.join("test_model.py")
        src_content = """
import torch
from torch import nn

class TempModel(nn.Module):
    def __init__(self, *, model_config):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
"""
        src_file.write(src_content)
        sys.path.insert(0, str(tmpdir))

        import test_model
        from test_model import TempModel

        CompiledTempModel = magi_compile(dynamic_arg_dims={"x": 0})(TempModel)
        model1 = CompiledTempModel(model_config=model_config)
        x = torch.randn(4, 10)
        output1 = model1(x)

        # Modify the source file to change the forward logic
        modified_src = """
import torch
from torch import nn

class TempModel(nn.Module):
    def __init__(self, *, model_config):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) * 2
"""
        src_file.write(modified_src)
        importlib.reload(test_model)
        from test_model import TempModel

        CompiledTempModel2 = magi_compile(dynamic_arg_dims={"x": 0})(TempModel)
        model2 = CompiledTempModel2(model_config=model_config)
        output2 = model2(x)

        # Outputs should be different due to the *2 multiplication
        assert not torch.allclose(output1, output2, rtol=1e-5, atol=1e-5)

        # Verify compiled output matches native forward
        native_output = model2.forward(x)
        assert_close(output2, native_output, rtol=1e-5, atol=1e-5)


class TestPerformanceImprovementConsistency:
    def test_simple_model_performance_improvement_consistency(self):
        """Test that a simple model achieves similar performance across all compilation levels."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.dim = 8
                self.layers = nn.ModuleList([nn.Linear(self.dim, self.dim) for _ in range(2)])

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for layer in self.layers:
                    res = x
                    x = layer(x)
                    x = torch.nn.functional.gelu(x)
                    x = x + res
                return x

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        SEQ_LEN = 8
        test_input = torch.randn(SEQ_LEN, 8).to(DEVICE)
        native_model = SimpleModel().to(DEVICE)

        @magi_compile(dynamic_arg_dims={"x": 0})
        class ClsSimpleModel(SimpleModel):
            pass

        cls_model = ClsSimpleModel().to(DEVICE)
        cls_model.load_state_dict(native_model.state_dict())

        inst_model = SimpleModel().to(DEVICE)
        inst_model.load_state_dict(native_model.state_dict())
        inst_model = magi_compile(inst_model, dynamic_arg_dims={"x": 0})

        # Basic correctness across levels
        with torch.no_grad():
            native_out = native_model(test_input)
            cls_out = cls_model(test_input)
            inst_out = inst_model(test_input)

        assert_close(cls_out, native_out, rtol=1e-3, atol=1e-3)
        assert_close(inst_out, native_out, rtol=1e-3, atol=1e-3)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support for stable timing")
    def test_simple_model_timing_class_function_instance_method(self):
        """Timing sanity on a heavier workload: Class / Function / Instance / Method entrypoints."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.dim = 128
                self.layers = nn.ModuleList([nn.Linear(self.dim, self.dim) for _ in range(4)])
                self.norms = nn.ModuleList([nn.LayerNorm(self.dim) for _ in range(4)])

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for i, layer in enumerate(self.layers):
                    res = x
                    x = self.norms[i](x)
                    x = layer(x)
                    x = torch.nn.functional.gelu(x)
                    x = layer(x)
                    x = x + res
                return x

        device = torch.device("cuda:0")
        seq_len = 256
        test_input = torch.randn(seq_len, 128, device=device)

        native = SimpleModel().to(device).eval()

        @magi_compile(dynamic_arg_dims={"x": 0})
        class ClsSimpleModel(SimpleModel):
            pass

        cls_model = ClsSimpleModel().to(device).eval()
        cls_model.load_state_dict(native.state_dict())

        inst_model = SimpleModel().to(device).eval()
        inst_model.load_state_dict(native.state_dict())
        inst_model = magi_compile(inst_model, dynamic_arg_dims={"x": 0})

        func_native = SimpleModel().to(device).eval()
        func_native.load_state_dict(native.state_dict())

        @magi_compile(dynamic_arg_dims={"x": 0})
        def func_entry(x: torch.Tensor) -> torch.Tensor:
            return func_native(x)

        mtd_model = SimpleModel().to(device).eval()
        mtd_model.load_state_dict(native.state_dict())
        mtd_model.forward = magi_compile(mtd_model.forward, dynamic_arg_dims={"x": 0})

        class NonModulePerf:
            def __init__(self, dim: int):
                self.dim = dim
                self.ln_weight = [torch.randn(dim, device=device) for _ in range(4)]
                self.ln_bias = [torch.randn(dim, device=device) for _ in range(4)]
                self.linear_weight = [torch.randn(dim, dim, device=device) for _ in range(4)]
                self.linear_bias = [torch.randn(dim, device=device) for _ in range(4)]

            def copy_from(self, other: "NonModulePerf"):
                self.ln_weight = [w.clone() for w in other.ln_weight]
                self.ln_bias = [b.clone() for b in other.ln_bias]
                self.linear_weight = [w.clone() for w in other.linear_weight]
                self.linear_bias = [b.clone() for b in other.linear_bias]

            def _run(self, x: torch.Tensor) -> torch.Tensor:
                for i in range(4):
                    res = x
                    x = torch.nn.functional.layer_norm(x, (self.dim,), self.ln_weight[i], self.ln_bias[i])
                    x = torch.nn.functional.linear(x, self.linear_weight[i], self.linear_bias[i])
                    x = torch.nn.functional.gelu(x)
                    x = torch.nn.functional.linear(x, self.linear_weight[i], self.linear_bias[i])
                    x = x + res
                return x

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self._run(x)

            def __call__(self, x: torch.Tensor) -> torch.Tensor:
                return self.forward(x)

            def step(self, x: torch.Tensor) -> torch.Tensor:
                return self._run(x)

        @magi_compile(dynamic_arg_dims={"x": 0})
        class CompiledNonModulePerf(NonModulePerf):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return super().forward(x)

        non_module_native = NonModulePerf(128)

        non_module_cls = CompiledNonModulePerf(128)
        non_module_cls.copy_from(non_module_native)

        non_module_inst_obj = NonModulePerf(128)
        non_module_inst_obj.copy_from(non_module_native)
        non_module_inst = magi_compile(non_module_inst_obj, dynamic_arg_dims={"x": 0})

        non_module_mtd_obj = NonModulePerf(128)
        non_module_mtd_obj.copy_from(non_module_native)
        non_module_mtd_obj.step = magi_compile(non_module_mtd_obj.step, dynamic_arg_dims={"x": 0})

        with torch.no_grad():
            class_result = cuda_benchmark(lambda: cls_model(test_input), compilation_warmup=3)
            func_result = cuda_benchmark(lambda: func_entry(test_input), compilation_warmup=3)
            inst_result = cuda_benchmark(lambda: inst_model(test_input), compilation_warmup=3)
            method_result = cuda_benchmark(lambda: mtd_model(test_input), compilation_warmup=3)

        print(class_result.summary("class"))
        print(func_result.summary("function"))
        print(inst_result.summary("instance"))
        print(method_result.summary("method"))

        t_class = class_result.median / 1000.0
        t_func = func_result.median / 1000.0
        t_inst = inst_result.median / 1000.0
        t_mtd = method_result.median / 1000.0

        compiled_times = [t_class, t_func, t_inst, t_mtd]
        max_compiled = max(compiled_times)
        min_compiled = min(compiled_times)
        assert max_compiled / min_compiled < 1.2, (
            "Magi entry timings diverged too much: "
            f"class={t_class:.4f}s, function={t_func:.4f}s, instance={t_inst:.4f}s, method={t_mtd:.4f}s"
        )

        # non-nn.Module callable class / instance / method timing sanity
        with torch.no_grad():
            non_module_class_result = cuda_benchmark(lambda: non_module_cls(test_input), compilation_warmup=3)
            non_module_instance_result = cuda_benchmark(lambda: non_module_inst(test_input), compilation_warmup=3)
            non_module_method_result = cuda_benchmark(lambda: non_module_mtd_obj.step(test_input), compilation_warmup=3)

        print(non_module_class_result.summary("non_module_class"))
        print(non_module_instance_result.summary("non_module_instance"))
        print(non_module_method_result.summary("non_module_method"))

        t_nm_class = non_module_class_result.median / 1000.0
        t_nm_inst = non_module_instance_result.median / 1000.0
        t_nm_mtd = non_module_method_result.median / 1000.0

        nm_times = [t_nm_class, t_nm_inst, t_nm_mtd]
        max_nm = max(nm_times)
        min_nm = min(nm_times)
        assert max_nm / min_nm < 1.2, (
            "Non-module entry timings diverged too much: "
            f"class={t_nm_class:.4f}s, instance={t_nm_inst:.4f}s, method={t_nm_mtd:.4f}s"
        )
