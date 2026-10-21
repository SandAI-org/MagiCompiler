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

from contextlib import contextmanager
from typing import Type

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from magi_compiler import magi_compile, magi_register_custom_op
from magi_compiler.config import OffloadPolicy, get_compile_config
from tests.model_definition import MLPConfig, RMSNorm


class TransformerWrapper(nn.Module):
    """
    A wrapper class simulating a Transformer Block.
    Accepts mlp_cls to support injecting dynamically defined classes.
    """

    def __init__(self, config: MLPConfig, mlp_cls: Type[nn.Module]):
        super().__init__()
        # Standard layer (should move to GPU)
        self.attention_proj = nn.Linear(config.hidden_size, config.hidden_size, dtype=config.params_dtype)

        # Compiled layer (should stay on CPU if offload is enabled)
        self.mlp = mlp_cls(config)

    def forward(self, x):
        x = self.mlp(x)
        x = my_attention(x, x, x)
        x = self.attention_proj(x)
        return x


@contextmanager
def set_cpu_offload(enable: bool, offload_policy: OffloadPolicy = OffloadPolicy.COST_EFFECTIVE):
    """
    Context manager to temporarily override the cpu_offload setting in global config.
    """
    config = get_compile_config()
    original_value = config.offload_config.model_cpu_offload
    config.offload_config.model_cpu_offload = enable
    original_offload_policy = config.offload_config.offload_policy
    config.offload_config.offload_policy = offload_policy
    try:
        yield
    finally:
        config.offload_config.model_cpu_offload = original_value
        config.offload_config.offload_policy = original_offload_policy


def create_offload_mlp_class():
    """
    Create MLP class at runtime so that @magi_compile decorator captures the *current* config state.

    This is necessary because the decorator runs at class definition time.
    By defining the class inside a function called within `set_cpu_offload(True)` context,
    we ensure the decorator sees `model_cpu_offload=True`.
    """

    @magi_compile(dynamic_arg_dims={"x": 0})
    class OffloadMLP(torch.nn.Module):
        config: MLPConfig

        def __init__(self, config: MLPConfig):
            super().__init__()
            self.config = config
            self.pre_norm = RMSNorm(config.hidden_size)
            self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, dtype=config.params_dtype)
            self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False, dtype=config.params_dtype)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.pre_norm(x).to(torch.bfloat16)
            x = self.up_proj(x).to(torch.float32)
            x = F.silu(x).to(torch.bfloat16)
            x = self.down_proj(x)
            return x

    return OffloadMLP


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support")
def test_cpu_offload_placement(device, mlp_config):
    """
    Test that the decorated module stays on CPU when .cuda() is called on parent,
    while other modules move correctly.
    """
    # Use the context manager to enable CPU offload
    with set_cpu_offload(True):
        # 1. Initialize the parent model
        OffloadMLP = create_offload_mlp_class()

        model = TransformerWrapper(mlp_config, mlp_cls=OffloadMLP)

        # Verify initial state (everything on CPU by default in PyTorch)
        assert model.attention_proj.weight.device.type == "cpu"
        assert model.mlp.up_proj.weight.device.type == "cpu"

        # 2. Move the model to GPU
        # This triggers the _apply hook in _magi_compile
        model.cuda()

        # 3. Verify devices
        # The standard layer should be on GPU
        assert model.attention_proj.weight.device.type == "cuda", "Standard layers should move to CUDA"

        # The compiled/offloaded layer should stay on CPU
        assert (
            model.mlp.up_proj.weight.device.type == "cpu"
        ), "Compiled MLP layer should remain on CPU due to offload configuration"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support")
def test_cpu_offload_manual_move(device, mlp_config):
    """
    Test that the offload hook only blocks the move ONCE.
    Subsequent calls to .to(device) on the specific module should allow movement.
    """
    with set_cpu_offload(True):
        OffloadMLP = create_offload_mlp_class()

        model = TransformerWrapper(mlp_config, mlp_cls=OffloadMLP)

        # 1. First move (Should trigger offload logic)
        model.cuda()
        assert model.mlp.up_proj.weight.device.type == "cpu"
        assert model.attention_proj.weight.device.type == "cuda"

        # 2. Check if the internal flag is set (optional debugging check)
        # Note: This relies on the implementation detail _magi_offloaded_once
        if hasattr(model.mlp, "_magi_offloaded_once"):
            assert model.mlp._magi_offloaded_once is True

        # 3. Second move (Should not take any effect)
        model.mlp.to(device)
        assert model.mlp.up_proj.weight.device.type == "cpu", "Subsequent .to() calls should not take effect"


@torch.library.custom_op("magi_test::add_scalar", mutates_args=())
def _add_scalar(x: torch.Tensor, value: float) -> torch.Tensor:
    return x + value


@_add_scalar.register_fake
def _(x, value):
    return torch.empty_like(x)


def create_offload_float_attribute_class():
    @magi_compile(dynamic_arg_dims={"x": 0})
    class FloatAttributeModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(8))
            self.shift = 0.25

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return _add_scalar(x * self.weight, self.shift)

    return FloatAttributeModule


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support")
def test_cpu_offload_keeps_python_scalar_inputs_on_cpu():
    """Dynamo passes an unspecialized Python float as a CPU float64 tensor and reads it back with ``.item()``.
    Moving it to CUDA with the other inputs failed the compile with ``aten._local_scalar_dense``."""
    with set_cpu_offload(True):
        model = create_offload_float_attribute_class()().cuda()
        x = torch.randn(4, 8, device="cuda")
        with torch.no_grad():
            out = model(x)
        torch.testing.assert_close(out, x * model.weight.detach().cuda() + 0.25)


@magi_register_custom_op("athena::my_attention", infer_output_meta_fn=["q"], is_subgraph_boundary=True)
def my_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return q + k + v


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support")
def test_cpu_offload_inference(device, mlp_config):
    """
    Test that the offload hook only blocks the move ONCE.
    Subsequent calls to .to(device) on the specific module should allow movement.
    """

    test_shapes = [
        (4, mlp_config.hidden_size),  # Small batch
        (8, mlp_config.hidden_size),  # Medium batch
        (16, mlp_config.hidden_size),  # Large batch
        # NOTE: compiler will specialize for single token, so we move it to the last
        (1, mlp_config.hidden_size),  # Single token
    ]
    with set_cpu_offload(True):
        OffloadMLP = create_offload_mlp_class()

        model = TransformerWrapper(mlp_config, mlp_cls=OffloadMLP)

        # 1. First move (Should trigger offload logic)
        model.cuda()
        assert model.mlp.up_proj.weight.device.type == "cpu"
        assert model.attention_proj.weight.device.type == "cuda"

        with torch.no_grad():
            for num_tokens, hidden_size in test_shapes:
                input_tensor = torch.randn(num_tokens, hidden_size, device=device, dtype=mlp_config.params_dtype)
                output = model(input_tensor)

                assert output.shape == (
                    num_tokens,
                    hidden_size,
                ), f"For input shape ({num_tokens}, {hidden_size}), output shape should be ({num_tokens}, {hidden_size}), but got {output.shape}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support")
def test_cpu_offload_heuristic(device, mlp_config):
    """
    Test that the heuristic scheduler is working correctly.
    """
    test_shapes = [
        (4, mlp_config.hidden_size),  # Small batch
        (8, mlp_config.hidden_size),  # Medium batch
        (16, mlp_config.hidden_size),  # Large batch
        # NOTE: compiler will specialize for single token, so we move it to the last
        (1, mlp_config.hidden_size),  # Single token
    ]
    with set_cpu_offload(True, OffloadPolicy.HEURISTIC):
        OffloadMLP = create_offload_mlp_class()
        model = TransformerWrapper(mlp_config, mlp_cls=OffloadMLP)
        model.cuda()
        assert model.mlp.up_proj.weight.device.type == "cpu"
        assert model.attention_proj.weight.device.type == "cuda"

        with torch.no_grad():
            for num_tokens, hidden_size in test_shapes:
                input_tensor = torch.randn(num_tokens, hidden_size, device=device, dtype=mlp_config.params_dtype)
                output = model(input_tensor)

                assert output.shape == (
                    num_tokens,
                    hidden_size,
                ), f"For input shape ({num_tokens}, {hidden_size}), output shape should be ({num_tokens}, {hidden_size}), but got {output.shape}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA support")
def test_cpu_offload_different_decoration_styles(device, mlp_config):
    """
    Test CPU offload with different decoration styles.
    """
    with set_cpu_offload(True):
        # 1. Class Decoration
        @magi_compile(dynamic_arg_dims={"x": 0})
        class OffloadMLPClass(torch.nn.Module):
            def __init__(self, config: MLPConfig):
                super().__init__()
                self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, dtype=config.params_dtype)
                self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False, dtype=config.params_dtype)

            def forward(self, x):
                return self.down_proj(torch.nn.functional.silu(self.up_proj(x)))

        # Test class level
        model_cls = TransformerWrapper(mlp_config, mlp_cls=OffloadMLPClass)
        model_cls.cuda()
        assert model_cls.mlp.up_proj.weight.device.type == "cpu"

        # 2. Instance Decoration (Should be skipped/not offloaded now)
        # class PlainMLP(torch.nn.Module):
        #     def __init__(self, config: MLPConfig):
        #         super().__init__()
        #         self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, dtype=config.params_dtype)
        #         self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False, dtype=config.params_dtype)
        #
        #     def forward(self, x):
        #         return self.down_proj(torch.nn.functional.silu(self.up_proj(x)))
        #
        # model_inst = TransformerWrapper(mlp_config, mlp_cls=PlainMLP)
        # magi_compile(model_inst.mlp, dynamic_arg_dims={"x": 0})
        # model_inst.cuda()
        # assert model_inst.mlp.up_proj.weight.device.type == "cpu"
        #
        # # 3. Method Decoration (Should be skipped/not offloaded now)
        # model_mtd = TransformerWrapper(mlp_config, mlp_cls=PlainMLP)
        # magi_compile(model_mtd.mlp.forward, dynamic_arg_dims={"x": 0})
        # model_mtd.cuda()
        # assert model_mtd.mlp.up_proj.weight.device.type == "cpu"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
