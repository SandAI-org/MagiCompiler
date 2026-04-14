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

"""
Tests for @magi_register_custom_op decorator functionality.

This module tests:
- Basic custom op registration (forward only)
- Custom op with infer_output_meta_fn for torch.compile tracing
- Custom op with autograd support (setup_context + backward)
- Full custom op with all components
- Multiple outputs support
- Integration with magi_compile decorator
"""

import tempfile
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.testing import assert_close

from magi_compiler.api import magi_compile, magi_register_custom_op
from magi_compiler.config import CompileConfig, CompileMode


class TestBasicRegistration:
    """Tests for basic custom op registration without autograd."""

    def test_forward_only(self):
        """Test registering a custom op with only forward implementation."""

        @magi_register_custom_op(name="test::forward_only_op", mutates_args=())
        def _forward_only_op(x: torch.Tensor) -> torch.Tensor:
            return x * 2 + 1

        x = torch.randn(4, 8)
        output = _forward_only_op(x)
        expected = x * 2 + 1

        assert_close(output, expected)

    def test_multiple_inputs(self):
        """Test custom op with multiple input tensors."""

        @magi_register_custom_op(name="test::multi_input_op", mutates_args=())
        def _multi_input_op(a: torch.Tensor, b: torch.Tensor, scale: float) -> torch.Tensor:
            return (a + b) * scale

        a = torch.randn(4, 8)
        b = torch.randn(4, 8)
        scale = 2.5
        output = _multi_input_op(a, b, scale)
        expected = (a + b) * scale

        assert_close(output, expected)


class TestInferOutputMeta:
    """Tests for custom op with infer_output_meta_fn."""

    def test_with_infer_output_meta(self):
        """Test that infer_output_meta_fn is correctly registered for tracing."""

        def _scaled_add_infer_output_meta(x: torch.Tensor, y: torch.Tensor, scale: float) -> torch.Tensor:
            return torch.empty_like(x)

        @magi_register_custom_op(
            name="test::scaled_add_op", mutates_args=(), infer_output_meta_fn=_scaled_add_infer_output_meta
        )
        def _scaled_add_op(x: torch.Tensor, y: torch.Tensor, scale: float) -> torch.Tensor:
            return (x + y) * scale

        x = torch.randn(4, 8)
        y = torch.randn(4, 8)
        scale = 3.0
        output = _scaled_add_op(x, y, scale)
        expected = (x + y) * scale

        assert_close(output, expected)

    def test_multiple_outputs_infer_meta(self):
        """Test infer_output_meta_fn with multiple outputs."""

        def _split_op_infer_output_meta(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            half_size = x.shape[-1] // 2
            return (x.new_empty((*x.shape[:-1], half_size)), x.new_empty((*x.shape[:-1], half_size)))

        @magi_register_custom_op(name="test::split_op", mutates_args=(), infer_output_meta_fn=_split_op_infer_output_meta)
        def _split_op(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            half_size = x.shape[-1] // 2
            # NOTE: Output cannot share the same memory with input
            return torch.clone(x[..., :half_size]), torch.clone(x[..., half_size:])

        x = torch.randn(4, 8)
        out1, out2 = _split_op(x)

        assert out1.shape == (4, 4)
        assert out2.shape == (4, 4)
        assert_close(out1, x[..., :4])
        assert_close(out2, x[..., 4:])


class TestAutograd:
    """Tests for custom op with autograd support."""

    def test_with_autograd(self):
        """Test custom op with setup_context and backward functions."""

        def _square_infer_output_meta(x: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        def _square_setup_context(ctx, inputs, output):
            (x,) = inputs
            ctx.save_for_backward(x)

        def _square_backward(ctx, grad_output):
            (x,) = ctx.saved_tensors
            return grad_output * 2 * x

        @magi_register_custom_op(
            name="test::square_op",
            mutates_args=(),
            infer_output_meta_fn=_square_infer_output_meta,
            setup_context_fn=_square_setup_context,
            backward_fn=_square_backward,
        )
        def _square_op(x: torch.Tensor) -> torch.Tensor:
            return x * x

        x = torch.randn(4, 8, requires_grad=True)
        output = _square_op(x)
        loss = output.sum()
        loss.backward()

        # Gradient of x^2 is 2x
        expected_grad = 2 * x
        assert_close(x.grad, expected_grad)

    def test_autograd_multiple_inputs(self):
        """Test autograd with multiple input tensors."""

        def _weighted_sum_infer_output_meta(a: torch.Tensor, b: torch.Tensor, weight: float) -> torch.Tensor:
            return torch.empty_like(a)

        def _weighted_sum_setup_context(ctx, inputs, output):
            a, b, weight = inputs
            ctx.save_for_backward(a, b)
            ctx.weight = weight

        def _weighted_sum_backward(ctx, grad_output):
            a, b = ctx.saved_tensors
            weight = ctx.weight
            grad_a = grad_output * weight
            grad_b = grad_output * (1 - weight)
            return grad_a, grad_b, None  # None for non-tensor input

        @magi_register_custom_op(
            name="test::weighted_sum_op",
            mutates_args=(),
            infer_output_meta_fn=_weighted_sum_infer_output_meta,
            setup_context_fn=_weighted_sum_setup_context,
            backward_fn=_weighted_sum_backward,
        )
        def _weighted_sum_op(a: torch.Tensor, b: torch.Tensor, weight: float) -> torch.Tensor:
            return a * weight + b * (1 - weight)

        a = torch.randn(4, 8, requires_grad=True)
        b = torch.randn(4, 8, requires_grad=True)
        weight = 0.7

        output = _weighted_sum_op(a, b, weight)
        loss = output.sum()
        loss.backward()

        expected_grad_a = torch.ones_like(a) * weight
        expected_grad_b = torch.ones_like(b) * (1 - weight)

        assert_close(a.grad, expected_grad_a)
        assert_close(b.grad, expected_grad_b)

    def test_autograd_multiple_outputs(self):
        """Test autograd with multiple output tensors."""

        def _split_scale_infer_output_meta(x: torch.Tensor, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
            half = x.shape[-1] // 2
            return (x.new_empty((*x.shape[:-1], half)), x.new_empty((*x.shape[:-1], half)))

        def _split_scale_setup_context(ctx, inputs, output):
            x, scale = inputs
            ctx.save_for_backward(x)
            ctx.scale = scale
            ctx.half = x.shape[-1] // 2

        def _split_scale_backward(ctx, grad_out1, grad_out2):
            (x,) = ctx.saved_tensors
            scale = ctx.scale
            # Reconstruct gradient for x
            grad_x = torch.cat([grad_out1 * scale, grad_out2 * scale], dim=-1)
            return grad_x, None

        @magi_register_custom_op(
            name="test::split_scale_op",
            mutates_args=(),
            infer_output_meta_fn=_split_scale_infer_output_meta,
            setup_context_fn=_split_scale_setup_context,
            backward_fn=_split_scale_backward,
        )
        def _split_scale_op(x: torch.Tensor, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
            half = x.shape[-1] // 2
            return x[..., :half] * scale, x[..., half:] * scale

        x = torch.randn(4, 8, requires_grad=True)
        scale = 2.0

        out1, out2 = _split_scale_op(x, scale)
        loss = out1.sum() + out2.sum()
        loss.backward()

        expected_grad = torch.ones_like(x) * scale
        assert_close(x.grad, expected_grad)


class TestAutoGeneratedName:
    """Tests for auto-generated operator name when name is not provided."""

    def test_auto_name_single_output(self):
        """Test auto-generated name with single tensor output."""

        @magi_register_custom_op()
        def _auto_name_single_op(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        def fn(x):
            return _auto_name_single_op(x)

        compiled_fn = torch.compile(fn, backend="eager")

        x = torch.randn(4, 8)
        output = compiled_fn(x)
        expected = x * 2

        assert_close(output, expected)

    def test_auto_name_multiple_outputs(self):
        """Test auto-generated name with multiple tensor outputs."""

        @magi_register_custom_op()
        def _auto_name_multi_out_op(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.clone(a + 1), torch.clone(b + 2)

        def fn(a, b):
            return _auto_name_multi_out_op(a, b)

        compiled_fn = torch.compile(fn, backend="eager")

        a = torch.randn(3, 5)
        b = torch.randn(3, 5)
        out1, out2 = compiled_fn(a, b)

        assert_close(out1, a + 1)
        assert_close(out2, b + 2)

    def test_auto_name_with_autograd(self):
        """Test auto-generated name with autograd support."""

        def _auto_grad_setup_context(ctx, inputs, output):
            (x,) = inputs
            ctx.save_for_backward(x)

        def _auto_grad_backward(ctx, grad_output):
            (x,) = ctx.saved_tensors
            return grad_output * 2 * x

        @magi_register_custom_op(setup_context_fn=_auto_grad_setup_context, backward_fn=_auto_grad_backward)
        def _auto_name_square_op(x: torch.Tensor) -> torch.Tensor:
            return x * x

        x = torch.randn(4, 8, requires_grad=True)
        output = _auto_name_square_op(x)
        loss = output.sum()
        loss.backward()

        expected_grad = 2 * x
        assert_close(x.grad, expected_grad)


class TestDefaultIdentityMetaFn:
    """Tests for the default identity meta function when infer_output_meta_fn is not provided."""

    def test_single_output_default_meta(self):
        """Test default meta function with single tensor output."""

        @magi_register_custom_op(name="test::default_meta_single")
        def _default_meta_single_op(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        def fn(x):
            return _default_meta_single_op(x)

        compiled_fn = torch.compile(fn, backend="eager")

        x = torch.randn(4, 8)
        output = compiled_fn(x)
        expected = x * 2

        assert_close(output, expected)

    def test_single_output_multiple_inputs_default_meta(self):
        """Test default meta function with multiple inputs but single tensor output."""

        @magi_register_custom_op(name="test::default_meta_multi_in")
        def _default_meta_multi_in_op(a: torch.Tensor, b: torch.Tensor, scale: float) -> torch.Tensor:
            return (a + b) * scale

        def fn(a, b, scale):
            return _default_meta_multi_in_op(a, b, scale)

        compiled_fn = torch.compile(fn, backend="eager")

        a = torch.randn(4, 8)
        b = torch.randn(4, 8)
        scale = 2.5
        output = compiled_fn(a, b, scale)
        expected = (a + b) * scale

        assert_close(output, expected)

    def test_multiple_outputs_default_meta(self):
        """Test default meta function with multiple tensor outputs."""

        @magi_register_custom_op(name="test::default_meta_multi_out")
        def _default_meta_multi_out_op(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # Clone to avoid aliasing issues
            return torch.clone(x * 2), torch.clone(y * 3)

        def fn(x, y):
            return _default_meta_multi_out_op(x, y)

        compiled_fn = torch.compile(fn, backend="eager")

        x = torch.randn(4, 8)
        y = torch.randn(4, 8)
        out1, out2 = compiled_fn(x, y)

        assert_close(out1, x * 2)
        assert_close(out2, y * 3)

    def test_three_outputs_default_meta(self):
        """Test default meta function with three tensor outputs."""

        @magi_register_custom_op(name="test::default_meta_three_out", mutates_args=())
        def _default_meta_three_out_op(
            a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return torch.clone(a + 1), torch.clone(b + 2), torch.clone(c + 3)

        def fn(a, b, c):
            return _default_meta_three_out_op(a, b, c)

        compiled_fn = torch.compile(fn, backend="eager")

        a = torch.randn(2, 4)
        b = torch.randn(2, 4)
        c = torch.randn(2, 4)
        out1, out2, out3 = compiled_fn(a, b, c)

        assert_close(out1, a + 1)
        assert_close(out2, b + 2)
        assert_close(out3, c + 3)

    def test_default_meta_with_non_tensor_args(self):
        """Test default meta function correctly skips non-tensor arguments."""

        @magi_register_custom_op(name="test::default_meta_mixed_args")
        def _default_meta_mixed_args_op(
            scale: float, x: torch.Tensor, offset: int, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.clone(x * scale + offset), torch.clone(y * scale + offset)

        def fn(scale, x, offset, y):
            return _default_meta_mixed_args_op(scale, x, offset, y)

        compiled_fn = torch.compile(fn, backend="eager")

        scale = 2.0
        x = torch.randn(3, 5)
        offset = 10
        y = torch.randn(3, 5)
        out1, out2 = compiled_fn(scale, x, offset, y)

        assert_close(out1, x * scale + offset)
        assert_close(out2, y * scale + offset)


class TestTorchCompileIntegration:
    """Tests for integration with torch.compile."""

    def test_custom_op_in_compiled_function(self):
        """Test that custom op works inside a torch.compile'd function."""

        def _double_infer_output_meta(x: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        @magi_register_custom_op(name="test::double_op", mutates_args=(), infer_output_meta_fn=_double_infer_output_meta)
        def _double_op(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        def fn(x):
            y = _double_op(x)
            return y + 1

        compiled_fn = torch.compile(fn, backend="eager")

        x = torch.randn(4, 8)
        output = compiled_fn(x)
        expected = x * 2 + 1

        assert_close(output, expected)

    def test_custom_op_with_autograd_in_compiled_function(self):
        """Test custom op with autograd inside torch.compile'd function."""

        def _cube_infer_output_meta(x: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        def _cube_setup_context(ctx, inputs, output):
            (x,) = inputs
            ctx.save_for_backward(x)

        def _cube_backward(ctx, grad_output):
            (x,) = ctx.saved_tensors
            return grad_output * 3 * x * x

        @magi_register_custom_op(
            name="test::cube_op",
            mutates_args=(),
            infer_output_meta_fn=_cube_infer_output_meta,
            setup_context_fn=_cube_setup_context,
            backward_fn=_cube_backward,
        )
        def _cube_op(x: torch.Tensor) -> torch.Tensor:
            return x * x * x

        def fn(x):
            return _cube_op(x)

        compiled_fn = torch.compile(fn, backend="eager")

        x = torch.randn(4, 8, requires_grad=True)
        output = compiled_fn(x)
        loss = output.sum()
        loss.backward()

        # Gradient of x^3 is 3x^2
        expected_grad = 3 * x * x
        assert_close(x.grad, expected_grad)


@pytest.fixture()
def magi_compile_config():
    """Fixture to set up a clean compile configuration for magi_compile tests."""
    compile_config = CompileConfig(compile_mode=CompileMode.TORCH_COMPILE, cache_root_dir=tempfile.mkdtemp())

    with patch("magi_compiler.api.get_compile_config") as mock_get_config, patch("torch.distributed.get_rank") as mock_rank:
        mock_get_config.return_value = compile_config
        mock_rank.return_value = 0
        yield compile_config

    import shutil

    shutil.rmtree(compile_config.cache_root_dir, ignore_errors=True)


class TestMagiCompileIntegration:
    """Tests for integration with magi_compile decorator."""

    def test_custom_op_in_magi_compiled_module(self, magi_compile_config):
        """Test that custom op works inside a magi_compile'd nn.Module."""

        def _triple_infer_output_meta(x: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        @magi_register_custom_op(name="test::triple_op", mutates_args=(), infer_output_meta_fn=_triple_infer_output_meta)
        def _triple_op(x: torch.Tensor) -> torch.Tensor:
            return x * 3

        @magi_compile(dynamic_arg_dims={"x": 0})
        class TripleModule(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return _triple_op(x) + 1

        model = TripleModule()

        x = torch.randn(4, 8)
        output = model(x)
        expected = x * 3 + 1
        assert_close(output, expected)

        # Test with different batch size to exercise dynamic shapes
        x2 = torch.randn(8, 8)
        output2 = model(x2)
        expected2 = x2 * 3 + 1
        assert_close(output2, expected2)

    def test_custom_op_with_autograd_in_magi_compiled_module(self, magi_compile_config):
        """Test custom op with autograd inside a magi_compile'd nn.Module."""

        def _square_v2_infer_output_meta(x: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        def _square_v2_setup_context(ctx, inputs, output):
            (x,) = inputs
            ctx.save_for_backward(x)

        def _square_v2_backward(ctx, grad_output):
            (x,) = ctx.saved_tensors
            return grad_output * 2 * x

        @magi_register_custom_op(
            name="test::square_v2_op",
            mutates_args=(),
            infer_output_meta_fn=_square_v2_infer_output_meta,
            setup_context_fn=_square_v2_setup_context,
            backward_fn=_square_v2_backward,
        )
        def _square_v2_op(x: torch.Tensor) -> torch.Tensor:
            return x * x

        @magi_compile(dynamic_arg_dims={"x": 0})
        class SquareModule(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return _square_v2_op(x)

        model = SquareModule()

        x = torch.randn(4, 8, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Gradient of x^2 is 2x
        expected_grad = 2 * x
        assert_close(x.grad, expected_grad)

    def test_custom_op_with_linear_in_magi_compiled_module(self, magi_compile_config):
        """Test custom op combined with nn.Linear inside a magi_compile'd module."""

        def _relu_custom_infer_output_meta(x: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        @magi_register_custom_op(
            name="test::relu_custom_op", mutates_args=(), infer_output_meta_fn=_relu_custom_infer_output_meta
        )
        def _relu_custom_op(x: torch.Tensor) -> torch.Tensor:
            return torch.relu(x)

        @magi_compile(dynamic_arg_dims={"x": 0})
        class LinearReluModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 8)

            @torch.no_grad()
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return _relu_custom_op(self.linear(x))

        model = LinearReluModule()

        x = torch.randn(4, 8)
        output = model(x)
        expected = torch.relu(model.linear(x))
        assert_close(output, expected)

    def test_multiple_custom_ops_in_magi_compiled_module(self, magi_compile_config):
        """Test multiple custom ops used together inside a magi_compile'd module."""

        def _add_one_infer_output_meta(x: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        def _mul_two_infer_output_meta(x: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        @magi_register_custom_op(name="test::add_one_op", mutates_args=(), infer_output_meta_fn=_add_one_infer_output_meta)
        def _add_one_op(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        @magi_register_custom_op(name="test::mul_two_op", mutates_args=(), infer_output_meta_fn=_mul_two_infer_output_meta)
        def _mul_two_op(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        @magi_compile(dynamic_arg_dims={"x": 0})
        class ChainedOpsModule(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # (x + 1) * 2
                return _mul_two_op(_add_one_op(x))

        model = ChainedOpsModule()

        x = torch.randn(4, 8)
        output = model(x)
        expected = (x + 1) * 2
        assert_close(output, expected)

    def test_custom_op_multiple_outputs_in_magi_compiled_module(self, magi_compile_config):
        """Test custom op with multiple outputs inside a magi_compile'd module."""

        def _split_v2_infer_output_meta(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            half = x.shape[-1] // 2
            return (x.new_empty((*x.shape[:-1], half)), x.new_empty((*x.shape[:-1], half)))

        @magi_register_custom_op(name="test::split_v2_op", mutates_args=(), infer_output_meta_fn=_split_v2_infer_output_meta)
        def _split_v2_op(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            half = x.shape[-1] // 2
            return torch.clone(x[..., :half]), torch.clone(x[..., half:])

        @magi_compile(dynamic_arg_dims={"x": 0})
        class SplitModule(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                a, b = _split_v2_op(x)
                return a + 1, b * 2

        model = SplitModule()

        x = torch.randn(4, 8)
        out1, out2 = model(x)

        assert out1.shape == (4, 4)
        assert out2.shape == (4, 4)
        assert_close(out1, x[..., :4] + 1)
        assert_close(out2, x[..., 4:] * 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
