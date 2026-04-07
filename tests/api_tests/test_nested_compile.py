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

"""Test nested compile scenarios: various combinations of torch.compile and magi_compile"""

import os

import pytest
import torch
import torch.nn as nn

from magi_compiler import magi_compile
from magi_compiler.config import CompileMode, get_compile_config

DEVICE = "cuda"
HIDDEN_SIZE = 8
TOLERANCE = 1e-3


# ============ Helper Functions ============


def is_torch_compiled(module: nn.Module) -> bool:
    """
    Check if a module is compiled by torch.compile

    Two ways:
    1. torch.compile(instance) -> OptimizedModule
    2. @torch.compile def forward -> forward has _torchdynamo_orig_callable

    Note: @torch.compiler.disable also sets _torchdynamo_orig_callable,
    but additionally sets _torchdynamo_disable=True, need to exclude
    """
    if type(module).__name__ == "OptimizedModule":
        return True
    forward_method = type(module).forward
    if hasattr(forward_method, "_torchdynamo_orig_callable"):
        if not getattr(forward_method, "_torchdynamo_disable", False):
            return True
    return False


def is_torch_disabled(module: nn.Module) -> bool:
    """Check if forward is decorated with @torch.compiler.disable"""
    return getattr(type(module).forward, "_torchdynamo_disable", False)


def assert_torch_compiled(module: nn.Module, msg: str = ""):
    assert is_torch_compiled(module), (
        f"Expected torch.compile'd. type={type(module).__name__}, "
        f"has _torchdynamo_orig_callable={hasattr(type(module).forward, '_torchdynamo_orig_callable')}. {msg}"
    )


def assert_not_torch_compiled_or_disabled(module: nn.Module, msg: str = ""):
    assert not is_torch_compiled(module), (
        f"Expected NOT torch.compile'd. type={type(module).__name__}, "
        f"has _torchdynamo_orig_callable={hasattr(type(module).forward, '_torchdynamo_orig_callable')}. {msg}"
    )


def assert_magi_compiled(module: nn.Module, msg: str = ""):
    assert hasattr(module, "_magi_state_for_forward"), f"Missing _magi_state_for_forward (MagiCompileState). {msg}"
    state = module._magi_state_for_forward
    assert state is not None, f"_magi_state_for_forward is None (compilation disabled). {msg}"
    assert state.jit_compiled_code is not None, f"jit_compiled_code is None. {msg}"


def assert_not_magi_compiled(module: nn.Module, msg: str = ""):
    if hasattr(module, "_magi_state_for_forward") and module._magi_state_for_forward is not None:
        assert module._magi_state_for_forward.jit_compiled_code is None, f"jit_compiled_code should be None. {msg}"


def assert_torch_disabled(module: nn.Module, msg: str = ""):
    assert is_torch_disabled(module), (
        f"Expected @torch.compiler.disable. "
        f"_torchdynamo_disable={getattr(type(module).forward, '_torchdynamo_disable', False)}. {msg}"
    )


# ============ Fixtures ============


@pytest.fixture(autouse=True)
def set_magi_compile_mode():
    """Set compile_mode=MAGI_COMPILE during tests"""
    config = get_compile_config()
    old_value = config.compile_mode
    config.compile_mode = CompileMode.MAGI_COMPILE
    config.cache_root_dir = os.environ.get("MAGI_COMPILE_CACHE_ROOT_DIR", config.cache_root_dir)
    print(f"set magi compile_mode: {config.compile_mode}, cache root dir: {config.cache_root_dir}")
    yield
    config.compile_mode = old_value


# ============ torch.compile Nested Behavior ============


def test_torch_compile_nested():
    """torch.compile nested: inner compiled OptimizedModule as opaque node"""

    class InnerBlock(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.linear = nn.Linear(hidden_size, hidden_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.linear(x))

    class OuterModel(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.inner = InnerBlock(hidden_size)
            self.output = nn.Linear(hidden_size, hidden_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.inner(x)
            return self.output(x)

    model = OuterModel(HIDDEN_SIZE).to(DEVICE)
    x = torch.randn(4, 16, HIDDEN_SIZE, device=DEVICE)

    with torch.no_grad():
        baseline = model(x)

    model.inner = torch.compile(model.inner, fullgraph=False, dynamic=True)
    assert_torch_compiled(model.inner)
    with torch.no_grad():
        inner_compiled_out = model(x)
    assert torch.allclose(baseline, inner_compiled_out, atol=TOLERANCE, rtol=TOLERANCE)

    compiled_model = torch.compile(model, fullgraph=False, dynamic=True)
    assert_torch_compiled(compiled_model)
    assert_torch_compiled(compiled_model.inner)
    with torch.no_grad():
        nested_out = compiled_model(x)

    assert torch.allclose(baseline, nested_out, atol=TOLERANCE, rtol=TOLERANCE)


def test_torch_compile_with_disable_inner():
    """torch.compile + @torch.compiler.disable: disabled functions cause graph break"""

    class InnerBlock(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.linear = nn.Linear(hidden_size, hidden_size)

        @torch.compiler.disable
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.linear(x))

    class OuterModel(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.inner = InnerBlock(hidden_size)
            self.output = nn.Linear(hidden_size, hidden_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.inner(x)
            return self.output(x)

    model = OuterModel(HIDDEN_SIZE).to(DEVICE)
    x = torch.randn(4, 16, HIDDEN_SIZE, device=DEVICE)

    with torch.no_grad():
        baseline = model(x)

    compiled_model = torch.compile(model, fullgraph=False, dynamic=True)
    assert_torch_compiled(compiled_model)
    assert_not_torch_compiled_or_disabled(compiled_model.inner)
    with torch.no_grad():
        compiled_out = compiled_model(x)

    assert torch.allclose(baseline, compiled_out, atol=TOLERANCE, rtol=TOLERANCE)


# ============ torch.compile + magi_compile Nested ============


def test_nested_torch_compile_magi_compile():
    """Outer torch.compile + inner magi_compile"""

    @magi_compile()
    class InnerMagiBlock(nn.Module):
        def __init__(self, hidden_size: int):
            super().__init__()
            self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
            self.linear2 = nn.Linear(hidden_size * 4, hidden_size)
            self.norm = nn.LayerNorm(hidden_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            x = self.norm(x)
            x = self.linear1(x)
            x = torch.nn.functional.gelu(x)
            x = self.linear2(x)
            return x + residual

    class OuterModel(nn.Module):
        def __init__(self, hidden_size: int, num_layers: int = 2):
            super().__init__()
            self.embed = nn.Linear(hidden_size, hidden_size)
            self.blocks = nn.ModuleList([InnerMagiBlock(hidden_size) for _ in range(num_layers)])
            self.output = nn.Linear(hidden_size, hidden_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.embed(x)
            for block in self.blocks:
                x = block(x)
            return self.output(x)

    num_layers = 2
    model = OuterModel(HIDDEN_SIZE, num_layers=num_layers).to(DEVICE)
    x = torch.randn(4, 16, HIDDEN_SIZE, device=DEVICE)

    for i, block in enumerate(model.blocks):
        assert hasattr(block, "_magi_wrapper_installed"), f"block {i} should be @magi_compile'd"
        assert block._magi_wrapper_installed is True

    with torch.no_grad():
        baseline = model(x)

    for i, block in enumerate(model.blocks):
        assert_magi_compiled(block)
        assert_not_torch_compiled_or_disabled(block)

    compiled_model = torch.compile(model, fullgraph=False, dynamic=True)
    assert_torch_compiled(compiled_model)
    assert compiled_model._orig_mod is model

    with torch.no_grad():
        compiled_out = compiled_model(x)

    for i, block in enumerate(model.blocks):
        assert block._magi_wrapper_installed is True
        assert_magi_compiled(block)

    assert torch.allclose(baseline, compiled_out, atol=TOLERANCE, rtol=TOLERANCE)


def test_nested_torch_compile_multiple_magi_compile():
    """Outer torch.compile with multiple magi_compile modules"""

    @magi_compile()
    class MagiBlock1(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.linear = nn.Linear(hidden_size, hidden_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.linear(x))

    @magi_compile()
    class MagiBlock2(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.linear = nn.Linear(hidden_size, hidden_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.linear(x))

    class OuterModel(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.block1 = MagiBlock1(hidden_size)
            self.block2 = MagiBlock2(hidden_size)
            self.output = nn.Linear(hidden_size, hidden_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.block1(x)
            x = self.block2(x)
            return self.output(x)

    model = OuterModel(HIDDEN_SIZE).to(DEVICE)
    x = torch.randn(4, 16, HIDDEN_SIZE, device=DEVICE)

    with torch.no_grad():
        baseline = model(x)

    assert_magi_compiled(model.block1)
    assert_magi_compiled(model.block2)
    assert_not_torch_compiled_or_disabled(model.block1)
    assert_not_torch_compiled_or_disabled(model.block2)

    compiled_model = torch.compile(model, fullgraph=False, dynamic=True)
    assert_torch_compiled(compiled_model)
    assert_not_torch_compiled_or_disabled(model.block1)
    assert_not_torch_compiled_or_disabled(model.block2)
    with torch.no_grad():
        compiled_out = compiled_model(x)

    assert torch.allclose(baseline, compiled_out, atol=TOLERANCE, rtol=TOLERANCE)


# ============ torch.compile Decorator + magi_compile Nested ============


def test_decorator_torch_compile_on_forward():
    """@torch.compile decorate forward: module type unchanged, but is_torch_compiled returns True"""

    class MyModel(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.linear = nn.Linear(hidden_size, hidden_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.linear(x))

    model = MyModel(HIDDEN_SIZE).to(DEVICE)
    x = torch.randn(4, 16, HIDDEN_SIZE, device=DEVICE)

    # eager baseline
    with torch.no_grad():
        baseline = model(x)

    # Create version with @torch.compile forward
    class CompiledModel(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.linear = nn.Linear(hidden_size, hidden_size)

        @torch.compile(fullgraph=False, dynamic=True)
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.linear(x))

    compiled_model = CompiledModel(HIDDEN_SIZE).to(DEVICE)
    compiled_model.load_state_dict(model.state_dict())

    assert type(compiled_model).__name__ == "CompiledModel"
    assert_torch_compiled(compiled_model)

    with torch.no_grad():
        out = compiled_model(x)

    assert torch.allclose(baseline, out, atol=TOLERANCE, rtol=TOLERANCE)


def test_decorator_nested_torch_compile_forward_magi_inner():
    """Outer forward @torch.compile + inner @magi_compile"""

    class InnerBlock(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.linear = nn.Linear(hidden_size, hidden_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.linear(x))

    class OuterModel(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.inner = InnerBlock(hidden_size)
            self.output = nn.Linear(hidden_size, hidden_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.inner(x)
            return self.output(x)

    model = OuterModel(HIDDEN_SIZE).to(DEVICE)
    x = torch.randn(4, 16, HIDDEN_SIZE, device=DEVICE)

    # eager baseline
    with torch.no_grad():
        baseline = model(x)

    # Create magi inner + torch.compile forward outer version
    MagiInnerBlock = magi_compile()(InnerBlock)

    class CompiledOuterModel(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.inner = MagiInnerBlock(hidden_size)
            self.output = nn.Linear(hidden_size, hidden_size)

        @torch.compile(fullgraph=False, dynamic=True)
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.inner(x)
            return self.output(x)

    compiled_model = CompiledOuterModel(HIDDEN_SIZE).to(DEVICE)
    compiled_model.load_state_dict(model.state_dict())

    assert_torch_compiled(compiled_model)

    with torch.no_grad():
        out = compiled_model(x)

    assert_magi_compiled(compiled_model.inner)
    assert_not_torch_compiled_or_disabled(compiled_model.inner)
    assert torch.allclose(baseline, out, atol=TOLERANCE, rtol=TOLERANCE)


# ============ torch._dynamo.config Correctness Verification ============


def test_dynamo_config_nested_patch_restore():
    """Verify config.patch() correctly restores to previous value when nested"""
    import torch._dynamo.config as config

    # Record initial value
    initial_value = config.assume_static_by_default

    # Simulate outer compile setting dynamic=True (assume_static_by_default=False)
    with config.patch(assume_static_by_default=False):
        assert config.assume_static_by_default is False, "outer patch should set value to False"

        # Simulate inner magi_compile restoring default (assume_static_by_default=True)
        with config.patch(assume_static_by_default=True):
            assert config.assume_static_by_default is True, "inner patch should set value to True"

        # After inner exits, should restore to outer value
        assert config.assume_static_by_default is False, "after inner exit should restore to outer value False"

    # After outer exits, should restore to initial value
    assert (
        config.assume_static_by_default == initial_value
    ), f"after outer exit should restore to initial value {initial_value}"


def test_dynamo_config_multiple_options_patch():
    """Verify correctness when patching multiple config options simultaneously"""
    import torch._dynamo.config as config

    # Record initial values
    initial_assume_static = config.assume_static_by_default
    initial_suppress_errors = config.suppress_errors
    initial_verbose = config.verbose

    # Patch multiple config options simultaneously
    with config.patch(
        assume_static_by_default=not initial_assume_static,
        suppress_errors=not initial_suppress_errors,
        verbose=not initial_verbose,
    ):
        # Verify all config options are modified
        assert config.assume_static_by_default == (not initial_assume_static), "assume_static_by_default should be modified"
        assert config.suppress_errors == (not initial_suppress_errors), "suppress_errors should be modified"
        assert config.verbose == (not initial_verbose), "verbose should be modified"

        # Nested patch for partial config options
        with config.patch(assume_static_by_default=initial_assume_static):
            assert config.assume_static_by_default == initial_assume_static, "inner should restore assume_static_by_default"
            # Other config options should keep outer patch values
            assert config.suppress_errors == (not initial_suppress_errors), "suppress_errors should keep outer value"
            assert config.verbose == (not initial_verbose), "verbose should keep outer value"

        # After inner exits, assume_static_by_default should restore to outer patch value
        assert config.assume_static_by_default == (
            not initial_assume_static
        ), "after inner exit should restore to outer patch value"

    # After outer exits, all config options should restore to initial values
    assert config.assume_static_by_default == initial_assume_static, "after outer exit assume_static_by_default should restore"
    assert config.suppress_errors == initial_suppress_errors, "after outer exit suppress_errors should restore"
    assert config.verbose == initial_verbose, "after outer exit verbose should restore"


def test_dynamo_config_restore_on_exception():
    """Verify config correctly restores when exception is raised inside with block"""
    import torch._dynamo.config as config

    # Record initial value
    initial_value = config.assume_static_by_default

    # Test single-layer patch restore on exception
    try:
        with config.patch(assume_static_by_default=not initial_value):
            assert config.assume_static_by_default == (not initial_value), "patch should take effect"
            raise RuntimeError("test exception")
    except RuntimeError:
        pass

    # After exception config should restore
    assert config.assume_static_by_default == initial_value, "after single exception should restore to initial value"

    # Test nested patch restore on inner exception
    try:
        with config.patch(assume_static_by_default=False):
            assert config.assume_static_by_default is False, "outer patch should take effect"
            try:
                with config.patch(assume_static_by_default=True):
                    assert config.assume_static_by_default is True, "inner patch should take effect"
                    raise ValueError("inner test exception")
            except ValueError:
                pass
            # After inner exception caught, should restore to outer value
            assert config.assume_static_by_default is False, "after inner exception should restore to outer value"
    except Exception:
        pytest.fail("outer should not catch exception")

    # Final should restore to initial value
    assert config.assume_static_by_default == initial_value, "after nested exception should restore to initial value"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
