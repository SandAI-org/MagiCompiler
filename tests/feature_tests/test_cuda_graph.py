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
Tests for CUDA Graph compilation modes.

This module tests:
- FULL mode: entire forward pass captured as a single graph
- PIECEWISE mode: forward pass split at custom ops into multiple graphs
- Parameter handling: nn.Parameter should be ignored in graph capture
"""

import os
from typing import Optional
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch.testing import assert_close

from magi_compiler.api import magi_compile, magi_register_custom_op
from magi_compiler.config import CompileConfig, CompileMode, CudaGraphMode
from magi_compiler.magi_backend.cuda_graph_mgr import CudaGraphMgr

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@magi_register_custom_op("athena::my_split_op", infer_output_meta_fn=["x"], is_subgraph_boundary=True)
def my_split_op(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


@pytest.fixture(autouse=True)
def cuda_graph_test_env(tmp_path):
    """Fixture to set up a clean CUDA graph test environment."""

    # Create an isolated CudaGraphMgr instance for testing
    test_mgr = CudaGraphMgr()
    test_mgr.cache = dict()

    with patch("magi_compiler.magi_backend.cuda_graph_mgr.cuda_graph_mgr", return_value=test_mgr):
        yield tmp_path, test_mgr


def _create_compile_config(cache_dir: str, cudagraph_mode: CudaGraphMode) -> CompileConfig:
    """Create a compile configuration for CUDA graph testing."""
    from magi_compiler.config import get_compile_config

    return CompileConfig(
        compile_mode=CompileMode.MAGI_COMPILE,
        backend="inductor",
        cudagraph_mode=cudagraph_mode,
        splitting_ops=get_compile_config().splitting_ops,
        cache_root_dir=cache_dir,
    )


class TestCudaGraphFullMode:
    """Tests for CudaGraphMode.FULL - entire forward as single graph."""

    def test_full_mode_basic(self, cuda_graph_test_env):
        """Test basic FULL mode functionality with graph reuse."""
        tmp_path, test_mgr = cuda_graph_test_env
        cache_dir = os.path.join(str(tmp_path), "cache_full")
        os.makedirs(cache_dir, exist_ok=True)

        compile_config = _create_compile_config(cache_dir, CudaGraphMode.FULL)

        with patch("magi_compiler.api.get_compile_config", return_value=compile_config), patch(
            "torch.distributed.get_rank", return_value=0
        ):

            @magi_compile(dynamic_arg_dims={"x": 0})
            class FullModeModel(nn.Module):
                def __init__(self, model_config: Optional[dict] = None):
                    super().__init__()
                    self.linear1 = nn.Linear(10, 20).cuda()
                    self.linear2 = nn.Linear(20, 5).cuda()

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    x = torch.relu(self.linear1(x))
                    return self.linear2(x)

            model = FullModeModel(model_config=None).cuda()

            with torch.no_grad():
                # Prepare test inputs
                x1 = torch.randn(4, 10).cuda()
                x2 = x1.clone()
                x3 = torch.randn(2, 10).cuda()
                x4 = torch.randn(6, 10).cuda()
                x5 = x1.clone()

                import magi_compiler.magi_backend.cuda_graph_mgr as cgm

                active_mgr = cgm.cuda_graph_mgr()

                # First run: capture graph
                output1 = model(x1)
                assert output1.shape == (4, 5)
                assert active_mgr.graph_count == 1
                assert active_mgr.tensor_entry_count == 1

                # Same shape input: reuse tensor and graph
                output2 = model(x2)
                assert_close(output1, output2, rtol=1e-4, atol=1e-4)
                assert active_mgr.tensor_entry_count == 1
                assert active_mgr.graph_count == 1

                # Smaller batch: reuse tensor, new graph
                output3 = model(x3)
                assert output3.shape == (2, 5)
                assert active_mgr.tensor_entry_count == 1
                assert active_mgr.graph_count == 2

                # Larger batch: expand tensor, invalidate previous graphs
                output4 = model(x4)
                assert output4.shape == (6, 5)
                assert active_mgr.tensor_entry_count == 1
                assert active_mgr.graph_count == 1

                # Return to original batch size: recapture graph
                output5 = model(x5)
                assert_close(output1, output5, rtol=1e-4, atol=1e-4)
                assert active_mgr.tensor_entry_count == 1
                assert active_mgr.graph_count == 2


class TestCudaGraphPiecewiseMode:
    """Tests for CudaGraphMode.PIECEWISE - split at custom ops."""

    def test_piecewise_mode_with_split_op(self, cuda_graph_test_env):
        """Test PIECEWISE mode with custom splitting ops."""
        tmp_path, test_mgr = cuda_graph_test_env
        cache_dir = os.path.join(str(tmp_path), "cache_piecewise")
        os.makedirs(cache_dir, exist_ok=True)

        compile_config = _create_compile_config(cache_dir, CudaGraphMode.PIECEWISE)

        with patch("magi_compiler.api.get_compile_config", return_value=compile_config), patch(
            "torch.distributed.get_rank", return_value=0
        ):

            @magi_compile(dynamic_arg_dims={"x": 0})
            class PiecewiseModel(nn.Module):
                def __init__(self, *, model_config):
                    super().__init__()
                    self.linear1 = nn.Linear(10, 20).cuda()
                    self.linear2 = nn.Linear(20, 5).cuda()

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    x = self.linear1(x)
                    x = torch.ops.athena.my_split_op(x)  # Split point
                    x = self.linear2(x)
                    return x

            model = PiecewiseModel(model_config=None).cuda()

            with torch.no_grad():
                x1 = torch.randn(4, 10).cuda()
                x2 = torch.randn(2, 10).cuda()
                x3 = torch.randn(6, 10).cuda()
                x4 = torch.randn(4, 10).cuda()
                x5 = x1.clone()

                import magi_compiler.magi_backend.cuda_graph_mgr as cgm

                active_mgr = cgm.cuda_graph_mgr()

                # First run: capture 2 sub-graphs (before and after split op)
                output1 = model(x1)
                assert output1.shape == (4, 5)
                assert active_mgr.tensor_entry_count == 2  # Two tensor entries for two sub-graphs
                assert active_mgr.graph_count == 2

                # Same input: reuse all
                output2 = model(x1)
                assert_close(output1, output2, rtol=1e-4, atol=1e-4)
                assert active_mgr.tensor_entry_count == 2
                assert active_mgr.graph_count == 2

                # Smaller batch: reuse tensors, new sub-graphs
                output3 = model(x2)
                assert output3.shape == (2, 5)
                assert active_mgr.tensor_entry_count == 2
                assert active_mgr.graph_count == 4

                # Larger batch: expand tensors, invalidate previous sub-graphs
                output4 = model(x3)
                assert output4.shape == (6, 5)
                assert active_mgr.tensor_entry_count == 2
                assert active_mgr.graph_count == 2

                # Return to batch=4: reuse tensors, recapture sub-graphs
                output5 = model(x4)
                assert output5.shape == (4, 5)
                assert active_mgr.tensor_entry_count == 2
                assert active_mgr.graph_count == 4

                # Same as first input: verify output consistency
                output6 = model(x5)
                assert_close(output1, output6, rtol=1e-4, atol=1e-4)
                assert active_mgr.tensor_entry_count == 2
                assert active_mgr.graph_count == 4


class TestCudaGraphParameterHandling:
    """Tests for nn.Parameter handling in CUDA graph capture."""

    def test_parameters_excluded_from_graph_inputs(self, cuda_graph_test_env):
        """Test that nn.Parameters are not included in graph input tensors."""
        tmp_path, test_mgr = cuda_graph_test_env
        cache_dir = os.path.join(str(tmp_path), "cache_params")
        os.makedirs(cache_dir, exist_ok=True)

        compile_config = _create_compile_config(cache_dir, CudaGraphMode.FULL)

        with patch("magi_compiler.api.get_compile_config", return_value=compile_config), patch(
            "torch.distributed.get_rank", return_value=0
        ):

            @magi_compile(dynamic_arg_dims={"x": 0})
            class ParamModel(nn.Module):
                def __init__(self, model_config: Optional[dict] = None):
                    super().__init__()
                    # Multiple nn.Parameters (simulating model weights)
                    self.weight1 = nn.Parameter(torch.randn(10, 20).cuda())
                    self.bias1 = nn.Parameter(torch.randn(20).cuda())
                    self.weight2 = nn.Parameter(torch.randn(20, 5).cuda())
                    self.bias2 = nn.Parameter(torch.randn(5).cuda())
                    self.nested_params = nn.ParameterList(
                        [nn.Parameter(torch.randn(3, 5).cuda()), nn.Parameter(torch.randn(4, 4).cuda())]
                    )

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    x = torch.matmul(x, self.weight1) + self.bias1
                    x = torch.relu(x)
                    x = torch.matmul(x, self.weight2) + self.bias2
                    x = x + torch.matmul(torch.ones_like(x[:, :3]), self.nested_params[0])
                    return x

            model = ParamModel(model_config=None).cuda()

            with torch.no_grad():
                x1 = torch.randn(4, 10).cuda()
                output1 = model(x1)
                assert output1.shape == (4, 5)

                import magi_compiler.magi_backend.cuda_graph_mgr as cgm

                active_mgr = cgm.cuda_graph_mgr()

                # Only 1 tensor entry (the input x), not the parameters
                assert active_mgr.tensor_entry_count == 1

                # Verify the static entry contains only the input tensor
                static_entry = next(iter(active_mgr.cache.values()))
                assert len(static_entry.input_tensors) == 1
                assert len(static_entry.output_tensors) == 1
                assert isinstance(static_entry.input_tensors[0], torch.Tensor)
                assert not isinstance(static_entry.input_tensors[0], nn.Parameter)

            # Additional verification: ArgsUtils extracts only input tensor
            from magi_compiler.magi_backend.cuda_graph_mgr import ArgsUtils

            input_obj = {"args": (x1,), "kwargs": {}}
            extracted_tensors, _, _ = ArgsUtils.recursive_extract_core(input_obj)
            assert len(extracted_tensors) == 1
            assert extracted_tensors[0].data_ptr() == x1.data_ptr()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
