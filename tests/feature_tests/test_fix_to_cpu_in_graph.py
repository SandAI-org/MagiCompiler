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

"""Test: _fix_graph_device_placement rewrites .to(device('cpu')) in FX graphs.

Root cause (commit e0c7277): _deep_cuda was removed, so Dynamo traces with CPU
tensors.  .to(x.device) gets specialised to .to(device('cpu')) as a literal
constant in the FX graph.  _fix_graph_device_placement must rewrite these nodes
to .to(device('cuda')) alongside the existing example_value metadata fix.
"""

import pytest
import torch
import torch.fx as fx
import torch.nn as nn


def _build_graph_with_to_cpu():
    """Build an FX graph that mirrors the offload-traced pattern:

    mapping_on_cpu = mapping.to(device('cpu'))
    out = x.index_select(0, mapping_on_cpu)
    """
    graph = fx.Graph()
    x = graph.placeholder("x")
    mapping = graph.placeholder("mapping")
    to_node = graph.call_method("to", args=(mapping, torch.device("cpu")))
    idx_node = graph.call_method("index_select", args=(x, 0, to_node))
    graph.output(idx_node)

    gm = fx.GraphModule(nn.Module(), graph)

    x.meta["example_value"] = torch.randn(8, 32, dtype=torch.bfloat16)
    mapping.meta["example_value"] = torch.randperm(8)
    to_node.meta["example_value"] = torch.randperm(8)
    idx_node.meta["example_value"] = torch.randn(8, 32, dtype=torch.bfloat16)

    return gm


def _is_cpu_device(val):
    if isinstance(val, torch.device):
        return val.type == "cpu"
    if isinstance(val, str):
        return val == "cpu"
    return False


def _apply_metadata_only_fix(gm, target_device=0):
    for node in gm.graph.nodes:
        ev = node.meta.get("example_value")
        if ev is not None and hasattr(ev, "device") and str(ev.device) == "cpu":
            node.meta["example_value"] = ev.to(target_device)
    gm.recompile()


def _apply_full_fix(gm, target_device=0):
    for node in gm.graph.nodes:
        if node.op == "call_method" and node.target == "to":
            new_args = list(node.args)
            changed = False
            for i, arg in enumerate(new_args):
                if _is_cpu_device(arg):
                    new_args[i] = torch.device("cuda", target_device)
                    changed = True
            if changed:
                node.args = tuple(new_args)
            if "device" in node.kwargs and _is_cpu_device(node.kwargs["device"]):
                node.update_kwarg("device", torch.device("cuda", target_device))
    _apply_metadata_only_fix(gm, target_device)


def _run_with_fake_tensors(gm, cuda_device=0):
    from torch._subclasses.fake_tensor import FakeTensorMode

    x_real = torch.randn(8, 32, dtype=torch.bfloat16, device=f"cuda:{cuda_device}")
    mapping_real = torch.randperm(8, device=f"cuda:{cuda_device}")

    with FakeTensorMode() as fm:
        x_fake = fm.from_tensor(x_real)
        mapping_fake = fm.from_tensor(mapping_real)
        with torch.no_grad():
            return gm(x_fake, mapping_fake)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestFixToCpuInGraph:
    def test_graph_has_to_cpu_node(self):
        gm = _build_graph_with_to_cpu()
        found = any(
            isinstance(arg, torch.device) and arg.type == "cpu"
            for node in gm.graph.nodes
            if node.op == "call_method" and node.target == "to"
            for arg in node.args
        )
        assert found, "Graph should contain .to(device('cpu'))"

    def test_metadata_only_fix_fails(self):
        """Fixing only example_values leaves .to(cpu) -> device mismatch."""
        gm = _build_graph_with_to_cpu()
        _apply_metadata_only_fix(gm)
        with pytest.raises(RuntimeError, match=r"[Dd]evice|FakeTensor"):
            _run_with_fake_tensors(gm)

    def test_full_fix_succeeds(self):
        """Rewriting .to(cpu) -> .to(cuda) + metadata fix -> no error."""
        gm = _build_graph_with_to_cpu()
        _apply_full_fix(gm)
        out = _run_with_fake_tensors(gm)
        assert str(out.device).startswith("cuda")

    def test_no_residual_to_cpu(self):
        gm = _build_graph_with_to_cpu()
        _apply_full_fix(gm)
        for node in gm.graph.nodes:
            if node.op == "call_method" and node.target == "to":
                for arg in node.args:
                    assert not _is_cpu_device(arg), f"Residual .to(cpu): {node}"

    def test_to_dtype_untouched(self):
        """Rewrite must NOT affect .to(dtype) calls."""
        graph = fx.Graph()
        x = graph.placeholder("x")
        to_bf16 = graph.call_method("to", args=(x, torch.bfloat16))
        graph.output(to_bf16)
        gm = fx.GraphModule(nn.Module(), graph)
        x.meta["example_value"] = torch.randn(4, 8)
        to_bf16.meta["example_value"] = torch.randn(4, 8, dtype=torch.bfloat16)

        _apply_full_fix(gm)

        for node in gm.graph.nodes:
            if node.op == "call_method" and node.target == "to":
                assert node.args[1] is torch.bfloat16
