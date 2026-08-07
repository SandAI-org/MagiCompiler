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

"""Tests for compile_artifacts.py serialization helpers.

Each test class reproduces the **real failure scenario** that the corresponding
class in ``compile_artifacts.py`` was written to fix, then verifies the fix.
"""

import math
import operator
import pickle

import pytest
import torch
import torch.fx as fx
from torch.utils._pytree import tree_map_only

_TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])
_IS_TORCH_212 = _TORCH_VERSION >= (2, 12)

from magi_compiler.magi_backend.compile_artifacts import (
    GraphNodeOpPatchUtils,
    GraphNodePicklePatchUtils,
    GraphPicklerPatchUtils,
    _deep_map_nodes,
    _import_by_qualname,
    _OpImportablePickleData,
)


def _make_graph_with_nodes(*names):
    """Return ``(graph, {name: node, ...})`` with placeholder nodes."""
    g = fx.Graph()
    nodes = {n: g.placeholder(n) for n in names}
    return g, nodes


# ═══════════════════════════════════════════════════════════════════════════
# Proof: slice(None, Node, None) really exists in FX graphs
# ═══════════════════════════════════════════════════════════════════════════


class TestSliceNodeProof:
    """Proves that ``fx.symbolic_trace`` (and by extension ``torch.compile``)
    produces ``operator.getitem`` nodes with ``slice(None, Node, None)`` in
    args, and that ``tree_map_only`` misses the Node inside the slice.
    """

    def test_symbolic_trace_produces_slice_with_node(self):
        """fx.symbolic_trace of ``x[:, :n]`` produces ``slice(None, Node('n'), None)``."""

        class SliceModel(torch.nn.Module):
            def forward(self, x, n):
                return x[:, :n]

        gm = fx.symbolic_trace(SliceModel())

        found_slice_with_node = False
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target is operator.getitem:
                for arg in node.args:
                    if isinstance(arg, tuple):
                        for elem in arg:
                            if isinstance(elem, slice) and isinstance(elem.stop, fx.Node):
                                found_slice_with_node = True

        assert found_slice_with_node, "Expected slice(None, Node, None) in FX graph but not found"

    def test_tree_map_only_misses_node_in_slice(self):
        """tree_map_only treats slice as opaque leaf — the bug that _deep_map_nodes fixes."""

        class SliceModel(torch.nn.Module):
            def forward(self, x, n):
                return x[:, :n]

        gm = fx.symbolic_trace(SliceModel())

        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target is operator.getitem:
                mapped = tree_map_only(fx.Node, lambda n: f"MAPPED_{n.name}", node.args)
                for arg in mapped:
                    if isinstance(arg, tuple):
                        for elem in arg:
                            if isinstance(elem, slice) and isinstance(elem.stop, fx.Node):
                                # tree_map_only left the Node unmapped — this IS the bug
                                assert True
                                return

        pytest.fail("Did not find the expected tree_map_only bug")

    def test_deep_map_nodes_fixes_slice_in_traced_graph(self):
        """_deep_map_nodes correctly maps the Node inside the traced slice."""

        class SliceModel(torch.nn.Module):
            def forward(self, x, n):
                return x[:, :n]

        gm = fx.symbolic_trace(SliceModel())

        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target is operator.getitem:
                mapped = _deep_map_nodes(node.args, fx.Node, lambda n: f"MAPPED_{n.name}")
                for arg in mapped:
                    if isinstance(arg, tuple):
                        for elem in arg:
                            if isinstance(elem, slice) and elem.stop is not None:
                                # _deep_map_nodes DID map the Node
                                assert isinstance(elem.stop, str)
                                assert elem.stop.startswith("MAPPED_")
                                return

        pytest.fail("Did not find slice with Node in traced graph")


# ═══════════════════════════════════════════════════════════════════════════
# _deep_map_nodes — fixes tree_map_only not descending into slice objects
# ═══════════════════════════════════════════════════════════════════════════


class TestDeepMapNodes:
    """Reproduce: ``tree_map_only`` treats ``slice`` as opaque leaf, so
    ``slice(None, <Node>, None)`` leaks raw Node refs into the pickle stream.

    ``_deep_map_nodes`` descends into slice/tuple/list/dict to fix this.
    """

    def test_tree_map_only_misses_slice__the_bug(self):
        """Reproduce the original bug: tree_map_only does NOT enter slices."""
        g, ns = _make_graph_with_nodes("x", "s0")
        args = (ns["x"], (slice(None, ns["s0"], None),))

        result = tree_map_only(torch.fx.Node, lambda n: n.name, args)

        # Top-level Node IS mapped
        assert result[0] == "x"
        # But Node inside slice is NOT — this is the bug
        assert isinstance(result[1][0].stop, torch.fx.Node)

    def test_deep_map_nodes_fixes_slice(self):
        """_deep_map_nodes correctly maps Node refs inside slices."""
        g, ns = _make_graph_with_nodes("x", "s0")
        args = (ns["x"], (slice(None, ns["s0"], None),))

        result = _deep_map_nodes(args, torch.fx.Node, lambda n: n.name)

        assert result[0] == "x"
        assert result[1][0].stop == "s0"

    def test_roundtrip_through_slice(self):
        """Simulate serialize→deserialize: Node→id→Node through slice."""
        g, ns = _make_graph_with_nodes("x", "s0")
        args = (ns["x"], (slice(None, ns["s0"], None),))

        fwd = {ns["x"]: "id_x", ns["s0"]: "id_s0"}
        pickled = _deep_map_nodes(args, torch.fx.Node, lambda n: fwd[n])
        assert pickled == ("id_x", (slice(None, "id_s0", None),))

        rev = {"id_x": ns["x"], "id_s0": ns["s0"]}
        restored = _deep_map_nodes(pickled, str, lambda s: rev.get(s, s))
        assert restored[0] is ns["x"]
        assert restored[1][0].stop is ns["s0"]

    def test_nested_containers(self):
        s = slice(None, [{"v": 5}], None)
        result = _deep_map_nodes(s, int, lambda n: n + 100)
        assert result == slice(None, [{"v": 105}], None)


# ═══════════════════════════════════════════════════════════════════════════
# Patch A: GraphPicklerPatchUtils — handles FakeTensorMode + sympy.Function
# ═══════════════════════════════════════════════════════════════════════════


class TestGraphPicklerPatchUtils:
    """Reproduce: ``FakeTensorMode`` holds a live ``ShapeEnv`` with symbolic
    variables, guards, and caches — standard ``pickle.dumps`` cannot handle it.

    Without ``GraphPicklerPatchUtils``, ``GraphPickler`` hits errors like
    "cannot pickle ShapeEnv", or produces a stale ``FakeTensorMode``
    disconnected from the session's ``ShapeEnv``.
    """

    def test_fake_tensor_mode_is_not_directly_picklable(self):
        """Reproduce the original failure: FakeTensorMode cannot be pickled."""
        from torch._subclasses import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        fm = FakeTensorMode(shape_env=ShapeEnv())
        with pytest.raises((TypeError, pickle.PicklingError, AttributeError)):
            pickle.dumps(fm)

    def test_restore_returns_session_fake_mode(self):
        """The fix: _restore_fake_mode extracts fake_mode from the unpickle state."""
        from torch._subclasses import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        fm = FakeTensorMode(shape_env=ShapeEnv())

        class MockState:
            fake_mode = fm

        assert GraphPicklerPatchUtils._restore_fake_mode(MockState()) is fm

    def test_reducer_override_intercepts_fake_mode(self):
        """make_patch_for_reducer_override returns a reducer tuple for FakeTensorMode."""
        from torch._subclasses import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        fm = FakeTensorMode(shape_env=ShapeEnv())

        patched = GraphPicklerPatchUtils.make_patch_for_reducer_override()

        class MockSelf:
            _unpickle_state = "token"

        result = patched(MockSelf(), fm)
        assert result == (GraphPicklerPatchUtils._restore_fake_mode, ("token",))

    def test_reducer_override_intercepts_sympy_function(self):
        """sympy.Function subclasses with _torch_unpickler are also handled."""
        import sympy

        class FakeSymFunc(sympy.Function):
            _torch_unpickler = lambda name: None  # noqa: E731
            _torch_handler_name = "test_handler"

        patched = GraphPicklerPatchUtils.make_patch_for_reducer_override()

        class MockSelf:
            _unpickle_state = "token"

        result = patched(MockSelf(), FakeSymFunc)
        assert result == (FakeSymFunc._torch_unpickler, ("test_handler",))

    def test_reducer_override_delegates_unknown_objects(self):
        """Objects not FakeTensorMode or sympy.Function pass through to original."""

        patched = GraphPicklerPatchUtils.make_patch_for_reducer_override()

        class MockSelf:
            _unpickle_state = "token"

        assert patched(MockSelf(), 42) is NotImplemented

    # ── Scenario: view tensor base.fake_mode ──

    @staticmethod
    def _make_dynamic_fake_tensor(shape=(2, 42, 64), dynamic_dims=(1,)):
        """Create a FakeTensor with SymInt dims for testing."""
        from torch._dynamo.source import ConstantSource
        from torch._subclasses import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv, StatelessSymbolicContext

        env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=env)
        sym_ctx = StatelessSymbolicContext(
            dynamic_sizes=[DimDynamic.DYNAMIC if i in dynamic_dims else DimDynamic.STATIC for i in range(len(shape))],
            constraint_sizes=[None] * len(shape),
        )
        real_t = torch.randn(*shape)
        with fake_mode:
            ft = fake_mode.from_tensor(real_t, symbolic_context=sym_ctx, source=ConstantSource("x"))
        return ft, fake_mode, env

    def test_view_tensor_base_fake_mode_not_cleared__the_bug(self):
        """Reproduce: _TensorPickleData clears top-level fake_mode but NOT base.fake_mode.

        For view tensors (e.g. transpose), MetaTensorDesc.base.fake_mode still
        holds a live FakeTensorMode. If the reducer replaces it with None, the
        deserialization fast-path fails and triggers an assertion error.
        """
        from torch._subclasses.meta_utils import MetaTensorDescriber

        ft, fake_mode, env = self._make_dynamic_fake_tensor()
        ft_view = ft.transpose(1, 2)
        assert ft_view._is_view()

        describer = MetaTensorDescriber(copy_data=False)
        desc = describer.describe_tensor(ft_view)

        # Top-level fake_mode is cleared by _TensorPickleData (that's normal)
        import dataclasses

        desc_cleared = dataclasses.replace(desc, fake_mode=None)
        assert desc_cleared.fake_mode is None

        # But base.fake_mode is NOT cleared — this is where FakeTensorMode leaks
        assert desc_cleared.base is not None
        assert desc_cleared.base.fake_mode is fake_mode  # still the live object!

    @pytest.mark.skipif(_IS_TORCH_212, reason="PyTorch 2.12 handles view tensors natively; see _pt212 variant")
    def test_view_tensor_bad_reducer_pt29(self):
        """PT 2.9: bad reducer (FakeTensorMode→None) breaks view tensor deserialization."""
        from unittest.mock import patch

        from torch._subclasses import FakeTensorMode
        from torch.fx._graph_pickler import GraphPickler, Options
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        ft, fake_mode, env = self._make_dynamic_fake_tensor()
        ft_view = ft.transpose(1, 2)

        orig_reducer = GraphPickler.reducer_override

        def bad_reducer(self, obj):
            if isinstance(obj, FakeTensorMode):
                return type(None), ()
            return orig_reducer(self, obj)

        with patch.object(GraphPickler, "reducer_override", bad_reducer):
            data = GraphPickler.dumps(ft_view, Options(ops_filter=None))

        env2 = ShapeEnv()
        fm2 = FakeTensorMode(shape_env=env2)
        with pytest.raises(AssertionError):
            GraphPickler.loads(data, fm2)

    @pytest.mark.skipif(not _IS_TORCH_212, reason="PyTorch 2.9 fails on bad reducer; see _pt29 variant")
    def test_view_tensor_bad_reducer_pt212(self):
        """PT 2.12: view tensors are restored into session FakeTensorMode even with a bad reducer."""
        from unittest.mock import patch

        from torch._subclasses import FakeTensorMode
        from torch.fx._graph_pickler import GraphPickler, Options
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        ft, fake_mode, env = self._make_dynamic_fake_tensor()
        ft_view = ft.transpose(1, 2)

        orig_reducer = GraphPickler.reducer_override

        def bad_reducer(self, obj):
            if isinstance(obj, FakeTensorMode):
                return type(None), ()
            return orig_reducer(self, obj)

        with patch.object(GraphPickler, "reducer_override", bad_reducer):
            data = GraphPickler.dumps(ft_view, Options(ops_filter=None))

        env2 = ShapeEnv()
        fm2 = FakeTensorMode(shape_env=env2)
        ft_loaded = GraphPickler.loads(data, fm2)

        assert ft_loaded.fake_mode is fm2
        assert ft_loaded.shape == ft_view.shape

    def test_view_tensor_fixed_reducer_succeeds(self):
        """The fix: FakeTensorMode → _restore_fake_mode(unpickle_state)
        correctly restores base.fake_mode, so view tensor deserialization works."""
        from unittest.mock import patch

        from torch._subclasses import FakeTensorMode
        from torch.fx._graph_pickler import GraphPickler, Options
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        ft, fake_mode, env = self._make_dynamic_fake_tensor()
        ft_view = ft.transpose(1, 2)

        fixed_reducer = GraphPicklerPatchUtils.make_patch_for_reducer_override()

        with patch.object(GraphPickler, "reducer_override", fixed_reducer):
            data = GraphPickler.dumps(ft_view, Options(ops_filter=None))

        env2 = ShapeEnv()
        fm2 = FakeTensorMode(shape_env=env2)
        ft_loaded = GraphPickler.loads(data, fm2)

        # Deserialization succeeds and fake_mode points to the session's mode
        assert ft_loaded.fake_mode is fm2
        assert ft_loaded.shape == ft_view.shape


# ═══════════════════════════════════════════════════════════════════════════
# Patch B: GraphNodePicklePatchUtils
#
# Fully replaces _NodePickleData.__init__ and unpickle to handle:
#   1. slice-embedded Nodes (deep-map)
#   2. un-picklable meta (whitelist strip)
#   3. targets via _OpPickleData.pickle() (standard path)
#   4. Triton kernel side-table (extract/restore per node)
# ═══════════════════════════════════════════════════════════════════════════


class TestGraphNodePicklePatchUtils:
    """Tests for the fully-replaced _NodePickleData.__init__ and unpickle."""

    # ── Scenario 1: slice with Node ──

    def test_slice_with_node_leaks_without_fix(self):
        """Reproduce: original _NodePickleData.__init__ uses tree_map_only,
        which leaves Node refs inside slice untouched → assert crash."""
        g, ns = _make_graph_with_nodes("x", "s0")
        getitem = g.call_function(operator.getitem, (ns["x"], (slice(None, ns["s0"], None),)))

        # Simulate what the original __init__ does
        mapping = {ns["x"]: "PD_x", ns["s0"]: "PD_s0"}
        result = tree_map_only(torch.fx.Node, lambda n: mapping[n], getitem.args)

        # The Node inside the slice is still a raw Node — this would crash GraphPickler
        assert isinstance(result[1][0].stop, torch.fx.Node)

    def test_patched_init_fixes_slice(self):
        """The fix: make_patch_for_init uses _deep_map_nodes which enters slices."""
        from torch.fx._graph_pickler import Options

        g, ns = _make_graph_with_nodes("x", "s0")
        getitem = g.call_function(operator.getitem, (ns["x"], (slice(None, ns["s0"], None),)))

        patched_init, stats = GraphNodePicklePatchUtils.make_patch_for_init()

        mapping = {ns["x"]: "PD_x", ns["s0"]: "PD_s0"}

        class FakeNPD:
            pass

        data = FakeNPD()
        patched_init(data, getitem, mapping, Options(ops_filter=None))

        assert data.args[0] == "PD_x"
        assert data.args[1][0].stop == "PD_s0"

    # ── Scenario 2: un-picklable source_fn_stack ──

    def test_source_fn_stack_is_not_picklable(self):
        """Reproduce: source_fn_stack contains callables → pickle.dumps fails."""

        def make_closure():
            x = 42
            return lambda: x  # noqa: E731

        meta_with_closure = {"source_fn_stack": [("fn", make_closure())]}
        with pytest.raises((pickle.PicklingError, AttributeError)):
            pickle.dumps(meta_with_closure)

    def test_nn_module_stack_with_closure_class_not_picklable(self):
        """Reproduce: nn_module_stack with closure-defined class → pickle fails."""

        def make_module_class():
            class _Inner(torch.nn.Module):
                pass

            return _Inner

        InnerCls = make_module_class()
        meta = {"nn_module_stack": {"m@1": ("layer", InnerCls)}}
        with pytest.raises((pickle.PicklingError, AttributeError)):
            pickle.dumps(meta)

    def test_patched_init_strips_unpicklable_meta(self):
        """The fix: make_patch_for_init strips meta to _META_WHITELIST."""
        from torch.fx._graph_pickler import Options

        g, ns = _make_graph_with_nodes("x")
        ns["x"].meta["source_fn_stack"] = [("relu", torch.relu)]
        ns["x"].meta["nn_module_stack"] = {"m@1": ("", torch.nn.Module)}
        ns["x"].meta["example_value"] = "keep_me"

        patched_init, stats = GraphNodePicklePatchUtils.make_patch_for_init()

        class FakeNPD:
            pass

        data = FakeNPD()
        patched_init(data, ns["x"], {}, Options(ops_filter=None))

        assert "source_fn_stack" not in data.meta
        assert "nn_module_stack" not in data.meta
        assert data.meta["example_value"] == "keep_me"

    def test_stats_accumulation(self):
        """Stats dict should accumulate drop counts across nodes."""
        from torch.fx._graph_pickler import Options

        g, ns = _make_graph_with_nodes("a", "b", "c")
        ns["a"].meta["source_fn_stack"] = [("relu", torch.relu)]
        ns["b"].meta["source_fn_stack"] = [("gelu", torch.nn.functional.gelu)]
        ns["b"].meta["nn_module_stack"] = {"m": ("", torch.nn.Module)}
        # c has no droppable keys

        patched_init, stats = GraphNodePicklePatchUtils.make_patch_for_init()

        for name in ["a", "b", "c"]:
            data = type("FakeNPD", (), {})()
            patched_init(data, ns[name], {}, Options(ops_filter=None))

        assert stats["total"] == 3
        assert stats["stripped"] == 2  # a and b had drops
        assert stats["dropped_keys"]["source_fn_stack"] == 2
        assert stats["dropped_keys"]["nn_module_stack"] == 1

    # ── Scenario 3: target via _OpPickleData.pickle() ──

    def test_patched_init_stores_op_pickle_data_target(self):
        """Patched init converts target via _OpPickleData.pickle() — standard path."""
        from torch.fx._graph_pickler import Options, _OpPickleData

        g = fx.Graph()
        x = g.placeholder("x")
        mul = g.call_function(torch.ops.aten.mul.Tensor, (x, 2))

        patched_init, _ = GraphNodePicklePatchUtils.make_patch_for_init()

        class FakeNPD:
            pass

        data = FakeNPD()
        patched_init(data, mul, {x: "PD_x"}, Options(ops_filter=None))

        # target is an _OpPickleData subclass (not the raw op)
        assert isinstance(data.target, _OpPickleData)
        assert hasattr(data.target, "unpickle")

    @pytest.mark.skipif(_IS_TORCH_212, reason="PyTorch 2.12 handles unknown ops natively; see _pt212 variant")
    def test_patched_init_with_einops_needs_patch_c_pt29(self):
        """PT 2.9: third-party functions like einops.rearrange need Patch C fallback."""
        from unittest.mock import patch

        einops = pytest.importorskip("einops")
        from torch.fx._graph_pickler import Options, _OpPickleData

        g = fx.Graph()
        x = g.placeholder("x")
        r = g.call_function(einops.rearrange, (x, "b (h d) -> b h d"), {"h": 2})

        patched_init, _ = GraphNodePicklePatchUtils.make_patch_for_init()
        patched_op_pickle = GraphNodeOpPatchUtils.make_patch_for_pickle()

        class FakeNPD:
            pass

        data = FakeNPD()
        with patch.object(_OpPickleData, "pickle", patched_op_pickle):
            patched_init(data, r, {x: "PD_x"}, Options(ops_filter=None))

        assert isinstance(data.target, _OpImportablePickleData)
        assert data.target.module_name == "einops.einops"

    @pytest.mark.skipif(not _IS_TORCH_212, reason="PyTorch 2.9 needs Patch C; see _pt29 variant")
    def test_patched_init_with_einops_native_pt212(self):
        """PT 2.12: third-party functions handled natively without Patch C."""
        einops = pytest.importorskip("einops")
        from torch.fx._graph_pickler import Options, _OpPickleData

        g = fx.Graph()
        x = g.placeholder("x")
        r = g.call_function(einops.rearrange, (x, "b (h d) -> b h d"), {"h": 2})

        patched_init, _ = GraphNodePicklePatchUtils.make_patch_for_init()

        class FakeNPD:
            pass

        data = FakeNPD()
        patched_init(data, r, {x: "PD_x"}, Options(ops_filter=None))

        assert isinstance(data.target, _OpPickleData)

    # ── Scenario 4: Triton kernel extraction ──

    def test_is_triton_node_detects_wrapper(self):
        """_is_triton_node identifies Triton kernel wrapper nodes."""

        class MockTritonOp:
            __name__ = "triton_kernel_wrapper_mutation"

            def __call__(self, **kwargs):
                pass

        g = fx.Graph()
        node = g.call_function(
            MockTritonOp(),
            (),
            {"kernel_idx": 0, "constant_args_idx": 0, "grid": [1], "tma_descriptor_metadata": {}, "kwargs": {}},
        )
        assert GraphNodePicklePatchUtils._is_triton_node(node) is True

    def test_is_triton_node_rejects_regular_op(self):
        g = fx.Graph()
        node = g.call_function(torch.relu, (g.placeholder("x"),))
        assert GraphNodePicklePatchUtils._is_triton_node(node) is False

    def test_is_triton_node_rejects_placeholder(self):
        g = fx.Graph()
        node = g.placeholder("x")
        assert GraphNodePicklePatchUtils._is_triton_node(node) is False

    def test_patched_init_extracts_triton_info(self):
        """Patched init extracts kernel module/qualname from the side table."""
        pytest.importorskip("flash_attn")
        from flash_attn.ops.triton.rotary import rotary_kernel
        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table, triton_kernel_wrapper_mutation
        from torch.fx._graph_pickler import Options

        k_idx = kernel_side_table.add_kernel(rotary_kernel)
        ca_idx = kernel_side_table.add_constant_args({"BLOCK_K": 32})

        g = fx.Graph()
        g.placeholder("x")
        triton_node = g.call_function(
            triton_kernel_wrapper_mutation,
            (),
            {"kernel_idx": k_idx, "constant_args_idx": ca_idx, "grid": [1], "tma_descriptor_metadata": {}, "kwargs": {}},
        )

        patched_init, _ = GraphNodePicklePatchUtils.make_patch_for_init()

        class FakeNPD:
            pass

        data = FakeNPD()
        patched_init(data, triton_node, {}, Options(ops_filter=None))

        assert hasattr(data, "_triton_kernel_info")
        assert data._triton_kernel_info["qualname"] == "rotary_kernel"
        assert hasattr(data, "_triton_constant_args")
        assert data._triton_constant_args == {"BLOCK_K": 32}


# ═══════════════════════════════════════════════════════════════════════════
# Patch C: GraphNodeOpPatchUtils — _OpPickleData.pickle safety net
# ═══════════════════════════════════════════════════════════════════════════


class TestGraphNodeOpPatchUtils:
    """Tests for the patched ``_OpPickleData.pickle``.

    The patch catches ``NotImplementedError`` for unknown op types
    and falls back to ``_OpImportablePickleData``.
    """

    @pytest.mark.skipif(_IS_TORCH_212, reason="PyTorch 2.12 handles unknown ops natively; see _pt212 variant")
    def test_original_pickle_unknown_op_pt29(self):
        """PT 2.9: _OpPickleData.pickle raises NotImplementedError for unknown third-party ops."""
        from torch.fx._graph_pickler import Options, _OpPickleData

        einops = pytest.importorskip("einops")

        with pytest.raises(NotImplementedError):
            _OpPickleData.pickle(einops.rearrange, Options(ops_filter=None))

    @pytest.mark.skipif(not _IS_TORCH_212, reason="PyTorch 2.9 raises for unknown ops; see _pt29 variant")
    def test_original_pickle_unknown_op_pt212(self):
        """PT 2.12: _OpPickleData.pickle handles unknown third-party ops natively."""
        from torch.fx._graph_pickler import Options, _OpPickleData

        einops = pytest.importorskip("einops")

        result = _OpPickleData.pickle(einops.rearrange, Options(ops_filter=None))
        assert isinstance(result, _OpPickleData)

    @pytest.mark.skipif(_IS_TORCH_212, reason="PyTorch 2.12 handles unknown ops natively; see _pt212 variant")
    def test_patched_pickle_catches_error_pt29(self):
        """PT 2.9: patched pickle falls back to _OpImportablePickleData for unknown ops."""
        from unittest.mock import patch

        from torch.fx._graph_pickler import Options, _OpPickleData

        einops = pytest.importorskip("einops")

        patched = GraphNodeOpPatchUtils.make_patch_for_pickle()

        with patch.object(_OpPickleData, "pickle", patched):
            result = _OpPickleData.pickle(einops.rearrange, Options(ops_filter=None))

        assert isinstance(result, _OpImportablePickleData)
        assert result.module_name == "einops.einops"
        assert result.qualname == "rearrange"

        restored = result.unpickle(None)
        assert restored is einops.rearrange

    @pytest.mark.skipif(not _IS_TORCH_212, reason="PyTorch 2.9 uses fallback; see _pt29 variant")
    def test_patched_pickle_passthrough_pt212(self):
        """PT 2.12: patched pickle passes through to native handling (no fallback needed)."""
        from unittest.mock import patch

        from torch.fx._graph_pickler import Options, _OpPickleData

        einops = pytest.importorskip("einops")

        patched = GraphNodeOpPatchUtils.make_patch_for_pickle()

        with patch.object(_OpPickleData, "pickle", patched):
            result = _OpPickleData.pickle(einops.rearrange, Options(ops_filter=None))

        assert isinstance(result, _OpPickleData)
        assert not isinstance(result, _OpImportablePickleData)

    def test_known_ops_pass_through(self):
        """torch ops should pass through the original path, not the fallback."""
        from unittest.mock import patch

        from torch.fx._graph_pickler import Options, _OpPickleData

        patched = GraphNodeOpPatchUtils.make_patch_for_pickle()

        with patch.object(_OpPickleData, "pickle", patched):
            result = _OpPickleData.pickle(torch.ops.aten.mul.Tensor, Options(ops_filter=None))

        # Should return a standard _OpPickleData subclass (not _OpImportablePickleData)
        assert isinstance(result, _OpPickleData)
        assert not isinstance(result, _OpImportablePickleData)


# ═══════════════════════════════════════════════════════════════════════════
# _import_by_qualname
# ═══════════════════════════════════════════════════════════════════════════


class TestImportByQualname:
    def test_import_torch_relu(self):
        assert _import_by_qualname("torch", "relu") is torch.relu

    def test_import_nested(self):
        assert _import_by_qualname("torch.nn.functional", "gelu") is torch.nn.functional.gelu

    def test_import_math_sqrt(self):
        assert _import_by_qualname("math", "sqrt") is math.sqrt

    def test_import_nonexistent_raises(self):
        with pytest.raises(AttributeError):
            _import_by_qualname("torch", "nonexistent_xyz")


# ═══════════════════════════════════════════════════════════════════════════
# Integration: Full serialize → deserialize round-trip
#
# Self-contained tests that build FX GraphModules with FakeTensor metadata,
# run serialize_compile_artifacts → deserialize_compile_artifacts,
# and verify the restored graph structure and metadata.
#
# These cover the same scenarios as learn/20, learn/21, learn/22 but run
# entirely in-process — no subprocess, no learn scripts, no CUDA required.
# ═══════════════════════════════════════════════════════════════════════════


class TestAOTIntegration:
    """End-to-end serialize → deserialize round-trip with real FX GraphModules.

    Each test manually constructs an FX graph with FakeTensor ``example_value``
    metadata, calls ``serialize_compile_artifacts``, then
    ``deserialize_compile_artifacts``, and verifies the restored result.

    ╔═══════════════════════════╦══════════════════════════════════════════════════════╗
    ║ Test                      ║ Patches exercised                                    ║
    ╠═══════════════════════════╬══════════════════════════════════════════════════════╣
    ║ test_basic_roundtrip      ║ A (FakeTensorMode) + B (meta strip) + C (op pickle)  ║
    ╠═══════════════════════════╬══════════════════════════════════════════════════════╣
    ║ test_einops_roundtrip     ║ A + B + C (einops → _OpImportablePickleData)         ║
    ╠═══════════════════════════╬══════════════════════════════════════════════════════╣
    ║ test_triton_roundtrip     ║ A + B (triton extract/restore per node) + C          ║
    ╠═══════════════════════════╬══════════════════════════════════════════════════════╣
    ║ test_slice_roundtrip      ║ A + B (deep-map slice-embedded Nodes) + C            ║
    ╚═══════════════════════════╩══════════════════════════════════════════════════════╝
    """

    @pytest.fixture(autouse=True)
    def _config(self, tmp_path):
        """Provide a clean magi CompileConfig scoped to each test."""
        import magi_compiler.config as cfg_mod

        saved = cfg_mod._GLOBAL_COMPILE_CONFIG
        cfg_mod._GLOBAL_COMPILE_CONFIG = None
        cfg = cfg_mod.get_compile_config()
        cfg.cache_root_dir = str(tmp_path)
        yield
        cfg_mod._GLOBAL_COMPILE_CONFIG = saved

    @staticmethod
    def _make_fake(shape=(2, 4)):
        """Create a FakeTensor with a fresh FakeTensorMode + ShapeEnv."""
        from torch._subclasses import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        fm = FakeTensorMode(shape_env=ShapeEnv())
        with fm:
            ft = fm.from_tensor(torch.randn(*shape))
        return ft, fm

    # ── learn/20 equivalent: basic model (Patch A + B) ──

    def test_basic_roundtrip(self):
        """placeholder → aten.mul → aten.add → output, with un-picklable meta.

        Verifies:
        - FakeTensorMode serialized via Patch A (persistent_id token)
        - Un-picklable meta (source_fn_stack, nn_module_stack) stripped by Patch B
        - Raw target (aten op) serialized via reducer_override
        - example_value preserved after round-trip
        """
        from magi_compiler.magi_backend.compile_artifacts import MagiSerializableFunction

        ft, fm = self._make_fake()

        g = torch.fx.Graph()
        x = g.placeholder("x")
        mul = g.call_function(torch.ops.aten.mul.Tensor, (x, 2))
        add = g.call_function(torch.ops.aten.add.Tensor, (mul, 1))
        g.output(add)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)

        with fm:
            ft_mul = ft * 2
            ft_add = ft_mul + 1

        nodes = list(gm.graph.nodes)
        nodes[0].meta.update(
            {
                "example_value": ft,
                # These are un-picklable — Patch B must strip them
                "source_fn_stack": [("fn", lambda: None)],
                "nn_module_stack": {"m": ("layer", type)},
            }
        )
        nodes[1].meta["example_value"] = ft_mul
        nodes[2].meta["example_value"] = ft_add

        fn = MagiSerializableFunction(gm, "test_basic", lambda *a: None)
        data = MagiSerializableFunction.serialize_compile_artifacts(fn)
        assert isinstance(data, bytes) and len(data) > 0

        restored = MagiSerializableFunction.deserialize_compile_artifacts(data)

        assert restored.model_tag == "test_basic"
        assert isinstance(restored.graph_module, torch.fx.GraphModule)

        r_nodes = list(restored.graph_module.graph.nodes)
        assert r_nodes[0].op == "placeholder"
        assert r_nodes[1].op == "call_function"

        # Raw target restored correctly
        assert r_nodes[1].target is torch.ops.aten.mul.Tensor

        # Un-picklable meta was stripped
        assert "source_fn_stack" not in r_nodes[0].meta
        assert "nn_module_stack" not in r_nodes[0].meta

        # example_value preserved with correct shape
        ev = r_nodes[0].meta.get("example_value")
        assert ev is not None and ev.shape == (2, 4)

    # ── learn/21 equivalent: einops model (Patch A + B, raw target natively pickled) ──

    def test_einops_roundtrip(self):
        """Graph with ``einops.rearrange`` as call_function target.

        Verifies:
        - Third-party function stored as raw target in _NodePickleData
        - Standard pickle handles module-level function natively
        - Restored target is the original einops.rearrange function
        """
        einops = pytest.importorskip("einops")
        from magi_compiler.magi_backend.compile_artifacts import MagiSerializableFunction

        ft, fm = self._make_fake(shape=(1, 8))

        g = torch.fx.Graph()
        x = g.placeholder("x")
        r = g.call_function(einops.rearrange, (x, "b (h d) -> b h d"), {"h": 2})
        g.output(r)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)

        with fm:
            ft_out = ft.view(1, 2, 4)

        nodes = list(gm.graph.nodes)
        nodes[0].meta["example_value"] = ft
        nodes[1].meta["example_value"] = ft_out

        fn = MagiSerializableFunction(gm, "test_einops", lambda *a: None)
        data = MagiSerializableFunction.serialize_compile_artifacts(fn)
        assert isinstance(data, bytes) and len(data) > 0

        restored = MagiSerializableFunction.deserialize_compile_artifacts(data)

        assert restored.model_tag == "test_einops"

        # Verify einops.rearrange was restored (natively by pickle)
        for node in restored.graph_module.graph.nodes:
            if node.op == "call_function":
                assert node.target is einops.rearrange

    # ── learn/22 equivalent: triton model (Patch A + B with triton per-node) ──

    def test_triton_roundtrip(self):
        """Graph with Triton kernel wrapper nodes → per-node extract/restore.

        Verifies:
        - Triton kernel info extracted in patched init (module + qualname)
        - After deserialization, kernel re-imported and re-registered
        - kernel_idx remapped so side-table lookup returns the original kernel
        """
        pytest.importorskip("flash_attn")
        from flash_attn.ops.triton.rotary import rotary_kernel
        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table, triton_kernel_wrapper_mutation

        from magi_compiler.magi_backend.compile_artifacts import MagiSerializableFunction

        ft, fm = self._make_fake()

        k_idx = kernel_side_table.add_kernel(rotary_kernel)
        ca_idx = kernel_side_table.add_constant_args({"BLOCK_K": 32})

        g = torch.fx.Graph()
        x = g.placeholder("x")
        g.call_function(
            triton_kernel_wrapper_mutation,
            (),
            {"kernel_idx": k_idx, "constant_args_idx": ca_idx, "grid": [1], "tma_descriptor_metadata": {}, "kwargs": {}},
        )
        g.output(x)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)

        list(gm.graph.nodes)[0].meta["example_value"] = ft

        fn = MagiSerializableFunction(gm, "test_triton", lambda *a: None)
        data = MagiSerializableFunction.serialize_compile_artifacts(fn)
        assert isinstance(data, bytes) and len(data) > 0

        restored = MagiSerializableFunction.deserialize_compile_artifacts(data)

        assert restored.model_tag == "test_triton"

        # Verify kernel was re-registered and index remapped correctly
        for node in restored.graph_module.graph.nodes:
            if node.op == "call_function":
                new_k_idx = node.kwargs.get("kernel_idx")
                if new_k_idx is not None:
                    assert kernel_side_table.get_kernel(new_k_idx) is rotary_kernel

    # ── slice-embedded Node roundtrip ──

    def test_slice_roundtrip(self):
        """Graph with ``operator.getitem(x, (slice(None), slice(None, node, None)))``.

        Verifies:
        - Node inside slice is correctly mapped during serialization
        - Node inside slice is correctly restored during deserialization
        - This is the scenario proven by TestSliceNodeProof
        """
        from magi_compiler.magi_backend.compile_artifacts import MagiSerializableFunction

        ft, fm = self._make_fake(shape=(2, 10, 64))
        with fm:
            ft_sliced = ft[:, :5, :]

        g = torch.fx.Graph()
        x = g.placeholder("x")
        n = g.placeholder("n")
        getitem = g.call_function(operator.getitem, (x, (slice(None, None, None), slice(None, n, None))))
        g.output(getitem)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)

        nodes = list(gm.graph.nodes)
        nodes[0].meta["example_value"] = ft
        # n is an int input — use a plain int FakeTensor equivalent
        nodes[1].meta["example_value"] = 5
        nodes[2].meta["example_value"] = ft_sliced

        fn = MagiSerializableFunction(gm, "test_slice", lambda *a: None)
        data = MagiSerializableFunction.serialize_compile_artifacts(fn)
        assert isinstance(data, bytes) and len(data) > 0

        restored = MagiSerializableFunction.deserialize_compile_artifacts(data)

        assert restored.model_tag == "test_slice"

        # Verify the getitem node's slice has a proper Node (not a dangling ref)
        for node in restored.graph_module.graph.nodes:
            if node.op == "call_function" and node.target is operator.getitem:
                slice_tuple = node.args[1]
                assert isinstance(slice_tuple[1], slice)
                # The stop of the slice should be the restored 'n' node
                assert isinstance(slice_tuple[1].stop, torch.fx.Node)
                assert slice_tuple[1].stop.name == "n"
