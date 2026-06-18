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

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib
import inspect
import pickle
from typing import TYPE_CHECKING
from unittest.mock import patch

import torch
from torch.utils._pytree import tree_map_only

from magi_compiler.utils import magi_logger

if TYPE_CHECKING:
    from magi_compiler.config import CompileConfig

try:
    from torch._dynamo.aot_compile import SerializableCallable
except ImportError:
    SerializableCallable = object

assert isinstance(SerializableCallable, type)


class MagiSerializableFunction(SerializableCallable):
    """
    A wrapper around a compiled function. It will forward the tensor
    inputs to the compiled function and return the result.
    It also implements a serialization interface to support PyTorch's precompile
    with custom backend, so that we can save and load the compiled function on
    disk. There's no need to wrap around the compiled function if we don't want
    to serialize them in particular cases.
    Right now serialization for the custom backend is done via
    serializing the Dynamo fx graph plus example inputs.
    """

    def __init__(
        self,
        graph_module,
        example_inputs,
        model_tag,
        optimized_call,
        model_idx: int = 0,
        traced_files: list | None = None,
        compile_config: CompileConfig | None = None,
    ):
        assert isinstance(graph_module, torch.fx.GraphModule)
        self.graph_module = graph_module
        self.example_inputs = example_inputs
        self.model_idx = model_idx
        self.model_tag = model_tag
        self.traced_files = traced_files or []
        self.compile_config = compile_config
        self.optimized_call = optimized_call

    def __call__(self, *args, **kwargs):
        return self.optimized_call(*args, **kwargs)

    # ───────────────────────────────────────────────────────────────
    #  Serialization
    # ───────────────────────────────────────────────────────────────

    @classmethod
    def serialize_compile_artifacts(cls, compiled_fn: MagiSerializableFunction) -> bytes:
        from torch.fx._graph_pickler import GraphPickler, Options, _NodePickleData, _OpPickleData

        state = compiled_fn.__dict__.copy()
        state.pop("optimized_call")
        if state.get("compile_config") is not None:
            state["compile_config"] = state["compile_config"].model_dump(mode="json")

        # Build monkey-patches for GraphPickler
        patched_reducer = GraphPicklerPatchUtils.make_patch_for_reducer_override()
        patched_node_init, _stats = GraphNodePicklePatchUtils.make_patch_for_init()
        patched_op_pickle = GraphNodeOpPatchUtils.make_patch_for_pickle()

        # Pickle under all patches
        state["example_inputs"] = tree_map_only(torch.Tensor, lambda _: None, state["example_inputs"])
        with (
            patch.object(GraphPickler, "reducer_override", patched_reducer),
            patch.object(_NodePickleData, "__init__", patched_node_init),
            patch.object(_OpPickleData, "pickle", patched_op_pickle),
        ):
            state["graph_module"] = GraphPickler.dumps(state["graph_module"], Options(ops_filter=None))
            state["example_inputs"] = GraphPickler.dumps(state["example_inputs"])

        return pickle.dumps(state)

    # ───────────────────────────────────────────────────────────────
    #  Deserialization
    # ───────────────────────────────────────────────────────────────

    @classmethod
    def deserialize_compile_artifacts(cls, data: bytes) -> MagiSerializableFunction:
        """Deserialize compile artifacts and rebuild the backend eagerly.

        Unpickles the graph, inputs, and config, then calls
        :meth:`rebuild_backend` to recompile the MagiBackend so the
        returned ``MagiSerializableFunction`` is immediately callable.
        """
        from torch._subclasses import FakeTensorMode
        from torch.fx._graph_pickler import GraphPickler, _NodePickleData
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        from magi_compiler.config import CompileConfig

        magi_logger.info("AOT deserialize: unpickling compile artifacts (%d bytes)", len(data))

        state = pickle.loads(data)

        # Backward compat: pop triton_kernel_info from old serialized artifacts.
        state.pop("triton_kernel_info", None)

        fake_mode = FakeTensorMode(shape_env=ShapeEnv())

        # Unpickle graph & inputs under node-level patches
        patched_unpickle = GraphNodePicklePatchUtils.make_patch_for_unpickle()
        with patch.object(_NodePickleData, "unpickle", patched_unpickle):
            state["graph_module"] = GraphPickler.loads(state["graph_module"], fake_mode)
            state["example_inputs"] = GraphPickler.loads(state["example_inputs"], fake_mode)

        # Reconstruct CompileConfig from the serialized artifact (self-contained).
        compile_config_data = state.get("compile_config")
        if compile_config_data is not None:
            compile_config = CompileConfig(**compile_config_data)
            magi_logger.info("AOT deserialize: CompileConfig restored from artifact")
        else:
            from magi_compiler.config import get_compile_config

            magi_logger.warning("AOT deserialize: 'compile_config' not found in artifact, falling back to global config.")
            compile_config = get_compile_config()
        state["compile_config"] = compile_config

        fn = cls(**state, optimized_call=None)
        fn.rebuild_backend()
        return fn

    def rebuild_backend(self) -> None:
        """Recompile the MagiBackend from deserialized state.

        Reconstructs ``CompileContext`` and ``MagiBackend`` from this object's
        attributes, runs graph splitting and inductor compilation, and
        populates ``self.optimized_call``.
        """
        from torch._guards import TracingContext, detect_fake_mode, tracing

        from magi_compiler.magi_backend import MagiBackend
        from magi_compiler.utils import OrderedSet

        # Fill None placeholders in example_inputs with FakeTensors from graph metadata.
        placeholder_fake_values = [
            node.meta.get("example_value") for node in self.graph_module.graph.nodes if node.op == "placeholder"
        ]
        compile_inputs = [inp if inp is not None else placeholder_fake_values[i] for i, inp in enumerate(self.example_inputs)]

        fake_mode = detect_fake_mode(compile_inputs)
        magi_backend = MagiBackend(
            self.compile_config,
            model_idx=self.model_idx,
            model_tag=self.model_tag,
            traced_files=OrderedSet(self.traced_files),
            inductor_compile_config={},
        )

        with tracing(TracingContext(fake_mode)):
            msf = magi_backend(self.graph_module, compile_inputs)

        self.optimized_call = msf.optimized_call
        magi_logger.info("AOT deserialize: backend rebuilt for model_tag=%s", self.model_tag)

    @property
    def co_name(self):
        """
        Used for depyf debugging.
        """
        return "MagiSerializableFunction"


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────


def _import_by_qualname(module_name, qualname):
    """Import an object by its module path and dot-separated qualified name."""
    mod = importlib.import_module(module_name)
    obj = mod
    for attr in qualname.split("."):
        obj = getattr(obj, attr)
    return obj


class _OpImportablePickleData:
    """Fallback for ops unknown to ``_OpPickleData.pickle``.

    Stores ``module`` + ``qualname`` strings; ``unpickle`` restores via import.
    Quacks like ``_OpPickleData`` (has ``unpickle(self, unpickle_state)``).
    """

    def __init__(self, module_name, qualname):
        self.module_name = module_name
        self.qualname = qualname

    def unpickle(self, unpickle_state):
        return _import_by_qualname(self.module_name, self.qualname)


def _deep_map_nodes(data, node_type, fn):
    """Recursively map instances of *node_type* through *fn* in nested structures.

    Unlike ``tree_map_only``, this descends into ``slice`` objects as well as
    tuples, lists, and dicts — fixing the bug where ``tree_map_only`` treats
    ``slice`` as an opaque leaf and misses ``slice(None, Node(s0), None)``.
    """
    if isinstance(data, node_type):
        return fn(data)
    if isinstance(data, tuple):
        return tuple(_deep_map_nodes(x, node_type, fn) for x in data)
    if isinstance(data, list):
        return [_deep_map_nodes(x, node_type, fn) for x in data]
    if isinstance(data, dict):
        return {k: _deep_map_nodes(v, node_type, fn) for k, v in data.items()}
    if isinstance(data, slice):
        return slice(
            _deep_map_nodes(data.start, node_type, fn),
            _deep_map_nodes(data.stop, node_type, fn),
            _deep_map_nodes(data.step, node_type, fn),
        )
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Patch A: GraphPickler.reducer_override  (fully replaced)
#
# Self-contained replacement that includes the standard GraphPickler
# type dispatching (FakeTensor, GraphModule, OperatorBase, SymInt, etc.)
# plus Magi-specific handlers (FakeTensorMode, sympy.Function).
# ─────────────────────────────────────────────────────────────────────────────


class GraphPicklerPatchUtils:
    """Fully replaces ``GraphPickler.reducer_override``.

    Combines:
    * **Standard handlers** from ``torch.fx._graph_pickler`` — FakeTensor,
      GraphModule, OperatorBase, ShapeEnv, SymInt, TracingContext, numpy.
    * **Magi extensions** — FakeTensorMode (→ session token), sympy.Function
      subclasses with ``_torch_unpickler``.
    """

    @staticmethod
    def _restore_fake_mode(unpickle_state):
        """Reducer target: called during deserialization to return session's FakeTensorMode."""
        return unpickle_state.fake_mode

    @classmethod
    def make_patch_for_reducer_override(cls):
        """Return a self-contained ``reducer_override`` (no original needed)."""
        import sympy
        from torch._subclasses import FakeTensorMode
        from torch._subclasses.fake_tensor import FakeTensor
        from torch.fx._graph_pickler import (
            _GraphModulePickleData,
            _OpPickleData,
            _ShapeEnvPickleData,
            _SymNodePickleData,
            _TensorPickleData,
            _TorchNumpyPickleData,
            _TracingContextPickleData,
        )
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        def reducer_override(self, obj):
            # ── Magi-specific extensions ──
            if inspect.isclass(obj) and issubclass(obj, sympy.Function) and hasattr(obj, "_torch_unpickler"):
                return obj._torch_unpickler, (obj._torch_handler_name,)
            if isinstance(obj, FakeTensorMode):
                return cls._restore_fake_mode, (self._unpickle_state,)

            # ── Standard GraphPickler handlers ──
            if isinstance(obj, FakeTensor):
                return _TensorPickleData.reduce_helper(self, obj)
            if isinstance(obj, torch.fx.GraphModule):
                return _GraphModulePickleData.reduce_helper(self, obj)
            if isinstance(obj, (torch._ops.OperatorBase, torch._ops.OpOverloadPacket)):
                return _OpPickleData.reduce_helper(self, obj)
            if isinstance(obj, ShapeEnv):
                return _ShapeEnvPickleData.reduce_helper(self, obj)
            if isinstance(obj, torch.SymInt):
                return _SymNodePickleData.reduce_helper(self, obj)
            if isinstance(obj, torch._guards.TracingContext):
                return _TracingContextPickleData.reduce_helper(self, obj)

            assert not isinstance(obj, torch.fx.Node), f"Leaked Node in pickle stream: {obj}"

            if reduce := _TorchNumpyPickleData.reduce_helper(self, obj):
                return reduce

            return NotImplemented

        return reducer_override


# ─────────────────────────────────────────────────────────────────────────────
# Patch B: _NodePickleData.__init__ / unpickle  (fully replaced)
#
# This patch *replaces* (not wraps) both __init__ and unpickle to fix
# three categories of issues:
#
#   1. Slice-embedded Nodes — tree_map_only treats slice as opaque leaf.
#      We use _deep_map_nodes instead.
#
#   2. Un-picklable meta — source_fn_stack, nn_module_stack, etc.
#      We strip meta to a whitelist.
#
#   3. Targets — stored via _OpPickleData.pickle() (standard path).
#      For unknown ops, our Patch C on _OpPickleData.pickle catches
#      NotImplementedError and returns _OpImportablePickleData.
#
#   4. Triton kernel side-table — for Triton wrapper nodes, we extract
#      kernel (module, qualname) and constant_args from the live side
#      table during init, and re-import / re-register during unpickle.
#      This replaces the standalone TritonKernelPatchUtils class.
# ─────────────────────────────────────────────────────────────────────────────


class GraphNodePicklePatchUtils:
    """Patches ``_NodePickleData.__init__`` and ``unpickle``.

    The patched ``__init__`` fully replaces the original to:

    * use ``_deep_map_nodes`` (not ``tree_map_only``) so Node refs inside
      ``slice`` objects are properly mapped;
    * store target via ``_OpPickleData.pickle(node.target, options)``
      (standard path); unknown ops are handled by Patch C which falls
      back to ``_OpImportablePickleData``;
    * strip ``node.meta`` to ``_META_WHITELIST``;
    * extract Triton kernel info from the live side table.

    The patched ``unpickle`` fully replaces the original to:

    * use ``_deep_map_nodes`` to resolve ``_NodePickleData`` refs in slices;
    * call ``self.target.unpickle(unpickle_state)`` to resolve the target;
    * re-import and re-register Triton kernels, remapping indices.
    """

    # Only ``example_value`` (FakeTensor / SymInt) is needed for AOT Inductor
    # re-compilation.  Everything else is either debug-only or un-picklable.
    _META_WHITELIST = frozenset({"example_value"})

    _TRITON_WRAPPER_TARGETS = frozenset({"triton_kernel_wrapper_mutation", "triton_kernel_wrapper_functional"})

    @classmethod
    def _is_triton_node(cls, node):
        """Check if a FX graph node calls a Triton kernel wrapper higher-order op."""
        if node.op != "call_function":
            return False
        target_name = getattr(node.target, "__name__", "") or getattr(node.target, "name", lambda: "")()
        return target_name in cls._TRITON_WRAPPER_TARGETS

    @classmethod
    def make_patch_for_init(cls):
        """Return ``(patched_init, stats_dict)`` for ``_NodePickleData.__init__``.

        The patched init **fully replaces** the original (does NOT call it).
        ``stats_dict`` accumulates meta-stripping statistics across all nodes;
        call ``log_stats(stats_dict)`` after serialization to emit a summary.
        """
        from torch.fx._graph_pickler import _OpPickleData

        stats = {"total": 0, "stripped": 0, "dropped_keys": {}}

        def patched_init(self_npd, node, mapping, options):
            # Map ALL Node refs including inside slices (fixes tree_map_only bug)
            self_npd.args = _deep_map_nodes(node.args, torch.fx.Node, lambda n: mapping[n])
            self_npd.kwargs = _deep_map_nodes(node.kwargs, torch.fx.Node, lambda n: mapping[n])

            self_npd.name = node.name
            self_npd.op = node.op
            self_npd.type = node.type

            # Standard path: convert target via _OpPickleData.pickle().
            # Unknown ops (einops etc.) are caught by our Patch C fallback.
            self_npd.target = _OpPickleData.pickle(node.target, options)

            # Strip meta to whitelist
            stats["total"] += 1
            before = set(node.meta)
            self_npd.meta = {k: v for k, v in node.meta.items() if k in cls._META_WHITELIST}
            dropped = before - set(self_npd.meta)
            if dropped:
                stats["stripped"] += 1
                for k in dropped:
                    stats["dropped_keys"][k] = stats["dropped_keys"].get(k, 0) + 1

            # Extract Triton kernel info from live side table
            if cls._is_triton_node(node):
                cls._extract_triton_info(self_npd, node)

        return patched_init, stats

    @classmethod
    def _extract_triton_info(cls, npd, node):
        """Read kernel module/qualname and constant_args from the global side table."""
        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

        k_idx = node.kwargs.get("kernel_idx")
        ca_idx = node.kwargs.get("constant_args_idx")

        if k_idx is not None:
            try:
                kernel = kernel_side_table.get_kernel(k_idx)
                mod_name = getattr(kernel, "__module__", None)
                qual_name = getattr(kernel, "__qualname__", None) or getattr(kernel, "__name__", None)
                if mod_name and qual_name:
                    npd._triton_kernel_info = {"module": mod_name, "qualname": qual_name}
                    magi_logger.info("AOT serialize: Triton kernel idx=%d → %s.%s", k_idx, mod_name, qual_name)
                else:
                    magi_logger.warning("Triton kernel idx=%d not importable (module=%s, name=%s)", k_idx, mod_name, qual_name)
            except (AssertionError, KeyError):
                magi_logger.warning("Triton kernel idx=%d not found in side table", k_idx)

        if ca_idx is not None:
            try:
                npd._triton_constant_args = kernel_side_table.get_constant_args(ca_idx)
            except (AssertionError, KeyError):
                magi_logger.warning("Triton constant_args idx=%d not found in side table", ca_idx)

    @classmethod
    def make_patch_for_unpickle(cls):
        """Return a patched ``unpickle`` that **fully replaces** the original.

        The patched unpickle:
        * deep-maps ``_NodePickleData`` refs inside slices;
        * calls ``self.target.unpickle(unpickle_state)`` to resolve the target;
        * re-imports Triton kernels and remaps side-table indices.
        """
        from torch.fx._graph_pickler import _NodePickleData

        # Caches shared across all nodes in one deserialization session.
        kernel_remap = {}  # (module, qualname) → new_kernel_idx
        ca_remap = {}  # frozenset(items) → new_constant_args_idx

        def patched_unpickle(self_npd, graph, mapping, unpickle_state):
            # Deep-map _NodePickleData refs (including inside slices)
            args = _deep_map_nodes(self_npd.args, _NodePickleData, lambda n: mapping[n])
            kwargs = _deep_map_nodes(self_npd.kwargs, _NodePickleData, lambda n: mapping[n])

            # Resolve target via standard _OpPickleData.unpickle path
            target = self_npd.target.unpickle(unpickle_state)

            # Restore Triton kernel side-table entries
            if hasattr(self_npd, "_triton_kernel_info"):
                kwargs = cls._restore_triton_info(self_npd, kwargs, kernel_remap, ca_remap)

            assert callable(target) or isinstance(target, str)
            node = graph.create_node(self_npd.op, target, args, kwargs, self_npd.name, self_npd.type)
            node.meta = self_npd.meta
            return node

        return patched_unpickle

    @classmethod
    def _restore_triton_info(cls, npd, kwargs, kernel_remap, ca_remap):
        """Re-import Triton kernel, re-register in side table, remap indices."""
        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

        info = npd._triton_kernel_info
        key = (info["module"], info["qualname"])

        if key not in kernel_remap:
            fqn = f"{info['module']}.{info['qualname']}"
            try:
                kernel_obj = _import_by_qualname(info["module"], info["qualname"])
            except (ImportError, AttributeError) as e:
                raise RuntimeError(
                    f"Cannot restore Triton kernel {fqn}: {e}. " f"Make sure the module is installed and importable."
                ) from e
            kernel_remap[key] = kernel_side_table.add_kernel(kernel_obj)
            magi_logger.info("AOT deserialize: Triton kernel %s → new_idx=%d", fqn, kernel_remap[key])

        kwargs = dict(kwargs)
        kwargs["kernel_idx"] = kernel_remap[key]

        if hasattr(npd, "_triton_constant_args"):
            ca_key = frozenset(npd._triton_constant_args.items())
            if ca_key not in ca_remap:
                ca_remap[ca_key] = kernel_side_table.add_constant_args(npd._triton_constant_args)
            kwargs["constant_args_idx"] = ca_remap[ca_key]

        return kwargs

    @staticmethod
    def log_stats(stats):
        """Log a summary of meta keys dropped during serialization."""
        dropped_summary = ", ".join(f"{k}({v})" for k, v in sorted(stats["dropped_keys"].items(), key=lambda x: -x[1]))
        magi_logger.info(
            "AOT serialize: meta stripped on %d/%d nodes.  Dropped: [%s]", stats["stripped"], stats["total"], dropped_summary
        )


# ─────────────────────────────────────────────────────────────────────────────
# Patch C: _OpPickleData.pickle
#
# Safety net for unknown op types (third-party functions like einops.rearrange,
# unknown OperatorBase subclasses, etc.) that _OpPickleData.pickle cannot
# handle.  When pickle() raises NotImplementedError, we return an
# _OpImportablePickleData that stores module+qualname and restores via import.
# ─────────────────────────────────────────────────────────────────────────────


class GraphNodeOpPatchUtils:
    """Patches ``_OpPickleData.pickle`` to catch ``NotImplementedError``
    for unknown op types and fall back to ``_OpImportablePickleData``.
    """

    @classmethod
    def make_patch_for_pickle(cls):
        """Return a patched ``pickle`` classmethod."""
        from torch.fx._graph_pickler import _OpPickleData

        original_pickle = _OpPickleData.pickle.__func__

        @classmethod
        def patched_pickle(klass, op, options):
            try:
                return original_pickle(klass, op, options)
            except NotImplementedError:
                mod = getattr(op, "__module__", "")
                qualname = getattr(op, "__qualname__", "") or getattr(op, "__name__", "")
                fqn = f"{mod}.{qualname}"
                magi_logger.info("AOT serialize: unknown op %s → importlib fallback", fqn)
                return _OpImportablePickleData(mod, qualname)

        return patched_pickle
