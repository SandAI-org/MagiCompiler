"""
FSDP Graph Capture Engine.

Core tracing engine that captures both PyTorch tensor operations and
FSDP communication collectives into a single unified FXGraph. This is
the heart of MagiCompiler's native full-graph capture capability.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from magicompiler.core.graph import FXGraph, Node, NodeKind
from magicompiler.fsdp.hooks import FSDPHookContext
from magicompiler.utils.patches import get_and_clear_records

logger = logging.getLogger(__name__)


class FSDPGraphCapture:
    """Captures the full computation graph including FSDP collectives.

    The tracing process:
      1. Patches FSDP to expose communication collectives.
      2. Runs a forward pass with ``torch.no_grad()`` and
         ``torch._dynamo.export`` (or equivalent tracing) to capture
         the full FX graph.
      3. Merges the recorded FSDP collectives into the graph as
         first-class communication nodes.
      4. Returns the unified ``FXGraph`` for further optimization.
    """

    def __init__(
        self,
        sample_inputs: Optional[Tuple[Any, ...]] = None,
        record_gradients: bool = False,
    ):
        self.sample_inputs = sample_inputs
        self.record_gradients = record_gradients

    def capture(
        self,
        model: nn.Module,
        *example_args: Any,
        **example_kwargs: Any,
    ) -> Tuple[FXGraph, Callable]:
        """Trace the model and produce an FXGraph with FSDP awareness.

        Args:
            model: The PyTorch model (may be FSDP-wrapped).
            *example_args: Example inputs for tracing.
            **example_kwargs: Example keyword inputs for tracing.

        Returns:
            A tuple ``(fx_graph, compiled_forward)`` where ``fx_graph`` is the
            unified computation+communication graph and ``compiled_forward`` is
            the executable forward function.
        """
        fx_graph = FXGraph()
        compiled_fn: Callable = model.forward

        if not example_args:
            # If no example args provided, we return the graph skeleton
            # that can be populated later.
            logger.info("No example inputs provided; returning empty graph skeleton.")
            return fx_graph, compiled_fn

        # Phase 1: Run with FSDP patches active to capture collectives
        logger.info("Starting FSDP full-graph capture...")

        # Try to use torch._dynamo.export for FX graph capture
        try:
            captured_graph = self._trace_with_dynamo(model, example_args, example_kwargs)
        except Exception as e:
            logger.warning(
                f"torch._dynamo export failed ({e}); "
                "falling back to manual tracing."
            )
            captured_graph = self._trace_manual(model, example_args, example_kwargs)

        # Phase 2: Build FXGraph from captured representation
        fx_graph = self._build_graph(captured_graph, model, example_args)

        # Phase 3: Create compiled forward from captured graph
        compiled_fn = self._build_compiled_fn(model, fx_graph)

        logger.info(
            f"Full-graph capture complete: {fx_graph.num_nodes} nodes "
            f"({len(fx_graph.comm_nodes)} communication nodes), "
            f"{len(fx_graph.edges)} edges."
        )
        return fx_graph, compiled_fn

    def _trace_with_dynamo(
        self,
        model: nn.Module,
        args: tuple,
        kwargs: dict,
    ) -> dict:
        """Use ``torch._dynamo.export`` to capture the FX graph.

        Returns a dict with keys ``"graph_module"`` (the FX traced module)
        and ``"collectives"`` (the FSDP collective records captured during
        tracing).
        """
        with FSDPHookContext(model):
            with torch.no_grad():
                try:
                    import torch._dynamo as dynamo

                    graph_module, guards = dynamo.export(
                        model,
                        *args,
                        **kwargs,
                        aten_graph=True,
                    )
                    collectives = get_and_clear_records()
                    return {
                        "graph_module": graph_module,
                        "collectives": collectives,
                        "guards": guards,
                    }
                except (ImportError, Exception) as e:
                    logger.debug(f"dynamo.export failed: {e}")
                    raise

    def _trace_manual(
        self,
        model: nn.Module,
        args: tuple,
        kwargs: dict,
    ) -> dict:
        """Fallback: manually trace the forward pass and record collectives.

        Returns a dict with the same schema as ``_trace_with_dynamo``:
        ``{"graph_module": None, "collectives": [...], "output": Tensor}``.
        """
        with FSDPHookContext(model):
            with torch.no_grad():
                output = model(*args, **kwargs)
                collective_records = get_and_clear_records()

        return {
            "graph_module": None,
            "collectives": collective_records,
            "output": output,
            "args": args,
        }

    def _build_graph(
        self,
        captured: dict,
        model: nn.Module,
        args: tuple,
    ) -> FXGraph:
        """Construct an FXGraph from the captured trace dict.

        The ``captured`` dict always has the following keys:
        - ``"graph_module"``: ``torch.fx.GraphModule`` or ``None`` (manual trace)
        - ``"collectives"``: list of FSDP collective records
        """
        graph = FXGraph()

        # 1. Add input nodes
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                inp_node = Node(
                    name=f"input_{i}",
                    kind=NodeKind.INPUT,
                    output_shape=arg.shape,
                    output_dtype=arg.dtype,
                )
                graph.add_node(inp_node)

        # 2. Extract FX graph nodes (if captured via dynamo)
        graph_module = captured.get("graph_module")
        if graph_module is not None and hasattr(graph_module, "graph"):
            for fx_node in graph_module.graph.nodes:
                node_kind = self._classify_fx_node(fx_node)
                mc_node = Node(
                    name=fx_node.name,
                    kind=node_kind,
                    target=fx_node.target,
                    output_shape=getattr(fx_node, "meta", {}).get("tensor_meta", None),
                )
                graph.add_node(mc_node)

                # Wire inputs
                for fx_arg in fx_node.args:
                    if isinstance(fx_arg, torch.fx.Node):
                        graph.add_edge_by_id(fx_arg.name, fx_node.name)

        # 3. Add FSDP collective nodes from captured records
        collectives = captured.get("collectives", [])
        for i, rec in enumerate(collectives):
            kind_map = {
                "all_gather": NodeKind.ALL_GATHER,
                "reduce_scatter": NodeKind.REDUCE_SCATTER,
                "all_reduce": NodeKind.ALL_REDUCE,
            }
            node_kind = kind_map.get(rec.get("kind", ""), NodeKind.PLACEHOLDER)

            comm_node = Node(
                name=f"fsdp_{rec.get('kind', 'comm')}_{i}",
                kind=node_kind,
                target=rec.get("kind"),
                output_shape=rec.get("tensor_shape"),
                output_dtype=rec.get("tensor_dtype"),
                shard_group=rec.get("group"),
            )
            graph.add_node(comm_node)

        # 4. Add output node
        output = captured.get("output")
        if output is not None and isinstance(output, torch.Tensor):
            out_node = Node(
                name="output",
                kind=NodeKind.OUTPUT,
                output_shape=output.shape,
                output_dtype=output.dtype,
            )
            graph.add_node(out_node)

        return graph

    def _build_compiled_fn(
        self,
        model: nn.Module,
        graph: FXGraph,
    ) -> Callable:
        """Build a compiled forward function from the captured graph.

        For now, returns the original model forward with FSDP hooks patched.
        In future versions, this will generate an optimized TorchScript or
        Inductor-optimized graph.
        """

        def _compiled_forward(*args: Any, **kwargs: Any) -> Any:
            with FSDPHookContext(model):
                with torch.no_grad():
                    return model(*args, **kwargs)

        return _compiled_forward

    @staticmethod
    def _classify_fx_node(fx_node: Any) -> NodeKind:
        """Classify a ``torch.fx.Node`` into a ``NodeKind``."""
        if fx_node.op == "placeholder":
            return NodeKind.INPUT
        elif fx_node.op == "output":
            return NodeKind.OUTPUT
        elif fx_node.op == "call_function":
            target = str(fx_node.target)
            if "all_gather" in target:
                return NodeKind.ALL_GATHER
            elif "reduce_scatter" in target:
                return NodeKind.REDUCE_SCATTER
            elif "all_reduce" in target:
                return NodeKind.ALL_REDUCE
            elif "loss" in target.lower():
                return NodeKind.LOSS
            elif "backward" in target.lower():
                return NodeKind.BACKWARD
            return NodeKind.COMPUTE
        elif fx_node.op == "call_module":
            return NodeKind.COMPUTE
        elif fx_node.op == "get_attr":
            return NodeKind.PARAMETER
        return NodeKind.PLACEHOLDER
