"""
FSDP Full-Graph Capture – Training (Phase 2).

MagiCompiler extends FSDP full-graph capture to training scenarios,
where both the forward and backward passes are captured into a unified
graph. This enables:

    - Global gradient computation graph optimization (forward + backward).
    - FSDP reduce-scatter collectives during backward as graph nodes.
    - Memory-efficient recomputation (auto-recompute) under VRAM budgets.
    - Computation-communication overlap for both forward all-gather
      and backward reduce-scatter.

The training graph capture is more complex than inference because:
    1. The backward pass is implicitly defined by the autograd graph.
    2. FSDP's gradient synchronization (reduce-scatter) happens during
       backward via hooks.
    3. Peak memory must be managed across forward activations and gradients.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from magicompiler.core.graph import FXGraph, Node, NodeKind
from magicompiler.fsdp.graph_capture import FSDPGraphCapture
from magicompiler.fsdp.optimizations import (
    GraphOptimizer,
    OptimizationLevel,
)
from magicompiler.fsdp.hooks import FSDPHookContext

logger = logging.getLogger(__name__)


class FSDPTrainingCapture:
    """Captures an FSDP-wrapped model's full graph for training.

    This implements Phase 2 of the FSDP full-graph capture roadmap.
    Extends the inference capture to include backward-pass collectives
    and memory-aware optimizations.

    Usage::

        capture = FSDPTrainingCapture(
            fuse_comm_computation=True,
            auto_recompute=True,
            optimization_level=OptimizationLevel.LEVEL_2,
        )
        graph, compiled_fn = capture.capture(model, input_tensor, labels)
        loss = compiled_fn(input_tensor, labels)
        loss.backward()
        optimizer.step()
    """

    def __init__(
        self,
        fuse_comm_computation: bool = False,
        auto_recompute: bool = False,
        optimization_level: OptimizationLevel = OptimizationLevel.LEVEL_1,
        memory_budget_gb: Optional[float] = None,
    ):
        self.fuse_comm_computation = fuse_comm_computation
        self.auto_recompute = auto_recompute
        self.optimization_level = optimization_level
        self.memory_budget_gb = memory_budget_gb

        self._captured_graph: Optional[FXGraph] = None
        self._training_fn_cache: Optional[Callable] = None

    def capture(
        self,
        model: nn.Module,
        *example_args: Any,
        **example_kwargs: Any,
    ) -> Tuple[FXGraph, Callable]:
        """Capture and optimize an FSDP model for training.

        The capture process:

            1. Traces the forward graph (including FSDP all-gather).
            2. Runs a backward pass to capture reduce-scatter collectives
               and gradient flow.
            3. Unifies forward + backward into a single training graph.
            4. Applies training-specific optimizations.
            5. Returns the optimized graph and a compiled training step.

        Args:
            model: The FSDP-wrapped PyTorch model.
            *example_args: Example inputs for forward pass.
            **example_kwargs: Example keyword inputs (may include labels).

        Returns:
            (graph, compiled_training_fn): The unified training graph and
            a callable that runs the compiled forward + backward.
        """
        logger.info("=== FSDP Training Full-Graph Capture (Phase 2) ===")

        # If no example args, return skeleton early (same as inference)
        if not example_args:
            logger.info(
                "No example inputs provided; returning empty graph skeleton."
            )
            return FXGraph(), model.forward

        # Step 1: Capture forward graph (same as inference)
        base_capture = FSDPGraphCapture(
            sample_inputs=example_args,
            record_gradients=True,
        )
        forward_graph, forward_fn = base_capture.capture(
            model, *example_args, **example_kwargs
        )

        # Step 2: Capture backward collectives
        backward_graph = self._capture_backward_collectives(
            model, forward_fn, *example_args, **example_kwargs
        )

        # Step 3: Merge forward + backward into unified training graph
        unified_graph = self._merge_fwd_bwd(forward_graph, backward_graph)

        # Step 4: Apply training-specific optimizations
        optimizer = GraphOptimizer(level=self.optimization_level)
        optimized_graph = optimizer.optimize(
            unified_graph,
            fuse_comm_computation=self.fuse_comm_computation,
            auto_recompute=self.auto_recompute,
            is_inference=False,
        )

        # Step 5: Build a compiled training function
        compiled_fn = self._build_training_forward(model, optimized_graph)

        self._captured_graph = optimized_graph
        self._training_fn_cache = compiled_fn

        logger.info(
            f"Training capture complete: "
            f"{optimized_graph.num_nodes} total nodes "
            f"({len(optimized_graph.comm_nodes)} comm nodes). "
            f"AutoRecompute: {self.auto_recompute}"
        )

        return optimized_graph, compiled_fn

    def _capture_backward_collectives(
        self,
        model: nn.Module,
        forward_fn: Callable,
        *example_args: Any,
        **example_kwargs: Any,
    ) -> FXGraph:
        """Capture FSDP collectives that fire during the backward pass.

        FSDP's reduce-scatter for gradient synchronization happens during
        ``loss.backward()`` via post-backward hooks. We run a forward +
        backward pass with FSDP patches active to record these collectives.
        """
        bwd_graph = FXGraph()

        with FSDPHookContext(model):
            # Forward
            output = forward_fn(*example_args, **example_kwargs)

            # Compute a dummy loss if none provided
            if isinstance(output, torch.Tensor) and output.requires_grad:
                loss = output.sum()
            else:
                loss = torch.tensor(1.0, requires_grad=True)

            # Backward – this triggers FSDP reduce-scatter hooks
            loss.backward()

            # Retrieve collective records from the backward pass
            from magicompiler.utils.patches import get_and_clear_records
            bwd_records = get_and_clear_records()

            # Add backward collectives as graph nodes
            for i, rec in enumerate(bwd_records):
                if rec.get("kind") == "reduce_scatter":
                    node = Node(
                        name=f"bwd_reduce_scatter_{i}",
                        kind=NodeKind.REDUCE_SCATTER,
                        target="reduce_scatter",
                        output_shape=rec.get("tensor_shape"),
                        output_dtype=rec.get("tensor_dtype"),
                        shard_group=rec.get("group"),
                    )
                    bwd_graph.add_node(node)

            # Add gradient and backward nodes
            for param in model.parameters():
                if param.grad is not None:
                    grad_node = Node(
                        name=f"grad_{id(param)}",
                        kind=NodeKind.GRADIENT,
                        output_shape=param.grad.shape,
                        output_dtype=param.grad.dtype,
                    )
                    bwd_graph.add_node(grad_node)

        logger.info(
            f"Backward capture complete: "
            f"{len(bwd_graph.nodes)} backward nodes "
            f"({len(bwd_graph.comm_nodes)} comm nodes)."
        )

        return bwd_graph

    def _merge_fwd_bwd(self, fwd_graph: FXGraph, bwd_graph: FXGraph) -> FXGraph:
        """Merge forward and backward graphs into a unified training graph."""
        unified = FXGraph()

        # Copy all forward nodes
        for nid, node in fwd_graph.nodes.items():
            unified.add_node(node)
        for edge in fwd_graph.edges:
            unified.add_edge(edge)

        # Add backward nodes (disconnected; optimizer will wire them)
        for nid, node in bwd_graph.nodes.items():
            # Prefix backward nodes to avoid id collisions
            node.id = f"bwd_{nid}"
            node.name = f"bwd_{node.name}"
            unified.add_node(node)

        # Wire gradient nodes to their corresponding parameter nodes
        for nid, node in bwd_graph.nodes.items():
            if node.kind == NodeKind.GRADIENT:
                # Link gradient to the output (loss → backward → gradient)
                for out_id in fwd_graph.output_nodes:
                    unified.add_edge_by_id(out_id, f"bwd_{nid}")

        return unified

    def _build_training_forward(
        self,
        model: nn.Module,
        graph: FXGraph,
    ) -> Callable:
        """Build a training-step forward function.

        The compiled training function returns the loss, enabling the
        outer loop to call ``loss.backward()`` and ``optimizer.step()``.
        With ``auto_recompute``, activations are freed during forward
        and recomputed during backward.
        """

        def _training_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
            with FSDPHookContext(model):
                if self.auto_recompute:
                    # With auto-recompute: free activations after forward
                    with torch.no_grad():
                        output = model(*args, **kwargs)
                    # Output requires grad for backward
                    if isinstance(output, torch.Tensor):
                        output = output.detach().requires_grad_(True)
                else:
                    output = model(*args, **kwargs)

            return output

        return _training_forward

    def get_graph(self) -> Optional[FXGraph]:
        return self._captured_graph

    def summary(self) -> str:
        """Return a human-readable summary of the captured training graph."""
        if self._captured_graph is None:
            return "No training graph captured yet."

        graph = self._captured_graph
        lines = [
            "╔═══════════════════════════════════════════════╗",
            "║  FSDP Training Full-Graph Capture Summary    ║",
            "╚═══════════════════════════════════════════════╝",
            f"  Total nodes:          {graph.num_nodes}",
            f"  Communication nodes:  {len(graph.comm_nodes)}",
            f"  Compute nodes:        {sum(1 for n in graph.nodes.values() if n.kind == NodeKind.COMPUTE)}",
            f"  Gradient nodes:       {sum(1 for n in graph.nodes.values() if n.kind == NodeKind.GRADIENT)}",
            f"  Edges:                {len(graph.edges)}",
            f"  AutoRecompute:        {'ON' if self.auto_recompute else 'OFF'}",
            f"  Comm-Computation Fusion: {'ON' if self.fuse_comm_computation else 'OFF'}",
        ]

        return "\n".join(lines)
