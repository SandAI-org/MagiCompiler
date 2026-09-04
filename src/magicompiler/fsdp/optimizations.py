"""
Graph Optimization Passes for MagiCompiler.

Optimization passes that operate on the unified ``FXGraph`` (computation +
FSDP communication collectives) to produce more efficient execution plans.

Available optimizations:
    - **Operator Fusion**: Fuse consecutive communication collectives
      and adjacent compute-communication patterns.
    - **Dead Code Elimination (DCE)**: Remove unused nodes.
    - **Compute-Communication Overlap**: Schedule all-gather to run
      concurrently with preceding computation using CUDA streams.
    - **AutoRecompute**: Trade compute for memory by recomputing
      activations during backward instead of storing them.
    - **Deterministic Alignment**: Ensure bitwise reproducibility.
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import List, Optional, Set

from magicompiler.core.graph import FXGraph, Node, NodeKind

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """Level of graph optimization aggressiveness."""

    LEVEL_0 = auto()  # No optimization (pass-through)
    LEVEL_1 = auto()  # Basic: DCE, simple fusion
    LEVEL_2 = auto()  # Aggressive: full fusion, overlap, recompute


class GraphOptimizer:
    """Applies optimization passes to a unified FXGraph.

    Each optimization pass is a method that transforms the graph in-place
    or produces a new optimized graph.
    """

    def __init__(self, level: OptimizationLevel = OptimizationLevel.LEVEL_1):
        self.level = level
        self._passes_run: List[str] = []

    def optimize(
        self,
        graph: FXGraph,
        fuse_comm_computation: bool = False,
        auto_recompute: bool = False,
        deterministic: bool = False,
        is_inference: bool = True,
    ) -> FXGraph:
        """Run the optimization pipeline on the given graph.

        The pipeline applies passes in order of increasing aggressiveness,
        based on the configured ``OptimizationLevel``.

        Args:
            graph: The input FXGraph to optimize.
            fuse_comm_computation: Whether to fuse/overlap communication
                and computation.
            auto_recompute: Whether to insert recomputation nodes.
            deterministic: Whether to enforce deterministic execution.
            is_inference: If True, only inference-safe passes are applied.

        Returns:
            The optimized FXGraph.
        """
        if self.level == OptimizationLevel.LEVEL_0:
            logger.info("Optimization level 0: no passes applied.")
            return graph

        passes = []
        passes.append(("dead_code_elimination", self._dce))

        if self.level.value >= OptimizationLevel.LEVEL_1.value:
            passes.append(("comm_fusion", self._fuse_communication))

        if self.level.value >= OptimizationLevel.LEVEL_2.value:
            passes.append(("compute_comm_fusion", self._fuse_compute_comm))

            if not is_inference and auto_recompute:
                passes.append(("auto_recompute", self._auto_recompute))

            if fuse_comm_computation:
                passes.append(
                    ("comm_computation_overlap", self._overlap_comm_compute)
                )

        if deterministic:
            passes.append(("deterministic_align", self._align_deterministic))

        # Run passes sequentially
        current = graph
        for name, pass_fn in passes:
            try:
                current = pass_fn(current)
                self._passes_run.append(name)
                logger.debug(f"Pass '{name}' completed.")
            except Exception as e:
                logger.warning(f"Pass '{name}' failed: {e}. Skipping.")

        logger.info(
            f"Optimization complete. Applied passes: {', '.join(self._passes_run)}"
        )
        return current

    # ── Individual optimization passes ───────────────────────────────

    def _dce(self, graph: FXGraph) -> FXGraph:
        """Dead Code Elimination: Remove nodes that do not contribute
        to any output or communication node.

        A node is live if it is:
            - An output node
            - A communication node (all-gather, reduce-scatter)
            - A parameter node
            - An input node
            - On a path from input to output
        """
        live: Set[str] = set()

        # Mark output nodes and their transitive predecessors
        def mark_predecessors(nid: str) -> None:
            if nid in live:
                return
            live.add(nid)
            for pred in graph.predecessors(nid):
                mark_predecessors(pred)

        for out_id in graph.output_nodes:
            mark_predecessors(out_id)

        # Always keep comm nodes and input nodes
        for nid, node in graph.nodes.items():
            if node.kind in (
                NodeKind.INPUT,
                NodeKind.ALL_GATHER,
                NodeKind.REDUCE_SCATTER,
                NodeKind.ALL_REDUCE,
                NodeKind.PARAMETER,
            ):
                mark_predecessors(nid)

        # Build a new graph with only live nodes
        new_graph = FXGraph()
        for nid, node in graph.nodes.items():
            if nid in live:
                new_graph.add_node(node)
        for edge in graph.edges:
            if edge.src_id in live and edge.dst_id in live:
                new_graph.add_edge(edge)

        removed = graph.num_nodes - new_graph.num_nodes
        if removed > 0:
            logger.info(f"DCE: removed {removed} dead nodes.")

        return new_graph

    def _fuse_communication(self, graph: FXGraph) -> FXGraph:
        """Fuse consecutive communication collectives of the same kind.

        For example, consecutive all-gather operations on the same process
        group can be combined into a single larger all-gather.
        """
        new_graph = FXGraph()

        # Copy all nodes (for now, simple identity)
        for nid, node in graph.nodes.items():
            new_graph.add_node(node)
        for edge in graph.edges:
            new_graph.add_edge(edge)

        # Fuse consecutive same-kind comm nodes
        topo = new_graph.topo_sort()
        to_remove: Set[str] = set()

        for i in range(len(topo) - 1):
            curr_id = topo[i]
            next_id = topo[i + 1]
            curr_node = new_graph.get_node(curr_id)
            next_node = new_graph.get_node(next_id)

            if curr_node is None or next_node is None:
                continue

            # Check if both are same-kind communication nodes
            if curr_node.kind == next_node.kind and curr_node.kind in (
                NodeKind.ALL_GATHER,
                NodeKind.REDUCE_SCATTER,
            ):
                # Only fuse if they share the same shard group
                if curr_node.shard_group == next_node.shard_group:
                    # Mark the second node for removal and rewire
                    to_remove.add(next_id)
                    # Rewire: all successors of next_id now come from curr_id
                    for succ_id in new_graph.successors(next_id):
                        new_graph.add_edge_by_id(curr_id, succ_id)
                    logger.debug(
                        f"Fused {next_node.name} into {curr_node.name}"
                    )

        # Remove fused nodes
        if to_remove:
            final_graph = FXGraph()
            for nid, node in new_graph.nodes.items():
                if nid not in to_remove:
                    final_graph.add_node(node)
            for edge in new_graph.edges:
                if edge.dst_id not in to_remove and edge.src_id not in to_remove:
                    final_graph.add_edge(edge)

            logger.info(
                f"Comm Fusion: fused {len(to_remove)} communication nodes."
            )
            return final_graph

        return new_graph

    def _fuse_compute_comm(self, graph: FXGraph) -> FXGraph:
        """Fuse adjacent compute and communication nodes into composite
        operations.

        For instance, a ``torch.matmul`` followed by an all-gather can be
        fused into a fused kernel that performs both in one pass.
        """
        # This is a placeholder for the actual fusion strategy.
        # In v1.2.0, this identifies fusion opportunities and marks them
        # in node metadata for the code generation backend.
        for nid, node in graph.nodes.items():
            if node.kind == NodeKind.COMPUTE:
                successors = graph.successors(nid)
                for succ_id in successors:
                    succ = graph.get_node(succ_id)
                    if succ and succ.kind in (
                        NodeKind.ALL_GATHER,
                        NodeKind.REDUCE_SCATTER,
                    ):
                        node.meta["fuse_with"] = succ_id
                        succ.meta["fused"] = True
                        logger.debug(
                            f"Fusion opportunity: {node.name} -> {succ.name}"
                        )

        return graph

    def _overlap_comm_compute(self, graph: FXGraph) -> FXGraph:
        """Mark communication nodes for overlap with computation.

        Communication nodes (all-gather) are scheduled to run on a separate
        CUDA stream, overlapping with preceding computation. In the graph,
        this is represented by annotating nodes with stream assignments.
        """
        topo = graph.topo_sort()
        for nid in topo:
            node = graph.get_node(nid)
            if node is None:
                continue
            if node.kind == NodeKind.ALL_GATHER:
                # Schedule all-gather on a separate stream
                node.meta["stream"] = "cuda:overlap"
                # The all-gather can overlap with the preceding compute
                predecessors = graph.predecessors(nid)
                for pred_id in predecessors:
                    pred = graph.get_node(pred_id)
                    if pred and pred.kind == NodeKind.COMPUTE:
                        pred.meta["overlap_with"] = nid
                        logger.debug(
                            f"Overlap: {pred.name} || {node.name}"
                        )

        overlap_count = sum(
            1 for n in graph.nodes.values()
            if n.meta.get('stream') == 'cuda:overlap'
        )
        logger.info(
            f"Overlap: scheduled {overlap_count} "
            f"communication nodes for overlap."
        )
        return graph

    def _auto_recompute(self, graph: FXGraph) -> FXGraph:
        """Insert recomputation nodes to trade compute for memory.

        During training, certain activations are freed after forward and
        recomputed during backward, reducing peak VRAM usage. This pass
        identifies which nodes' outputs can be recomputed (instead of stored).
        """
        topo = graph.topo_sort()

        for nid in topo:
            node = graph.get_node(nid)
            if node is None:
                continue

            # Mark compute nodes with large outputs for recomputation
            if node.kind == NodeKind.COMPUTE and node.output_shape is not None:
                # Simple heuristic: recompute if output has > 1M elements
                numel = 1
                for dim in node.output_shape:
                    numel *= dim
                if numel > 1_000_000:  # > 1M elements
                    node.meta["recompute"] = True
                    node.meta["recompute_priority"] = "high" if numel > 10_000_000 else "medium"
                    logger.debug(
                        f"AutoRecompute: {node.name} "
                        f"(shape={node.output_shape}, elements={numel})"
                    )

        recompute_count = sum(
            1 for n in graph.nodes.values()
            if n.meta.get("recompute")
        )
        logger.info(
            f"AutoRecompute: marked {recompute_count} nodes for recomputation."
        )
        return graph

    def _align_deterministic(self, graph: FXGraph) -> FXGraph:
        """Align operations for deterministic/reproducible execution.

        Marks all nodes with a deterministic execution flag. This ensures
        that the compiled graph adheres to PyTorch 2.12's deterministic
        settings (``torch.use_deterministic_algorithms(True)``).
        """
        for nid, node in graph.nodes.items():
            node.meta["deterministic"] = True
            # Collectives must use deterministic algorithm
            if node.kind in (
                NodeKind.ALL_GATHER,
                NodeKind.REDUCE_SCATTER,
                NodeKind.ALL_REDUCE,
            ):
                node.meta["deterministic_algo"] = "non_default"
                logger.debug(f"Deterministic: {node.name}")

        logger.info(
            "Deterministic alignment: all nodes marked for reproducibility."
        )
        return graph
