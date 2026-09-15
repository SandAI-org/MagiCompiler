"""Tests for MagiCompiler graph optimization passes."""

import pytest
from magicompiler.core.graph import FXGraph, Node, NodeKind
from magicompiler.fsdp.optimizations import (
    GraphOptimizer,
    OptimizationLevel,
)


def _simple_graph() -> FXGraph:
    """Create a simple graph: input -> compute -> all_gather -> compute -> output."""
    graph = FXGraph()

    inp = Node(name="input", kind=NodeKind.INPUT)
    c1 = Node(name="layer1", kind=NodeKind.COMPUTE, target="linear")
    ag = Node(name="ag_0", kind=NodeKind.ALL_GATHER,
              shard_group="g0")
    c2 = Node(name="layer2", kind=NodeKind.COMPUTE, target="relu")
    out = Node(name="output", kind=NodeKind.OUTPUT)

    graph.add_node(inp)
    graph.add_node(c1)
    graph.add_node(ag)
    graph.add_node(c2)
    graph.add_node(out)

    graph.add_edge_by_id(inp.id, c1.id)
    graph.add_edge_by_id(c1.id, ag.id)
    graph.add_edge_by_id(ag.id, c2.id)
    graph.add_edge_by_id(c2.id, out.id)

    return graph


def _graph_with_dead_nodes() -> FXGraph:
    """Graph with nodes that don't contribute to output."""
    graph = FXGraph()

    inp = Node(name="input", kind=NodeKind.INPUT)
    live = Node(name="live", kind=NodeKind.COMPUTE, target="linear")
    dead = Node(name="dead", kind=NodeKind.COMPUTE, target="dead_op")
    ag = Node(name="ag_0", kind=NodeKind.ALL_GATHER, shard_group="g0")
    out = Node(name="output", kind=NodeKind.OUTPUT)

    graph.add_node(inp)
    graph.add_node(live)
    graph.add_node(dead)
    graph.add_node(ag)
    graph.add_node(out)

    graph.add_edge_by_id(inp.id, live.id)
    graph.add_edge_by_id(live.id, ag.id)
    graph.add_edge_by_id(ag.id, out.id)
    graph.add_edge_by_id(inp.id, dead.id)  # dead node, no path to output

    return graph


class TestGraphOptimizer:
    """Tests for the GraphOptimizer class."""

    def test_level_0_no_optimization(self) -> None:
        """Level 0 should return the graph unchanged."""
        graph = _simple_graph()
        original_nodes = graph.num_nodes

        opt = GraphOptimizer(level=OptimizationLevel.LEVEL_0)
        optimized = opt.optimize(graph)

        assert optimized.num_nodes == original_nodes

    def test_level_1_pass_through(self) -> None:
        """Level 1 optimizations should run without errors."""
        graph = _simple_graph()
        opt = GraphOptimizer(level=OptimizationLevel.LEVEL_1)
        optimized = opt.optimize(graph)

        assert optimized.num_nodes > 0

    def test_level_2_pass_through(self) -> None:
        """Level 2 optimizations should run without errors."""
        graph = _simple_graph()
        opt = GraphOptimizer(level=OptimizationLevel.LEVEL_2)
        optimized = opt.optimize(
            graph,
            fuse_comm_computation=True,
            auto_recompute=False,
            is_inference=True,
        )

        assert optimized.num_nodes > 0

    def test_dead_code_elimination(self) -> None:
        """DCE should remove unreachable nodes."""
        graph = _graph_with_dead_nodes()
        original_nodes = graph.num_nodes

        opt = GraphOptimizer(level=OptimizationLevel.LEVEL_1)
        optimized = opt.optimize(graph)

        assert optimized.num_nodes < original_nodes, (
            f"DCE should remove dead nodes. "
            f"Before: {original_nodes}, After: {optimized.num_nodes}"
        )

        # The "dead" node should be removed
        dead_ids = [
            nid for nid, n in optimized.nodes.items()
            if n.name == "dead"
        ]
        assert len(dead_ids) == 0, "Dead node should have been removed"

    def test_comm_fusion_basic(self) -> None:
        """Test fusion of consecutive same-kind communication nodes."""
        graph = FXGraph()

        inp = Node(name="input", kind=NodeKind.INPUT)
        ag1 = Node(name="ag_0", kind=NodeKind.ALL_GATHER, shard_group="g0")
        ag2 = Node(name="ag_1", kind=NodeKind.ALL_GATHER, shard_group="g0")
        compute = Node(name="compute", kind=NodeKind.COMPUTE)
        out = Node(name="output", kind=NodeKind.OUTPUT)

        graph.add_node(inp)
        graph.add_node(ag1)
        graph.add_node(ag2)
        graph.add_node(compute)
        graph.add_node(out)

        graph.add_edge_by_id(inp.id, ag1.id)
        graph.add_edge_by_id(ag1.id, ag2.id)
        graph.add_edge_by_id(ag2.id, compute.id)
        graph.add_edge_by_id(compute.id, out.id)

        opt = GraphOptimizer(level=OptimizationLevel.LEVEL_1)
        optimized = opt.optimize(graph)

        assert optimized.num_nodes <= graph.num_nodes

    def test_compute_comm_overlap_marking(self) -> None:
        """Compute-communication overlap should mark nodes for async execution."""
        graph = _simple_graph()

        opt = GraphOptimizer(level=OptimizationLevel.LEVEL_2)
        optimized = opt.optimize(
            graph,
            fuse_comm_computation=True,
        )

        overlap_nodes = [
            n for n in optimized.nodes.values()
            if n.meta.get("stream") == "cuda:overlap"
        ]
        assert len(overlap_nodes) >= 1, (
            "At least one node should be marked for overlap"
        )

    def test_auto_recompute(self) -> None:
        """AutoRecompute should mark compute nodes for recomputation."""
        graph = FXGraph()

        inp = Node(name="input", kind=NodeKind.INPUT)
        c1 = Node(name="big_layer", kind=NodeKind.COMPUTE,
                  output_shape=(32, 1024, 1024))  # 32M elements (> 1M)
        c2 = Node(name="small_layer", kind=NodeKind.COMPUTE,
                  output_shape=(32, 128))  # 4K elements, won't be recomputed
        out = Node(name="output", kind=NodeKind.OUTPUT)

        graph.add_node(inp)
        graph.add_node(c1)
        graph.add_node(c2)
        graph.add_node(out)

        graph.add_edge_by_id(inp.id, c1.id)
        graph.add_edge_by_id(c1.id, c2.id)
        graph.add_edge_by_id(c2.id, out.id)

        opt = GraphOptimizer(level=OptimizationLevel.LEVEL_2)
        optimized = opt.optimize(
            graph,
            auto_recompute=True,
            is_inference=False,
        )

        recompute_nodes = [
            n for n in optimized.nodes.values()
            if n.meta.get("recompute")
        ]
        assert len(recompute_nodes) >= 1, (
            "At least one node should be marked for recomputation"
        )

    def test_deterministic_alignment(self) -> None:
        """Deterministic alignment should mark all nodes."""
        graph = _simple_graph()

        opt = GraphOptimizer(level=OptimizationLevel.LEVEL_1)
        optimized = opt.optimize(
            graph,
            deterministic=True,
        )

        for nid, node in optimized.nodes.items():
            assert node.meta.get("deterministic") is True, (
                f"Node {nid} should be deterministic"
            )

    def test_training_mode_optimizations(self) -> None:
        """Training mode should apply training-specific passes."""
        graph = _simple_graph()

        opt = GraphOptimizer(level=OptimizationLevel.LEVEL_2)
        optimized = opt.optimize(
            graph,
            fuse_comm_computation=True,
            auto_recompute=True,
            is_inference=False,
        )

        assert optimized.num_nodes > 0
        overlap = [
            n for n in optimized.nodes.values()
            if n.meta.get("stream") == "cuda:overlap"
        ]
        assert len(overlap) >= 1


class TestOptimizationLevel:
    """Tests for OptimizationLevel enum."""

    def test_level_ordering(self) -> None:
        assert OptimizationLevel.LEVEL_0.value < OptimizationLevel.LEVEL_1.value
        assert OptimizationLevel.LEVEL_1.value < OptimizationLevel.LEVEL_2.value
