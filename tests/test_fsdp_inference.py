"""Tests for FSDP Inference Full-Graph Capture (Phase 1)."""

import pytest
import torch
import torch.nn as nn

from magicompiler.fsdp.inference import FSDPInferenceCapture
from magicompiler.fsdp.optimizations import OptimizationLevel


class SimpleMLP(nn.Module):
    """A simple MLP for testing FSDP full-graph capture."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ComplexModel(nn.Module):
    """A more complex model with multiple submodules, residual-like structure."""

    def __init__(self, dim: int = 32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim * 2),
        )
        self.dec = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, 5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc(x)
        return self.dec(h)


class TestFSDPInferenceCaptureUnit:
    """Unit tests for FSDPInferenceCapture (no actual FSDP wrapping)."""

    def test_create_capture(self) -> None:
        capture = FSDPInferenceCapture()
        assert capture.fuse_comm_computation is False
        assert capture.optimization_level == OptimizationLevel.LEVEL_1

    def test_create_capture_with_fusion(self) -> None:
        capture = FSDPInferenceCapture(
            fuse_comm_computation=True,
            optimization_level=OptimizationLevel.LEVEL_2,
        )
        assert capture.fuse_comm_computation is True
        assert capture.optimization_level == OptimizationLevel.LEVEL_2

    def test_capture_no_example_args(self) -> None:
        """Should return a skeleton graph when no example args provided."""
        model = SimpleMLP()
        capture = FSDPInferenceCapture()
        graph, compiled_fn = capture.capture(model)
        assert graph.num_nodes == 0
        assert compiled_fn is not None

    def test_capture_with_sample(self) -> None:
        """Capture graph with sample inputs."""
        model = SimpleMLP(dim=16)
        x = torch.randn(2, 16)

        capture = FSDPInferenceCapture()
        graph, compiled_fn = capture.capture(model, x)

        # Should at least have input and output nodes
        assert graph.num_nodes >= 6, (
            f"Expected >= 6 nodes (input, 4 compute layers, output), "
            f"got {graph.num_nodes}"
        )
        assert len(graph.input_nodes) >= 1
        assert len(graph.output_nodes) >= 1
        assert compiled_fn is not None

    def test_capture_output_is_callable(self) -> None:
        """The compiled function should be callable."""
        model = SimpleMLP(dim=16)
        x = torch.randn(2, 16)

        capture = FSDPInferenceCapture()
        _, compiled_fn = capture.capture(model, x)

        output = compiled_fn(x)
        assert isinstance(output, torch.Tensor)
        assert output.shape == (2, 10)

    def test_capture_complex_model(self) -> None:
        """Capture a more complex model."""
        model = ComplexModel(dim=16)
        x = torch.randn(2, 16)

        capture = FSDPInferenceCapture(
            optimization_level=OptimizationLevel.LEVEL_1,
        )
        graph, compiled_fn = capture.capture(model, x)

        # Complex model has ~6+ compute layers (enc 4 + dec 3)
        assert graph.num_nodes >= 8
        output = compiled_fn(x)
        assert output.shape == (2, 5)

    def test_summary_before_capture(self) -> None:
        """Summary should indicate no capture before capture()."""
        capture = FSDPInferenceCapture()
        summary = capture.summary()
        assert "No graph captured yet" in summary

    def test_summary_after_capture(self) -> None:
        """Summary should show graph stats after capture."""
        model = SimpleMLP(dim=16)
        x = torch.randn(2, 16)

        capture = FSDPInferenceCapture()
        capture.capture(model, x)
        summary = capture.summary()
        assert "FSDP Inference" in summary
        assert "Total nodes" in summary

    def test_get_graph(self) -> None:
        """get_graph() should return the captured graph."""
        model = SimpleMLP(dim=16)
        x = torch.randn(2, 16)

        capture = FSDPInferenceCapture()
        capture.capture(model, x)
        graph = capture.get_graph()
        assert graph is not None
        assert graph.num_nodes > 0

    def test_deterministic_flag(self) -> None:
        """Test that the deterministic alignment pass runs cleanly.

        This exercises the ``deterministic=True`` path through
        ``GraphOptimizer._align_deterministic``, which marks every
        node with ``meta["deterministic"] = True``.
        """
        from magicompiler.fsdp.optimizations import GraphOptimizer, OptimizationLevel

        model = SimpleMLP(dim=16)
        x = torch.randn(2, 16)

        capture = FSDPInferenceCapture(
            fuse_comm_computation=False,
            optimization_level=OptimizationLevel.LEVEL_2,
        )
        graph, compiled_fn = capture.capture(model, x)

        # Apply the deterministic pass via the optimizer
        opt = GraphOptimizer(level=OptimizationLevel.LEVEL_1)
        det_graph = opt.optimize(graph, deterministic=True)

        # Every node should be marked deterministic
        for nid, node in det_graph.nodes.items():
            assert node.meta.get("deterministic") is True, (
                f"Node {nid} should have deterministic=True"
            )

        # Compiled function should still run
        output1 = compiled_fn(x)
        output2 = compiled_fn(x)
        assert torch.allclose(output1, output2)

    def test_no_grad_inference(self) -> None:
        """Inference should run under torch.no_grad()."""
        model = SimpleMLP(dim=16)
        x = torch.randn(2, 16)

        capture = FSDPInferenceCapture()
        _, compiled_fn = capture.capture(model, x)

        with torch.no_grad():
            output = compiled_fn(x)

        assert output.requires_grad is False, (
            "Inference output should not require gradients"
        )


class TestFSDPInferenceGraphStructure:
    """Test the structure of captured graphs."""

    def test_graph_has_valid_topo_order(self) -> None:
        """Topological order should respect edge dependencies."""
        model = SimpleMLP(dim=16)
        x = torch.randn(2, 16)

        capture = FSDPInferenceCapture()
        graph, _ = capture.capture(model, x)

        topo = graph.topo_sort()
        assert len(topo) == graph.num_nodes

        # Verify each predecessor comes before successor
        positions = {nid: i for i, nid in enumerate(topo)}
        for edge in graph.edges:
            assert positions[edge.src_id] < positions[edge.dst_id], (
                f"Edge {edge.src_id} -> {edge.dst_id} violates topological order"
            )

    def test_graph_is_dag(self) -> None:
        """Graph should be a DAG (no cycles)."""
        model = ComplexModel(dim=16)
        x = torch.randn(2, 16)

        capture = FSDPInferenceCapture()
        graph, _ = capture.capture(model, x)

        topo = graph.topo_sort()
        assert len(topo) == graph.num_nodes, (
            f"Graph has {graph.num_nodes} nodes but topological sort "
            f"only returned {len(topo)}. Cycle detected!"
        )

    def test_graph_nodes_have_unique_ids(self) -> None:
        """All graph nodes must have unique IDs."""
        model = SimpleMLP(dim=16)
        x = torch.randn(2, 16)

        capture = FSDPInferenceCapture()
        graph, _ = capture.capture(model, x)

        ids = list(graph.nodes.keys())
        assert len(ids) == len(set(ids)), "Duplicate node IDs detected!"

    def test_dot_export_simple(self) -> None:
        """Test DOT export is valid."""
        model = SimpleMLP(dim=16)
        x = torch.randn(2, 16)

        capture = FSDPInferenceCapture()
        graph, _ = capture.capture(model, x)

        dot = graph.to_dot()
        assert dot.startswith("digraph")
        assert dot.count("->") == len(graph.edges)
