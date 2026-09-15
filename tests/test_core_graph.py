"""Tests for the MagiCompiler core graph IR (FXGraph, Node, Edge)."""

import pytest
from magicompiler.core.graph import FXGraph, Node, NodeKind, Edge


class TestNode:
    """Node creation and classification."""

    def test_create_node_default(self) -> None:
        node = Node()
        assert node.kind == NodeKind.PLACEHOLDER
        assert node.id.startswith("n_")
        assert node.name == ""

    def test_create_compute_node(self) -> None:
        node = Node(
            name="my_op",
            kind=NodeKind.COMPUTE,
            target="torch.matmul",
            output_shape=(32, 64),
        )
        assert node.name == "my_op"
        assert node.kind == NodeKind.COMPUTE
        assert node.target == "torch.matmul"
        assert node.output_shape == (32, 64)

    def test_create_comm_node_all_gather(self) -> None:
        node = Node(
            name="ag_0",
            kind=NodeKind.ALL_GATHER,
            target="all_gather",
            shard_world_size=4,
        )
        assert node.kind == NodeKind.ALL_GATHER
        assert node.shard_world_size == 4

    def test_create_comm_node_reduce_scatter(self) -> None:
        node = Node(
            name="rs_0",
            kind=NodeKind.REDUCE_SCATTER,
            target="reduce_scatter",
            shard_world_size=4,
        )
        assert node.kind == NodeKind.REDUCE_SCATTER

    def test_node_repr(self) -> None:
        node = Node(name="test_op", kind=NodeKind.COMPUTE, target="matmul")
        assert "COMPUTE" in repr(node)
        assert "test_op" in repr(node)


class TestEdge:
    """Edge creation."""

    def test_create_edge(self) -> None:
        edge = Edge(src_id="n1", dst_id="n2")
        assert edge.src_id == "n1"
        assert edge.dst_id == "n2"

    def test_edge_with_indices(self) -> None:
        edge = Edge(src_id="n1", dst_id="n2", src_idx=1, dst_idx=2)
        assert edge.src_idx == 1
        assert edge.dst_idx == 2

    def test_edge_with_meta(self) -> None:
        edge = Edge(src_id="n1", dst_id="n2", meta={"stream": "cuda:0"})
        assert edge.meta["stream"] == "cuda:0"


class TestFXGraph:
    """FXGraph management and operations."""

    def test_empty_graph(self) -> None:
        graph = FXGraph()
        assert graph.num_nodes == 0
        assert len(graph.nodes) == 0
        assert len(graph.topo_sort()) == 0

    def test_add_nodes(self) -> None:
        graph = FXGraph()
        inp = Node(name="input", kind=NodeKind.INPUT)
        compute = Node(name="compute", kind=NodeKind.COMPUTE)
        out = Node(name="output", kind=NodeKind.OUTPUT)

        graph.add_node(inp)
        graph.add_node(compute)
        graph.add_node(out)

        assert graph.num_nodes == 3
        assert len(graph.input_nodes) == 1
        assert len(graph.output_nodes) == 1

    def test_add_comm_nodes_collected(self) -> None:
        graph = FXGraph()
        ag = Node(name="ag_0", kind=NodeKind.ALL_GATHER)
        rs = Node(name="rs_0", kind=NodeKind.REDUCE_SCATTER)
        compute = Node(name="compute", kind=NodeKind.COMPUTE)

        graph.add_node(ag)
        graph.add_node(rs)
        graph.add_node(compute)

        assert len(graph.comm_nodes) == 2
        assert graph.is_comm_node(ag.id)
        assert graph.is_comm_node(rs.id)
        assert not graph.is_comm_node(compute.id)

    def test_add_edges(self) -> None:
        graph = FXGraph()
        n1 = Node(name="a", kind=NodeKind.COMPUTE)
        n2 = Node(name="b", kind=NodeKind.COMPUTE)
        graph.add_node(n1)
        graph.add_node(n2)

        graph.add_edge(Edge(src_id=n1.id, dst_id=n2.id))
        assert len(graph.edges) == 1
        assert graph.successors(n1.id) == [n2.id]
        assert graph.predecessors(n2.id) == [n1.id]

    def test_add_edge_by_id(self) -> None:
        graph = FXGraph()
        n1 = Node(name="a", kind=NodeKind.COMPUTE)
        n2 = Node(name="b", kind=NodeKind.COMPUTE)
        graph.add_node(n1)
        graph.add_node(n2)

        graph.add_edge_by_id(n1.id, n2.id)
        assert len(graph.edges) == 1

    def test_topo_sort(self) -> None:
        graph = FXGraph()
        a = Node(name="a", kind=NodeKind.INPUT)
        b = Node(name="b", kind=NodeKind.COMPUTE)
        c = Node(name="c", kind=NodeKind.COMPUTE)
        d = Node(name="d", kind=NodeKind.OUTPUT)

        graph.add_node(a)
        graph.add_node(b)
        graph.add_node(c)
        graph.add_node(d)

        graph.add_edge_by_id(a.id, b.id)
        graph.add_edge_by_id(b.id, c.id)
        graph.add_edge_by_id(c.id, d.id)

        topo = graph.topo_sort()
        assert len(topo) == 4
        # a must come before b, b before c, c before d
        assert topo.index(a.id) < topo.index(b.id)
        assert topo.index(b.id) < topo.index(c.id)
        assert topo.index(c.id) < topo.index(d.id)

    def test_get_node(self) -> None:
        graph = FXGraph()
        n = Node(name="test", kind=NodeKind.COMPUTE)
        nid = graph.add_node(n)
        assert graph.get_node(nid) is n
        assert graph.get_node("nonexistent") is None

    def test_to_dot(self) -> None:
        graph = FXGraph()
        inp = Node(name="x", kind=NodeKind.INPUT)
        op = Node(name="matmul", kind=NodeKind.COMPUTE)
        ag = Node(name="ag_0", kind=NodeKind.ALL_GATHER)
        out = Node(name="out", kind=NodeKind.OUTPUT)

        graph.add_node(inp)
        graph.add_node(op)
        graph.add_node(ag)
        graph.add_node(out)

        graph.add_edge_by_id(inp.id, op.id)
        graph.add_edge_by_id(op.id, ag.id)
        graph.add_edge_by_id(ag.id, out.id)

        dot = graph.to_dot()
        assert "digraph" in dot
        assert "MagiCompilerGraph" in dot
        assert "ALL_GATHER" in dot
        assert "matmul" in dot

    def test_repr(self) -> None:
        graph = FXGraph()
        graph.add_node(Node(name="a", kind=NodeKind.COMPUTE))
        graph.add_node(Node(name="b", kind=NodeKind.ALL_GATHER))
        r = repr(graph)
        assert "FXGraph" in r
        assert "nodes=2" in r
        assert "comm_nodes=1" in r
