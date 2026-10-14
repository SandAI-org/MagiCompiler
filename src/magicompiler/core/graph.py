"""
Core Graph Intermediate Representation for MagiCompiler.

Defines the unified computation graph that captures both PyTorch operators
and FSDP communication collectives (all-gather, reduce-scatter, all-reduce)
as first-class nodes, enabling holistic cross-boundary optimizations.
"""

from __future__ import annotations

import dataclasses
import uuid
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set


class NodeKind(Enum):
    """Categorizes each node in the FXGraph."""

    # Computation
    COMPUTE = auto()  # torch operator / module call
    PARAMETER = auto()  # parameter access / shard

    # FSDP Communication Collectives
    ALL_GATHER = auto()  # gather full parameters from shards
    REDUCE_SCATTER = auto()  # reduce and scatter gradients
    ALL_REDUCE = auto()  # all-reduce (used in some FSDP variants)

    # Graph Control
    INPUT = auto()
    OUTPUT = auto()
    PLACEHOLDER = auto()

    # Training-specific
    LOSS = auto()
    BACKWARD = auto()
    GRADIENT = auto()


@dataclasses.dataclass(frozen=True)
class Edge:
    """A directed edge between two graph nodes."""

    src_id: str
    dst_id: str
    src_idx: int = 0  # which output of src
    dst_idx: int = 0  # which input of dst
    meta: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Node:
    """A single node in the computation graph.

    Each node can represent either a PyTorch tensor operation or an
    FSDP communication collective. The unified representation allows
    MagiCompiler to optimize across both domains.
    """

    id: str = dataclasses.field(default_factory=lambda: f"n_{uuid.uuid4().hex[:12]}")
    kind: NodeKind = NodeKind.PLACEHOLDER
    name: str = ""

    # The callable (torch op, FSDP collective, etc.)
    target: Optional[Any] = None

    # Inputs as (node_id, output_idx) pairs
    args: List[tuple] = dataclasses.field(default_factory=list)

    # Output metadata
    output_shape: Optional[tuple] = None
    output_dtype: Optional[Any] = None

    # For FSDP: shard metadata
    shard_group: Optional[Any] = None  # process group
    shard_rank: int = 0
    shard_world_size: int = 1

    # For training: gradient info
    grad_shape: Optional[tuple] = None

    # Extra metadata
    meta: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"Node({self.name}, kind={self.kind.name}, "
            f"id={self.id[:8]}..., target={self.target})"
        )


class FXGraph:
    """The unified computation graph for MagiCompiler.

    Captures both regular PyTorch operations and FSDP communication
    collectives in a single directed acyclic graph (DAG), enabling
    holistic optimization passes that cross traditional FSDP boundaries.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, Node] = {}
        self._edges: List[Edge] = []
        self._input_nodes: List[str] = []
        self._output_nodes: List[str] = []
        self._comm_nodes: List[str] = []  # FSDP collective nodes

    # ── Node Management ──────────────────────────────────────────────

    def add_node(self, node: Node) -> str:
        """Register a node in the graph. Returns its id."""
        self._nodes[node.id] = node
        if node.kind == NodeKind.INPUT:
            self._input_nodes.append(node.id)
        elif node.kind == NodeKind.OUTPUT:
            self._output_nodes.append(node.id)
        if node.kind in (
            NodeKind.ALL_GATHER,
            NodeKind.REDUCE_SCATTER,
            NodeKind.ALL_REDUCE,
        ):
            self._comm_nodes.append(node.id)
        return node.id

    def get_node(self, node_id: str) -> Optional[Node]:
        return self._nodes.get(node_id)

    @property
    def nodes(self) -> Dict[str, Node]:
        return dict(self._nodes)

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    # ── Edge Management ──────────────────────────────────────────────

    def add_edge(self, edge: Edge) -> None:
        self._edges.append(edge)

    def add_edge_by_id(
        self, src_id: str, dst_id: str,
        src_idx: int = 0, dst_idx: int = 0,
    ) -> None:
        self._edges.append(Edge(src_id=src_id, dst_id=dst_id,
                                 src_idx=src_idx, dst_idx=dst_idx))

    def successors(self, node_id: str) -> List[str]:
        return [e.dst_id for e in self._edges if e.src_id == node_id]

    def predecessors(self, node_id: str) -> List[str]:
        return [e.src_id for e in self._edges if e.dst_id == node_id]

    @property
    def edges(self) -> List[Edge]:
        return list(self._edges)

    # ── Graph Properties ─────────────────────────────────────────────

    @property
    def comm_nodes(self) -> List[str]:
        return list(self._comm_nodes)

    @property
    def input_nodes(self) -> List[str]:
        return list(self._input_nodes)

    @property
    def output_nodes(self) -> List[str]:
        return list(self._output_nodes)

    def topo_sort(self) -> List[str]:
        """Topological sort of all node ids."""
        in_degree: Dict[str, int] = {}
        for nid in self._nodes:
            in_degree[nid] = 0
        for e in self._edges:
            if e.dst_id in in_degree:
                in_degree[e.dst_id] += 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order = []
        while queue:
            nid = queue.pop(0)
            order.append(nid)
            for succ in self.successors(nid):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        return order

    def is_comm_node(self, node_id: str) -> bool:
        node = self._nodes.get(node_id)
        return node is not None and node.kind in (
            NodeKind.ALL_GATHER,
            NodeKind.REDUCE_SCATTER,
            NodeKind.ALL_REDUCE,
        )

    # ── Visualization ────────────────────────────────────────────────

    def to_dot(self) -> str:
        """Export graph to DOT format for visualization."""
        lines = ["digraph MagiCompilerGraph {"]
        lines.append("  rankdir=LR;")
        lines.append('  node [shape=box, style="rounded,filled"];')

        kind_colors = {
            NodeKind.COMPUTE: "#AED6F1",
            NodeKind.ALL_GATHER: "#A9DFBF",
            NodeKind.REDUCE_SCATTER: "#F9E79F",
            NodeKind.ALL_REDUCE: "#F5B7B1",
            NodeKind.INPUT: "#D5D8DC",
            NodeKind.OUTPUT: "#D5D8DC",
            NodeKind.PARAMETER: "#D7BDE2",
            NodeKind.LOSS: "#F0B27A",
            NodeKind.BACKWARD: "#F1948A",
            NodeKind.GRADIENT: "#85C1E9",
        }

        for nid, node in self._nodes.items():
            color = kind_colors.get(node.kind, "#FFFFFF")
            label = f"{node.name}\\n{node.kind.name}"
            lines.append(f'  "{nid}" [label="{label}", fillcolor="{color}"];')

        for e in self._edges:
            lines.append(f'  "{e.src_id}" -> "{e.dst_id}";')

        lines.append("}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"FXGraph(nodes={self.num_nodes}, "
            f"comm_nodes={len(self._comm_nodes)}, "
            f"edges={len(self._edges)})"
        )
