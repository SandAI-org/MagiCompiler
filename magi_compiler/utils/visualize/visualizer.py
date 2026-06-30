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

import os
from typing import Any, Dict, List, Tuple

import graphviz
import torch
import torch.fx

from magi_compiler.config import get_compile_config
from magi_compiler.utils import envs, magi_logger


class FX_NODE_OP:
    PLACEHOLDER = "placeholder"
    OUTPUT = "output"
    CALL_MODULE = "call_module"
    CALL_FUNCTION = "call_function"
    CALL_METHOD = "call_method"
    GET_ATTR = "get_attr"
    DEFAULT = "default"


FIXED_NODE_STYLES = {
    FX_NODE_OP.PLACEHOLDER: {"shape": "rectangle", "style": "filled,bold", "fillcolor": "#E8F4FD", "color": "#2196F3"},
    FX_NODE_OP.OUTPUT: {"shape": "rectangle", "style": "filled,bold", "fillcolor": "#FCE4EC", "color": "#E91E63"},
    FX_NODE_OP.CALL_MODULE: {"shape": "ellipse", "style": "filled,bold", "fillcolor": "#FFF8E1", "color": "#FFC107"},
    FX_NODE_OP.CALL_FUNCTION: {"shape": "ellipse", "style": "filled", "fillcolor": "#E8F5E9", "color": "#4CAF50"},
    FX_NODE_OP.CALL_METHOD: {"shape": "ellipse", "style": "filled", "fillcolor": "#F3E5F5", "color": "#9C27B0"},
    FX_NODE_OP.GET_ATTR: {"shape": "ellipse", "style": "filled", "fillcolor": "#FFCCBC", "color": "#FF5722"},
    FX_NODE_OP.DEFAULT: {"shape": "ellipse", "style": "filled", "fillcolor": "#F5F5F5", "color": "#666666"},
    "call_function.linear": {"shape": "ellipse", "style": "filled,bold", "fillcolor": "#E8F5E9", "color": "#44FF00"},
}


def build_node_to_code_map(graph: torch.fx.Graph) -> Dict[torch.fx.Node, str]:
    node_to_code = {}

    python_code = graph.python_code(root_module="self", verbose=False)
    code_lines = python_code.src.strip().split("\n")
    lineno_map = python_code._lineno_map

    node_index_map = {idx: node for idx, node in enumerate(graph.nodes)}

    for line_num, node_idx in lineno_map.items():
        if node_idx is None or node_idx not in node_index_map:
            continue
        node = node_index_map[node_idx]
        if 0 <= line_num < len(code_lines):
            code_line = code_lines[line_num].strip()
            if code_line and not code_line.startswith(("wrap(", "#", "pass")):
                node_to_code[node] = code_line

    for node in graph.nodes:
        if node not in node_to_code:
            assert node.op == FX_NODE_OP.PLACEHOLDER, f"Unexpected missing code for {node.op=}, {node.target=}"
            node_to_code[node] = ""
    return node_to_code


# Graphviz aborts with a "syntax error ... missing endquote? longer than 16384?"
# when any single quoted string (i.e. a node label) exceeds 16384 characters.
# Keep labels comfortably below that ceiling.
_MAX_LABEL_LEN = 8000


def _sanitize_label(text: str) -> str:
    # Graphviz treats backslashes as escapes inside quoted strings. A literal
    # backslash in a target/meta repr (or one left at the end of a truncated
    # string) can escape the closing quote and produce
    # "syntax error in line N". Replace backslashes so the label is inert.
    return text.replace("\\", "/")


def _truncate_label(text: str) -> str:
    if len(text) <= _MAX_LABEL_LEN:
        return text
    omitted = len(text) - _MAX_LABEL_LEN
    return text[:_MAX_LABEL_LEN] + f"\n...[truncated {omitted} chars]"


def extract_fx_graph_structure(graph: torch.fx.Graph, simple_desc: bool = False) -> Tuple[List[Dict], List[Dict]]:
    import textwrap

    def wrap_str_to_multi_lines(text, width=30):
        return textwrap.fill(text, width=width, break_long_words=True, replace_whitespace=False)

    nodes, edges = [], []
    node_to_code = build_node_to_code_map(graph)

    for node in graph.nodes:
        name_str = str(node.name)
        name_str = wrap_str_to_multi_lines(name_str)

        if node.op == FX_NODE_OP.GET_ATTR:
            # torch._inductor.exc.InductorError: DataDependentOutputException: aten._local_scalar_dense.default
            meta_str = f"get_attr: {node.target} (skip fake tensor)"
        else:
            tensor_meta = node.meta.get("tensor_meta") or node.meta.get("val") or node.meta.get("example_value")
            meta_str = tensor_meta_to_str(tensor_meta)
            meta_str = wrap_str_to_multi_lines(meta_str)

        target_str = target_to_str(node.target)
        if hasattr(node, "original_target"):
            original_target_str = target_to_str(node.original_target)
            target_str += f"\nOriginal: {original_target_str}"
        target_str = wrap_str_to_multi_lines(target_str)

        node_code = node_to_code.get(node, "empty")
        node_code = wrap_str_to_multi_lines(node_code)

        style_info = FIXED_NODE_STYLES.get(node.op, FIXED_NODE_STYLES[FX_NODE_OP.DEFAULT])
        if node.op == FX_NODE_OP.CALL_FUNCTION and f"call_function.{node.target.__name__}" in FIXED_NODE_STYLES:
            style_info = FIXED_NODE_STYLES[f"call_function.{node.target.__name__}"]

        node_label = f"Op: {node.op}\nTarget: {target_str}\nName: {name_str}\nMeta: {meta_str}\nCode: {node_code}"
        if simple_desc:
            node_label = f"Op: {node.op}\nTarget: {target_str}\nName: {name_str}"
        node_label = _truncate_label(_sanitize_label(node_label))

        nodes.append({"id": node.name, "style": style_info, "node_label": node_label})

        def traverse_args(args_kwargs):
            if isinstance(args_kwargs, (tuple, list)):
                for arg in args_kwargs:
                    traverse_args(arg)
            elif isinstance(args_kwargs, dict):
                for val in args_kwargs.values():
                    traverse_args(val)
            elif isinstance(args_kwargs, torch.fx.Node):
                d = {"source": args_kwargs.name, "target": node.name}
                edges.append(d) if d not in edges else None

        traverse_args(node.args)
        traverse_args(node.kwargs)

    return nodes, edges


def target_to_str(target: Any) -> str:
    res = str(target)
    if isinstance(target, str):
        res = target
    elif hasattr(target, "_op"):
        res = str(target._op)
    elif callable(target):
        res = getattr(target, "__name__")
    return res


def tensor_meta_to_str(tensor_meta: Any) -> str:
    if type(tensor_meta) in [int, float, str, bool]:
        return str(tensor_meta)
    elif isinstance(tensor_meta, (list, tuple)):
        return f"[{', '.join([tensor_meta_to_str(t) for t in tensor_meta])}]"
    elif isinstance(tensor_meta, torch.Tensor):
        d = {}
        d["shape"] = tensor_meta.shape if hasattr(tensor_meta, "shape") else "N/A"
        d["size"] = tensor_meta.size() if hasattr(tensor_meta, "size") else "N/A"
        d["ndim"] = tensor_meta.ndim if hasattr(tensor_meta, "ndim") else "N/A"
        d["numel"] = tensor_meta.numel() if hasattr(tensor_meta, "numel") else "N/A"
        d["stride"] = tensor_meta.stride if hasattr(tensor_meta, "stride") else "N/A"
        d["stride"] = tensor_meta.stride() if hasattr(tensor_meta, "stride") and callable(tensor_meta.stride) else "N/A"
        d["is_contiguous"] = tensor_meta.is_contiguous() if hasattr(tensor_meta, "is_contiguous") else "N/A"
        d["dtype"] = str(tensor_meta.dtype) if hasattr(tensor_meta, "dtype") else "N/A"
        d["device"] = str(tensor_meta.device) if hasattr(tensor_meta, "device") else "N/A"
        return ", ".join([f"{k}: {v}" for k, v in d.items()])
    else:
        return str(tensor_meta)


def create_fx_graph_dot(nodes: list[Dict], edges: list[Dict]) -> graphviz.Digraph:
    dot = graphviz.Digraph(
        name="fx_graph",
        format="pdf",
        graph_attr={"rankdir": "TD", "nodesep": "0.1", "ranksep": "0.1", "overlap": "false", "splines": "spline"},
        node_attr={
            "fontname": "Helvetica",
            "fontsize": "10",
            "shape": "rect",
            "style": "rounded,filled",
            "fixedsize": "false",
            "margin": "0.2",
        },
    )

    for node in nodes:
        dot.node(node["id"], node["node_label"], **node["style"])

    for edge in edges:
        dot.edge(edge["source"], edge["target"])

    return dot


def get_fx_graph_path(sub_dir: str = "", filename: str = "") -> str:
    cache_root_dir = get_compile_config().cache_root_dir
    # Unify with magi_depyf output
    fx_graph_dir = os.path.join(cache_root_dir, "magi_depyf", "visualizations")
    os.makedirs(fx_graph_dir, exist_ok=True)
    if sub_dir:
        fx_graph_sub_dir = os.path.join(fx_graph_dir, sub_dir)
        os.makedirs(fx_graph_sub_dir, exist_ok=True)
        if filename:
            return os.path.join(fx_graph_sub_dir, filename)
        return fx_graph_sub_dir
    if filename:
        return os.path.join(fx_graph_dir, filename)
    return fx_graph_dir


def save_fx_graph_visualization(graph: torch.fx.Graph, sub_dir: str = "", filename: str = "fx_graph"):
    """
    Save FX graph visualization as PDF.

    Args:
        graph: The FX graph or GraphModule to visualize
        sub_dir: Optional subdirectory under the fx_graph_views folder
        filename: Filename for the output PDF (without extension)
    """
    if not envs.MAGI_ENABLE_FX_GRAPH_VIZ:
        magi_logger.info("FX graph visualization is disabled. Set MAGI_ENABLE_FX_GRAPH_VIZ=true to enable it.")
        return

    if isinstance(graph, torch.fx.GraphModule):
        graph = graph.graph

    assert envs.MAGI_FX_GRAPH_VIZ_NODE_DESC in {
        "simple",
        "detailed",
    }, f"Invalid MAGI_FX_GRAPH_VIZ_NODE_DESC: {envs.MAGI_FX_GRAPH_VIZ_NODE_DESC}"
    simple_desc = envs.MAGI_FX_GRAPH_VIZ_NODE_DESC == "simple"

    file_path = get_fx_graph_path(sub_dir=sub_dir, filename=filename)

    # Visualization is purely diagnostic; never let a render failure abort
    # compilation/inference.
    try:
        nodes, edges = extract_fx_graph_structure(graph, simple_desc=simple_desc)
        dot = create_fx_graph_dot(nodes, edges)
        dot.render(filename=file_path, view=False, cleanup=True)
        magi_logger.info("FX graph visualization saved to: %s.pdf", file_path)
    except Exception as e:
        magi_logger.warning("Skipping FX graph visualization (%s): %s", file_path, e)
