"""
MagiCompiler – Next-generation PyTorch compiler with native FSDP full-graph capture.

MagiCompiler breaks the limitations of PyTorch's FSDP hook-based layer-wise capture,
bringing FSDP natively into the compilation process for both inference and training.
"""

from magicompiler.core.graph import FXGraph, Node, Edge, NodeKind
from magicompiler.core.compiler import magicompile, MagiCompiler
from magicompiler.fsdp.graph_capture import FSDPGraphCapture
from magicompiler.fsdp.hooks import patch_fsdp, unpatch_fsdp
from magicompiler.fsdp.inference import FSDPInferenceCapture
from magicompiler.fsdp.training import FSDPTrainingCapture
from magicompiler.fsdp.optimizations import GraphOptimizer, OptimizationLevel

__version__ = "1.2.0a1"
__all__ = [
    "FXGraph",
    "Node",
    "Edge",
    "NodeKind",
    "magicompile",
    "MagiCompiler",
    "FSDPGraphCapture",
    "patch_fsdp",
    "unpatch_fsdp",
    "FSDPInferenceCapture",
    "FSDPTrainingCapture",
    "GraphOptimizer",
    "OptimizationLevel",
]
