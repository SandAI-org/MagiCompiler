"""
MagiCompiler – the main compilation engine.

Orchestrates the full compilation pipeline:
  1. Graph capture (tracing with FSDP collective awareness)
  2. Graph optimization (fusion, elimination, comm-computation overlap)
  3. Code generation (TorchScript / Inductor / custom backend)
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from magicompiler.core.graph import FXGraph, NodeKind
from magicompiler.fsdp.graph_capture import FSDPGraphCapture
from magicompiler.fsdp.hooks import patch_fsdp, unpatch_fsdp
from magicompiler.fsdp.inference import FSDPInferenceCapture
from magicompiler.fsdp.training import FSDPTrainingCapture
from magicompiler.fsdp.optimizations import (
    GraphOptimizer,
    OptimizationLevel,
)

logger = logging.getLogger(__name__)


class CompilationMode(Enum):
    """MagiCompiler compilation modes."""

    FSDP_INFERENCE = "fsdp-inference"
    FSDP_TRAINING = "fsdp-training"
    EAGER = "eager"  # fallback: no FSDP capture


class MagiCompiler:
    """The MagiCompiler compilation engine.

    Args:
        mode: Compilation mode (fsdp-inference, fsdp-training, eager).
        capture_full_graph: If True, capture the entire computation graph
            including FSDP communication collectives.
        fuse_comm_computation: If True, attempt to overlap communication
            with computation (e.g., all-gather with preceding compute).
        auto_recompute: If True, insert recomputation during training
            to reduce peak VRAM under memory constraints.
        optimization_level: Level of graph optimizations to apply.
        deterministic: If True, enforce deterministic execution.
    """

    def __init__(
        self,
        mode: Union[str, CompilationMode] = CompilationMode.FSDP_INFERENCE,
        capture_full_graph: bool = True,
        fuse_comm_computation: bool = False,
        auto_recompute: bool = False,
        optimization_level: OptimizationLevel = OptimizationLevel.LEVEL_1,
        deterministic: bool = False,
    ) -> None:
        self.mode = CompilationMode(mode) if isinstance(mode, str) else mode
        self.capture_full_graph = capture_full_graph
        self.fuse_comm_computation = fuse_comm_computation
        self.auto_recompute = auto_recompute
        self.optimization_level = optimization_level
        self.deterministic = deterministic

        # Internal state
        self._graph: Optional[FXGraph] = None
        self._original_model: Optional[torch.nn.Module] = None
        self._compiled_fn: Optional[Callable] = None

    def compile(self, model: torch.nn.Module) -> Callable:
        """Compile a model with FSDP full-graph capture.

        Args:
            model: A PyTorch model (potentially FSDP-wrapped).

        Returns:
            A callable that executes the compiled model.
        """
        self._original_model = model

        if not self.capture_full_graph:
            logger.info("Full-graph capture disabled; falling back to eager mode.")
            return model.forward

        # Step 1: Patch FSDP to expose communication collectives
        logger.info("Patching FSDP for full-graph capture...")
        patch_fsdp()

        try:
            if self.mode == CompilationMode.FSDP_INFERENCE:
                capture = FSDPInferenceCapture(
                    fuse_comm_computation=self.fuse_comm_computation,
                    optimization_level=self.optimization_level,
                )
                self._graph, self._compiled_fn = capture.capture(model)

            elif self.mode == CompilationMode.FSDP_TRAINING:
                capture = FSDPTrainingCapture(
                    fuse_comm_computation=self.fuse_comm_computation,
                    auto_recompute=self.auto_recompute,
                    optimization_level=self.optimization_level,
                )
                self._graph, self._compiled_fn = capture.capture(model)

            else:
                logger.warning(f"Unknown mode '{self.mode}'; using eager.")
                self._compiled_fn = model.forward

        finally:
            # Restore original FSDP behaviour
            unpatch_fsdp()

        # Step 2: Apply graph optimizations
        if self._graph is not None:
            optimizer = GraphOptimizer(level=self.optimization_level)
            self._graph = optimizer.optimize(
                self._graph,
                fuse_comm_computation=self.fuse_comm_computation,
                auto_recompute=self.auto_recompute,
                deterministic=self.deterministic,
            )

        logger.info(
            f"Compilation complete: mode={self.mode.value}, "
            f"graph_nodes={len(self._graph.nodes) if self._graph else 0}"
        )
        return self._compiled_fn or model.forward

    @property
    def graph(self) -> Optional[FXGraph]:
        return self._graph

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._compiled_fn is None:
            raise RuntimeError(
                "Module has not been compiled yet. Call .compile(model) first."
            )
        return self._compiled_fn(*args, **kwargs)


# ── Convenience API ──────────────────────────────────────────────────

from weakref import WeakValueDictionary
_MAGICOMPILER_INSTANCES: WeakValueDictionary[int, MagiCompiler] = WeakValueDictionary()


def magicompile(
    model: torch.nn.Module,
    mode: str = "fsdp-inference",
    capture_full_graph: bool = True,
    fuse_comm_computation: bool = False,
    auto_recompute: bool = False,
    optimization_level: str = "level_1",
    deterministic: bool = False,
) -> Callable:
    """Compile a PyTorch model with MagiCompiler's FSDP full-graph capture.

    This is the main entry point for MagiCompiler. It patches FSDP to expose
    communication collectives, traces the full computation graph, applies
    optimizations, and returns a compiled callable.

    Args:
        model: The PyTorch model to compile (may be FSDP-wrapped).
        mode: Compilation mode:
            - ``"fsdp-inference"``: Full-graph capture for inference.
            - ``"fsdp-training"``: Full-graph capture for training.
            - ``"eager"``: No FSDP capture, standard eager execution.
        capture_full_graph: Whether to trace FSDP collectives into the graph.
        fuse_comm_computation: Whether to fuse / overlap communication
            with computation.
        auto_recompute: Whether to automatically insert recomputation
            to reduce peak VRAM during training.
        optimization_level: ``"level_0"`` (none), ``"level_1"`` (basic),
            or ``"level_2"`` (aggressive).
        deterministic: Enforce deterministic execution.

    Returns:
        A callable that executes the compiled model.

    Example::

        import torch
        from magicompiler import magicompile

        model = MyModel().cuda()
        compiled_model = magicompile(
            model,
            mode="fsdp-inference",
            capture_full_graph=True,
            fuse_comm_computation=True,
        )
        output = compiled_model(input_tensor)
    """
    opt_level_map = {
        "level_0": OptimizationLevel.LEVEL_0,
        "level_1": OptimizationLevel.LEVEL_1,
        "level_2": OptimizationLevel.LEVEL_2,
    }
    opt_level = opt_level_map.get(optimization_level, OptimizationLevel.LEVEL_1)

    compiler = MagiCompiler(
        mode=mode,
        capture_full_graph=capture_full_graph,
        fuse_comm_computation=fuse_comm_computation,
        auto_recompute=auto_recompute,
        optimization_level=opt_level,
        deterministic=deterministic,
    )

    compiled_fn = compiler.compile(model)
    _MAGICOMPILER_INSTANCES[id(compiled_fn)] = compiler
    return compiled_fn
