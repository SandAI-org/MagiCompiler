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

import json
import os
from enum import Enum, unique
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .utils import compute_hash


@unique
class CompileMode(Enum):
    """
    The compilation approach used for torch.compile-based compilation of the model.

    NONE: No torch.compile compilation is applied, model runs in fully eager pytorch mode. The model runs as-is.
    TORCH_COMPILE: The standard `torch.compile` compilation pipeline.
    MAGI_COMPILE: Custom Inductor-based backend with caching, piecewise compilation, shape specialization, and custom passes.
    """

    NONE = 'NONE'
    TORCH_COMPILE = 'TORCH_COMPILE'
    MAGI_COMPILE = 'MAGI_COMPILE'


@unique
class CudaGraphMode(Enum):
    """
    Constants for the cudagraph mode in CompileConfig.
    Different from the CUDAGraphMode for llm, PIECEWISE and FULL modes are enough for diffusion models.

    NONE: No cudagraph is used.
    PIECEWISE: Cudagraph is used for piecewise compilation.
    FULL: Cudagraph is used for full compilation.
    """

    NONE = 'NONE'
    PIECEWISE = 'PIECEWISE'
    FULL = 'FULL'


class PassConfig(BaseModel):
    """Configuration for custom Inductor passes."""

    # TODO: Add custom fusion passes (RMSNorm/SiluMul+quant, Attention+quant, AllReduce fusion).
    # TODO: Add no-op elimination pass.
    # TODO: Add sequence parallelism pass and async TP pass.
    # TODO: Add Ulysses overlap pass.
    enable_sage_attn: bool = Field(False, description="Whether to replace flash attention with sage attention.")

    @property
    def hash(self) -> str:
        return compute_hash(self.model_dump(mode="json"))

    # Compatible with torch pass
    def uuid(self) -> str:
        return self.hash


@unique
class RecomputePolicy(Enum):
    """
    Defines the strategy for activation recomputation (rematerialization) to trade off
    memory usage against computational overhead.

    HANDCRAFT:
        A manual strategy where the user controls the trade-off via a `memory_budget`
        parameter. This parameter acts as a threshold (0.0 to 1.0) determining the
        target percentage of activations to save.

    HEURISTIC:
        A rule-based strategy that selectively saves activations from compute-bound
        operators (e.g., MatMul, Attention). Conversely, outputs from memory-bound
        or element-wise operators are prioritized for recomputation to save memory.

    AUTOSEARCH (Work In Progress):
        An automated strategy that searches for the optimal set of saved tensors based
        on available device memory. It prioritizes saving tensors with high computational
        cost relative to their memory footprint.
    """

    HANDCRAFT = "HANDCRAFT"
    HEURISTIC = "HEURISTIC"
    AUTOSEARCH = "AUTOSEARCH"


class RecomputeConfig(BaseModel):
    recompute_policy: RecomputePolicy = Field(RecomputePolicy.HEURISTIC, description="Recompute policy.")
    custom_compute_sensitive_ops: list[str] = Field(
        default_factory=list, description="Custom compute sensitive ops, registered by @magi_register_custom_op"
    )
    memory_budget: float = Field(0.5, description="Activation memory budget for recomputation, only used for handcraft.")


@unique
class OffloadPolicy(Enum):
    """
    The policy for offloading the model to CPU.

    BASE:
        The base policy for offloading the model to CPU.
        Offload all the submodules to CPU.
    COST_EFFECTIVE:
        The cost effective policy for offloading the model to CPU.
        Offload the submodules to CPU based on the cost effective policy.
    HEURISTIC:
        The heuristic policy for offloading the model to CPU.
        Offload the submodules to CPU based on the heuristic policy.
    """

    BASE = "BASE"
    COST_EFFECTIVE = "COST_EFFECTIVE"
    HEURISTIC = "HEURISTIC"


class OffloadConfig(BaseModel):
    model_cpu_offload: bool = Field(False, description="Whether to offload the model to CPU.")
    gpu_resident_weight_ratio: float = Field(
        0.3, description="The ratio of GPU memory to keep when offloading the model to CPU."
    )
    offload_policy: OffloadPolicy = Field(
        OffloadPolicy.COST_EFFECTIVE, description="The policy for offloading the model to CPU."
    )
    bandwidth_safety_factor: float = Field(0.9, description="The safety factor for the H2D bandwidth.")


class CompileConfig(BaseSettings):
    """Top-level configuration consumed by ``magi_compile`` and the MagiCompiler backend.

    All fields can be overridden via environment variables with a ``MAGI_COMPILE_``
    prefix (e.g. ``MAGI_COMPILE_AOT=1``, ``MAGI_COMPILE_BACKEND=eager``).
    Priority: user ``config_patch`` > env var > hardcoded default.
    """

    model_config = SettingsConfigDict(
        env_prefix="MAGI_COMPILE_",
        populate_by_name=True,
        cli_parse_args=True,
        cli_ignore_unknown_args=True,
        cli_implicit_flags=True,
    )

    # ---- Basic configs ----
    backend: Literal["inductor", "eager"] = Field(
        "inductor", description="TorchInductor backend to use. 'inductor' for optimized codegen, 'eager' for debugging."
    )
    compile_mode: CompileMode = Field(
        default=CompileMode.MAGI_COMPILE,
        description=(
            "Compilation strategy: NONE (eager), TORCH_COMPILE (vanilla torch.compile), "
            "or MAGI_COMPILE (piecewise compilation with caching and custom passes)."
        ),
    )
    cache_root_dir: str = Field(
        default=os.path.expanduser("~/.cache/magi_compiler"),
        description="Root directory for persisting compiled artifacts and debug dumps.",
    )

    # ---- Compilation mode ----
    aot: bool = Field(
        default=False,
        description=(
            "Enable AOT (Ahead-Of-Time) compilation. Persists compiled artifacts to disk "
            "and loads from cache on startup to skip Dynamo tracing."
        ),
    )
    disable_cache: bool = Field(False, description="Force re-compilation by ignoring any cached piecewise compiled artifacts.")

    # ---- CPU Offload ----
    offload_config: OffloadConfig = Field(
        OffloadConfig(), description="Configuration for CPU offloading of model weights and activations."
    )

    # ---- Inductor configs ----
    enable_inductor_max_autotune: bool = Field(False, description="Enable Inductor max_autotune for kernel selection.")
    enable_inductor_coordinate_descent_tuning: bool = Field(
        False, description="Enable Inductor coordinate_descent_tuning for kernel selection."
    )
    compile_sizes: list[int] = Field(
        default_factory=list,
        description=(
            "Explicit sequence lengths to pre-compile. An empty list means a single dynamic-shape compilation is used."
        ),
    )
    splitting_ops: list[str] = Field(
        default_factory=list,
        description=(
            "Custom operator names at which the FX graph is split into piecewise sub-graphs. "
            "Each sub-graph between two splitting ops is compiled independently by Inductor."
        ),
    )

    # ---- torch.compile options keys ----
    post_grad_pass: str = Field(
        "post_grad_custom_post_pass", description="Key name in torch.compile options for the post-grad custom pass."
    )
    custom_partitioner_fn: str = Field(
        "custom_partitioner_fn", description="Key name in torch.compile options for the custom graph partitioner function."
    )

    # ---- Pass configs ----
    pass_config: PassConfig = Field(
        PassConfig(), description="Configuration for custom post-grad Inductor passes (e.g. sage attention replacement)."
    )

    # ---- Recompute configs ----
    recompute_config: RecomputeConfig = Field(
        RecomputeConfig(), description="Activation recomputation (rematerialization) strategy and budget."
    )

    # ---- CUDA Graph configs ----
    cudagraph_mode: CudaGraphMode = Field(
        CudaGraphMode.NONE,
        description=(
            "CUDA Graph capture mode. NONE disables capture, PIECEWISE captures each sub-graph independently, "
            "FULL captures the entire compiled graph as a single CUDA Graph."
        ),
    )

    @property
    def hash(self) -> str:
        return compute_hash(self.model_dump(mode="json"))

    def __str__(self, indent: int = 4):
        data = self.model_dump(mode="json")
        formatted = json.dumps(data, indent=indent, ensure_ascii=False, sort_keys=False)

        # add configuration class name as title
        class_name = self.__class__.__name__
        return f"{class_name}:\n{formatted}".replace('"', "")

    def __repr__(self, indent: int = 4):
        return self.__str__(indent=indent)


def model_rank_dir_name(model_idx: int, model_tag: str | None) -> str:
    """Directory name for a model instance: ``model_{idx}[_{tag}]_rank_{rank}``."""
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if model_tag:
        return f"model_{model_idx}_{model_tag}_rank_{rank}"
    return f"model_{model_idx}_rank_{rank}"


def debug_dump_path(cache_root_dir: str, model_idx: int, model_tag: str | None = None) -> Path:
    from datetime import datetime

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    return Path(cache_root_dir) / "magi_depyf" / run_id / model_rank_dir_name(model_idx, model_tag)


def cache_dump_path(cache_root_dir: str, \
                    model_idx: int | None = None, model_tag: str | None = None) -> Path:
    if not model_idx and not model_tag:
        # Inductor cache path
        return Path(cache_root_dir) / "inductor_cache"
    else:
        # Magi cache path
        assert model_idx and model_tag, "model_idx, model_tag are required for magi_cache path"
        return Path(cache_root_dir) / "magi_cache" / model_rank_dir_name(model_idx, model_tag)


def inductor_compile_config_hash(inductor_compile_config: dict[str, Any]) -> str:
    """Hash covering an Inductor compile config dict (pass managers, etc.)."""
    if not inductor_compile_config:
        return ""
    serialized: dict[str, Any] = {}
    for key, value in inductor_compile_config.items():
        if hasattr(value, "uuid") and callable(getattr(value, "uuid", None)):
            try:
                serialized[key] = value.uuid()
            except (AttributeError, RuntimeError):
                serialized[key] = str(value)
        else:
            try:
                json.dumps(value)
                serialized[key] = value
            except (TypeError, ValueError):
                serialized[key] = str(value)
    return compute_hash(serialized)


_GLOBAL_COMPILE_CONFIG = None


def get_compile_config() -> CompileConfig:
    """Return the global default :class:`CompileConfig` singleton.

    This serves as the starting point for per-model configuration:
    * Users can modify it directly via ``get_compile_config().field = value``
      to change the default for all future models.
    * ``@magi_compile(config_patch=...)`` deep-copies this and applies
      per-model overrides.
    * ``@magi_register_custom_op`` registers splitting ops and compute-
      sensitive ops into this global config so they propagate to all models.
    """
    global _GLOBAL_COMPILE_CONFIG
    if _GLOBAL_COMPILE_CONFIG is None:
        _GLOBAL_COMPILE_CONFIG = CompileConfig()
    return _GLOBAL_COMPILE_CONFIG
