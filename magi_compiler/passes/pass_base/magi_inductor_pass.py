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

import functools
from collections.abc import Iterable

import torch
from torch.fx.experimental.symbolic_shapes import has_free_symbols

from .inductor_pass import InductorPass

DEFAULT_CONV_HEAVY_THRESHOLD = 300


class MagiInductorPass(InductorPass):
    """
    Base class for inductor passes.
    """

    # If a pass needs to modify any Inductor configuration (``torch._inductor.config``),
    # it **MUST** declare all affected config keys here. The declared keys will be
    # automatically snapshotted and restored after the subgraph compilation ends
    # to prevent global leakage.
    #
    # Note: Only passes running in ``PostGradPassManager`` are allowed to mutate Inductor
    # configs. Passes running in ``FullGraphPassManager`` **MUST NOT** modify them, as full-graph
    # passes do not trigger compilation and cannot be patched/isolated.
    inductor_config_keys_potentially_mutated_by_this_pass: tuple[str, ...] = ()

    def is_dynamic(self, graph: torch.fx.Graph) -> bool:
        """Determine if the graph has dynamic shapes by checking if any placeholder carries free symbols."""
        placeholder_vals = (n.meta.get("val", n.meta.get("example_value")) for n in graph.nodes if n.op == "placeholder")
        return any(v is not None and has_free_symbols(v) for v in placeholder_vals)

    def is_conv_heavy(self, graph: torch.fx.Graph, threshold: int = DEFAULT_CONV_HEAVY_THRESHOLD) -> bool:
        """Determine if the graph is convolution-heavy (dense in convolutions)."""
        nnodes = len(list(graph.nodes))
        nconv = sum(1 for n in graph.nodes if n.target == torch.ops.aten.convolution.default)
        return nnodes < threshold * nconv


def snapshot_original_inductor_configs(passes: Iterable, inductor_compile_config: dict) -> None:
    """Snapshot the original values of global Inductor configs that passes potentially mutate.

    The captured original values are stored in ``inductor_compile_config``. When ``standalone_compile``
    calls ``compile_fx``, it automatically passes this config as ``config_patches`` to Inductor's
    ``config.patch`` context manager. No matter what values the passes temporarily set these fields to
    during compilation, they will be safely restored to their pre-compilation state on scope exit.
    """
    cfg = torch._inductor.config
    for pass_ in passes:
        for key in getattr(pass_, "inductor_config_keys_potentially_mutated_by_this_pass", ()):
            snapshot = functools.reduce(getattr, key.split("."), cfg)
            inductor_compile_config.setdefault(key, snapshot)
