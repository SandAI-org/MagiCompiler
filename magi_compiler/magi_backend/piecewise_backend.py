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

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch.fx as fx

from magi_compiler.config import CompileConfig
from magi_compiler.utils import magi_logger

if TYPE_CHECKING:
    from .magi_backend import CompilerManager


@dataclasses.dataclass
class ConcreteSizeEntry:
    runtime_shape: int
    compiled: bool = False
    runnable: Callable = None  # type: ignore


class PiecewiseBackend:
    def __init__(
        self,
        graph: fx.GraphModule,
        compiled_graph_for_general_shape: Callable,
        compile_config: CompileConfig,
        inductor_compile_config: dict[str, Any],
        piecewise_compile_index: int,
        piecewise_submodule_number: int,
        sym_shape_indices: list[int],
        compiler_manager: "CompilerManager",
    ):
        """
        The backend for piecewise compilation. It mainly handles the compilation of static shapes and dispatching based on runtime shape.

        We will compile `self.graph` once for the general shape, and then compile for different shapes specified in `compile_config.compile_sizes`.
        """
        self.graph = graph
        self.compiled_graph_for_general_shape = compiled_graph_for_general_shape
        self.compile_config = compile_config
        self.inductor_compile_config = inductor_compile_config
        self.piecewise_compile_index = piecewise_compile_index
        self.piecewise_submodule_number = piecewise_submodule_number
        self.compiler_manager = compiler_manager
        self.sym_shape_indices = sym_shape_indices

        self.is_first_graph = piecewise_compile_index == 0
        self.is_last_graph = piecewise_compile_index == piecewise_submodule_number - 1
        self.is_first_run = True

        # to_be_compiled_sizes tracks the remaining sizes to compile,
        # and updates during the compilation process, so we need to copy it
        self.to_be_compiled_sizes: set[int] = set(self.compile_config.compile_sizes)

        # the entries for different shapes that we need to compile
        self.concrete_size_entries: dict[int, ConcreteSizeEntry] = {}
        for shape in self.to_be_compiled_sizes:
            self.concrete_size_entries[shape] = ConcreteSizeEntry(
                runtime_shape=shape, runnable=self.compiled_graph_for_general_shape
            )

    def check_for_ending_compilation(self):
        if self.is_last_graph and not self.to_be_compiled_sizes:
            self.compiler_manager.save_to_file()

    def __call__(self, *args) -> Any:
        import os

        self._call_n = getattr(self, "_call_n", 0) + 1
        if os.environ.get("GAGA4_SUBMOD_EAGER_FX", "").strip() in {"1", "true", "yes"}:
            if (
                int(os.environ.get("RANK", "0")) == 0
                and self.piecewise_compile_index == 0
                and self._call_n == 3
                and not getattr(self, "_ptr_probed", False)
            ):
                self._ptr_probed = True
                import torch

                ops = {}
                attrs = []
                for n in self.graph.graph.nodes:
                    ops[n.op] = ops.get(n.op, 0) + 1
                    if n.op == "get_attr":
                        try:
                            obj = self.graph
                            for part in str(n.target).split("."):
                                obj = getattr(obj, part)
                            if isinstance(obj, torch.Tensor):
                                attrs.append(
                                    f"{n.target} {tuple(obj.shape)} {obj.device} "
                                    f"ptr=0x{int(obj.data_ptr()):x} "
                                    f"fin={bool(torch.isfinite(obj.float()).all())} "
                                    f"mean={float(obj.float().mean()):.4g}"
                                )
                            else:
                                attrs.append(f"{n.target} type={type(obj).__name__}")
                        except Exception as exc:
                            attrs.append(f"{n.target} ERR {exc}")
                print(f"[fx-attr] idx=0 ops={ops} n_attr={len(attrs)} {attrs[:16]}", flush=True)

                captured = []
                inplace = []
                targets = {}

                def _walk(o, loc):
                    if isinstance(o, torch.Tensor):
                        captured.append(
                            f"{loc} {tuple(o.shape)} {o.device} ptr=0x{int(o.data_ptr()):x} "
                            f"dt={o.dtype}"
                        )
                    elif isinstance(o, (tuple, list)):
                        for i, x in enumerate(o):
                            _walk(x, f"{loc}[{i}]")
                    elif isinstance(o, dict):
                        for k, v in o.items():
                            _walk(v, f"{loc}.{k}")

                for n in self.graph.graph.nodes:
                    tgt = str(n.target)
                    if n.op in {"call_function", "call_method"}:
                        key = f"{n.op}:{tgt}"
                        targets[key] = targets.get(key, 0) + 1
                        if tgt.endswith("_") or ".copy_" in tgt or tgt.endswith("_.default"):
                            inplace.append(key)
                    _walk(n.args, f"{n.name}.args")
                    _walk(n.kwargs, f"{n.name}.kwargs")
                n_buf = sum(1 for _ in self.graph.named_buffers())
                n_par = sum(1 for _ in self.graph.named_parameters())
                top = sorted(targets.items(), key=lambda kv: -kv[1])[:24]
                print(
                    f"[fx-cap] n_captured={len(captured)} n_inplace={len(inplace)} "
                    f"n_buf={n_buf} n_par={n_par} captured={captured[:12]} "
                    f"inplace={inplace[:16]} top={top}",
                    flush=True,
                )

                w = None
                wi = -1
                for i, a in enumerate(args):
                    if isinstance(a, torch.Tensor) and a.is_floating_point() and a.numel() > 1024:
                        if w is None or a.numel() > w.numel():
                            w, wi = a, i
                if w is not None:
                    backup = w.detach().clone()
                    clean = self.graph(*args)
                    w.add_(100)
                    poisoned = self.graph(*args)
                    w.copy_(backup)

                    def _t0(x):
                        if isinstance(x, torch.Tensor):
                            return x
                        if isinstance(x, (tuple, list)):
                            for y in x:
                                t = _t0(y)
                                if t is not None:
                                    return t
                        return None

                    c, psn = _t0(clean), _t0(poisoned)
                    if c is not None and psn is not None and c.shape == psn.shape:
                        d = (c.detach().float() - psn.detach().float()).abs()
                        print(
                            f"[poison-fx] arg={wi} shape={tuple(w.shape)} "
                            f"ptr=0x{int(w.data_ptr()):x} "
                            f"max_delta={float(d.max()):.6g} "
                            f"reads_live_args={float(d.max()) > 1.0}",
                            flush=True,
                        )
            return self.graph(*args)
        if self.is_first_run:
            self.is_first_run = False
            self.check_for_ending_compilation()
            compiled_out = self.compiled_graph_for_general_shape(*args)
        elif len(self.sym_shape_indices) == 0:
            compiled_out = self.compiled_graph_for_general_shape(*args)
        elif args[self.sym_shape_indices[0]] not in self.concrete_size_entries:
            compiled_out = self.compiled_graph_for_general_shape(*args)
        else:
            compiled_out = None

        want = {0, 6, 32}
        if (
            compiled_out is not None
            and os.environ.get("GAGA4_SUBMOD_COMPARE", "").strip() in {"1", "true", "yes"}
            and int(os.environ.get("RANK", "0")) == 0
            and self.piecewise_compile_index in want
            and self._call_n == 3
        ):
            try:
                import torch

                eager_out = self.graph(*args)

                def _flat(x):
                    if isinstance(x, torch.Tensor):
                        return [x]
                    if isinstance(x, (tuple, list)):
                        out = []
                        for y in x:
                            out.extend(_flat(y))
                        return out
                    return []

                cs, es = _flat(compiled_out), _flat(eager_out)
                bits = []
                for i, (c, e) in enumerate(zip(cs, es)):
                    if c.shape != e.shape:
                        bits.append(f"o{i} shape {tuple(c.shape)} vs {tuple(e.shape)}")
                        continue
                    d = (c.detach().float() - e.detach().float()).abs()
                    bits.append(
                        f"o{i} max={float(d.max()):.6g} mean={float(d.mean()):.6g} "
                        f"cstd={float(c.float().std()):.4g} estd={float(e.float().std()):.4g}"
                    )
                print(
                    f"[submod-cmp] idx={self.piecewise_compile_index} call={self._call_n} "
                    f"n={len(cs)} {' '.join(bits)}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[submod-cmp] idx={self.piecewise_compile_index} call={self._call_n} FAIL {type(exc).__name__}: {exc}",
                    flush=True,
                )
        if compiled_out is not None:
            return compiled_out

        if len(self.sym_shape_indices) == 0:
            magi_logger.info("No symbolic shape indices found, falling back to general shape compiled graph")
            return self.compiled_graph_for_general_shape(*args)
        runtime_shape = args[self.sym_shape_indices[0]]
        if runtime_shape not in self.concrete_size_entries:
            # we don't need to do anything for this shape
            return self.compiled_graph_for_general_shape(*args)

        entry = self.concrete_size_entries[runtime_shape]

        if not entry.compiled:
            entry.compiled = True
            self.to_be_compiled_sizes.remove(runtime_shape)
            # args are real arguments
            entry.runnable = self.compiler_manager.compile(
                self.graph,
                args,
                self.inductor_compile_config,
                graph_index=self.piecewise_compile_index,
                num_graphs=self.piecewise_submodule_number,
                runtime_shape=runtime_shape,
            )

            # finished compilations for all required shapes
            if self.is_last_graph and not self.to_be_compiled_sizes:
                self.check_for_ending_compilation()

        return entry.runnable(*args)
