# Copyright (c) 2026 SandAI. All Rights Reserved.
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

import collections
import operator
from typing import Any, Dict

import torch
from torch.fx import GraphModule, Node
from torch.fx.node import map_arg

from magi_compiler.config import CompileConfig
from magi_compiler.offload.profiler import OffloadProfiler
from magi_compiler.offload.scheduler import OffloadRuntimeContext, SchedulerFactory
from magi_compiler.utils.nvtx import add_nvtx_event

from ..magi_depyf.timeline import observe_lifecycle


class OffloadExecutor:
    def __init__(self, graph_module: GraphModule, compile_config: CompileConfig):
        self.graph_module = graph_module
        self.compile_config = compile_config

        self.compute_stream = torch.cuda.current_stream()
        self.h2d_stream = torch.cuda.Stream()

        self.warmup = True
        self.second_call = False
        self.buffers: Dict[str, torch.Tensor] = {}
        self.persistent_weights: Dict[str, torch.Tensor] = {}
        self.submod_0_weight_handoff: Dict[Node, torch.Tensor] = {}

        self._analyze_graph()
        self.profiler = OffloadProfiler()

        common_args = {
            "submod_nodes": self.submod_nodes,
            "submod_weights_map": self.submod_weights_map,
            "name_node_map": self.name_node_map,
            "weight_sizes": self.submod_weight_sizes,
        }

        self.scheduler = SchedulerFactory.create(self.compile_config, common_args)

    def _analyze_graph(self):
        self.submod_nodes = [n for n in self.graph_module.graph.nodes if n.op == "call_module"]

        self.placeholder_nodes = []
        self.arg_index_weight = {}
        self.user_counts = collections.defaultdict(int)
        self.name_node_map = {}

        placeholder_idx = 0
        for node in self.graph_module.graph.nodes:
            for input_node in node.all_input_nodes:
                self.user_counts[input_node] += 1

            if node.op == "placeholder":
                is_w = isinstance(node.meta.get("example_value"), torch.nn.Parameter)
                self.arg_index_weight[placeholder_idx] = is_w
                self.placeholder_nodes.append(node)
                self.name_node_map[node.name] = node
                placeholder_idx += 1

        self.submod_weights_map = {}
        self.submod_weight_sizes = {}

        for node in self.submod_nodes:
            weight_names = []
            size = 0
            for arg in node.args:
                if isinstance(arg, Node) and self._is_weight_node(arg):
                    if arg.name in self.name_node_map:
                        weight_names.append(arg.name)
                        val = arg.meta.get("example_value")
                        if val is not None:
                            size += val.numel() * val.element_size()

            self.submod_weights_map[node.name] = weight_names
            self.submod_weight_sizes[node.name] = size

    def _is_weight_node(self, node: Node) -> bool:
        return node.op == "placeholder" and isinstance(node.meta.get("example_value"), torch.nn.Parameter)

    def _prepare_inputs(self, args) -> Dict[Node, Any]:
        env = {}
        args = list(args)
        submod_0 = self.submod_nodes[0]

        for i, node in enumerate(self.placeholder_nodes):
            arg_val = args[i]
            is_weight = self.arg_index_weight[i]

            # case 1: input tensor
            if not is_weight:
                if isinstance(arg_val, torch.Tensor):
                    arg_val = arg_val.to("cuda", non_blocking=False)
                env[node] = arg_val
                continue

            # case 2: kept weight
            if self.scheduler.is_kept(node.name):
                if node.name not in self.persistent_weights:
                    t = arg_val.to("cuda", non_blocking=False) if arg_val.device.type == "cpu" else arg_val
                    self.persistent_weights[node.name] = t
                env[node] = self.persistent_weights[node.name]
                continue

            # case 3: submod 0 weight
            if submod_0 in node.users:
                if self.warmup and arg_val.device.type == "cpu":
                    self.buffers[node.name] = arg_val
                    arg_val = arg_val.to("cuda", non_blocking=False)
                elif not self.warmup:
                    if node in self.submod_0_weight_handoff:
                        arg_val = self.submod_0_weight_handoff[node]
                        del self.submod_0_weight_handoff[node]

            env[node] = arg_val

        return env

    def _finalize_warmup(self):
        profile_results = self.profiler.summarize()
        self.scheduler.schedule_kept_weights(profile_results)
        self.warmup = False

    def __call__(self, *args):
        if len(self.submod_nodes) == 0:
            return self.graph_module(*args)

        env = self._prepare_inputs(args)
        current_user_counts = self.user_counts.copy()
        runtime_ctx = OffloadRuntimeContext(
            env=env,
            h2d_stream=self.h2d_stream,
            compute_stream=self.compute_stream,
            buffers=self.buffers,
            submod_0_handoff=self.submod_0_weight_handoff,
            need_profile=self.second_call or self.warmup,
        )
        need_profile = self.second_call

        for node in self.graph_module.graph.nodes:
            if node.op == "placeholder":
                continue

            elif node.op == "call_module":
                self.scheduler.prefetch(node.name, runtime_ctx)

                if need_profile:
                    if torch.distributed.is_initialized():
                        torch.distributed.barrier()
                    self.profiler.start_compute_profile(node.name, self.compute_stream)

                with add_nvtx_event(node.name):
                    with torch.cuda.stream(self.compute_stream):
                        s_args = map_arg(node.args, lambda n: env[n])
                        s_kwargs = map_arg(node.kwargs, lambda n: env[n])
                        env[node] = getattr(self.graph_module, node.target)(*s_args, **s_kwargs)
                        del s_args, s_kwargs

                if need_profile:
                    if torch.distributed.is_initialized():
                        torch.distributed.barrier()
                    self.profiler.end_compute_profile(node.name, self.compute_stream)

            elif node.op == "call_function":
                # ... (Standard execution logic same as before)
                if node.target == operator.getitem:
                    parent_node, idx = node.args
                    env[node] = env[parent_node][idx]
                else:
                    with torch.cuda.stream(self.compute_stream):
                        f_args = map_arg(node.args, lambda n: env[n])
                        f_kwargs = map_arg(node.kwargs, lambda n: env[n])
                        env[node] = node.target(*f_args, **f_kwargs)

            elif node.op == "output":
                if self.second_call:
                    self._finalize_warmup()
                    self.second_call = False
                if self.warmup:
                    self.second_call = True
                    self.warmup = False

                return map_arg(node.args[0], lambda n: env[n])

            # Memory Management
            for input_node in node.all_input_nodes:
                current_user_counts[input_node] -= 1
                if current_user_counts[input_node] == 0:
                    if input_node in env:
                        tensor_obj = env[input_node]
                        if isinstance(tensor_obj, torch.Tensor) and tensor_obj.is_cuda:
                            tensor_obj.record_stream(self.compute_stream)
                        del env[input_node]
        return None


class OffloadWrapper:
    @observe_lifecycle("offload_wrap")
    def __init__(self, graph_module: torch.fx.GraphModule, compile_config: CompileConfig):
        self.executor = OffloadExecutor(graph_module, compile_config)

    def __call__(self, *args):
        return self.executor(*args)
