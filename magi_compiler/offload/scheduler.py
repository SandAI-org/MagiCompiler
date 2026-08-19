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

import abc
import collections
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Type

import torch
from torch.fx import Node

from magi_compiler.config import CompileConfig
from magi_compiler.utils import magi_logger


@dataclass
class OffloadRuntimeContext:
    """
    Wrap the dynamic runtime context needed during Offload
    """

    env: Dict[Node, Any]  # Tensor environment of the current computation graph
    h2d_stream: torch.cuda.Stream  # CUDA stream for data transfer
    compute_stream: torch.cuda.Stream  # CUDA stream for computation
    buffers: Dict[str, torch.Tensor]  # GPU copies stored during Warmup
    submod_0_handoff: Dict[Node, torch.Tensor]  # Container for weights used by Submod 0 for the next iteration
    need_profile: bool = False


SchedulerType = Type["OffloadScheduler"]


class SchedulerFactory:
    _REGISTRY: Dict[str, SchedulerType] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(scheduler_cls: SchedulerType):
            cls._REGISTRY[name] = scheduler_cls
            return scheduler_cls

        return decorator

    @classmethod
    def create(cls, compile_config: CompileConfig, common_args: Dict[str, Any]) -> "OffloadScheduler":
        offload_policy_name = compile_config.offload_config.offload_policy.value

        scheduler_cls = cls._REGISTRY.get(offload_policy_name)
        if not scheduler_cls:
            raise ValueError(f"Unknown offload policy: {offload_policy_name}. " f"Available: {list(cls._REGISTRY.keys())}")

        return scheduler_cls(compile_config=compile_config, **common_args)


class OffloadScheduler(abc.ABC):
    def __init__(
        self,
        compile_config: CompileConfig,
        submod_nodes: List[Node],
        submod_weights_map: Dict[str, List[str]],
        name_node_map: Dict[str, Node],
        weight_sizes: Dict[str, int],
    ):
        """
        :param config: Compile configuration
        :param submod_nodes: List of submodule nodes in execution order
        :param submod_weights_map: {submod_name: [weight_names...]}
        :param name_node_map: {weight_name: weight_node} for looking up Node objects
        """
        self.compile_config = compile_config
        self.submod_nodes = submod_nodes
        self.submod_num = len(submod_nodes)
        self.submod_weights_map = submod_weights_map
        self.name_node_map = name_node_map
        self.weight_sizes = weight_sizes

        self.kept_weights: Set[str] = set()

    @abc.abstractmethod
    def schedule_kept_weights(self, profile_data: Dict[str, float]):
        """
        Static decision: determine which weights to keep on GPU based on profile data
        """
        pass

    @abc.abstractmethod
    def prefetch(self, current_node_name: str, ctx: OffloadRuntimeContext):
        """

        :param env: Tensor environment of the current execution environment
        :param h2d_stream: CUDA stream for data transfer
        :param buffers: GPU copies stored during Warmup
        :param submod_0_next_iter_weights: Container for weights used by Submod 0 for the next iteration
        """
        pass

    def is_kept(self, weight_name: str) -> bool:
        return weight_name in self.kept_weights

    def get_keep_weight_size(self) -> int:
        return sum(
            [
                self.name_node_map[w_name].meta.get("example_value").numel()
                * self.name_node_map[w_name].meta.get("example_value").element_size()
                for w_name in self.kept_weights
            ]
        )


@SchedulerFactory.register("BASE")
class BaseScheduler(OffloadScheduler):
    """
    Basic strategy implementation:
    1. schedule_kept_weights: Default to not keep any weights on GPU (gpu_resident_weight_ratio=0.0)
    2. prefetch: Inherit the original "Odd/Even + Next-Next" logic
    """

    def __init__(
        self,
        compile_config: CompileConfig,
        submod_nodes: List[Node],
        submod_weights_map: Dict[str, List[str]],
        name_node_map: Dict[str, Node],
        weight_sizes: Dict[str, int],
    ):
        super().__init__(compile_config, submod_nodes, submod_weights_map, name_node_map, weight_sizes)
        self.submod_cpu_names = set()

    def schedule_kept_weights(self, profile_data: Dict[str, float]):
        self.kept_weights.clear()
        # offload all submodules to CPU
        self.submod_cpu_names = set(self.submod_nodes[i].name for i in range(len(self.submod_nodes)))

    def prefetch(self, current_node_name: str, ctx: OffloadRuntimeContext):
        env = ctx.env
        h2d_stream = ctx.h2d_stream
        buffers = ctx.buffers
        submod_0_next_iter_weights = ctx.submod_0_handoff
        compute_stream = ctx.compute_stream

        try:
            idx = int(current_node_name.split('_')[-1])
        except ValueError:
            return

        max_lookahead = self.compile_config.offload_config.max_prefetch_lookahead
        target_node = None
        is_next_iter = False

        for offset in range(1, max_lookahead + 1):
            candidate_idx = (idx + offset) % self.submod_num
            candidate_node = self.submod_nodes[candidate_idx]

            if self.weight_sizes.get(candidate_node.name, 0) > 0:
                target_node = candidate_node

                if (idx + offset) >= self.submod_num:
                    is_next_iter = True
                else:
                    is_next_iter = False

                break

        if target_node is None:
            return

        last_target = getattr(self, "last_prefetched_target", None)

        if idx == 0:
            last_target = None

        if target_node.name == last_target:
            return

        self.last_prefetched_target = target_node.name

        compute_stream.wait_stream(h2d_stream)

        weight_names = self.submod_weights_map.get(target_node.name, [])

        with torch.cuda.stream(h2d_stream):
            for w_name in weight_names:
                if w_name in self.kept_weights:
                    continue

                w_node = self.name_node_map.get(w_name)
                if not w_node:
                    continue

                if is_next_iter:
                    if w_name in buffers:
                        submod_0_next_iter_weights[w_node] = buffers[w_name].to("cuda", non_blocking=True)

                elif w_node in env and hasattr(env[w_node], 'device') and env[w_node].device.type == "cpu":
                    env[w_node] = env[w_node].to("cuda", non_blocking=True)


@SchedulerFactory.register("COST_EFFECTIVE")
class CostEffectiveScheduler(BaseScheduler):
    """
    Cost-effective strategy implementation:
    Determine which weights to keep on GPU based on the ratio of time to size
    """

    def schedule_kept_weights(self, profile_data: Dict[str, float]):
        weight_sizes = self.weight_sizes
        gpu_resident_weight_ratio = self.compile_config.offload_config.gpu_resident_weight_ratio
        if gpu_resident_weight_ratio <= 0.0:
            return

        candidates = []
        total_weight_size = 0

        for name, timing in profile_data.items():
            if name == "h2d_bandwidth":
                continue
            duration = timing.get("compute", 0.0)
            size = weight_sizes.get(name, 0)
            ratio = duration / size if size > 0 else float('inf')
            candidates.append((name, ratio, size))
            total_weight_size += size

        candidates.sort(key=lambda x: x[1])

        current_kept_size = 0
        limit = total_weight_size * gpu_resident_weight_ratio

        self.kept_weights.clear()

        for name, ratio, s in candidates:
            if name == "submod_0":
                continue
            if current_kept_size + s <= limit:
                current_kept_size += s
                for w_name in self.submod_weights_map.get(name, []):
                    self.kept_weights.add(w_name)
                self.submod_cpu_names.add(name)
            else:
                break

        magi_logger.info(f"schedule_kept_weights size {current_kept_size} keep_submod_names: {self.submod_cpu_names}")


@SchedulerFactory.register("HEURISTIC")
class HeuristicScheduler(BaseScheduler):
    def __init__(
        self,
        compile_config: CompileConfig,
        submod_nodes: List[Node],
        submod_weights_map: Dict[str, List[str]],
        name_node_map: Dict[str, Node],
        weight_sizes: Dict[str, int],
    ):
        super().__init__(compile_config, submod_nodes, submod_weights_map, name_node_map, weight_sizes)

        # safety margin (ms), default 0.1ms, to prevent pipeline bubbles due to profile fluctuations
        self.safety_margin = getattr(compile_config, "prefetch_margin_ms", 0.1)

        # prefetch schedule: { trigger_node_name : target_submod_index }
        # when execute the trigger_node_name, prefetch the target_submod_index
        self.prefetch_schedule = collections.defaultdict()
        self.submod_load_events: Dict[str, torch.cuda.Event] = collections.defaultdict(torch.cuda.Event)
        for i in range(len(self.submod_nodes)):
            self.submod_load_events[self.submod_nodes[i].name] = torch.cuda.Event()

    def schedule_kept_weights(self, profile_data: Dict[str, Any]):
        """
        Core logic: build the prefetch schedule based on the profile data.
        profile_data format:
        {
            "submod_name": {
                "compute": 10.5,
                "h2d": 5.2
            },
            "h2d_bandwidth": 5.2
        }
        """
        self.kept_weights.clear()
        self.prefetch_schedule.clear()
        weight_sizes = self.weight_sizes

        len(self.submod_nodes)
        compute_times = []
        h2d_times = []

        # GiB/s
        EST_BANDWIDTH = profile_data.get("h2d_bandwidth") * self.compile_config.offload_config.bandwidth_safety_factor

        for i, node in enumerate(self.submod_nodes):
            data = profile_data.get(node.name, {})
            c_time = data.get("compute", 0.0)

            w_size = weight_sizes.get(node.name, 0)
            # w_size is in bytes, EST_BANDWIDTH is in GB/s (1e9 bytes/s)
            # h_time should be in milliseconds
            h_time = (w_size / (EST_BANDWIDTH * 1e6)) if EST_BANDWIDTH > 0 else 0

            compute_times.append(c_time)
            h2d_times.append(h_time)

        keep_submod_names = collections.defaultdict(list)

        schedule_nodes = collections.defaultdict(dict)

        submod_cpu_name = set()
        cpu_weight_size = 0

        for i in range(len(self.submod_nodes) - 1, -1, -1):
            if weight_sizes.get(self.submod_nodes[i].name, 0.0) > 0.0:
                offload_idx = i
                break
        # traverse submod_nodes in reverse order to ensure the latest start of transmission
        i = offload_idx - 1
        schedule_node_compute_time = 0
        while i >= 0:
            end_idx = i
            schedule_node_compute_time = 0

            while schedule_node_compute_time < self.safety_margin + h2d_times[offload_idx] and i >= 0:
                schedule_node_compute_time += compute_times[i]
                i -= 1

            if i < 0 and schedule_node_compute_time < self.safety_margin + h2d_times[offload_idx]:
                break

            submod_cpu_name.add(self.submod_nodes[offload_idx].name)
            cpu_weight_size += weight_sizes.get(self.submod_nodes[offload_idx].name, 0)

            schedule_nodes[offload_idx].update(
                {
                    "compute_start_idx": i + 1,
                    "compute_end_idx": end_idx,
                    "h2d_idx": offload_idx,
                    "compute_time": schedule_node_compute_time,
                    "h2d_time": h2d_times[offload_idx],
                    "ratio": schedule_node_compute_time / (self.safety_margin + h2d_times[offload_idx]),
                }
            )

            offload_idx = i + 1

            while weight_sizes.get(self.submod_nodes[offload_idx].name, 0.0) == 0.0 and offload_idx <= end_idx:
                offload_idx += 1

        gpu_resident_weight_ratio = self.compile_config.offload_config.gpu_resident_weight_ratio

        total_weight_size = sum(weight_sizes.values())
        limit = total_weight_size * gpu_resident_weight_ratio
        keep_gpu_size = total_weight_size - cpu_weight_size

        while limit < keep_gpu_size:
            schedule_nodes_list = [(i, s_node["ratio"]) for i, s_node in schedule_nodes.items()]
            schedule_nodes_list.sort(key=lambda x: x[1], reverse=True)
            changed = False
            for s_node_idx, _ in schedule_nodes_list:
                s_node = schedule_nodes[s_node_idx]
                compute_start_idx = s_node["compute_start_idx"]

                for i in range(s_node["compute_start_idx"] + 1, s_node["compute_end_idx"] + 1):
                    if self.submod_nodes[i].name in submod_cpu_name or weight_sizes.get(self.submod_nodes[i].name, 0.0) == 0.0:
                        continue

                    offload_idx = i
                    schedule_node_compute_time = sum(compute_times[compute_start_idx:i])
                    schedule_nodes[offload_idx].update(
                        {
                            "compute_start_idx": compute_start_idx,
                            "compute_end_idx": max(i - 1, compute_start_idx),
                            "h2d_idx": offload_idx,
                            "compute_time": schedule_node_compute_time,
                            "h2d_time": h2d_times[offload_idx],
                            "ratio": schedule_node_compute_time / (self.safety_margin + h2d_times[offload_idx]),
                        }
                    )

                    submod_cpu_name.add(self.submod_nodes[i].name)
                    keep_gpu_size -= weight_sizes.get(self.submod_nodes[i].name, 0)

                    schedule_nodes[s_node_idx].update(
                        {
                            "compute_start_idx": i,
                            "compute_end_idx": s_node["compute_end_idx"],
                            "h2d_idx": s_node_idx,
                            "compute_time": s_node["compute_time"] - schedule_node_compute_time,
                            "h2d_time": h2d_times[s_node_idx],
                            "ratio": (s_node["compute_time"] - schedule_node_compute_time)
                            / (self.safety_margin + h2d_times[s_node_idx]),
                        }
                    )

                    changed = True
                    break

                if changed:
                    break

            if not changed:
                break

        self.submod_cpu_names = submod_cpu_name

        for i, s_node in schedule_nodes.items():
            self.prefetch_schedule[self.submod_nodes[s_node["compute_start_idx"]].name] = s_node["h2d_idx"]

        # add weights of submod that are not on CPU
        keep_submod_names = set()
        for i in range(len(self.submod_nodes)):
            if self.submod_nodes[i].name not in submod_cpu_name:
                for w_name in self.submod_weights_map.get(self.submod_nodes[i].name, []):
                    self.kept_weights.add(w_name)
                keep_submod_names.add(self.submod_nodes[i].name)

    def prefetch(self, current_node_name: str, ctx: OffloadRuntimeContext):
        """
        1. check if the current node is on CPU, if yes, wait for the submod load event
        2. if not on CPU, skip
        3. prefetch the weights of the submod that the current node is pointing to
        """
        if ctx.need_profile:
            return super().prefetch(current_node_name, ctx)

        env = ctx.env
        h2d_stream = ctx.h2d_stream
        buffers = ctx.buffers
        submod_0_next_iter_weights = ctx.submod_0_handoff
        compute_stream = ctx.compute_stream

        if current_node_name in self.submod_cpu_names:
            compute_stream.wait_event(self.submod_load_events[current_node_name])

        h2d_idx = self.prefetch_schedule.get(current_node_name, None)
        if h2d_idx is None:
            return

        compute_complete_event = torch.cuda.Event()
        compute_complete_event.record(compute_stream)
        h2d_stream.wait_event(compute_complete_event)

        h2d_node = self.submod_nodes[h2d_idx]
        is_last_step = h2d_idx == len(self.submod_nodes) - 1

        weight_names = self.submod_weights_map.get(h2d_node.name, [])

        with torch.cuda.stream(h2d_stream):
            for w_name in weight_names:
                if w_name in self.kept_weights:
                    continue

                w_node = self.name_node_map.get(w_name)

                if w_node in env and hasattr(env[w_node], 'device') and env[w_node].device.type == "cpu":
                    env[w_node] = env[w_node].to("cuda", non_blocking=True)
                elif is_last_step and w_name in buffers:
                    submod_0_next_iter_weights[w_node] = buffers[w_name].to("cuda", non_blocking=True)

            self.submod_load_events[h2d_node.name].record(h2d_stream)
