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

from typing import Dict

import torch

from magi_compiler.utils.device import accelerator_module, accelerator_type


class OffloadProfiler:
    def __init__(self):
        self.device_module = accelerator_module()
        self.compute_events: Dict[str, Dict[str, torch.cuda.Event]] = {}
        self.timings: Dict[str, Dict[str, float]] = {}

    def start_compute_profile(self, name: str, stream: torch.cuda.Stream):
        if name not in self.compute_events:
            self.compute_events[name] = {}
        start_event = self.device_module.Event(enable_timing=True)
        start_event.record(stream)
        self.compute_events[name]["start"] = start_event

    def end_compute_profile(self, name: str, stream: torch.cuda.Stream):
        end_event = self.device_module.Event(enable_timing=True)
        end_event.record(stream)
        self.compute_events[name]["end"] = end_event

    def get_h2d_bandwidth(self, size_mb=1024, iters=3, warmup=3, dtype=torch.float32, device=None):
        device = device or accelerator_type()
        self.device_module.synchronize()

        num_elements = size_mb * 1024 * 1024 // torch.tensor([], dtype=dtype).element_size()

        cpu_tensor = torch.empty(num_elements, dtype=dtype, pin_memory=True)
        gpu_tensor = torch.empty(num_elements, dtype=dtype, device=device)

        # warmup
        for _ in range(warmup):
            gpu_tensor.copy_(cpu_tensor, non_blocking=True)
        self.device_module.synchronize()

        start_event = self.device_module.Event(enable_timing=True)
        end_event = self.device_module.Event(enable_timing=True)

        start_event.record()

        for _ in range(iters):
            gpu_tensor.copy_(cpu_tensor, non_blocking=True)

        end_event.record()
        self.device_module.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event)
        elapsed_s = elapsed_ms / 1000.0

        total_bytes = size_mb * 1024 * 1024 * iters
        bandwidth = total_bytes / elapsed_s / 1e9  # GB/s

        return bandwidth

    def broadcast_obj(self, obj, src=0):
        obj_list = [obj]
        torch.distributed.broadcast_object_list(obj_list, src=src)
        return obj_list[0]

    def summarize(self) -> Dict[str, Dict[str, float]]:
        self.device_module.synchronize()
        results = {}
        for name, evs in self.compute_events.items():
            if name not in results:
                results[name] = {}
            if "start" in evs and "end" in evs:
                results[name]["compute"] = evs["start"].elapsed_time(evs["end"])

        h2d_bandwidth = self.get_h2d_bandwidth()
        results["h2d_bandwidth"] = h2d_bandwidth

        if torch.distributed.is_initialized():
            h2d_bandwidth = self.broadcast_obj(h2d_bandwidth)
            results = self.broadcast_obj(results)

        self.timings = results
        return results
