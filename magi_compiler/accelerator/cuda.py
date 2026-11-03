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

from functools import cached_property

import torch

from .base import Platform


class CudaPlatform(Platform):
    device_type = "cuda"
    nvtx = torch.cuda.nvtx
    graph_cls = torch.cuda.CUDAGraph
    has_symm_mem = True

    @cached_property
    def sm(self) -> tuple[int, int]:
        return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else (0, 0)

    def host_register(self, ptr: int, nbytes: int) -> None:
        torch.cuda.check_error(torch.cuda.cudart().cudaHostRegister(ptr, nbytes, 0))

    def emit_nvtx(self, record_shapes: bool = False):
        return torch.autograd.profiler.emit_nvtx(record_shapes=record_shapes)

    def profiler_start(self) -> None:
        torch.cuda.cudart().cudaProfilerStart()

    def profiler_stop(self) -> None:
        torch.cuda.cudart().cudaProfilerStop()
