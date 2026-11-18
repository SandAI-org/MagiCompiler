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
from typing import Optional

import torch

from .base import Platform


class CudaPlatform(Platform):
    device_type = "cuda"
    dispatch_key = "CUDA"
    nvtx = torch.cuda.nvtx
    graph_cls = torch.cuda.CUDAGraph
    has_symm_mem = True

    @cached_property
    def sm(self) -> tuple[int, int]:
        return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else (0, 0)

    def host_register(self, ptr: int, nbytes: int) -> None:
        torch.cuda.check_error(torch.cuda.cudart().cudaHostRegister(ptr, nbytes, 0))

    def host_unregister(self, ptr: int) -> None:
        torch.cuda.cudart().cudaHostUnregister(ptr)

    def emit_nvtx(self, record_shapes: bool = False):
        return torch.autograd.profiler.emit_nvtx(record_shapes=record_shapes)

    def profiler_start(self) -> None:
        torch.cuda.cudart().cudaProfilerStart()

    def profiler_stop(self) -> None:
        torch.cuda.cudart().cudaProfilerStop()

    def has_active_nvlink(self, device: int) -> Optional[bool]:
        """Whether NVML reports an active NVLink on this CUDA device; None when NVML cannot tell.

        Looked up by PCI bus id, so ``CUDA_VISIBLE_DEVICES`` renumbering does not matter.
        """
        try:
            import pynvml

            props = torch.cuda.get_device_properties(device)
            pci_bus_id = f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
            pynvml.nvmlInit()
        except Exception:
            return None
        try:
            handle = pynvml.nvmlDeviceGetHandleByPciBusId(pci_bus_id)
            for link in range(pynvml.NVML_NVLINK_MAX_LINKS):
                try:
                    if pynvml.nvmlDeviceGetNvLinkState(handle, link) == pynvml.NVML_FEATURE_ENABLED:
                        return True
                except pynvml.NVMLError:
                    # NotSupported on a GPU without NVLink, InvalidArgument past its last link.
                    return False
            return False
        except pynvml.NVMLError:
            return None
        finally:
            pynvml.nvmlShutdown()
