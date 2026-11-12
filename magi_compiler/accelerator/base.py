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

import contextlib
from typing import Optional

import torch


class _NoNvtx:
    """``torch.cuda.nvtx`` surface for devices without NVTX: profiler ranges become no-ops."""

    @staticmethod
    def range_push(msg: str) -> None:
        pass

    @staticmethod
    def range_pop() -> None:
        pass


class Platform:
    """Device facts magi_compiler branches on; any other attribute forwards to ``torch.<device_type>``
    (``PLATFORM.synchronize()``, ``PLATFORM.Stream``, ``PLATFORM.graph(...)``...)."""

    device_type: str
    # Dispatch key of its tensors, which custom-op kernels register under.
    dispatch_key: str
    nvtx = _NoNvtx
    # CUDA compute capability, which the CUTLASS / TMA paths gate on; (0, 0) keeps them all off.
    sm: tuple[int, int] = (0, 0)
    # Symmetric memory, which fsdp_config.transport="copy_engine" needs.
    has_symm_mem = False
    # torch.cuda.graph's default.
    graph_capture_error_mode = "global"

    def device(self, index: Optional[int] = None) -> torch.device:
        return torch.device(self.device_type, self.current_device() if index is None else index)

    def host_register(self, ptr: int, nbytes: int) -> None:
        """Page-lock ``nbytes`` of host memory at ``ptr`` in place."""
        raise NotImplementedError

    def emit_nvtx(self, record_shapes: bool = False) -> contextlib.AbstractContextManager:
        """NVTX ranges around every autograd op, for nsys."""
        return contextlib.nullcontext()

    def profiler_start(self) -> None:
        """Open the vendor profiler's capture window (nsys ``--capture-range``)."""

    def profiler_stop(self) -> None:
        pass

    def __getattr__(self, name: str):
        return getattr(torch.get_device_module(self.device_type), name)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.device_type})"
