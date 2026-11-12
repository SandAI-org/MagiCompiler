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

import ctypes
from functools import cached_property

from .base import Platform


class MusaPlatform(Platform):
    device_type = "musa"
    dispatch_key = "PrivateUse1"
    # Under the default "global" mode torch_musa fails every capture, even a bare ``x * 2``.
    graph_capture_error_mode = "relaxed"
    runtime_pointwise_autotune = False
    # MCCL hangs on all_gather_into_tensor_coalesced; keep FSDP weight gathers unbucketed (plain all_gather works).
    coalesced_all_gather = False

    def __init__(self) -> None:
        import torch_musa  # noqa: F401  registers torch.musa and the "musa" device type

    @property
    def graph_cls(self) -> type:
        return self.MUSAGraph

    @cached_property
    def _musart(self) -> ctypes.CDLL:
        return ctypes.CDLL("libmusart.so")

    def host_register(self, ptr: int, nbytes: int) -> None:
        # torch_musa reports memory registered before its device context exists as unpinned.
        self.current_stream()
        err = self._musart.musaHostRegister(ctypes.c_void_p(ptr), ctypes.c_size_t(nbytes), 0)
        if err != 0:
            raise RuntimeError(f"musaHostRegister failed with error {err}")
