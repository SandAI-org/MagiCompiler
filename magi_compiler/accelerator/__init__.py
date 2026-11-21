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

"""Process-wide accelerator: the only place magi_compiler asks which device it runs on.

MUSA wins when ``torch_musa`` is installed, CUDA otherwise, CPU-only hosts included (there every capability reads as
absent). Detection only looks for the package, so importing this module neither initializes the device runtime nor
breaks ``fork``.
"""

import functools
import importlib.util

import torch

from .base import Platform

__all__ = ["PLATFORM", "Platform", "get_platform", "pin_memory_in_place"]


@functools.cache
def get_platform() -> Platform:
    if importlib.util.find_spec("torch_musa"):
        from .musa import MusaPlatform

        return MusaPlatform()
    from .cuda import CudaPlatform

    return CudaPlatform()


def pin_memory_in_place(tensor: torch.Tensor) -> torch.Tensor:
    """Page-lock a CPU tensor's storage where it is, without the copy ``Tensor.pin_memory()`` makes.

    Pinned tensors are skipped: registering memory twice leaves a sticky runtime error behind that the next kernel
    launch reports as its own.
    """
    if tensor.device.type == "cpu" and not tensor.is_pinned():
        get_platform().host_register(tensor.data_ptr(), tensor.numel() * tensor.element_size())
    return tensor


def __getattr__(name: str):
    if name == "PLATFORM":
        return get_platform()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
