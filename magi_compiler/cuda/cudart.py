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
import os

import torch

from magi_compiler.utils.device import accelerator_type

_cudart = None
_musart = None


def init_cudart():
    global _cudart
    if _cudart is not None:
        return _cudart
    candidates = ["libcudart.so", "libcudart.so.11.0", "libcudart.so.12.0, libcudart.so.13"]
    try:
        cuda_path = os.path.dirname(torch.utils.cpp_extension._find_cuda_home())
        candidates.append(os.path.join(cuda_path, "lib64", "libcudart.so"))
    except:
        pass
    for lib in candidates:
        try:
            _cudart = ctypes.CDLL(lib)
            return _cudart
        except OSError:
            continue
    return None


def init_musart():
    global _musart
    if _musart is None:
        try:
            _musart = ctypes.CDLL("libmusart.so")
        except OSError:
            pass
    return _musart


def pin_memory_in_place(tensor: torch.Tensor):
    """
    Pin memory in-place using cudaHostRegister (musaHostRegister on MUSA).
    """
    if tensor.device.type != "cpu":
        return tensor
    # Pick by accelerator, not by which library loads: a MUSA host may also ship libcudart.
    if accelerator_type() == "musa":
        lib = init_musart()
        host_register = lib.musaHostRegister if lib is not None else None
    else:
        lib = init_cudart()
        host_register = lib.cudaHostRegister if lib is not None else None
    if host_register is None:
        return tensor

    ptr = tensor.data_ptr()
    size = tensor.numel() * tensor.element_size()
    res = host_register(ctypes.c_void_p(ptr), ctypes.c_size_t(size), 0)

    # 0: success, 712: already registered
    # https://www.autodesk.com/support/technical/article/caas/sfdcarticles/sfdcarticles/GPU-and-CUDA-call-failed-error-messages-with-Arnold-5-3-5-and-Maya-2024.html
    if res == 0 or res == 712:
        return tensor
    else:
        raise RuntimeError(f"{host_register.__name__} failed with error code {res}")
