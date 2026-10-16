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

"""GPU device introspection helpers.

Centralised so that pass-manager / FX passes / runtime modules don't all
re-implement the same try/except dance around ``torch.cuda``.
"""

from typing import Tuple


def device_capability(device: int = 0) -> Tuple[int, int]:
    """Return ``(major, minor)`` for the given CUDA device.

    Falls back to ``(0, 0)`` when CUDA is unavailable / not initialised /
    raises any error during introspection — callers compare against a
    minimum cap so a zero pair always means "feature unsupported", which
    is the safe behaviour on CPU-only hosts and during static analysis.
    """
    try:
        import torch as _torch

        if _torch.cuda.is_available():
            return _torch.cuda.get_device_capability(device)
    except Exception:
        pass
    return (0, 0)


def device_capability_major(device: int = 0) -> int:
    """Convenience wrapper: just the major-capability int (0 if no CUDA)."""
    return device_capability(device)[0]


def accelerator_module():
    """``torch.cuda`` on NVIDIA GPUs, the out-of-tree backend module (e.g. ``torch_musa``) elsewhere."""
    import torch as _torch

    return _torch.get_device_module()


def accelerator_type() -> str:
    """Device type of the current accelerator, e.g. ``"cuda"`` or ``"musa"``."""
    import torch as _torch

    acc = _torch.accelerator.current_accelerator()
    return acc.type if acc is not None else "cuda"


def is_host_scalar(val) -> bool:
    """0-dim CPU tensor, e.g. a Python float lifted by Dynamo.

    Device kernels accept these as-is, and they must stay on the host: a fake one moved to the device loses
    its constant value, so a later ``.item()`` becomes data-dependent.
    """
    import torch as _torch

    return isinstance(val, _torch.Tensor) and val.dim() == 0 and val.device.type == "cpu"
