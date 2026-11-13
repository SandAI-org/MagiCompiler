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

from typing import Optional, Tuple


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


def has_active_nvlink(device: int) -> Optional[bool]:
    """Whether NVML reports an active NVLink on this CUDA device; None when NVML cannot tell.

    Looked up by PCI bus id, so ``CUDA_VISIBLE_DEVICES`` renumbering does not matter.
    """
    try:
        import pynvml
        import torch as _torch

        props = _torch.cuda.get_device_properties(device)
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
