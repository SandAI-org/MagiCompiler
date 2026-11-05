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

"""NUMA topology lookups and the calling thread's page-placement policy (Linux)."""

from __future__ import annotations

import ctypes
import platform
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Literal, Sequence

import torch

from .logger import magi_logger

NumaPagePolicy = Literal["preferred", "interleave"]

# Kernel uapi linux/mempolicy.h.
_MPOL_MODE_BY_POLICY = {"preferred": 1, "interleave": 3}
_NODE_MASK_BITS = 1024
_NodeMask = ctypes.c_ulong * (_NODE_MASK_BITS // 64)


@dataclass(frozen=True)
class _MempolicySyscallNumbers:
    get_mempolicy: int
    set_mempolicy: int


# glibc wraps neither call, so they go through syscall(2); numbers from asm/unistd_64.h and asm-generic/unistd.h.
_MEMPOLICY_SYSCALL_NUMBERS_BY_MACHINE = {
    "x86_64": _MempolicySyscallNumbers(get_mempolicy=239, set_mempolicy=238),
    "aarch64": _MempolicySyscallNumbers(get_mempolicy=236, set_mempolicy=237),
}


@dataclass(frozen=True)
class _SavedMemoryPolicy:
    mode: int
    node_mask: ctypes.Array


def online_numa_nodes() -> list[int]:
    """Online NUMA nodes, e.g. ``[0, 1]``; ``[]`` when the kernel reports none."""
    try:
        with open("/sys/devices/system/node/online") as node_list_file:
            node_list = node_list_file.read().strip()
    except OSError:
        return []
    nodes = []
    for node_range in node_list.split(","):
        first_node, _, last_node = node_range.partition("-")
        nodes.extend(range(int(first_node), int(last_node or first_node) + 1))
    return nodes


def cuda_device_numa_node(device_index: int) -> int | None:
    """The NUMA node a CUDA device's PCIe link attaches to, or None when the platform does not say."""
    properties = torch.cuda.get_device_properties(device_index)
    pci_address = f"{properties.pci_domain_id:04x}:{properties.pci_bus_id:02x}:{properties.pci_device_id:02x}.0"
    try:
        with open(f"/sys/bus/pci/devices/{pci_address}/numa_node") as numa_node_file:
            numa_node = int(numa_node_file.read())
    except OSError:
        return None
    # -1 is the kernel's "no NUMA affinity".
    return numa_node if numa_node >= 0 else None


@contextmanager
def numa_page_policy(*, policy: NumaPagePolicy, nodes: Sequence[int]) -> Iterator[None]:
    """Place pages this thread allocates inside the block per ``policy`` over ``nodes``,
    then restore the thread's previous policy (e.g. one set by ``numactl``).

    Placement only affects performance, so when the policy cannot be set the block
    still runs with the current policy, after a warning.
    """
    saved_policy = _replace_thread_memory_policy(policy=policy, nodes=nodes)
    try:
        yield
    finally:
        if saved_policy is not None:
            _set_thread_memory_policy(mode=saved_policy.mode, node_mask=saved_policy.node_mask)


def _replace_thread_memory_policy(*, policy: NumaPagePolicy, nodes: Sequence[int]) -> _SavedMemoryPolicy | None:
    """Install ``policy``; return the policy it replaced, or None if nothing was installed."""
    syscall_numbers = _MEMPOLICY_SYSCALL_NUMBERS_BY_MACHINE.get(platform.machine())
    if syscall_numbers is None:
        magi_logger.warning("numa: no mempolicy syscalls known for %s; pages keep their placement", platform.machine())
        return None
    saved_policy = _get_thread_memory_policy()
    requested_node_mask = _NodeMask()
    for node in nodes:
        requested_node_mask[node // 64] |= 1 << (node % 64)
    if saved_policy is None or not _set_thread_memory_policy(mode=_MPOL_MODE_BY_POLICY[policy], node_mask=requested_node_mask):
        magi_logger.warning(
            "numa: could not set the %s policy over nodes %s (errno %d); pages keep their placement",
            policy,
            list(nodes),
            ctypes.get_errno(),
        )
        return None
    return saved_policy


def _get_thread_memory_policy() -> _SavedMemoryPolicy | None:
    mode = ctypes.c_int()
    node_mask = _NodeMask()
    syscall_number = _MEMPOLICY_SYSCALL_NUMBERS_BY_MACHINE[platform.machine()].get_mempolicy
    # get_mempolicy(mode, nodemask, maxnode, addr=NULL, flags=0): the calling thread's policy.
    if _libc().syscall(ctypes.c_long(syscall_number), ctypes.byref(mode), node_mask, _max_node(), None, ctypes.c_ulong(0)):
        return None
    return _SavedMemoryPolicy(mode=mode.value, node_mask=node_mask)


def _set_thread_memory_policy(*, mode: int, node_mask: ctypes.Array) -> bool:
    syscall_number = _MEMPOLICY_SYSCALL_NUMBERS_BY_MACHINE[platform.machine()].set_mempolicy
    return _libc().syscall(ctypes.c_long(syscall_number), ctypes.c_int(mode), node_mask, _max_node()) == 0


def _max_node() -> ctypes.c_ulong:
    # One past the last mask bit, as libnuma passes it.
    return ctypes.c_ulong(_NODE_MASK_BITS + 1)


def _libc() -> ctypes.CDLL:
    return ctypes.CDLL(None, use_errno=True)
