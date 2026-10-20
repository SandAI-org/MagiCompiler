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

"""Node-meta keys the offload passes share.

``node.meta`` is the channel from the FX passes to the scheduler passes (through
``meta["custom"]``, see below).  Keys live here so a typo cannot silently leave a
weight on the device.
"""

from __future__ import annotations

import torch.fx as fx

# On the node that produces a weight's shard: the host-pool slot its bytes were
# moved to.  Set by binding only.
HOST_SLOT = "magi_host_offload_slot"

# Local bucket planning separates gathers by this tag; rank 0's adopted plan may mix them.
# Kept under ``meta["custom"]``, the field AOT preserves into scheduler-node origins.
HOST_OFFLOADED = "magi_host_offloaded"


def host_slot(node: fx.Node) -> int | None:
    """The host-pool slot of the shard this node produces, or None."""
    return node.meta.get(HOST_SLOT)


def is_host_offloaded(node: fx.Node) -> bool:
    return bool((node.meta.get("custom") or {}).get(HOST_OFFLOADED))


def mark_host_slot(node: fx.Node, slot: int) -> None:
    node.meta[HOST_SLOT] = slot


def mark_host_offloaded(node: fx.Node) -> None:
    node.meta["custom"] = {**(node.meta.get("custom") or {}), HOST_OFFLOADED: True}
