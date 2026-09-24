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

``node.meta`` is the only channel that survives bucketing, which rebuilds nodes.
Keys live here so a typo cannot silently leave a weight on the device.
"""

from __future__ import annotations

import torch.fx as fx

# On the node that produces a weight's shard: the host-pool slot its bytes were
# moved to.  Set by binding only.
HOST_SLOT = "magi_host_offload_slot"

# On a node downstream of an offloaded weight, so a later pass can keep
# offloaded and resident work apart -- the FSDP bucketing reads this to avoid
# putting both kinds in one all-gather.
HOST_OFFLOADED = "magi_host_offloaded"


def host_slot(node: fx.Node) -> int | None:
    """The host-pool slot of the shard this node produces, or None."""
    return node.meta.get(HOST_SLOT)


def is_host_offloaded(node: fx.Node) -> bool:
    return bool(node.meta.get(HOST_OFFLOADED))


def mark_host_slot(node: fx.Node, slot: int) -> None:
    node.meta[HOST_SLOT] = slot


def mark_host_offloaded(node: fx.Node) -> None:
    node.meta[HOST_OFFLOADED] = True
