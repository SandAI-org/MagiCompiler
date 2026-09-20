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

"""Node-meta keys the FSDP overlap passes share.

``node.meta`` is the only channel that survives bucketing, which rebuilds nodes.
Keys live here so a typo cannot silently drop a weight back to NCCL.
"""

from __future__ import annotations

import torch.fx as fx

# All-gather that gathers a SimpleFSDP weight.
WEIGHT_AG = "magi_fsdp_weight_ag"

# Weight gather whose Shard(0) does not divide across the mesh.
UNEVEN_SHARD = "magi_fsdp_uneven_shard"

# Weight gather whose shard now lives in symmetric memory. Set by binding only.
CE_BOUND = "magi_ce_bound"


def is_weight_ag(node: fx.Node) -> bool:
    return bool(node.meta.get(WEIGHT_AG))


def is_uneven_shard(node: fx.Node) -> bool:
    return bool(node.meta.get(UNEVEN_SHARD))


def is_ce_bound(node: fx.Node) -> bool:
    return bool(node.meta.get(CE_BOUND))


def mark_weight_ag(node: fx.Node, *, uneven: bool) -> None:
    """Tag a newly built all-gather as a weight gather.

    ``uneven`` is keyword-only: omitting it would default to the unsafe answer.
    """
    node.meta[WEIGHT_AG] = True
    node.meta[UNEVEN_SHARD] = uneven


def mark_ce_bound(node: fx.Node) -> None:
    node.meta[CE_BOUND] = True
