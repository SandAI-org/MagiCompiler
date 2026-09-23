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

"""Compile-time CPU offload of model weights.

What a weight looks like in the graph is a ``WeightSource`` --
``PlainParamSource`` for an unsharded model, ``FsdpShardSource`` for a
SimpleFSDP one.  Everything else -- the host pool, the ``magi::h2d_load``
splice, the placement pass -- is written once against that.
"""

from .binder import apply_weight_offload, bind_weights_to_host, insert_h2d_loads
from .h2d_reorder import H2dLoadReorder
from .host_first import handoff_if_pending, patch_materialize
from .node_meta import HOST_OFFLOADED, HOST_SLOT, host_slot, is_host_offloaded, mark_host_offloaded, mark_host_slot
from .offload_cache import CacheValidity, OffloadCache
from .ops import H2D_OPS, is_h2d_load, slots_of
from .sources import FsdpShardSource, OffloadCandidate, PlainParamSource, WeightSource, shard_holder

__all__ = [
    "apply_weight_offload",
    "bind_weights_to_host",
    "insert_h2d_loads",
    "CacheValidity",
    "OffloadCache",
    "H2dLoadReorder",
    "FsdpShardSource",
    "OffloadCandidate",
    "PlainParamSource",
    "WeightSource",
    "shard_holder",
    "HOST_OFFLOADED",
    "HOST_SLOT",
    "H2D_OPS",
    "handoff_if_pending",
    "host_slot",
    "patch_materialize",
    "is_h2d_load",
    "is_host_offloaded",
    "mark_host_offloaded",
    "mark_host_slot",
    "slots_of",
]
