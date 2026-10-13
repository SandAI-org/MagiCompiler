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

Layout, in the order a compile meets it:

* ``host_first``  -- model build: weights materialize in pinned host memory.
* ``graph/``      -- FX weight pipeline: bind parked weights, splice loads in.
* ``schedule/``   -- Inductor scheduler pass: place each load, buy residency.
* ``cache/``      -- host-slot sidecar next to the piecewise compile cache.
* ``runtime/``    -- what the compiled graph calls: host pool, ``magi::h2d_load``.
"""

from .cache.offload_cache import CacheValidity, OffloadCache
from .graph.bind import bind_weights_to_host, insert_h2d_loads
from .graph.weight_source import FsdpShardSource, PlainParamSource
from .host_first import handoff_if_pending, patch_materialize
from .node_meta import is_host_offloaded
from .runtime import host_pool
from .schedule.h2d_reorder import H2dLoadReorder

__all__ = [
    "bind_weights_to_host",
    "insert_h2d_loads",
    "CacheValidity",
    "OffloadCache",
    "H2dLoadReorder",
    "FsdpShardSource",
    "PlainParamSource",
    "handoff_if_pending",
    "host_pool",
    "patch_materialize",
    "is_host_offloaded",
]
