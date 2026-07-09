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

"""Whole-graph FSDP all-gather / compute overlap (``enable_fsdp_fullgraph_overlap``).

The pipeline, all for the ``disable_graph_split=True`` path (there is no FX-level
graph split here despite the historical name):

1. :mod:`.redistribute_lowering` -- lower SimpleFSDP weight ``prim_redistribute``
   into explicit ``all_gather`` + ``wait`` collectives.
2. :mod:`.bucket_all_gather` -- coalesce the per-region weight gathers into one
   ``all_gather_into_tensor_coalesced`` per (region, group, dtype), optionally
   size-capped. :mod:`.lower_and_bucket` is the entry point wrapping steps 1-2.
3. :mod:`.reorder` -- the scheduler-level ``FsdpOverlapReorder`` pass that hoists
   each weight all-gather launch into upstream compute so the collective is hidden.

The per-op cost model that feeds the reorder its compute/comm timings lives in the
general :mod:`magi_compiler.profiling` package (``ProfilingRuntimeEstimator`` +
``register_benchmark_inputs``), not here -- it is not FSDP-specific.
"""

from .bucket_all_gather import bucket_weight_all_gather_coalesced_per_region
from .lower_and_bucket import lower_and_bucket_full_graph
from .redistribute_lowering import lower_prim_redistribute_to_collectives
from .reorder import FsdpOverlapReorder

__all__ = [
    "bucket_weight_all_gather_coalesced_per_region",
    "lower_prim_redistribute_to_collectives",
    "lower_and_bucket_full_graph",
    "FsdpOverlapReorder",
]
