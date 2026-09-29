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

"""Shared scheduling core for transfer/compute overlap passes.

A pass that hides some transfer behind compute -- an FSDP weight all-gather, a
host-to-device weight load -- subclasses ``OverlapPass``, turns each transfer
into a ``TransferTask``, and places the tasks with one of the strategies in
``scheduler``.  Indexing, cost reads, rebuild/validation, in-flight memory
accounting and cross-rank agreement are written once, here.

* ``graph``     -- ``SnodeGraph``: the order, real-dep readers, rebuild, validate.
* ``cost``      -- ``CostView``: clamped runtimes, the compute time axis.
* ``task``      -- ``TransferTask``: one transfer's placement problem and answer.
* ``scheduler`` -- ``alap_schedule`` (time axis) and ``index_sweep`` (snode axis).
* ``memory``    -- ``InflightMap`` and device-peak accounting.
* ``rank_sync`` -- keeping a collective-moving placement identical on every rank.
* ``base``      -- ``OverlapPass``: the wants/plan/place/commit/report template.
"""

from .base import DEFAULT_WINDOW_MARGIN_NS, OverlapPass
from .cost import CostView
from .graph import SnodeGraph
from .memory import InflightMap, device_peak, inflight_peak, live_starts, peak_point, peak_with
from .scheduler import alap_schedule, enforce_monotone, index_sweep, issue_index, latest_finish, unhoisted_index
from .task import TransferTask

__all__ = [
    "DEFAULT_WINDOW_MARGIN_NS",
    "OverlapPass",
    "CostView",
    "SnodeGraph",
    "TransferTask",
    "InflightMap",
    "device_peak",
    "inflight_peak",
    "live_starts",
    "peak_point",
    "peak_with",
    "alap_schedule",
    "enforce_monotone",
    "index_sweep",
    "issue_index",
    "latest_finish",
    "unhoisted_index",
]
