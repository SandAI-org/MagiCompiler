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
into a ``TransferTask``, and places the tasks with ``scheduler.schedule_transfers_alap``.  Indexing, cost reads, rebuild/validation, in-flight memory
accounting and cross-rank agreement are written once, here.

* ``graph``     -- ``SnodeGraph``: the order, real-dep readers, rebuild, topological check.
* ``cost``      -- ``CostView``: clamped runtimes, the compute time axis.
* ``task``      -- ``TransferTask`` (one transfer's problem) and ``ScheduleResult`` (the placement).
* ``scheduler`` -- ``schedule_transfers_alap``: the deadline chain on the compute time axis.
* ``memory``    -- ``InflightMemoryBudget`` and weight-storage peak accounting.
* ``rank_sync`` -- keeping a collective-moving placement identical on every rank.
* ``base``      -- ``OverlapPass``: the build-tasks / assign-launch-indices / commit / log template.
"""

from .base import DEFAULT_WINDOW_MARGIN_NS, OverlapCallState, OverlapPass
from .cost import CostView
from .graph import SnodeGraph
from .memory import (
    InflightMemoryBudget,
    inflight_peak_bytes,
    inflight_peak_bytes_and_index,
    peak_bytes_with_resident_weights,
    weight_storage_peak_bytes,
)
from .scheduler import (
    enforce_launch_order,
    latest_finish_times_ns,
    latest_legal_launch_index,
    schedule_transfers_alap,
    unhoisted_index,
)
from .task import ScheduleResult, TransferTask

__all__ = [
    "DEFAULT_WINDOW_MARGIN_NS",
    "OverlapCallState",
    "OverlapPass",
    "CostView",
    "SnodeGraph",
    "ScheduleResult",
    "TransferTask",
    "InflightMemoryBudget",
    "weight_storage_peak_bytes",
    "inflight_peak_bytes",
    "inflight_peak_bytes_and_index",
    "peak_bytes_with_resident_weights",
    "schedule_transfers_alap",
    "enforce_launch_order",
    "latest_legal_launch_index",
    "latest_finish_times_ns",
    "unhoisted_index",
]
