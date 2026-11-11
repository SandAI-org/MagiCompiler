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

"""The compute time transfers on one shared link leave free for the next ones placed.

Two transfers on one physical link slow each other down, so ``LinkSharing`` makes
them take turns: every collective in ``LINK_COLLECTIVE_OPS`` (weight all-gathers,
graph all-to-alls) queues on one link stream (``RouteCollectivesToLinkStream``),
and without NVLink H2D loads queue there too.  A pass placing transfers on such a
link then sees the collectives already there as busy spans of the compute time
axis: it places its transfers on the time they leave free.  A transfer longer than
the gap between two of them may run into the next one, which on one stream holds
that collective back; the scheduler charges the overrun as exposure.
"""

from __future__ import annotations

import enum
from bisect import bisect_right
from typing import Collection, Sequence

from torch._inductor.scheduler import BaseSchedulerNode

from ..snode_utils import link_collective_node
from .cost import CostView
from .graph import SnodeGraph


class LinkSharing(enum.Enum):
    """Which transfers take turns on one link; the value names the link-stream op overload."""

    OFF = "off"  # every transfer keeps its own stream and is placed as if alone
    PCIE = "pcie"  # no NVLink: loads, all-gathers and all-to-alls all cross the GPU's PCIe link
    NVLINK = "nvlink"  # all-gathers and all-to-alls share NVLink; loads keep PCIe to themselves

    @property
    def collectives_take_turns(self) -> bool:
        return self is not LinkSharing.OFF

    @property
    def loads_take_turns_with_collectives(self) -> bool:
        return self is LinkSharing.PCIE


def link_busy_spans_ns(
    graph: SnodeGraph,
    cost: CostView,
    compute_prefix_ns: Sequence[float],
    *,
    excluded_snodes: Collection[BaseSchedulerNode] = (),
) -> list[tuple[float, float]]:
    """Where each link collective, other than ``excluded_snodes``, occupies the link on the compute time axis.

    The collectives run back to back in program order, each from its launch for its
    profiled time.  A span ends at the collective's first wait at the latest: compute
    stalls there until the collective is done, so its remainder takes no compute time.
    """
    busy_spans = []
    link_free_ns = 0.0
    for snode in graph.order:
        if snode in excluded_snodes or link_collective_node(snode) is None:
            continue
        start_ns = max(compute_prefix_ns[graph.original_index_by_node[snode]], link_free_ns)
        end_ns = start_ns + cost(snode)
        waits = graph.waits_of(snode)
        if waits:
            first_wait_ns = compute_prefix_ns[graph.original_index_by_node[waits[0]]]
            end_ns = max(start_ns, min(end_ns, first_wait_ns))
        busy_spans.append((start_ns, end_ns))
        link_free_ns = end_ns
    return busy_spans


def free_time_axis(
    compute_prefix_ns: Sequence[float], busy_spans: Sequence[tuple[float, float]]
) -> tuple[list[float], list[tuple[float, float]]]:
    """The compute axis with every busy span cut out, and each span's position on it.

    A span collapses to a point on the free axis; as a blocked window it ends the
    gap a transfer hides in, since on one stream the collective waits for it.
    ``busy_spans`` must be sorted and disjoint, as ``link_busy_spans_ns`` returns them.
    """
    span_starts = [start_ns for start_ns, _ in busy_spans]
    busy_ns_before_span = [0.0]
    for start_ns, end_ns in busy_spans:
        busy_ns_before_span.append(busy_ns_before_span[-1] + end_ns - start_ns)

    def free_ns_at(time_ns: float) -> float:
        span_index = bisect_right(span_starts, time_ns) - 1
        if span_index < 0:
            return time_ns
        start_ns, end_ns = busy_spans[span_index]
        return time_ns - busy_ns_before_span[span_index] - (min(time_ns, end_ns) - start_ns)

    free_prefix_ns = [free_ns_at(time_ns) for time_ns in compute_prefix_ns]
    blocked_windows_ns = [(free_ns_at(start_ns), free_ns_at(end_ns)) for start_ns, end_ns in busy_spans]
    return free_prefix_ns, blocked_windows_ns
