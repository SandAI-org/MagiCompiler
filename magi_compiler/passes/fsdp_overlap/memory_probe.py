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

"""Where the estimated peak memory of a schedule comes from (``fsdp_config.memory_probe``).

A reorder pass that leaves the order alone and logs Inductor's own estimate
for the order it is handed: the peak, the snode it falls on, and the bytes live
there split by what produced them.  Installed between the overlap passes it
shows what each one did to the peak; Inductor's own before/after estimate in
``reorder_compute_and_comm_for_overlap`` measures the list it started with,
not the reordered one, and cannot.

A gathered weight counts as *prefetched* at a step while none of its real
readers has run yet: that is the share the launch position, not the model,
decides.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.utils import contains_wait

from magi_compiler.utils import magi_logger

from ..snode_utils import is_multi_output, is_weight_gather
from ..weight_offload.schedule.h2d_snode import is_h2d_load

GiB = 2**30


@dataclass
class MemoryTimeline:
    """Inductor's memory events for one order, each tagged with its producer's kind.

    Events, not buffers: Inductor charges a multi-output kernel's whole allocation
    to the kernel and each unpack frees its own share, so no single buffer's
    lifetime says how many bytes are live.
    """

    curve: list[int]  # bytes live at each step (len = len(order) + 1)
    events: list[tuple[int, str, int]] = field(default_factory=list)  # (step, kind, +alloc / -free)
    prefetch: list[tuple[int, int, int]] = field(default_factory=list)  # gathered bytes over [launch, first read)

    @property
    def peak(self) -> int:
        return max(self.curve, default=0)

    @property
    def peak_step(self) -> int:
        return max(range(len(self.curve)), key=self.curve.__getitem__) if self.curve else 0

    def live_at(self, step: int) -> dict[str, int]:
        out: dict[str, int] = {}
        for t, kind, delta in self.events:
            if t <= step:
                out[kind] = out.get(kind, 0) + delta
        early = sum(size for lo, hi, size in self.prefetch if lo <= step < hi)
        if early:
            out["ag_prefetched"] = early
            out["ag"] = out.get("ag", 0) - early
        return out

    def kind_peak(self, kind: str) -> tuple[int, int]:
        """(most bytes of ``kind`` live at once, the step it happens at)."""
        delta = [0] * (len(self.curve) + 1)
        for t, k, d in self.events:
            if k == kind:
                delta[t] += d
        best = best_step = cur = 0
        for t in range(len(self.curve)):
            cur += delta[t]
            if cur > best:
                best, best_step = cur, t
        return best, best_step


def _kind(snode: BaseSchedulerNode, kind_of_buf: dict) -> str:
    if is_weight_gather(snode):
        return "ag"
    if is_h2d_load(snode):
        return "h2d"
    if is_multi_output(snode):
        # The unpack of a multi-output kernel: the gathered or loaded bytes it frees
        # were allocated by its parent.
        for dep in snode.unmet_dependencies:
            if dep.name in kind_of_buf:
                return kind_of_buf[dep.name]
    return "other"


def memory_timeline(order: list[BaseSchedulerNode]) -> MemoryTimeline:
    """Inductor's estimate for ``order``, kept per event.  Needs ``V.graph``."""
    from torch._inductor.memory import compute_memory_timeline, get_freeable_input_buf
    from torch._inductor.utils import OrderedSet
    from torch._inductor.virtualized import V

    graph_inputs = OrderedSet(V.graph.graph_inputs.keys())
    graph_outputs = OrderedSet(V.graph.get_output_names())
    infos, node_to_step, _ = compute_memory_timeline(order, get_freeable_input_buf(order, graph_inputs), graph_outputs)

    kind_of_snode: dict[BaseSchedulerNode, str] = {}
    kind_of_buf: dict[str, str] = {}
    step_of_buf: dict[str, int] = {}
    for step, s in enumerate(order):
        k = _kind(s, kind_of_buf)
        kind_of_snode[s] = k
        for name in (*s.get_buffer_names(), s.get_name()):
            kind_of_buf[name] = k
            step_of_buf[name] = step

    curve_delta = [0] * (len(order) + 2)
    events, prefetch = [], []
    for info in infos:
        # A graph output is never freed (Inductor marks it end_step -1).
        end = info.end_step if info.end_step >= info.start_step else len(order)
        curve_delta[info.start_step] += info.size_alloc
        curve_delta[end + 1] -= info.size_free
        buf = info.buffer
        producer = getattr(buf, "defining_op", None)
        kind = kind_of_snode.get(producer, "input" if producer is None else "other")
        events.append((info.start_step, kind, info.size_alloc))
        events.append((end + 1, kind, -info.size_free))
        if kind != "ag":
            continue
        real = [
            node_to_step[u]
            for u in getattr(buf.mpi_buffer, "succ_nodes", ())
            if u in node_to_step and not contains_wait(u) and not is_multi_output(u)
        ]
        first_use = min(real, default=end)
        if info.size_alloc:
            prefetch.append((info.start_step, first_use, info.size_alloc))
        elif info.size_free and producer is not None:
            launched = min(
                (step_of_buf[d.name] for d in producer.unmet_dependencies if d.name in step_of_buf), default=info.start_step
            )
            prefetch.append((launched, first_use, info.size_free))

    curve, cur = [], 0
    for t in range(len(order) + 1):
        cur += curve_delta[t]
        curve.append(cur)
    return MemoryTimeline(curve=curve, events=events, prefetch=prefetch)


def describe(tag: str, order: list[BaseSchedulerNode], timeline: MemoryTimeline) -> str:
    step = timeline.peak_step
    live = timeline.live_at(step)
    at = order[step].get_name() if step < len(order) else "<end>"
    ag_peak, ag_step = timeline.kind_peak("ag")
    parts = ", ".join(
        f"{k} {live.get(k, 0) / GiB:.2f}" for k in ("ag_prefetched", "ag", "h2d", "input", "other") if live.get(k, 0)
    )
    return (
        f"memory probe [{tag}]: est. peak {timeline.peak / GiB:.2f} GiB at step {step}/{len(order)} ({at}); "
        f"live GiB there: {parts or 'none'}; most gathered weight live at once {ag_peak / GiB:.2f} GiB at step {ag_step}"
    )


class MemoryProbe:
    """Reorder pass that only logs; see the module docstring."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.name = f"memory probe [{tag}]"

    def __call__(self, snodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
        try:
            magi_logger.info("%s", describe(self.tag, snodes, memory_timeline(snodes)))
        except Exception as exc:  # noqa: BLE001 - a diagnostic never fails a compile
            magi_logger.warning("%s: could not estimate (%s)", self.name, exc)
        return snodes
