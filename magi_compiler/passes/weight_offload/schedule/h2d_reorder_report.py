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

"""What ``H2dLoadReorder`` logs about the placement it settled on."""

from __future__ import annotations

import logging
from collections import defaultdict

from magi_compiler.utils import magi_logger

from .load_plan import device_peak, inflight_peak, live_starts


def magi_logger_enabled_for_debug() -> bool:
    """The per-load report builds a string per line; skip it when nobody reads it."""
    return logging.getLogger("magi_compiler").isEnabledFor(logging.DEBUG)


class H2dReorderReport:
    """Logging half of ``H2dLoadReorder``.

    Reads ``bandwidth_bytes_per_ns``, ``bus_utilization`` and
    ``_hideable_bytes`` off the reorder pass it is mixed into.
    """

    def _report(self, plans, targets, index_of, given_back, budget, n_snodes, prefix) -> None:
        resident = [p for p in plans if p.promoted]
        exposed = sum(p.exposed for p in plans)
        by_budget = sum(p.exposed for p in plans if p.budget_bound)
        inflight = inflight_peak(plans, live_starts(plans, targets, index_of))
        moved = sum(1 for load, target in targets.items() if target != index_of[load])
        on_bus = sum(p.nbytes for p in plans if not p.promoted)
        total = sum(p.nbytes for p in plans)
        hideable = self._hideable_bytes(plans, prefix)
        compute_ns = hideable / self.bus_utilization / self.bandwidth_bytes_per_ns
        magi_logger.info(
            "h2d load reorder: %.0fms of compute at %.1f GB/s can overlap %.1f MiB of the %.1f MiB "
            "offloaded (%.0f%% bus utilization assumed), so %.1f MiB had to become resident; it did "
            "over %d weight(s), leaving %.1f MiB on the bus, %.1fms of transfer. Hoisted %d/%d "
            "load(s); in-flight peak %.1f MiB of the %.0f MiB budget; %.1f MiB of weight on the "
            "device at peak; %.1fms still exposed%s",
            compute_ns / 1e6,
            self.bandwidth_bytes_per_ns,
            hideable / 2**20,
            total / 2**20,
            self.bus_utilization * 100,
            given_back / 2**20,
            len(resident),
            on_bus / 2**20,
            on_bus / self.bandwidth_bytes_per_ns / 1e6,
            moved,
            len(plans),
            inflight / 2**20,
            budget / 2**20,
            device_peak(plans, targets, index_of) / 2**20,
            exposed / 1e6,
            f", {by_budget / 1e6:.1f}ms of it because the in-flight budget and not the bus ran out" if by_budget > 0 else "",
        )
        self._log_density(plans, prefix)
        if inflight > budget:
            magi_logger.warning(
                "h2d load reorder: %.1f MiB of load buffers are live at once, over the %.0f MiB "
                "budget the sweep was supposed to hold -- the in-flight accounting and the emitted "
                "live ranges disagree, so treat the peak this pass reports as unreliable",
                inflight / 2**20,
                budget / 2**20,
            )
        self._log_placement(plans, targets, index_of, n_snodes)

    def _log_density(self, plans, prefix) -> None:
        """How full the bus is over every prefix of the graph, after residency.

        Measured on prefixes, not on the gap between neighbouring deadlines: a
        load can be hoisted anywhere upstream, so what has to hold is that the
        bytes due by each deadline fit the compute available by then.  Per-gap
        densities read as noise for exactly that reason -- two loads four snodes
        apart show one empty window and one impossible one, and nothing is wrong.

        A max above 1.0 is a schedule that does not exist: that prefix has more
        transfer than compute and the excess is exposed wherever it is placed.
        """
        if not magi_logger_enabled_for_debug():
            return
        by_wait: dict[int, int] = defaultdict(int)
        for p in plans:
            if not p.promoted:
                by_wait[p.wait_idx] += p.nbytes
        ratios = []
        cumulative = 0
        for wait in sorted(by_wait):
            cumulative += by_wait[wait]
            compute = prefix[min(wait, len(prefix) - 1)]
            if compute > 0:
                ratios.append(cumulative / self.bandwidth_bytes_per_ns / compute)
        if not ratios:
            return
        magi_logger.debug(
            "h2d load reorder: bus occupancy over %d prefix(es): first %.2f, median %.2f, worst "
            "%.2f at prefix %d/%d (1.0 means the transfers due by then exactly fill the compute "
            "available by then, and above 1.0 cannot be hidden at any placement)",
            len(ratios),
            ratios[0],
            sorted(ratios)[len(ratios) // 2],
            max(ratios),
            ratios.index(max(ratios)) + 1,
            len(ratios),
        )

    @staticmethod
    def _weights_of(plan) -> str:
        from ..runtime import host_pool

        names = [host_pool.name_of(s) for s in plan.slots]
        names = [n for n in names if n] or ["?"]
        return ", ".join(names[:3]) + (f", +{len(names) - 3} more" if len(names) > 3 else "")

    def _log_placement(self, plans, targets, index_of, n_snodes) -> None:
        if not magi_logger_enabled_for_debug():
            return
        magi_logger.debug(
            "h2d load placement (%d loads over %d snodes; 'at' is where the load ended up, "
            "'last_user' the last snode that still reads its bytes, 'floor' the dep floor and the "
            "in-flight floor):",
            len(plans),
            n_snodes,
        )
        for p in sorted(plans, key=lambda p: index_of[p.load]):
            if p.promoted:
                verdict = f"RESIDENT: {p.need / 1e6:.1f}ms off the bus"
            elif p.exposed <= 0:
                verdict = "hidden"
            elif p.budget_bound:
                verdict = f"EXPOSED {p.exposed / 1e6:.1f}ms (in-flight budget)"
            else:
                verdict = f"EXPOSED {p.exposed / 1e6:.1f}ms (bus/compute)"
            at = targets.get(p.load, index_of[p.load])
            magi_logger.debug(
                "  %-10s %2d slot(s) %7.1f MiB  at %5d (from %5d, floor %5d/%5d)  last_user %5d  " "need %6.1fms  %-38s  %s",
                p.load.get_name(),
                len(p.slots),
                p.nbytes / 2**20,
                at,
                index_of[p.load],
                p.lower,
                p.budget_floor,
                p.last_user,
                p.need / 1e6,
                verdict,
                self._weights_of(p),
            )

    def _log_shortfall(self, total, hideable, resident, exposed) -> None:
        """Say so when the step simply has no compute for the bytes left on the bus."""
        unavoidable = total - resident - hideable
        if unavoidable <= 0 or exposed <= 0:
            return
        magi_logger.warning(
            "h2d load reorder: %.1f MiB of weight is offloaded and %.1f MiB of it is resident, "
            "leaving %.1f MiB on the bus, but %.0f ms of compute at %.1f GB/s can only overlap "
            "%.1f MiB of it -- %.1f MiB (%.1fms) is exposed wherever the loads are placed, and only "
            "more residency removes it",
            total / 2**20,
            resident / 2**20,
            (total - resident) / 2**20,
            hideable / self.bus_utilization / self.bandwidth_bytes_per_ns / 1e6,
            self.bandwidth_bytes_per_ns,
            hideable / 2**20,
            unavoidable / 2**20,
            unavoidable / self.bandwidth_bytes_per_ns / 1e6,
        )
