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

"""The shape every transfer/compute overlap pass shares.

Installed in Inductor's ``reorder_for_compute_comm_overlap_passes``: Inductor
calls it with the scheduler-node order and takes back whatever it returns, and
does NOT repair that order -- it must be topological.  A subclass says which
transfers exist and how to place them; this class owns the rest:

    wants -> plan -> place -> rebuild + validate -> agree -> on_commit -> report

``plan`` returning ``None`` leaves the graph untouched.  ``ctx`` is a per-call
dict the hooks pass state through, so nothing call-specific lives on the pass
object (which Inductor deep-copies into its cache key).
"""

from __future__ import annotations

import copy

from torch._inductor.scheduler import BaseSchedulerNode

from magi_compiler.utils import magi_logger

from .cost import CostView, default_cost_fn
from .graph import SnodeGraph
from .task import TransferTask

DEFAULT_WINDOW_MARGIN_NS = 5_000.0
"""Headroom added to every transfer's window, absorbing estimator error and
launch latency so the wait rarely stalls."""


class OverlapPass:
    """Callable reorder pass; subclass and implement ``wants``, ``plan``, ``place``."""

    name = "overlap reorder"  # log prefix

    def __init__(self, cost_fn=None) -> None:
        # snode -> ns.  Normally the SnodeCostTable filled by the SnodeCostProfile
        # pass ahead of this one; Inductor's estimate_op_runtime when run alone.
        self._cost_fn = cost_fn if cost_fn is not None else default_cost_fn()

    def __deepcopy__(self, memo):
        # Through memo, so this copy and the profile pass's copy share one table.
        new = type(self).__new__(type(self))
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            new.__dict__[key] = copy.deepcopy(value, memo) if key == "_cost_fn" else value
        return new

    # -- the template -------------------------------------------------------
    def __call__(self, snodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
        order = list(snodes)
        if not self.wants(order):
            return order
        graph = SnodeGraph(order)
        cost = CostView(self._cost_fn)
        ctx: dict = {}
        tasks = self.plan(graph, cost, ctx)
        if tasks is None:
            return order
        self.place(graph, cost, tasks, ctx)

        new_order = graph.rebuild(tasks)
        ok = graph.validate(new_order)
        if not ok:
            magi_logger.warning("%s: rebuilt order failed validation; leaving graph unchanged", self.name)
        committed = self.agree(ok, ctx)
        if committed:
            self.on_commit(graph, tasks, ctx)
            order[:] = new_order
        self.report(graph, cost, tasks, committed, ctx)
        return order

    # -- hooks ----------------------------------------------------------------
    def wants(self, order: list[BaseSchedulerNode]) -> bool:
        """Cheap check that the graph holds anything this pass moves."""
        raise NotImplementedError

    def plan(self, graph: SnodeGraph, cost: CostView, ctx: dict) -> list[TransferTask] | None:
        """One task per transfer, in program order; ``None`` to leave the graph alone."""
        raise NotImplementedError

    def place(self, graph: SnodeGraph, cost: CostView, tasks: list[TransferTask], ctx: dict) -> None:
        """Set ``target`` on every task that moves."""
        raise NotImplementedError

    def agree(self, ok: bool, ctx: dict) -> bool:
        """Whether to commit a rebuilt order this rank found valid (``ok``)."""
        return ok

    def on_commit(self, graph: SnodeGraph, tasks: list[TransferTask], ctx: dict) -> None:
        """Side effects that may only happen once the new order is committed."""

    def report(self, graph: SnodeGraph, cost: CostView, tasks: list[TransferTask], committed: bool, ctx: dict) -> None:
        """Log what was done."""
