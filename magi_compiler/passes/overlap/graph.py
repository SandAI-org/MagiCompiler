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

"""The scheduler-node order an overlap pass reads, indexed once.

Every question a placement asks of the graph -- who reads this buffer, where
may this group legally go, is the rebuilt order still topological -- is asked
against REAL data dependencies only.  Inductor's fake ``WeakDep`` edges order
collectives on the comm stream; they are advisory, and the passes exist to
cross them (see ``snode_utils.earliest_legal_index``).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable, Iterator

from torch._inductor.comms import _is_fake_dep
from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.utils import contains_wait

from ..snode_utils import earliest_legal_index, is_multi_output, issues_transfer, validate_topological_order

Classify = Callable[[BaseSchedulerNode], "tuple[str | None, bool]"]


class SnodeGraph:
    """``order`` plus the maps every overlap pass would otherwise rebuild."""

    def __init__(self, order: list[BaseSchedulerNode]) -> None:
        self.order = order
        self.index_of = {s: i for i, s in enumerate(order)}
        self.buf_to_snode = {b: s for s in order for b in s.get_buffer_names()}
        self.users: dict[str, set] = defaultdict(set)
        for s in order:
            for d in s.unmet_dependencies:
                if not _is_fake_dep(d):
                    self.users[d.name].add(s)

    def __len__(self) -> int:
        return len(self.order)

    def _sorted(self, snodes: Iterable[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
        return sorted(snodes, key=self.index_of.__getitem__)

    def real_producers(self, snode: BaseSchedulerNode) -> Iterator[BaseSchedulerNode | None]:
        """The producer of each buffer ``snode`` really reads; None for a graph input."""
        for d in snode.unmet_dependencies:
            if not _is_fake_dep(d):
                yield self.buf_to_snode.get(d.name)

    def unpacks_of(self, snode: BaseSchedulerNode) -> list[BaseSchedulerNode]:
        """The ``MultiOutput`` snodes ``snode``'s result reaches its readers through, in program order."""
        return self._sorted({u for name in snode.get_buffer_names() for u in self.users.get(name, ()) if is_multi_output(u)})

    def waits_of(self, snode: BaseSchedulerNode) -> list[BaseSchedulerNode]:
        """The waits on ``snode``'s result, directly or through its unpacks, in program order."""
        names = {name for s in (snode, *self.unpacks_of(snode)) for name in s.get_buffer_names()}
        return self._sorted({u for name in names for u in self.users.get(name, ()) if contains_wait(u)})

    def prep_chain(self, group: Iterable[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
        """Upstream shard prep that may move with a transfer, in program order.

        Prep reads only graph inputs or other prep: H2D loads and their waits,
        casts, pads.  A real collective is not prep; it has a plan of its own.
        A producer left out raises the transfer's legal floor instead.
        """
        # Deferred: weight_offload's package init imports this one.
        from ..weight_offload.schedule.h2d_snode import is_h2d_load

        members = set(group)
        stack = list(members)
        reached: set[BaseSchedulerNode] = set()
        while stack:
            for prod in self.real_producers(stack.pop()):
                if prod is None or prod in members or prod in reached:
                    continue
                if issues_transfer(prod) and not is_h2d_load(prod):
                    continue
                reached.add(prod)
                stack.append(prod)

        # Program order, so every producer is classified before its consumers and
        # one pass settles the cascade of a dropped node's dependents.
        keep: set[BaseSchedulerNode] = set()
        for s in self._sorted(reached):
            if all(prod is None or prod is s or prod in keep for prod in self.real_producers(s)):
                keep.add(s)
        return self._sorted(keep)

    def earliest_legal_index(self, group: Iterable[BaseSchedulerNode], *, skip: Iterable[BaseSchedulerNode] = ()) -> int:
        return earliest_legal_index(group=group, index_of=self.index_of, buf_to_snode=self.buf_to_snode, skip=skip)

    def validate(self, new_order: list[BaseSchedulerNode]) -> bool:
        return validate_topological_order(new_order, self.buf_to_snode)

    def walk_users(self, start: Iterable[BaseSchedulerNode], classify: Classify) -> list[tuple[str, BaseSchedulerNode]]:
        """Breadth of real readers reachable from ``start``'s buffers.

        ``classify(u)`` answers ``(label, descend)``: a non-None label records
        ``u`` under it, and ``descend`` continues through ``u``'s own buffers.
        Each reader is classified once, however many paths reach it.
        """
        stack = [b for s in start for b in s.get_buffer_names()]
        found: list[tuple[str, BaseSchedulerNode]] = []
        seen: set = set()
        while stack:
            for u in self.users.get(stack.pop(), ()):
                if u in seen:
                    continue
                seen.add(u)
                label, descend = classify(u)
                if label is not None:
                    found.append((label, u))
                if descend:
                    stack.extend(u.get_buffer_names())
        return found

    def last_user_index(self, group: list[BaseSchedulerNode], waits: list[BaseSchedulerNode]) -> int:
        """The last snode still reading what the group or its waits produce."""
        last = max(self.index_of[w] for w in waits)
        skip = set(group)
        for src in (*group, *waits):
            for name in src.get_buffer_names():
                for u in self.users.get(name, ()):
                    if u not in skip:
                        last = max(last, self.index_of[u])
        return last

    def rebuild(self, tasks) -> list[BaseSchedulerNode]:
        """Apply every placed task in ONE stable sort.

        Targets live in the original index space, so incremental moves would
        shift them.  A task's group sorts to ``target - 0.5`` -- just before the
        node originally at ``target`` -- members keep their internal order, and
        everything else keeps its original index.  Tasks with no target stay put;
        a snode in two groups follows the earlier task (``tasks`` in program order).
        """
        member_target = {m: t.target for t in reversed(tasks) if t.target is not None for m in t.group}
        index_of = self.index_of

        def _key(s):
            target = member_target.get(s)
            return (target - 0.5, index_of[s]) if target is not None else (index_of[s], 0.0)

        return sorted(self.order, key=_key)
