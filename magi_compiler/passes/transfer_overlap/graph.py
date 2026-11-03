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
cross them (see ``SnodeGraph.earliest_legal_index``).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable, Iterator, Mapping

from torch._inductor.comms import _is_fake_dep
from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.utils import contains_wait

from magi_compiler.utils import magi_logger

from ..snode_utils import is_multi_output, issues_comm

UserClassifier = Callable[[BaseSchedulerNode], "tuple[str | None, bool]"]


class SnodeGraph:
    """``order`` plus the maps every overlap pass would otherwise rebuild."""

    def __init__(self, order: list[BaseSchedulerNode]) -> None:
        self.order = order
        self.original_index_by_node = {s: i for i, s in enumerate(order)}
        self.producer_by_buffer = {b: s for s in order for b in s.get_buffer_names()}
        # Only real data readers: fake ordering edges (WeakDep / StarDep) are left out.
        self.users_by_buffer: dict[str, set] = defaultdict(set)
        for s in order:
            for d in s.unmet_dependencies:
                if not _is_fake_dep(d):
                    self.users_by_buffer[d.name].add(s)

    def __len__(self) -> int:
        return len(self.order)

    def _sort_by_graph_order(self, snodes: Iterable[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
        return sorted(snodes, key=self.original_index_by_node.__getitem__)

    def data_producers(self, snode: BaseSchedulerNode) -> Iterator[BaseSchedulerNode | None]:
        """The producer of each buffer ``snode`` really reads; None for a graph input."""
        for d in snode.unmet_dependencies:
            if not _is_fake_dep(d):
                yield self.producer_by_buffer.get(d.name)

    def unpacks_of(self, snode: BaseSchedulerNode) -> list[BaseSchedulerNode]:
        """The ``MultiOutput`` snodes ``snode``'s result reaches its readers through, in program order."""
        readers = {u for name in snode.get_buffer_names() for u in self.users_by_buffer.get(name, ())}
        return self._sort_by_graph_order(u for u in readers if is_multi_output(u))

    def waits_of(self, snode: BaseSchedulerNode) -> list[BaseSchedulerNode]:
        """The waits on ``snode``'s result, directly or through its unpacks, in program order."""
        names = {name for s in (snode, *self.unpacks_of(snode)) for name in s.get_buffer_names()}
        readers = {u for name in names for u in self.users_by_buffer.get(name, ())}
        return self._sort_by_graph_order(u for u in readers if contains_wait(u))

    def movable_prep_nodes(self, moving_nodes: Iterable[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
        """Upstream shard prep that may move with a transfer, in program order.

        Prep reads only graph inputs or other prep: H2D loads and their waits,
        casts, pads.  A real collective is not prep; it has a plan of its own.
        A producer left out raises the transfer's legal floor instead.
        """
        members = set(moving_nodes)
        pending_nodes = list(members)
        reachable_producers: set[BaseSchedulerNode] = set()
        while pending_nodes:
            for producer in self.data_producers(pending_nodes.pop()):
                if producer is None or producer in members or producer in reachable_producers:
                    continue
                if issues_comm(producer):
                    continue
                reachable_producers.add(producer)
                pending_nodes.append(producer)

        # Program order, so every producer is classified before its consumers and
        # one pass settles the cascade of a dropped node's dependents.
        movable_prep_nodes: set[BaseSchedulerNode] = set()
        for node in self._sort_by_graph_order(reachable_producers):
            producers = self.data_producers(node)
            if all(producer is None or producer is node or producer in movable_prep_nodes for producer in producers):
                movable_prep_nodes.add(node)
        return self._sort_by_graph_order(movable_prep_nodes)

    def earliest_legal_index(
        self, moving_nodes: Iterable[BaseSchedulerNode], *, ignored_producers: Iterable[BaseSchedulerNode] = ()
    ) -> int:
        """First boundary after external data producers, excluding ``ignored_producers``."""
        excluded_nodes = set(moving_nodes) | set(ignored_producers)
        earliest_index = 0
        for node in moving_nodes:
            for dependency in node.unmet_dependencies:  # buffer dependencies
                if _is_fake_dep(dependency):  # WeakDep / StarDep -- ordering hint, not data
                    continue
                producer = self.producer_by_buffer.get(dependency.name)
                if producer is None or producer in excluded_nodes:
                    continue
                earliest_index = max(earliest_index, self.original_index_by_node.get(producer, 0) + 1)
        return earliest_index

    def is_topological_order(self, new_order: list[BaseSchedulerNode]) -> bool:
        """Whether every real producer still precedes its readers in ``new_order``."""
        new_index_by_node = {node: index for index, node in enumerate(new_order)}
        for node in new_order:
            node_index = new_index_by_node[node]
            for dependency in node.unmet_dependencies:  # buffer dependencies
                if _is_fake_dep(dependency):  # WeakDep / StarDep -- advisory ordering, not data
                    continue
                producer = self.producer_by_buffer.get(dependency.name)
                if producer is node:  # fused snode may name its own internal buffers
                    continue
                if producer is not None and new_index_by_node.get(producer, -1) >= node_index:
                    magi_logger.debug(
                        "validate fail: %s@%d needs buffer-dep %s@%d (buf %s)",
                        node.get_name(),
                        node_index,
                        producer.get_name(),
                        new_index_by_node.get(producer, -1),
                        dependency.name,
                    )
                    return False
        return True

    def collect_classified_users(
        self, start: Iterable[BaseSchedulerNode], classify: UserClassifier
    ) -> list[tuple[str, BaseSchedulerNode]]:
        """Breadth of real readers reachable from ``start``'s buffers.

        ``classify(u)`` answers ``(classification, should_descend)``: a non-None
        classification records ``u`` under it, and ``should_descend`` continues
        through ``u``'s own buffers.  Each reader is classified once, however
        many paths reach it.
        """
        pending_buffer_names = [b for s in start for b in s.get_buffer_names()]
        classified_users: list[tuple[str, BaseSchedulerNode]] = []
        visited_users: set = set()
        while pending_buffer_names:
            for user in self.users_by_buffer.get(pending_buffer_names.pop(), ()):
                if user in visited_users:
                    continue
                visited_users.add(user)
                classification, should_descend = classify(user)
                if classification is not None:
                    classified_users.append((classification, user))
                if should_descend:
                    pending_buffer_names.extend(user.get_buffer_names())
        return classified_users

    def last_user_index(self, moving_nodes: list[BaseSchedulerNode], waits: list[BaseSchedulerNode]) -> int:
        """The last snode still reading what the moving nodes or their waits produce."""
        last_index = max(self.original_index_by_node[w] for w in waits)
        moving_node_set = set(moving_nodes)
        for source in (*moving_nodes, *waits):
            for name in source.get_buffer_names():
                for user in self.users_by_buffer.get(name, ()):
                    if user not in moving_node_set:
                        last_index = max(last_index, self.original_index_by_node[user])
        return last_index

    def rebuild_node_order(self, tasks, launch_index_by_anchor: Mapping[BaseSchedulerNode, int]) -> list[BaseSchedulerNode]:
        """Apply every placed task in ONE stable sort.

        Launch indices are in the original index space, so incremental moves
        would shift them.  A task's moving nodes sort to ``launch_index - 0.5`` --
        just before the node originally there -- members keep their internal
        order, and everything else keeps its original index.  Tasks without a
        launch index stay put; a snode moving with two tasks follows the earlier
        one (``tasks`` in program order).
        """
        launch_index_by_member: dict = {}
        for task in reversed(tasks):
            launch_index = launch_index_by_anchor.get(task.anchor)
            if launch_index is None:
                continue
            for member in task.moving_nodes:
                launch_index_by_member[member] = launch_index

        def _sort_key(node):
            original_index = self.original_index_by_node[node]
            launch_index = launch_index_by_member.get(node)
            if launch_index is None:
                return (original_index, 0.0)
            # Members keep their internal order through the original index.
            return (launch_index - 0.5, original_index)

        return sorted(self.order, key=_sort_key)
