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

"""Route link collectives onto their link stream at codegen, after every pass has placed them.

The IR stays a ``_CollectiveKernel``: bucketing, the reorder passes, cost profiling
and Inductor's wait/buffer semantics see an ordinary functional collective.  Only the
Python kernel name the wrapper emits changes, to the ``magi::link_*`` op with the
same schema (see ``runtime.link_collective_op``).
"""

from __future__ import annotations

from collections import Counter

from torch._inductor.scheduler import BaseSchedulerNode

from magi_compiler.utils import magi_logger

from ...snode_utils import link_collective_node
from ...transfer_overlap import LinkSharing
from ..runtime.link_collective_op import link_kernel_name


class RouteCollectivesToLinkStream:
    """Reorder pass that never reorders.  Install it last: positions are final by then."""

    def __init__(self, link_sharing: LinkSharing) -> None:
        self.link_sharing = link_sharing

    def __call__(self, snodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
        routed_count_by_op: Counter = Counter()
        for snode in snodes:
            collective_node = link_collective_node(snode)
            if collective_node is None:
                continue
            collective_node.python_kernel_name = link_kernel_name(collective_node.op_overload, self.link_sharing)
            routed_count_by_op[collective_node.op_overload.name()] += 1
        if routed_count_by_op:
            magi_logger.info(
                "collective routing: queued on the %s link stream: %s",
                self.link_sharing.value,
                ", ".join(f"{count} {op_name}" for op_name, count in sorted(routed_count_by_op.items())),
            )
        return snodes
