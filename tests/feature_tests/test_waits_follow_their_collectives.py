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

"""``waits_follow_their_collectives`` on a synthetic snode list, Inductor's predicates stubbed by kind."""

from types import SimpleNamespace

from magi_compiler.passes import snode_utils


class FakeSnode:
    def __init__(self, name, kind, reads=()):
        self.name, self.kind = name, kind
        self.read_writes = SimpleNamespace(reads=[SimpleNamespace(name=r) for r in reads])

    def get_buffer_names(self):
        return {self.name}


def test_each_wait_moves_right_after_its_collective(monkeypatch):
    monkeypatch.setattr(snode_utils, "contains_collective", lambda s: s.kind == "comm")
    monkeypatch.setattr(snode_utils, "contains_wait", lambda s: s.kind == "wait")
    a2a, gather, mm = FakeSnode("a2a", "comm"), FakeSnode("gather", "comm"), FakeSnode("mm", "compute")
    # A wait on a collective launched by an earlier graph has nothing to follow and stays put.
    upstream_wait = FakeSnode("upstream_wait", "wait", ["graph_input"])
    order = [
        a2a,
        gather,
        mm,
        upstream_wait,
        FakeSnode("a2a_wait", "wait", ["a2a"]),
        FakeSnode("gather_wait", "wait", ["gather"]),
    ]
    assert [s.name for s in snode_utils.waits_follow_their_collectives(order)] == [
        "a2a",
        "a2a_wait",
        "gather",
        "gather_wait",
        "mm",
        "upstream_wait",
    ]
