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

from contextlib import contextmanager

_pass_context = None


class PassContext:
    def __init__(self, runtime_shape: int | None, subgraph_index: int | None = None, expect_restart_analysis: bool = False):
        self.runtime_shape = runtime_shape
        self.subgraph_index = subgraph_index
        # The cache recorded a RestartAnalysis for this subgraph and assert_cache_hit is on, so this compile must
        # restart before lowering; reaching the post-grad passes means it is compiling from scratch.
        self.expect_restart_analysis = expect_restart_analysis


def get_pass_context() -> PassContext:
    """Get the current pass context."""
    assert _pass_context is not None
    return _pass_context


@contextmanager
def pass_context(runtime_shape: int | None, subgraph_index: int | None = None, expect_restart_analysis: bool = False):
    """
    A context manager that stores the current pass context, usually it is a list of sizes to specialize.
    """
    global _pass_context
    prev_context = _pass_context
    _pass_context = PassContext(runtime_shape, subgraph_index, expect_restart_analysis)
    try:
        yield
    finally:
        _pass_context = prev_context
