# Copyright (c) 2025 SandAI. All Rights Reserved.
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

"""Unit tests for the profiling benchmark-input registry
(``magi_compiler.profiling.benchmark_inputs``).

Pure-CPU: the registry is a plain module-level dict/set, no torch/GPU/distributed.
"""

import pytest

from magi_compiler.profiling import benchmark_inputs as bi
from magi_compiler.profiling import get_benchmark_inputs_hook, op_has_internal_collective, register_benchmark_inputs


@pytest.fixture
def clean_registry():
    """Snapshot + restore the global registry so tests don't leak into each other."""
    hooks = dict(bi._BENCHMARK_INPUT_HOOKS)
    coll = set(bi._INTERNAL_COLLECTIVE_OPS)
    yield
    bi._BENCHMARK_INPUT_HOOKS.clear()
    bi._BENCHMARK_INPUT_HOOKS.update(hooks)
    bi._INTERNAL_COLLECTIVE_OPS.clear()
    bi._INTERNAL_COLLECTIVE_OPS.update(coll)


def test_unregistered_returns_none(clean_registry):
    assert get_benchmark_inputs_hook("ns::never_registered") is None
    assert op_has_internal_collective("ns::never_registered") is False


def test_register_and_get(clean_registry):
    def hook(fx_node, realize):
        return None

    register_benchmark_inputs("ns::op_a", hook)
    assert get_benchmark_inputs_hook("ns::op_a") is hook
    # not flagged as internal-collective by default
    assert op_has_internal_collective("ns::op_a") is False


def test_internal_collective_flag(clean_registry):
    register_benchmark_inputs("ns::coll_op", lambda n, r: None, has_internal_collective=True)
    assert op_has_internal_collective("ns::coll_op") is True
    assert get_benchmark_inputs_hook("ns::coll_op") is not None

    # a hook registered WITHOUT the flag must not be marked
    register_benchmark_inputs("ns::plain_op", lambda n, r: None)
    assert op_has_internal_collective("ns::plain_op") is False


def test_register_overrides_previous(clean_registry):
    def hook1(n, r):
        return "1"

    def hook2(n, r):
        return "2"

    register_benchmark_inputs("ns::op_b", hook1)
    assert get_benchmark_inputs_hook("ns::op_b") is hook1
    register_benchmark_inputs("ns::op_b", hook2)
    assert get_benchmark_inputs_hook("ns::op_b") is hook2


def test_hook_is_callable_returning_none(clean_registry):
    """A no-op hook (returns None -> fall back to generic realize) is valid."""
    register_benchmark_inputs("ns::noop", lambda fx_node, realize: None, has_internal_collective=True)
    hook = get_benchmark_inputs_hook("ns::noop")
    assert hook(object(), lambda x: x) is None
    assert op_has_internal_collective("ns::noop") is True
