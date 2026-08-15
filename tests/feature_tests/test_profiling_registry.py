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

"""Unit tests for the profiling materialize-input registry
(``magi_compiler.profiling.materialize_inputs``).

Pure-CPU: the registry is a plain module-level dict/set, no torch/GPU/distributed.
"""

import pytest

from magi_compiler.profiling import apply_materialize_inputs, get_materialize_inputs_hook
from magi_compiler.profiling import materialize_inputs as mi
from magi_compiler.profiling import op_has_internal_collective, register_materialize_inputs


@pytest.fixture
def clean_registry():
    """Snapshot + restore the global registry so tests don't leak into each other."""
    hooks = dict(mi._MATERIALIZE_INPUT_HOOKS)
    coll = set(mi._INTERNAL_COLLECTIVE_OPS)
    yield
    mi._MATERIALIZE_INPUT_HOOKS.clear()
    mi._MATERIALIZE_INPUT_HOOKS.update(hooks)
    mi._INTERNAL_COLLECTIVE_OPS.clear()
    mi._INTERNAL_COLLECTIVE_OPS.update(coll)


def test_unregistered_returns_none(clean_registry):
    assert get_materialize_inputs_hook("ns::never_registered") is None
    assert op_has_internal_collective("ns::never_registered") is False


def test_register_and_get(clean_registry):
    def hook(q, k, v, cp_split_sizes):
        return None

    register_materialize_inputs("ns::op_a", hook)
    assert get_materialize_inputs_hook("ns::op_a") is hook
    # not flagged as internal-collective by default
    assert op_has_internal_collective("ns::op_a") is False


def test_internal_collective_flag(clean_registry):
    register_materialize_inputs("ns::coll_op", has_internal_collective=True)
    assert op_has_internal_collective("ns::coll_op") is True
    assert get_materialize_inputs_hook("ns::coll_op") is None

    # a hook registered WITHOUT the flag must not be marked
    register_materialize_inputs("ns::plain_op", lambda *a, **k: None)
    assert op_has_internal_collective("ns::plain_op") is False


def test_register_overrides_previous(clean_registry):
    def hook1(q):
        return (q,)

    def hook2(q):
        return (q,)

    register_materialize_inputs("ns::op_b", hook1)
    assert get_materialize_inputs_hook("ns::op_b") is hook1
    register_materialize_inputs("ns::op_b", hook2)
    assert get_materialize_inputs_hook("ns::op_b") is hook2


def test_hook_is_callable_returning_none(clean_registry):
    """A no-op hook (returns None -> keep generic realize) is valid."""
    register_materialize_inputs("ns::noop", lambda *a, **k: None, has_internal_collective=True)
    hook = get_materialize_inputs_hook("ns::noop")
    assert hook(object()) is None
    assert op_has_internal_collective("ns::noop") is True


def test_apply_materialize_inputs_same_signature_as_op(clean_registry):
    """Hook sees realized custom-op args and returns a rewritten positional tuple."""

    def hook(q, k, v, cp_split_sizes):
        seq = q[0]
        return q, k, v, [seq] * len(cp_split_sizes)

    args, kwargs = apply_materialize_inputs(hook, ((16,), (16,), (16,), [0, 0]), {})
    assert kwargs == {}
    assert args[3] == [16, 16]
    assert apply_materialize_inputs(None, (1,), {}) == ((1,), {})
    assert apply_materialize_inputs(lambda *a, **k: None, (1, 2), {"x": 3}) == ((1, 2), {"x": 3})
