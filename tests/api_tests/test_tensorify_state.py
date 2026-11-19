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

from types import SimpleNamespace

from torch._dynamo.symbolic_convert import TensorifyState

from magi_compiler._api import _compilation_context


def test_compilation_context_neither_reads_nor_leaves_float_specializations(tmp_path):
    """A float specialization another compile left set must not reach this one, and this one's must not outlive it:
    Dynamo names every frame's floats from "zf0", so a stale entry specializes the next frame's on its first attempt."""
    state = SimpleNamespace(
        compile_config=SimpleNamespace(cache_root_dir=str(tmp_path)),
        model_tag="tensorify_state",
        traced_files=set(),
        original_code_for_hook=test_compilation_context_neither_reads_nor_leaves_float_specializations.__code__,
    )
    TensorifyState.specialize("zf0")
    try:
        with _compilation_context(state):
            assert TensorifyState.empty()
            TensorifyState.specialize("zf1")
        assert TensorifyState.empty()
    finally:
        TensorifyState.clear()
