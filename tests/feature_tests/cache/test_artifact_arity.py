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

"""Before loading a cached artifact, the compile cache reads its input count from the generated wrapper and skips a
load whose count differs from the caller's.

Graph partition emits that wrapper as ``Runner.call(self, args)``; piecewise compiles turn the partition off, which
emits a module-level ``call(args)``. A reader that only matched the first form returned no count, so a mismatched
artifact loaded and failed when called.
"""

import pytest
import torch

from magi_compiler.magi_backend.piecewise_compiler import _read_generated_code_expected_arity


def _add3(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return a * b + c


@pytest.mark.parametrize("graph_partition", [True, False])
def test_reads_the_input_count_of_a_saved_artifact(tmp_path, graph_partition):
    gm = torch.fx.symbolic_trace(_add3)
    inputs = [torch.randn(4, 8) for _ in range(3)]
    artifact = torch._inductor.standalone_compile(
        gm, inputs, dynamic_shapes="from_example_inputs", options={"config_patches": {"graph_partition": graph_partition}}
    )
    artifact.save(path=str(tmp_path), format="unpacked")
    assert _read_generated_code_expected_arity(str(tmp_path)) == len(inputs)


@pytest.mark.parametrize(
    "wrapper",
    [
        "class Runner:\n    def call(self, args):\n        arg0_1, arg1_1 = args\n",
        "def call(args):\n    arg0_1, arg1_1 = args\n",
    ],
    ids=["partitioned", "unpartitioned"],
)
def test_reads_either_wrapper_form(tmp_path, wrapper):
    (tmp_path / "kernel.py").write_text("def triton_poi_fused_add_0(in_ptr0, out_ptr0):\n    pass\n")
    (tmp_path / "wrapper.py").write_text(wrapper)
    assert _read_generated_code_expected_arity(str(tmp_path)) == 2
