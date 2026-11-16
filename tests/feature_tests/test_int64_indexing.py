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

"""Index dtype of Triton kernels compiled from a small first input.

Inductor picks int32 indexing whenever the first compile's example sizes fit and guards that
choice, but MagiCompiler never checks those guards: a graph first compiled on a small input
(e.g. a 10s video) overflowed its int32 offsets on a larger one (e.g. a 30s video).
``PassConfig.force_int64_indexing_for_dynamic_shapes`` makes dynamic-shape kernels index in int64
while static-shape kernels keep int32.
"""

import pytest
import torch
import torch.nn as nn
from torch._inductor.utils import run_and_get_code

from magi_compiler import magi_compile

HIDDEN = 64
SMALL_SEQ_LEN = 16


class _DynamicAndStaticKernels(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(HIDDEN))

    def forward(self, x):
        # x * 2 depends on the dynamic dim; weight * 3 only on the static parameter shape.
        return x * 2, self.weight * 3


def _compile_and_get_triton_kernels(tmp_path, force_int64_indexing: bool) -> list[str]:
    def configure(compile_config):
        # A cache hit would skip codegen and leave no kernel source to inspect.
        compile_config.cache_root_dir = str(tmp_path)
        compile_config.pass_config.force_int64_indexing_for_dynamic_shapes = force_int64_indexing
        return compile_config

    # magi_compile patches the class in place, so each config needs its own class.
    @magi_compile(dynamic_arg_dims={"x": 0}, config_patch=configure)
    class CompiledModel(_DynamicAndStaticKernels):
        pass

    model = CompiledModel().cuda()
    x = torch.randn(SMALL_SEQ_LEN, HIDDEN, device="cuda")
    with torch.no_grad():
        (dynamic_out, static_out), source_codes = run_and_get_code(model, x)
    torch.testing.assert_close(dynamic_out, x * 2)
    torch.testing.assert_close(static_out, model.weight * 3)
    return [kernel for code in source_codes for kernel in code.split("async_compile.triton(")[1:]]


def _split_static_and_dynamic_kernel(kernels: list[str]) -> tuple[str, str]:
    # Inductor bakes a static numel into the kernel body; a dynamic one stays a runtime argument.
    static_kernels = [kernel for kernel in kernels if f"xnumel = {HIDDEN}\n" in kernel]
    dynamic_kernels = [kernel for kernel in kernels if f"xnumel = {HIDDEN}\n" not in kernel]
    assert len(static_kernels) == 1 and len(dynamic_kernels) == 1, kernels
    return static_kernels[0], dynamic_kernels[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dynamic_kernel_uses_int64_and_static_kernel_keeps_int32(tmp_path):
    kernels = _compile_and_get_triton_kernels(tmp_path, force_int64_indexing=True)
    static_kernel, dynamic_kernel = _split_static_and_dynamic_kernel(kernels)

    assert "'xnumel': 'i64'" in dynamic_kernel
    assert "xoffset = tl.program_id(0).to(tl.int64) * XBLOCK" in dynamic_kernel
    assert "xoffset = tl.program_id(0) * XBLOCK" in static_kernel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_disabled_falls_back_to_inductor_int32_from_first_input(tmp_path):
    kernels = _compile_and_get_triton_kernels(tmp_path, force_int64_indexing=False)
    _, dynamic_kernel = _split_static_and_dynamic_kernel(kernels)

    assert "'xnumel': 'i32'" in dynamic_kernel
    assert "xoffset = tl.program_id(0) * XBLOCK" in dynamic_kernel
