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

"""External helper module used by ``test_register_triton_op.py``.

The helpers below intentionally live in their own module so that, when the
test file imports them and calls them inside a ``magi_register_custom_op``-
decorated function, the helpers' ``__globals__`` are *this* module, not the
test module. That exercises the truly cross-module rebuild path in
``rewrite_fn_with_wrap_triton``.
"""

from __future__ import annotations

"""
External helper module for test_register_triton_op.py to verify
cross-module triton kernel introspection.
"""
import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def external_neg_kernel(
        in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, -x, mask=mask)

    @triton.jit
    def external_double_kernel(
        in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x * 2, mask=mask)

    def external_neg_launcher(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        n = x.numel()
        external_neg_kernel[((n + 127) // 128,)](x, out, n, BLOCK_SIZE=128)
        return out

    def external_double_launcher(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        n = x.numel()
        external_double_kernel[((n + 127) // 128,)](x, out, n, BLOCK_SIZE=128)
        return out

    def maybe_capture(kernel):
        """Third-party-style thin wrapper around a triton kernel.

        Some libraries return objects with a ``.fn`` attribute pointing back to
        the underlying ``JITFunction``; we mimic that pattern here so the test
        can confirm ``rewrite_fn_with_wrap_triton`` still recognises the
        underlying kernel when users write ``maybe_capture(kernel)[grid](...)``.
        """

        class _Captured:
            def __init__(self, k):
                self.fn = k  # introspector recognises objects with .fn

            def __getitem__(self, grid):
                return self.fn[grid]

        return _Captured(kernel)
