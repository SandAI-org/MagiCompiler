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


from __future__ import annotations

"""
This test suite covers the ``triton_op`` auto-detection and registration path.

Coverage Matrix & Sections:
---------------------------
SECTION 1: Direct Kernel Launch Patterns
  - Flat direct kernel call (``kernel[grid](...)``)
  - Multiple kernels in sequence
  - Kernel launched inside a closure
  - Multilevel nesting & Helper functions launching kernels

SECTION 2: Wrapped, Dynamic & Exotic Retrievals
  - Helper launchers (local, cross-module, 3rd party wrappers)
  - ``wrap_triton`` idempotency (Mixing wrapped and bare kernels safely)
  - Explicit ``extra_triton_kernels`` override & deduplication
  - Staticmethod / Classmethod kernels
  - Dynamically fetched / runtime-imported kernels

SECTION 3: Autotune, Heuristics & Autograd in Triton
  - ``@triton.autotune`` kernels (single & multiple configs)
  - ``@triton.heuristics`` rejection & graceful fallback
  - Autograd combined with Triton kernels

SECTION 4: End-to-End Tracing
  - Pure Inductor see-through proof (AOT graph verification)
"""

import pytest
import torch
from torch.testing import assert_close

triton = pytest.importorskip("triton")
tl = pytest.importorskip("triton.language")

from magi_compiler.api import magi_register_custom_op  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="triton kernels require CUDA")


# ---------------------------------------------------------------------------
# Module-level kernels (so they live in fn.__globals__ for several scenarios)
# ---------------------------------------------------------------------------


@triton.jit
def _cos_kernel(in_ptr0, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    output = tl.cos(x)
    tl.store(out_ptr + offsets, output, mask=mask)


@triton.jit
def _scale_kernel(in_ptr0, out_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    output = x * scale
    tl.store(out_ptr + offsets, output, mask=mask)


@triton.jit
def _add_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, a + b, mask=mask)


@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": 128}, num_warps=4), triton.Config({"BLOCK_SIZE": 256}, num_warps=4)],
    key=["n_elements"],
)
@triton.jit
def _autotuned_cos_kernel(in_ptr0, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    tl.store(out_ptr + offsets, tl.cos(x), mask=mask)


def _grid_1d(n: int):
    return ((n + 127) // 128,)


# Module-level frozen dataclass fixtures used by the dataclass+triton tests
# below. Defined at module scope (not inside the test methods) so that
# ``typing.get_type_hints`` / ``eval`` on the function's stringified
# annotations (PEP 563 / ``from __future__ import annotations``) can find
# them via ``fn.__globals__``.
from dataclasses import dataclass as _dc_dataclass  # noqa: E402


@_dc_dataclass(frozen=True)
class _DcCosCfg:
    block_size: int


@_dc_dataclass(frozen=True)
class _DcKernelCfg:
    block_size: int
    extra_offset: float


@_dc_dataclass(frozen=True)
class _DcOuterCfg:
    kernel: _DcKernelCfg
    scale: float


@_dc_dataclass(frozen=True)
class _DcShapeCfg:
    out_dim: int


@_dc_dataclass(frozen=True)
class _DcProjCfg:
    shape: _DcShapeCfg
    block_size: int


# that by defining it at module scope but in its own helper that fn calls.


def _scale_launcher(x: torch.Tensor, factor: float) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    _scale_kernel[_grid_1d(n)](x, out, n, factor, BLOCK_SIZE=128)
    return out


def _add_launcher(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a)
    n = a.numel()
    _add_kernel[_grid_1d(n)](a, b, out, n, BLOCK_SIZE=128)
    return out


def _make_cos_kernel():
    @triton.jit
    def _kernel(in_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK
        offsets = block_start + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(in_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, tl.cos(x), mask=mask)

    return _kernel


def _inner_launcher(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    _cos_kernel[_grid_1d(n)](x, out, n, BLOCK_SIZE=128)
    return out


def _dispatch_launcher(x: torch.Tensor) -> torch.Tensor:
    return _inner_launcher(x)


@triton.heuristics({"BLOCK_SIZE": lambda args: 128})
@triton.jit
def _heuristics_top_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


@triton.autotune(configs=[triton.Config({}, num_warps=4)], key=["n_elements"])
@triton.heuristics({"BLOCK_SIZE": lambda args: 128})
@triton.jit
def _autotune_then_heuristics_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


class _KernelHolder:
    """Holder with kernel exposed via classmethod / staticmethod. The
    introspector cannot statically follow ``Holder.get()`` to a kernel at
    decoration time; users must use ``extra_triton_kernels=`` instead.
    """

    @staticmethod
    def get_static():
        return _scale_kernel

    @classmethod
    def get_class(cls):
        return _scale_kernel


# ============================================================================
# SECTION 1: Direct Kernel Launch Patterns
# ============================================================================


class TestFlatDirectKernel:
    def test_basic_cos(self):
        @magi_register_custom_op(name="magi_test::flat_cos")
        def mycos(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            n = x.numel()
            _cos_kernel[_grid_1d(n)](x, out, n, BLOCK_SIZE=128)
            return out

        x = torch.randn(1024, device="cuda")
        out = mycos(x)
        assert_close(out, torch.cos(x), atol=1e-5, rtol=1e-5)

    def test_op_is_triton_op(self):
        """Sanity: the registered op should be a triton_op-style CustomOpDef
        and torch.compile should be able to see through it."""

        @magi_register_custom_op(name="magi_test::seethrough_cos")
        def mycos(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            n = x.numel()
            _cos_kernel[_grid_1d(n)](x, out, n, BLOCK_SIZE=128)
            return out

        compiled = torch.compile(mycos, backend="inductor", fullgraph=True)
        x = torch.randn(2048, device="cuda")
        out = compiled(x)
        assert_close(out, torch.cos(x), atol=1e-5, rtol=1e-5)


class TestMultiKernelSequence:
    def test_chain(self):
        @magi_register_custom_op(name="magi_test::cos_then_scale")
        def fn(x: torch.Tensor, scale: float) -> torch.Tensor:
            tmp = torch.empty_like(x)
            out = torch.empty_like(x)
            n = x.numel()
            _cos_kernel[_grid_1d(n)](x, tmp, n, BLOCK_SIZE=128)
            _scale_kernel[_grid_1d(n)](tmp, out, n, scale, BLOCK_SIZE=128)
            return out

        x = torch.randn(1024, device="cuda")
        out = fn(x, 2.5)
        assert_close(out, torch.cos(x) * 2.5, atol=1e-5, rtol=1e-5)


class TestKernelInsideClosure:
    def test_closure_kernel(self):
        def make_op(kernel):
            @magi_register_custom_op(name=f"magi_test::closure_{id(kernel)}")
            def op(x: torch.Tensor) -> torch.Tensor:
                out = torch.empty_like(x)
                n = x.numel()
                kernel[_grid_1d(n)](x, out, n, BLOCK=128)
                return out

            return op

        kernel = _make_cos_kernel()
        op = make_op(kernel)
        x = torch.randn(2048, device="cuda")
        assert_close(op(x), torch.cos(x), atol=1e-5, rtol=1e-5)


# extra_triton_kernels escape hatch (scenario 9-style: kernel hidden behind
# an attribute access the introspector cannot trace).


class TestMultiLevelNesting:
    def test_fn_to_dispatch_to_launcher_to_kernel(self):
        @magi_register_custom_op(name="magi_test::multi_level_cos")
        def fn(x: torch.Tensor) -> torch.Tensor:
            return _dispatch_launcher(x)

        x = torch.randn(2048, device="cuda")
        assert_close(fn(x), torch.cos(x), atol=1e-5, rtol=1e-5)

    def test_introspection_walks_all_levels(self):
        from torch._higher_order_ops.triton_kernel_wrap import TraceableTritonKernelWrapper

        from magi_compiler._triton_introspect import introspect_fn, rewrite_fn_with_wrap_triton

        def fn(x):
            return _dispatch_launcher(x)

        kernels = list(introspect_fn(fn).bare_triton_kernels)
        assert _cos_kernel in kernels

        rewritten = rewrite_fn_with_wrap_triton(fn, kernels)
        rebuilt_dispatch = rewritten.__globals__["_dispatch_launcher"]
        rebuilt_inner = rebuilt_dispatch.__globals__["_inner_launcher"]
        assert isinstance(rebuilt_inner.__globals__["_cos_kernel"], TraceableTritonKernelWrapper)


# Third-party "thin wrapper" pattern: some libraries return objects with a
# ``.fn`` attribute pointing at the underlying triton kernel; the introspector
# already knows how to unwrap that, so kernels invoked via
# ``maybe_capture(kernel)[grid](...)`` should still register as a triton_op.


class TestFactoryInsideFn:
    def test_factory_inside_fn_runtime(self):
        @magi_register_custom_op(name="magi_test::factory_inside_fn")
        def fn(x: torch.Tensor) -> torch.Tensor:
            kernel = _make_cos_kernel()
            out = torch.empty_like(x)
            n = x.numel()
            kernel[_grid_1d(n)](x, out, n, BLOCK=128)
            return out

        x = torch.randn(1024, device="cuda")
        assert_close(fn(x), torch.cos(x), atol=1e-5, rtol=1e-5)


# True cross-module launcher: helpers and kernels live in
# ``tests/api_tests/_triton_external_helpers.py``. The decorated function
# imports them, so ``_rebuild`` has to descend into a helper whose
# ``__globals__`` is a *different* module dict than ``fn.__globals__``.


class TestNnModuleSelfKernel:
    def test_kernel_on_self(self):
        from torch import nn

        class CosModule(nn.Module):
            def __init__(self, kernel):
                super().__init__()
                self._kernel = kernel
                self.fn = self._build_fn()

            def _build_fn(self):
                kernel = self._kernel

                @magi_register_custom_op(name=f"magi_test::module_self_kernel_{id(self)}", extra_triton_kernels=[kernel])
                def op(x: torch.Tensor) -> torch.Tensor:
                    out = torch.empty_like(x)
                    n = x.numel()
                    kernel[_grid_1d(n)](x, out, n, BLOCK_SIZE=128)
                    return out

                return op

            def forward(self, x):
                return self.fn(x)

        mod = CosModule(_cos_kernel).to("cuda")
        x = torch.randn(1024, device="cuda")
        assert_close(mod(x), torch.cos(x), atol=1e-5, rtol=1e-5)


# Factory created *inside* fn (kernel is a local variable, not a closure
# captured from outside). The introspector detects the bare ``kernel[grid]``
# call but the actual kernel object lives only in the runtime locals, so
# rewrite has nothing to shadow. This must still execute correctly because
# ``wrap_triton`` is optional for runtime correctness (only required for
# torch.compile traceability).


# ============================================================================
# SECTION 2: Wrapped, Dynamic & Exotic Retrievals
# ============================================================================


class TestHelperLauncher:
    def test_helper_launcher(self):
        @magi_register_custom_op(name="magi_test::add_via_launcher")
        def add_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return _add_launcher(a, b)

        a = torch.randn(2048, device="cuda")
        b = torch.randn(2048, device="cuda")
        assert_close(add_op(a, b), a + b, atol=1e-5, rtol=1e-5)


class TestCrossModuleLauncher:
    def test_scale_via_external_launcher(self):
        @magi_register_custom_op(name="magi_test::scale_via_external")
        def scale_op(x: torch.Tensor, factor: float) -> torch.Tensor:
            return _scale_launcher(x, factor)

        x = torch.randn(2048, device="cuda")
        assert_close(scale_op(x, 0.25), x * 0.25, atol=1e-5, rtol=1e-5)


class TestThirdPartyThinWrapper:
    def test_thin_wrapper_kernel(self):
        from tests.api_tests._triton_external_helpers import maybe_capture

        @magi_register_custom_op(
            name="magi_test::cos_via_thin_wrapper",
            # Even though the introspector handles ``.fn``-style wrappers, we
            # also pass the raw kernel as ``extra_triton_kernels`` to confirm
            # the deduplication path works with this style of call.
            extra_triton_kernels=[_cos_kernel],
        )
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            n = x.numel()
            wrapped = maybe_capture(_cos_kernel)
            wrapped[_grid_1d(n)](x, out, n, BLOCK_SIZE=128)
            return out

        x = torch.randn(1024, device="cuda")
        assert_close(fn(x), torch.cos(x), atol=1e-5, rtol=1e-5)


# wrap_triton idempotency: if the user already wrote ``wrap_triton(kernel)``
# explicitly, we must not produce a wrap_triton(wrap_triton(kernel)).


class TestTrueCrossModuleLauncher:
    def test_external_neg_launcher(self):
        from tests.api_tests._triton_external_helpers import external_neg_launcher

        @magi_register_custom_op(name="magi_test::true_cross_module_neg")
        def fn(x: torch.Tensor) -> torch.Tensor:
            return external_neg_launcher(x)

        x = torch.randn(2048, device="cuda")
        assert_close(fn(x), -x, atol=1e-5, rtol=1e-5)

    def test_rewrite_descends_into_other_module(self):
        from torch._higher_order_ops.triton_kernel_wrap import TraceableTritonKernelWrapper

        from magi_compiler._triton_introspect import introspect_fn, rewrite_fn_with_wrap_triton
        from tests.api_tests._triton_external_helpers import external_double_kernel, external_double_launcher

        def fn(x):
            # Bare Name call so the introspector can follow it across modules
            # via ``called_functions``.
            return external_double_launcher(x)

        kernels = list(introspect_fn(fn).bare_triton_kernels)
        assert external_double_kernel in kernels

        rewritten = rewrite_fn_with_wrap_triton(fn, kernels)

        # ``external_double_launcher`` was captured from the enclosing test
        # method's locals, so it lives in ``fn``'s closure (NOT in
        # ``__globals__``). The rewrite pass must still descend into it and
        # produce a rebuilt copy whose globals reference the wrap_triton-
        # aware kernel.
        rebuilt_launcher = None
        for cell in rewritten.__closure__ or ():
            try:
                contents = cell.cell_contents
            except ValueError:
                continue
            if callable(contents) and getattr(contents, "__name__", None) == ("external_double_launcher"):
                rebuilt_launcher = contents
                break
        assert rebuilt_launcher is not None, (
            "expected rewrite_fn_with_wrap_triton to keep the launcher in " "the rewritten function's closure"
        )
        assert isinstance(rebuilt_launcher.__globals__["external_double_kernel"], TraceableTritonKernelWrapper), (
            "rewrite_fn_with_wrap_triton should rebuild cross-module helpers "
            "so the kernel reference inside them is wrap_triton-aware."
        )

        # The ORIGINAL helper module's globals must NOT be mutated; only the
        # rebuilt copy carries the wrapper.
        from tests.api_tests import _triton_external_helpers as ext_mod

        assert not isinstance(
            ext_mod.external_double_launcher.__globals__["external_double_kernel"], TraceableTritonKernelWrapper
        ), (
            "rewrite_fn_with_wrap_triton must not mutate the helper's home "
            "module globals (other unrelated callers would be affected)."
        )


class TestMixedWrappedAndBareKernels:
    """When the user has manually wrapped some kernels with ``wrap_triton``
    but left others bare (a common state during incremental migration), the
    decorator must wrap only the bare ones (no double-wrap) and the op must
    still run.
    """

    def test_mixed_wrapped_and_bare(self):
        from torch.library import wrap_triton

        @magi_register_custom_op(name="magi_test::mixed_wrap_state")
        def myop(x: torch.Tensor) -> torch.Tensor:
            n = x.numel()
            mid = torch.empty_like(x)
            wrap_triton(_cos_kernel)[_grid_1d(n)](x, mid, n, BLOCK_SIZE=128)
            out = torch.empty_like(x)
            _scale_kernel[_grid_1d(n)](mid, out, n, 2.0, BLOCK_SIZE=128)
            return out

        x = torch.randn(512, device="cuda")
        out = myop(x)
        assert_close(out, torch.cos(x) * 2.0, atol=1e-5, rtol=1e-5)


class TestWrapTritonIdempotent:
    def test_user_already_wrapped(self):
        from torch.library import wrap_triton

        @magi_register_custom_op(name="magi_test::cos_user_wrapped")
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            n = x.numel()
            wrap_triton(_cos_kernel)[_grid_1d(n)](x, out, n, BLOCK_SIZE=128)
            return out

        x = torch.randn(1024, device="cuda")
        assert_close(fn(x), torch.cos(x), atol=1e-5, rtol=1e-5)

    def test_rewrite_does_not_double_wrap(self):
        """Direct unit test: passing the already-wrapped kernel back through
        ``rewrite_fn_with_wrap_triton`` must not produce a double wrapper."""
        from torch._higher_order_ops.triton_kernel_wrap import TraceableTritonKernelWrapper
        from torch.library import wrap_triton

        from magi_compiler._triton_introspect import rewrite_fn_with_wrap_triton

        wrapped_kernel = wrap_triton(_cos_kernel)

        def fn(x):
            out = torch.empty_like(x)
            n = x.numel()
            wrapped_kernel[_grid_1d(n)](x, out, n, BLOCK_SIZE=128)
            return out

        # Pass the wrapped kernel as the "kernels" argument; the rewrite path
        # should pass it through ``_resolve_kernel`` and not re-wrap.
        rewritten = rewrite_fn_with_wrap_triton(fn, [wrapped_kernel])
        # The closure cell for ``wrapped_kernel`` (or the rebuilt globals
        # entry, depending on closure capture order) must still be a single
        # TraceableTritonKernelWrapper, not nested.
        seen = []
        if rewritten.__closure__ is not None:
            for cell in rewritten.__closure__:
                try:
                    seen.append(cell.cell_contents)
                except ValueError:
                    pass
        seen.extend(rewritten.__globals__.values())
        wrappers = [v for v in seen if isinstance(v, TraceableTritonKernelWrapper)]
        assert wrappers, "expected at least one wrap_triton wrapper to be present"
        for w in wrappers:
            inner = getattr(w, "kernel", None) or getattr(w, "fn", None)
            assert not isinstance(
                inner, TraceableTritonKernelWrapper
            ), "rewrite_fn_with_wrap_triton produced a double-wrapped kernel"


# infer_output_meta_fn override: both the ``list[str]`` shorthand and the
# explicit ``Callable`` form should be honoured even when we go down the
# triton_op path (because triton_op pre-registers ``fn`` itself as the fake).


class TestExtraTritonKernels:
    def test_explicit_kernel_list(self):
        kernels_holder = type("KH", (), {})()
        kernels_holder.k = _cos_kernel

        @magi_register_custom_op(name="magi_test::cos_via_extra", extra_triton_kernels=[_cos_kernel])
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            n = x.numel()
            kernels_holder.k[_grid_1d(n)](x, out, n, BLOCK_SIZE=128)
            return out

        x = torch.randn(1024, device="cuda")
        assert_close(fn(x), torch.cos(x), atol=1e-5, rtol=1e-5)


# Fallback: no triton kernels => still works (custom_op path).


class TestExtraTritonKernelsDedup:
    def test_dedup_in_resolve_and_rewrite(self):
        from magi_compiler._triton_introspect import introspect_fn, rewrite_fn_with_wrap_triton

        def fn(x):
            out = torch.empty_like(x)
            n = x.numel()
            _cos_kernel[_grid_1d(n)](x, out, n, BLOCK_SIZE=128)
            return out

        introspection = introspect_fn(fn, extra_triton_kernels=[_cos_kernel])
        resolved_bare = list(introspection.bare_triton_kernels)
        # Should appear exactly once even though it's both passed explicitly
        # and discovered by introspection.
        assert resolved_bare.count(_cos_kernel) == 1
        assert len(resolved_bare) == 1

        rewritten = rewrite_fn_with_wrap_triton(fn, resolved_bare)
        from torch._higher_order_ops.triton_kernel_wrap import TraceableTritonKernelWrapper

        wrapped = rewritten.__globals__["_cos_kernel"]
        assert isinstance(wrapped, TraceableTritonKernelWrapper)

    def test_dedup_e2e(self):
        @magi_register_custom_op(name="magi_test::dedup_cos", extra_triton_kernels=[_cos_kernel])
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            n = x.numel()
            _cos_kernel[_grid_1d(n)](x, out, n, BLOCK_SIZE=128)
            return out

        x = torch.randn(1024, device="cuda")
        assert_close(fn(x), torch.cos(x), atol=1e-5, rtol=1e-5)

        # Confirm we still went down the triton_op path even though the kernel
        # was specified twice (auto-detected + extra_triton_kernels).
        assert TestTritonOpRegistryAssertion._was_registered_as_triton_op(
            "magi_test::dedup_cos"
        ), "expected the op to be registered as a triton_op"


class TestExtraTritonKernelsForStaticOrClassmethod:
    """``staticmethod`` / ``classmethod`` selectors are opaque to source
    introspection. ``extra_triton_kernels`` keeps the op on the triton_op
    path even so.
    """

    def test_staticmethod_selected_kernel(self):
        @magi_register_custom_op(name="magi_test::sm_kernel", extra_triton_kernels=[_scale_kernel])
        def myop(x: torch.Tensor) -> torch.Tensor:
            kernel = _KernelHolder.get_static()
            out = torch.empty_like(x)
            n = x.numel()
            kernel[_grid_1d(n)](x, out, n, 2.0, BLOCK_SIZE=128)
            return out

        x = torch.randn(256, device="cuda")
        out = myop(x)
        assert_close(out, x * 2.0)

    def test_classmethod_selected_kernel(self):
        @magi_register_custom_op(name="magi_test::cm_kernel", extra_triton_kernels=[_scale_kernel])
        def myop(x: torch.Tensor) -> torch.Tensor:
            kernel = _KernelHolder.get_class()
            out = torch.empty_like(x)
            n = x.numel()
            kernel[_grid_1d(n)](x, out, n, 3.0, BLOCK_SIZE=128)
            return out

        x = torch.randn(256, device="cuda")
        out = myop(x)
        assert_close(out, x * 3.0)


class TestExtraTritonKernelsForRuntimeImport:
    """A kernel imported inside the function body (runtime import) is invisible
    to source introspection. ``extra_triton_kernels`` works around that.
    """

    def test_runtime_imported_kernel(self):
        # The kernel object lives at module scope (we can't actually do a fresh
        # ``import`` in a way that hides it from source scanning AND lets the
        # function still call it). Simulate the runtime-import case by stuffing
        # the kernel into a local ``import``-like alias derived from globals,
        # so source introspection cannot statically resolve it.
        @magi_register_custom_op(name="magi_test::runtime_import_kernel", extra_triton_kernels=[_cos_kernel])
        def myop(x: torch.Tensor) -> torch.Tensor:
            module_globals = globals()
            # Indirect lookup hides the kernel from static introspection of
            # ``myop``'s globals/closure.
            kernel = module_globals["_cos_kernel"]
            out = torch.empty_like(x)
            n = x.numel()
            kernel[_grid_1d(n)](x, out, n, BLOCK_SIZE=128)
            return out

        x = torch.randn(256, device="cuda")
        out = myop(x)
        assert_close(out, torch.cos(x))


class TestNoTritonFallback:
    def test_no_kernel_uses_custom_op(self):
        @magi_register_custom_op(name="magi_test::pure_python_op")
        def fn(x: torch.Tensor) -> torch.Tensor:
            return x * 2 + 1

        x = torch.randn(8, 8)
        assert_close(fn(x), x * 2 + 1)


# Triton path + autograd combination.


class TestIntrospection:
    def test_introspect_fn_flat(self):
        from magi_compiler._triton_introspect import introspect_fn

        def fn(x):
            out = torch.empty_like(x)
            n = x.numel()
            _cos_kernel[_grid_1d(n)](x, out, n, BLOCK_SIZE=128)
            return out

        assert _cos_kernel in introspect_fn(fn).bare_triton_kernels

    def test_introspect_fn_nested(self):
        from magi_compiler._triton_introspect import introspect_fn

        def fn(a, b):
            return _add_launcher(a, b)

        assert _add_kernel in introspect_fn(fn).bare_triton_kernels

    def test_rewrite_replaces_kernel_with_wrap_triton(self):
        from torch._higher_order_ops.triton_kernel_wrap import TraceableTritonKernelWrapper

        from magi_compiler._triton_introspect import introspect_fn, rewrite_fn_with_wrap_triton

        def fn(x):
            out = torch.empty_like(x)
            n = x.numel()
            _cos_kernel[_grid_1d(n)](x, out, n, BLOCK_SIZE=128)
            return out

        kernels = list(introspect_fn(fn).bare_triton_kernels)
        rewritten = rewrite_fn_with_wrap_triton(fn, kernels)

        # _cos_kernel name in the rewritten globals should now point to a
        # TraceableTritonKernelWrapper, not the bare JITFunction.
        assert isinstance(rewritten.__globals__["_cos_kernel"], TraceableTritonKernelWrapper)
        # Originals untouched.
        from triton.runtime.jit import JITFunction

        assert isinstance(_cos_kernel, JITFunction)

    def test_rewrite_propagates_through_helpers(self):
        from torch._higher_order_ops.triton_kernel_wrap import TraceableTritonKernelWrapper

        from magi_compiler._triton_introspect import introspect_fn, rewrite_fn_with_wrap_triton

        def fn(a, b):
            return _add_launcher(a, b)

        kernels = list(introspect_fn(fn).bare_triton_kernels)
        rewritten = rewrite_fn_with_wrap_triton(fn, kernels)

        rebuilt_launcher = rewritten.__globals__["_add_launcher"]
        assert isinstance(rebuilt_launcher.__globals__["_add_kernel"], TraceableTritonKernelWrapper)

    def test_introspect_fn_dot_run(self):
        """``kernel.run(*args, grid=...)`` is Triton's low-level launch
        API. It is what ``kernel[grid](*args)`` desugars to and what
        PyTorch Inductor's generated code uses. Verify the AST scanner
        recognises it as a bare kernel launch.
        """
        from magi_compiler._triton_introspect import introspect_fn

        def fn(x):
            out = torch.empty_like(x)
            n = x.numel()
            _cos_kernel.run(x, out, n, BLOCK_SIZE=128, grid=_grid_1d(n), warmup=False)
            return out

        assert _cos_kernel in introspect_fn(fn).bare_triton_kernels

    def test_introspect_fn_dotted_module_attr(self):
        """``mod.kernel[grid](...)`` references a kernel through a module
        attribute. The collector must record ``"mod.kernel"`` and the
        resolver must walk the dotted path via ``getattr`` to recover
        the underlying ``JITFunction``.
        """
        from magi_compiler._triton_introspect import introspect_fn
        from tests.api_tests import _triton_external_helpers as ext

        def fn(x):
            out = torch.empty_like(x)
            n = x.numel()
            ext.external_neg_kernel[_grid_1d(n)](x, out, n, BLOCK_SIZE=128)
            return out

        assert ext.external_neg_kernel in introspect_fn(fn).bare_triton_kernels

    def test_introspect_fn_class_attr(self):
        """``Holder.kernel[grid](...)`` references a kernel through a
        class attribute. Same mechanism as module attributes -- the
        collector records ``"Holder.kernel"`` and the resolver walks
        it via ``getattr``.
        """
        from magi_compiler._triton_introspect import introspect_fn

        class Holder:
            kernel = _cos_kernel

        def fn(x):
            out = torch.empty_like(x)
            n = x.numel()
            Holder.kernel[_grid_1d(n)](x, out, n, BLOCK_SIZE=128)
            return out

        assert _cos_kernel in introspect_fn(fn).bare_triton_kernels

    def test_introspect_fn_dot_run_with_dotted_receiver(self):
        """Combination of A.5 and dotted lookup: ``mod.kernel.run(...)``."""
        from magi_compiler._triton_introspect import introspect_fn
        from tests.api_tests import _triton_external_helpers as ext

        def fn(x):
            out = torch.empty_like(x)
            n = x.numel()
            ext.external_double_kernel.run(
                x, out, n, BLOCK_SIZE=128, grid=_grid_1d(n), warmup=False
            )
            return out

        assert ext.external_double_kernel in introspect_fn(fn).bare_triton_kernels


# Multi-level nesting: fn -> dispatch -> launcher -> kernel.
# Verifies that kernels several call-graph hops away are still detected and
# that ``rewrite_fn_with_wrap_triton`` rebuilds every helper along the path.


class TestInferOutputMetaOverride:
    def test_meta_list_form(self):
        @magi_register_custom_op(name="magi_test::triton_meta_list", infer_output_meta_fn=["x"])
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            n = x.numel()
            _cos_kernel[_grid_1d(n)](x, out, n, BLOCK_SIZE=128)
            return out

        x = torch.randn(1024, device="cuda")
        assert_close(fn(x), torch.cos(x), atol=1e-5, rtol=1e-5)
        # And inside torch.compile (forces the fake/meta path to be used).
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        assert_close(compiled(x), torch.cos(x), atol=1e-5, rtol=1e-5)

    def test_meta_callable_form(self):
        called = {"count": 0}

        def custom_meta(x: torch.Tensor) -> torch.Tensor:
            called["count"] += 1
            return torch.empty_like(x)

        @magi_register_custom_op(name="magi_test::triton_meta_callable", infer_output_meta_fn=custom_meta)
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            n = x.numel()
            _cos_kernel[_grid_1d(n)](x, out, n, BLOCK_SIZE=128)
            return out

        x = torch.randn(1024, device="cuda")
        assert_close(fn(x), torch.cos(x), atol=1e-5, rtol=1e-5)
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        assert_close(compiled(x), torch.cos(x), atol=1e-5, rtol=1e-5)
        # Tracing through torch.compile should have invoked the user-provided
        # meta at least once.
        assert called["count"] >= 1


# Explicit registry-level assertion that we actually went down the
# ``torch.library.triton_op`` path (i.e. Inductor would be able to inline
# the kernel), distinguishing it from the silent custom_op fallback.


class TestTritonOpRegistryAssertion:
    """Verify we actually take the ``torch.library.triton_op`` registration
    path (so Inductor / make_fx can see through the op) instead of silently
    falling back to plain ``custom_op`` (which would be opaque)."""

    @staticmethod
    def _was_registered_as_triton_op(op_or_name) -> bool:
        # ``triton_op`` installs a torch_dispatch on FunctionalTensorMode that
        # decomposes the op into ``triton_kernel_wrapper_mutation`` calls.
        # Plain ``custom_op`` does not.
        from torch._library.custom_ops import OPDEFS
        from torch._subclasses.functional_tensor import FunctionalTensorMode

        if isinstance(op_or_name, str):
            opdef = OPDEFS.get(op_or_name)
            if opdef is None:
                return False
        else:
            opdef = op_or_name
        dispatch_fns = getattr(opdef, "_torch_dispatch_fns", {}) or {}
        return FunctionalTensorMode in dispatch_fns

    def test_registered_as_triton_op(self):
        @magi_register_custom_op(name="magi_test::registry_cos")
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            n = x.numel()
            _cos_kernel[_grid_1d(n)](x, out, n, BLOCK_SIZE=128)
            return out

        assert self._was_registered_as_triton_op("magi_test::registry_cos"), (
            "magi_test::registry_cos should have been registered via "
            "torch.library.triton_op (so make_fx decomposes it into "
            "triton_kernel_wrapper_mutation), not via plain custom_op."
        )

    def test_pure_python_op_not_registered_as_triton(self):
        @magi_register_custom_op(name="magi_test::registry_pure_python")
        def fn(x: torch.Tensor) -> torch.Tensor:
            return x * 2 + 1

        assert not self._was_registered_as_triton_op("magi_test::registry_pure_python"), (
            "magi_test::registry_pure_python has no triton kernels; it should "
            "have fallen back to the custom_op path and remain opaque to "
            "make_fx."
        )


# extra_triton_kernels deduplication: a kernel that is *both* auto-detected
# and listed in ``extra_triton_kernels`` should appear exactly once after
# resolution and must not be wrap_triton-wrapped twice.


# ============================================================================
# SECTION 3: Autotune, Heuristics & Autograd in Triton
# ============================================================================


class TestAutotuneKernels:
    def test_autotuned(self):
        @magi_register_custom_op(name="magi_test::autotuned_cos")
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            n = x.numel()
            # autotuner picks BLOCK_SIZE; grid uses meta lambda
            _autotuned_cos_kernel[(triton.cdiv(n, 128),)](x, out, n)
            return out

        x = torch.randn(2048, device="cuda")
        assert_close(fn(x), torch.cos(x), atol=1e-5, rtol=1e-5)


class TestMultipleAutotuneKernelsSameOp:
    """A single op may launch several differently-autotuned kernels (a common
    FlashAttention / Mamba pattern). Verify both kernels are detected and the
    op runs end-to-end through the triton_op path.
    """

    def test_two_autotune_kernels_in_same_op(self):
        # Build a *second* autotuned kernel locally so we can be sure both
        # kernel objects appear in the op's call graph.
        @triton.autotune(
            configs=[triton.Config({"BLOCK_SIZE": 128}, num_warps=4), triton.Config({"BLOCK_SIZE": 256}, num_warps=4)],
            key=["n_elements"],
        )
        @triton.jit
        def _autotuned_sin_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr + offsets, mask=mask)
            tl.store(out_ptr + offsets, tl.sin(x), mask=mask)

        @magi_register_custom_op(name="magi_test::two_autotune_kernels", extra_triton_kernels=[_autotuned_sin_kernel])
        def myop(x: torch.Tensor) -> torch.Tensor:
            n = x.numel()
            mid = torch.empty_like(x)
            _autotuned_cos_kernel[_grid_1d(n)](x, mid, n)
            out = torch.empty_like(x)
            _autotuned_sin_kernel[_grid_1d(n)](mid, out, n)
            return out

        x = torch.randn(2048, device="cuda")
        out = myop(x)
        assert_close(out, torch.sin(torch.cos(x)), atol=1e-4, rtol=1e-4)


class TestHeuristicsRejection:
    """``torch.library.wrap_triton`` only accepts ``JITFunction`` and
    ``Autotuner``. A top-level ``@triton.heuristics`` produces a
    ``Heuristics`` instance that fails ``wrap_triton`` with a confusing
    error. ``@magi_register_custom_op`` rejects this case up front with a
    clearer message, while still accepting the recommended layering of
    ``@triton.autotune -> @triton.heuristics -> @triton.jit``.
    """

    def test_top_level_heuristics_rejected_with_clear_message(self):
        """Bare ``@triton.heuristics`` on a kernel referenced from the op
        body must be rejected at registration time, not deep inside
        ``wrap_triton``."""
        with pytest.raises(RuntimeError, match="triton.heuristics"):

            @magi_register_custom_op(name="magi_test::heuristics_top")
            def myop(x: torch.Tensor) -> torch.Tensor:
                out = torch.empty_like(x)
                n = x.numel()
                _heuristics_top_kernel[_grid_1d(n)](x, out, n)
                return out

    def test_top_level_heuristics_via_extra_triton_kernels_rejected(self):
        """Same constraint applies when the user passes the offending kernel
        through the ``extra_triton_kernels`` escape hatch (no auto-detection
        involved)."""
        with pytest.raises(RuntimeError, match="triton.heuristics"):

            @magi_register_custom_op(name="magi_test::heuristics_extra", extra_triton_kernels=[_heuristics_top_kernel])
            def myop(x: torch.Tensor) -> torch.Tensor:
                # Body doesn't reference the kernel at all; rejection comes
                # purely from the extra_triton_kernels list.
                return x.clone()

    def test_autotune_outside_heuristics_is_accepted(self):
        """The recommended layering ``@triton.autotune -> @triton.heuristics
        -> @triton.jit`` produces an ``Autotuner`` at the top level and is
        accepted (and end-to-end functional)."""

        @magi_register_custom_op(name="magi_test::autotune_over_heuristics")
        def myop(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            n = x.numel()
            _autotune_then_heuristics_kernel[_grid_1d(n)](x, out, n)
            return out

        x = torch.randn(512, device="cuda")
        out = myop(x)
        assert_close(out, x)


# #15 / #16: kernels not statically discoverable -> extra_triton_kernels=


class TestTritonWithAutograd:
    def test_triton_with_backward(self):
        def setup_ctx(ctx, inputs, output):
            (x,) = inputs
            ctx.save_for_backward(x)

        def backward(ctx, grad_out):
            (x,) = ctx.saved_tensors
            # d/dx cos(x) = -sin(x)
            return grad_out * (-torch.sin(x))

        @magi_register_custom_op(name="magi_test::triton_cos_grad", setup_context_fn=setup_ctx, backward_fn=backward)
        def mycos(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            n = x.numel()
            _cos_kernel[_grid_1d(n)](x, out, n, BLOCK_SIZE=128)
            return out

        x = torch.randn(1024, device="cuda", requires_grad=True)
        out = mycos(x)
        loss = out.sum()
        loss.backward()
        assert_close(x.grad, -torch.sin(x.detach()), atol=1e-5, rtol=1e-5)


# ============================================================================
# SECTION 4: Dataclass + Triton Bridge
# ----------------------------------------------------------------------------
# The dataclass-aware registration path lowers each dataclass parameter into
# flat primitive leaves before handing the function off to torch.library.
# This section verifies that the triton_op auto-detection still kicks in on
# that lowered path, including nested dataclasses, custom meta functions,
# autograd hooks, per-field grads, and ``is_compute_sensitive``.
# ============================================================================


class TestDataclassWithTritonKernel:
    def test_dataclass_input_with_triton(self):
        @magi_register_custom_op(name="magi_test::dc_cos")
        def fn(x: torch.Tensor, cfg: _DcCosCfg) -> torch.Tensor:
            out = torch.empty_like(x)
            n = x.numel()
            _cos_kernel[_grid_1d(n)](x, out, n, BLOCK_SIZE=cfg.block_size)
            return out

        x = torch.randn(1024, device="cuda")
        cfg = _DcCosCfg(block_size=128)
        assert_close(fn(x, cfg), torch.cos(x), atol=1e-5, rtol=1e-5)

        # The dataclass-aware path registers an inner op under the requested
        # name; that inner op should still be a triton_op so Inductor can see
        # through it.
        assert TestTritonOpRegistryAssertion._was_registered_as_triton_op(
            "magi_test::dc_cos"
        ), "dataclass+triton path should still register the inner op as a triton_op"


class TestNestedDataclassWithTritonKernel:
    def test_two_level_nested_dc_with_triton(self):
        """Outer dataclass containing an inner dataclass; both are lowered
        into flat primitive parameters."""

        @magi_register_custom_op(name="magi_test::nested_dc_cos_scale")
        def fn(x: torch.Tensor, cfg: _DcOuterCfg) -> torch.Tensor:
            tmp = torch.empty_like(x)
            out = torch.empty_like(x)
            n = x.numel()
            _cos_kernel[_grid_1d(n)](x, tmp, n, BLOCK_SIZE=cfg.kernel.block_size)
            _scale_kernel[_grid_1d(n)](tmp, out, n, cfg.scale, BLOCK_SIZE=cfg.kernel.block_size)
            return out + cfg.kernel.extra_offset

        x = torch.randn(1024, device="cuda")
        cfg = _DcOuterCfg(kernel=_DcKernelCfg(block_size=128, extra_offset=0.5), scale=2.5)
        out = fn(x, cfg)
        expected = torch.cos(x) * 2.5 + 0.5
        assert_close(out, expected, atol=1e-5, rtol=1e-5)

        # Sanity-check the param_mapping_tree exposes the expected lowered
        # leaf names.
        plan = fn._magi_param_mapping_tree
        cfg_node = plan[1]
        assert cfg_node[0] == "dataclass" and cfg_node[1] == "cfg"
        flat_names: list[str] = []

        def _collect(node):
            if node[0] == "primitive":
                flat_names.append(node[2])
            else:
                for child in node[3]:
                    _collect(child)

        _collect(cfg_node)
        assert {"cfg__kernel__block_size", "cfg__kernel__extra_offset", "cfg__scale"}.issubset(flat_names)

        assert TestTritonOpRegistryAssertion._was_registered_as_triton_op(
            "magi_test::nested_dc_cos_scale"
        ), "nested-dataclass + triton path should still register the inner op as a triton_op"

    def test_nested_dc_with_triton_and_meta_fn(self):
        """User-supplied meta function expressed in nested-dataclass terms,
        combined with a triton kernel call."""

        def _meta(x: torch.Tensor, cfg: _DcProjCfg) -> torch.Tensor:
            return x.new_empty((*x.shape[:-1], cfg.shape.out_dim))

        @magi_register_custom_op(name="magi_test::nested_dc_cos_proj", infer_output_meta_fn=_meta)
        def fn(x: torch.Tensor, cfg: _DcProjCfg) -> torch.Tensor:
            sliced = x[..., : cfg.shape.out_dim].contiguous()
            out = torch.empty_like(sliced)
            n = sliced.numel()
            _cos_kernel[_grid_1d(n)](sliced, out, n, BLOCK_SIZE=cfg.block_size)
            return out

        x = torch.randn(2, 8, device="cuda")
        cfg = _DcProjCfg(shape=_DcShapeCfg(out_dim=3), block_size=128)
        out = fn(x, cfg)
        expected = torch.cos(x[..., :3].contiguous())
        assert out.shape == (2, 3)
        assert_close(out, expected, atol=1e-5, rtol=1e-5)


class TestDataclassWithTritonKernelAndBackward:
    def test_triton_dc_backward_basic(self):
        """End-to-end backward against a dc + triton op: use the cos kernel
        (analytical grad: -sin(x)) so we can verify exact grads."""

        def _setup(ctx, inputs, output):
            x, cfg = inputs
            assert isinstance(cfg, _DcCosCfg)
            ctx.save_for_backward(x)
            ctx.block_size = cfg.block_size

        def _bwd(ctx, grad_out):
            (x,) = ctx.saved_tensors
            return grad_out * (-torch.sin(x)), None

        @magi_register_custom_op(name="magi_test::dc_cos_grad", setup_context_fn=_setup, backward_fn=_bwd)
        def mycos(x: torch.Tensor, cfg: _DcCosCfg) -> torch.Tensor:
            out = torch.empty_like(x)
            n = x.numel()
            _cos_kernel[_grid_1d(n)](x, out, n, BLOCK_SIZE=cfg.block_size)
            return out

        x = torch.randn(1024, device="cuda", requires_grad=True)
        cfg = _DcCosCfg(block_size=128)
        out = mycos(x, cfg)
        out.sum().backward()
        assert_close(x.grad, -torch.sin(x.detach()), atol=1e-5, rtol=1e-5)

        assert TestTritonOpRegistryAssertion._was_registered_as_triton_op("magi_test::dc_cos_grad")

    def test_triton_nested_dc_backward(self):
        """Nested dataclass + triton + backward. The bridge must spread the
        whole-nested-dc ``None`` grad over every flat slot under that
        dataclass."""

        def _setup(ctx, inputs, output):
            x, cfg = inputs
            assert isinstance(cfg, _DcOuterCfg)
            assert isinstance(cfg.kernel, _DcKernelCfg)
            ctx.save_for_backward(x)
            ctx.scale = cfg.scale

        def _bwd(ctx, grad_out):
            (x,) = ctx.saved_tensors
            return grad_out * (-torch.sin(x)) * ctx.scale, None

        @magi_register_custom_op(name="magi_test::nested_dc_cos_grad", setup_context_fn=_setup, backward_fn=_bwd)
        def fn(x: torch.Tensor, cfg: _DcOuterCfg) -> torch.Tensor:
            tmp = torch.empty_like(x)
            out = torch.empty_like(x)
            n = x.numel()
            _cos_kernel[_grid_1d(n)](x, tmp, n, BLOCK_SIZE=cfg.kernel.block_size)
            _scale_kernel[_grid_1d(n)](tmp, out, n, cfg.scale, BLOCK_SIZE=cfg.kernel.block_size)
            return out + cfg.kernel.extra_offset

        x = torch.randn(1024, device="cuda", requires_grad=True)
        cfg = _DcOuterCfg(kernel=_DcKernelCfg(block_size=128, extra_offset=0.5), scale=2.5)
        out = fn(x, cfg)
        out.sum().backward()
        expected = -torch.sin(x.detach()) * 2.5
        assert_close(x.grad, expected, atol=1e-5, rtol=1e-5)

    def test_triton_dc_backward_with_per_field_grad(self):
        """User returns per-field grads (as a same-shape dataclass with
        ``None`` leaves) for the dc slot. The triton path must still work."""

        def _setup(ctx, inputs, output):
            x, cfg = inputs
            ctx.save_for_backward(x)
            ctx.block_size = cfg.block_size

        def _bwd(ctx, grad_out):
            (x,) = ctx.saved_tensors
            return (grad_out * (-torch.sin(x)), _DcCosCfg(block_size=None))

        @magi_register_custom_op(name="magi_test::dc_cos_per_field_grad", setup_context_fn=_setup, backward_fn=_bwd)
        def mycos(x: torch.Tensor, cfg: _DcCosCfg) -> torch.Tensor:
            out = torch.empty_like(x)
            n = x.numel()
            _cos_kernel[_grid_1d(n)](x, out, n, BLOCK_SIZE=cfg.block_size)
            return out

        x = torch.randn(512, device="cuda", requires_grad=True)
        out = mycos(x, _DcCosCfg(block_size=128))
        out.sum().backward()
        assert_close(x.grad, -torch.sin(x.detach()), atol=1e-5, rtol=1e-5)

    def test_triton_dc_backward_with_dict_grad(self):
        """User returns the dataclass slot's grad as a plain ``dict``; the
        bridge must spread it through ``__getitem__``-style access into the
        underlying flat slots."""

        def _setup(ctx, inputs, output):
            x, cfg = inputs
            ctx.save_for_backward(x)
            ctx.block_size = cfg.block_size

        def _bwd(ctx, grad_out):
            (x,) = ctx.saved_tensors
            return grad_out * (-torch.sin(x)), {"block_size": None}

        @magi_register_custom_op(name="magi_test::dc_cos_dict_grad", setup_context_fn=_setup, backward_fn=_bwd)
        def mycos(x: torch.Tensor, cfg: _DcCosCfg) -> torch.Tensor:
            out = torch.empty_like(x)
            n = x.numel()
            _cos_kernel[_grid_1d(n)](x, out, n, BLOCK_SIZE=cfg.block_size)
            return out

        x = torch.randn(512, device="cuda", requires_grad=True)
        out = mycos(x, _DcCosCfg(block_size=128))
        out.sum().backward()
        assert_close(x.grad, -torch.sin(x.detach()), atol=1e-5, rtol=1e-5)


class TestDataclassTritonComputeSensitiveSmoke:
    """The dataclass-aware bridge composes cleanly with
    ``is_compute_sensitive=True`` on the triton path: registration succeeds,
    the op runs, and its name lands in the compute-sensitive registry.
    """

    def test_dataclass_triton_compute_sensitive(self):
        from magi_compiler.config import get_compile_config

        @magi_register_custom_op(name="magi_test::dc_triton_cs", is_compute_sensitive=True)
        def myop(x: torch.Tensor, cfg: _DcCosCfg) -> torch.Tensor:
            out = torch.empty_like(x)
            n = x.numel()
            _cos_kernel[_grid_1d(n)](x, out, n, BLOCK_SIZE=cfg.block_size)
            return out

        x = torch.randn(256, device="cuda")
        out = myop(x, _DcCosCfg(block_size=128))
        assert_close(out, torch.cos(x))
        assert "magi_test::dc_triton_cs" in get_compile_config().recompute_config.custom_compute_sensitive_ops


# Direct unit tests for the introspection / rewrite helpers.


class TestInductorSeesTritonKernel:
    """The whole point of the triton_op auto-detection is that
    ``torch.compile`` (Inductor) sees through the op to the underlying
    triton kernel rather than treating it as opaque. Verify by inspecting
    the FX graph captured by Inductor for the wrap_triton-functional HOP.
    """

    def test_triton_kernel_visible_in_aot_graph(self):
        from torch._functorch.aot_autograd import aot_function

        @magi_register_custom_op(name="magi_test::inductor_visible_cos")
        def mycos(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            n = x.numel()
            _cos_kernel[_grid_1d(n)](x, out, n, BLOCK_SIZE=128)
            return out

        # Run the op through AOTAutograd directly with a custom forward
        # compiler that just records the post-functionalization graph. This
        # is exactly the layer where ``triton_op``s decompose into the
        # ``triton_kernel_wrapper_functional`` HOP; the presence of that
        # node in the captured graph proves Inductor (which runs *after*
        # AOTAutograd) sees the underlying triton kernel rather than an
        # opaque ``torch.ops.magi_test.inductor_visible_cos`` call.
        captured_graphs: list[str] = []

        def _capture(gm, _example_inputs):
            captured_graphs.append(gm.code)
            return gm.forward

        x = torch.randn(1024, device="cuda")
        torch._dynamo.reset()
        compiled_aot = aot_function(mycos, fw_compiler=_capture, bw_compiler=_capture)
        out = compiled_aot(x)
        assert_close(out, torch.cos(x), atol=1e-5, rtol=1e-5)

        joined = "\n".join(captured_graphs)
        assert "triton_kernel_wrapper_functional" in joined or "triton_kernel_wrapper_mutation" in joined, (
            "AOT graph did not decompose magi_test::inductor_visible_cos "
            "into the triton_kernel_wrapper HOP; Inductor will treat it "
            "as opaque. Captured AOT graph:\n" + joined
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
