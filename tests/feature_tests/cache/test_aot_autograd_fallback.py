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

"""Tests for the eager backward lowering fix in PiecewiseCompiler.

Root cause (PyTorch Issue #152022, fixed in PR #185635 but not in our version):
standalone_compile() calls save_cache_artifacts() immediately after compile_fx(),
but for training graphs AOTAutograd lazily defers backward lowering to the first
.backward() call.  This means aot_autograd_artifacts is empty when save() runs.

Our fix: _force_eager_backward_lowering() monkey-patches AOTConfig.__post_init__
to set force_non_lazy_backward_lowering=True, which makes AOTAutograd compile
the backward eagerly before save_cache_artifacts() runs.

Test 1: ``test_lazy_backward_causes_empty_aot_artifacts``
    Reproduces the bug: AOTConfig with force_non_lazy_backward_lowering=False
    → backward compiled lazily → aot_autograd_artifacts stays empty → save() asserts.

Test 2: ``test_eager_backward_populates_aot_artifacts``
    With _force_eager_backward_lowering(), AOTConfig gets
    force_non_lazy_backward_lowering=True → backward compiled eagerly →
    aot_autograd_artifacts is populated.

Test 3: ``test_context_manager_restores_post_init``
    Verifies _force_eager_backward_lowering() properly restores __post_init__
    after exiting the context.
"""
from __future__ import annotations

import dataclasses
from collections import defaultdict

import pytest


@dataclasses.dataclass
class FakeCacheInfo:
    """Minimal mock of torch.compiler._cache.CacheInfo."""

    artifacts: defaultdict = dataclasses.field(default_factory=lambda: defaultdict(list))

    @property
    def aot_autograd_artifacts(self):
        return self.artifacts["aot_autograd"]


class FakeCompiledGraph:
    """Minimal mock of standalone_compile result."""

    def __init__(self, *, aot_autograd_key: str | None = None):
        ci = FakeCacheInfo()
        ci.artifacts["inductor"] = ["fake_inductor_key"]
        if aot_autograd_key:
            ci.artifacts["aot_autograd"] = [aot_autograd_key]
        self._artifacts = (b"fake_bytes", ci)

    def save(self, *, path: str, format: str):
        """Mimic the real save() assert from standalone_compile.py:74."""
        _, cache_info = self._artifacts
        assert (
            len(cache_info.aot_autograd_artifacts) == 1
        ), f"Expected 1 aot_autograd artifact, got {len(cache_info.aot_autograd_artifacts)}: {cache_info}"


def test_lazy_backward_causes_empty_aot_artifacts():
    """Reproduce: without the fix, AOTConfig.force_non_lazy_backward_lowering=False
    causes backward to be lazy, leaving aot_autograd_artifacts empty → save() fails.

    This simulates the exact assertion error seen in the H200 monolithic bake:
    'AssertionError: CacheInfo(..., aot_autograd_artifacts=[], ...)'
    """
    from torch._functorch._aot_autograd.schemas import AOTConfig

    # Simulate creating AOTConfig without the fix
    config = AOTConfig(
        fw_compiler=lambda *a, **k: None,
        bw_compiler=lambda *a, **k: None,
        partition_fn=lambda *a, **k: None,
        decompositions={},
        num_params_buffers=0,
        aot_id=0,
        keep_inference_input_mutations=False,
    )

    # Without fix: force_non_lazy_backward_lowering is False
    assert config.force_non_lazy_backward_lowering is False, "Default should be False (lazy backward)"

    # This causes aot_autograd_artifacts to be empty → save() fails
    graph = FakeCompiledGraph(aot_autograd_key=None)  # no backward artifact
    with pytest.raises(AssertionError, match="aot_autograd"):
        graph.save(path="/tmp/fake", format="unpacked")


def test_eager_backward_populates_aot_artifacts():
    """With _force_eager_backward_lowering(), AOTConfig gets
    force_non_lazy_backward_lowering=True, enabling eager backward compilation.
    This means aot_autograd_artifacts will be populated before save() runs."""
    from torch._functorch._aot_autograd.schemas import AOTConfig

    from magi_compiler.magi_backend.piecewise_compiler import _force_eager_backward_lowering

    with _force_eager_backward_lowering():
        config = AOTConfig(
            fw_compiler=lambda *a, **k: None,
            bw_compiler=lambda *a, **k: None,
            partition_fn=lambda *a, **k: None,
            decompositions={},
            num_params_buffers=0,
            aot_id=0,
            keep_inference_input_mutations=False,
        )

        # With fix: force_non_lazy_backward_lowering is True
        assert config.force_non_lazy_backward_lowering is True, "Fix should set force_non_lazy_backward_lowering=True"

    # Simulate what happens when backward IS compiled eagerly:
    # aot_autograd_artifacts is populated → save() succeeds
    graph = FakeCompiledGraph(aot_autograd_key="real_backward_key")
    graph.save(path="/tmp/fake", format="unpacked")  # should NOT raise


def test_context_manager_restores_post_init():
    """_force_eager_backward_lowering() must restore __post_init__ on exit."""
    from torch._functorch._aot_autograd.schemas import AOTConfig

    from magi_compiler.magi_backend.piecewise_compiler import _force_eager_backward_lowering

    orig_post_init = AOTConfig.__post_init__

    with _force_eager_backward_lowering():
        # Inside: post_init is patched
        assert AOTConfig.__post_init__ is not orig_post_init

    # Outside: post_init is restored
    assert AOTConfig.__post_init__ is orig_post_init

    # Verify default behavior is back to lazy
    config = AOTConfig(
        fw_compiler=lambda *a, **k: None,
        bw_compiler=lambda *a, **k: None,
        partition_fn=lambda *a, **k: None,
        decompositions={},
        num_params_buffers=0,
        aot_id=0,
        keep_inference_input_mutations=False,
    )
    assert config.force_non_lazy_backward_lowering is False
