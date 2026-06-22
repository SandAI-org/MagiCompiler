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

"""Decision-logic tests for ``ND_TilingWorkaroundPass``.

When applicable, the pass flips three ``torch._inductor.config`` triton keys
(``prefer_nd_tiling`` / ``max_tiles`` / ``tile_reductions``) ON. The binary
``enable_nd_tiling_workaround`` config controls registration:

  * ``True`` (default) -> register the pass; its internal heuristics then decide:
    apply iff dynamic shapes AND PyTorch < 2.11.0 AND conv-heavy.
  * ``False`` -> pass not registered at all.

These tests assert the registered pass's heuristic decision. The shared
base-class utilities (``is_dynamic``, ``is_conv_heavy``, config
snapshot/anti-leakage) are tested in ``test_magi_inductor_pass.py``; the
end-to-end speedup in ``tests/perf_tests/test_nd_tiling_perf_workaround.py``.
"""

import pytest
import torch

from magi_compiler.config import PassConfig
from magi_compiler.passes.piecewise_graph.nd_tiling_workaround import ND_TilingWorkaroundPass
from tests.feature_tests.conftest import build_graph_module, dynamic_tensor, static_tensor


@pytest.fixture(autouse=True)
def _restore_inductor_config():
    """Snapshot/restore the three triton keys around every test.

    The pass mutates the process-global ``torch._inductor.config`` directly, so
    without this fixture one test could leak into the next.
    """
    cfg = torch._inductor.config
    saved = (cfg.triton.prefer_nd_tiling, cfg.triton.max_tiles, cfg.triton.tile_reductions)
    try:
        yield
    finally:
        cfg.triton.prefer_nd_tiling, cfg.triton.max_tiles, cfg.triton.tile_reductions = saved


def _make_pass(*, is_target_torch_version=True):
    """Build the pass and pin its version gate.

    The pass reads ``torch.__version__`` at construction and caches whether it is
    a target version (< 2.11.0) in ``is_target_torch_version``. Tests override that
    cached flag directly so the version branch is exercised regardless of the
    installed torch.
    """
    pass_ = ND_TilingWorkaroundPass()
    pass_.is_target_torch_version = is_target_torch_version
    return pass_


def _assert_injected(injected):
    cfg = torch._inductor.config
    if injected:
        assert cfg.triton.prefer_nd_tiling is True
        assert cfg.triton.max_tiles == 3
        assert cfg.triton.tile_reductions is True
    else:
        assert cfg.triton.prefer_nd_tiling is False
        assert cfg.triton.max_tiles is None
        assert cfg.triton.tile_reductions is False


def _auto_eligible_graph(fake_mode):
    """Dynamic, conv-heavy graph (nnodes < 300 * nconv): the workaround applies."""
    return build_graph_module(fake_mode, placeholder_vals=[dynamic_tensor(fake_mode)], n_conv=1, n_filler=5)


@pytest.mark.parametrize("value", [True, False])
def test_config_field_binary(value):
    """enable_nd_tiling_workaround accepts True/False (default True covered in test_magi_inductor_pass)."""
    assert PassConfig(enable_nd_tiling_workaround=value).enable_nd_tiling_workaround is value


def test_auto_injects_when_all_conditions_met(fake_mode):
    pass_ = _make_pass(is_target_torch_version=True)
    gm = _auto_eligible_graph(fake_mode)
    pass_(gm.graph)
    _assert_injected(True)


def test_auto_skips_on_static_shapes(fake_mode):
    pass_ = _make_pass(is_target_torch_version=True)
    gm = build_graph_module(fake_mode, placeholder_vals=[static_tensor(fake_mode)], n_conv=0, n_filler=5)
    pass_(gm.graph)
    _assert_injected(False)


def test_auto_skips_on_fixed_version(fake_mode):
    """Dynamic shapes but PyTorch >= 2.11.0: native coalesce path handles it."""
    pass_ = _make_pass(is_target_torch_version=False)
    gm = _auto_eligible_graph(fake_mode)
    pass_(gm.graph)
    _assert_injected(False)


def test_auto_skips_when_graph_not_conv_heavy(fake_mode):
    """``nnodes >= 300 * nconv`` (conv-sparse graph): low conv ratio, ND-tiling gives little, so skip."""
    pass_ = _make_pass(is_target_torch_version=True)
    gm = build_graph_module(fake_mode, placeholder_vals=[dynamic_tensor(fake_mode)], n_conv=1, n_filler=320)
    pass_(gm.graph)
    _assert_injected(False)
