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
(``prefer_nd_tiling`` / ``max_tiles`` / ``tile_reductions``) ON. Whether it does
so is driven by the ``enable_nd_tiling_workaround`` config:

  * ``True``  -> force on, skip heuristics
  * ``False`` -> pass not registered at all
  * ``None``  -> auto: on iff dynamic shapes AND PyTorch < 2.11.0 AND not conv-heavy

These tests assert that mapping. The shared base-class utilities (``is_dynamic``,
``is_conv_heavy``, config snapshot/anti-leakage) are tested in
``test_magi_inductor_pass.py``; the end-to-end speedup in
``tests/perf_tests/test_dynamic_nd_tiling_perf.py``.
"""

import pytest
import torch
from torch.torch_version import TorchVersion

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


def _make_pass(*, force_on=False, version="2.9.1"):
    return ND_TilingWorkaroundPass(force_on=force_on, torch_version=TorchVersion(version))


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
    """Dynamic graph with 1 conv + plenty of filler nodes (nnodes >= 300 * nconv)."""
    return build_graph_module(fake_mode, placeholder_vals=[dynamic_tensor(fake_mode)], n_conv=1, n_filler=320)


# ── config field tri-state ───────────────────────────────────────────────


@pytest.mark.parametrize("value", [True, False, None])
def test_config_field_tristate(value):
    """enable_nd_tiling_workaround accepts True/False/None."""
    assert PassConfig(enable_nd_tiling_workaround=value).enable_nd_tiling_workaround is value


def test_config_field_default_is_none():
    assert PassConfig().enable_nd_tiling_workaround is None


# ── ND-tiling injection decision ─────────────────────────────────────────


def test_force_on_injects_even_when_static_and_fixed_version(fake_mode):
    """force_on=True flips the config regardless of graph/version."""
    pass_ = _make_pass(force_on=True, version="2.11.0")
    gm = build_graph_module(fake_mode, placeholder_vals=[static_tensor(fake_mode)], n_conv=1, n_filler=0)
    pass_(gm.graph)
    _assert_injected(True)


def test_auto_injects_when_all_conditions_met(fake_mode):
    pass_ = _make_pass(version="2.9.1")
    gm = _auto_eligible_graph(fake_mode)
    pass_(gm.graph)
    _assert_injected(True)


def test_auto_skips_on_static_shapes(fake_mode):
    pass_ = _make_pass(version="2.9.1")
    gm = build_graph_module(fake_mode, placeholder_vals=[static_tensor(fake_mode)], n_conv=0, n_filler=5)
    pass_(gm.graph)
    _assert_injected(False)


def test_auto_skips_on_fixed_version(fake_mode):
    """Dynamic shapes but PyTorch >= 2.11.0: native coalesce path handles it."""
    pass_ = _make_pass(version="2.11.0")
    gm = _auto_eligible_graph(fake_mode)
    pass_(gm.graph)
    _assert_injected(False)


def test_auto_skips_when_graph_too_conv_dense(fake_mode):
    """``nnodes < 300 * nconv`` (conv-dense graph): the heuristic bails out."""
    pass_ = _make_pass(version="2.9.1")
    gm = build_graph_module(fake_mode, placeholder_vals=[dynamic_tensor(fake_mode)], n_conv=1, n_filler=5)
    pass_(gm.graph)
    _assert_injected(False)
