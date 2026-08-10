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
    apply iff dynamic shapes AND conv-heavy.
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
        (cfg.triton.prefer_nd_tiling, cfg.triton.max_tiles, cfg.triton.tile_reductions) = saved


from magi_compiler.utils.envs import IS_PT_212


def _assert_injected(injected):
    cfg = torch._inductor.config
    if injected:
        assert cfg.triton.prefer_nd_tiling is True
        expected_max_tiles = 2 if IS_PT_212 else 3
        assert cfg.triton.max_tiles == expected_max_tiles
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
    pass_ = ND_TilingWorkaroundPass()
    gm = _auto_eligible_graph(fake_mode)
    pass_(gm.graph)
    _assert_injected(True)


def test_auto_skips_on_static_shapes(fake_mode):
    pass_ = ND_TilingWorkaroundPass()
    gm = build_graph_module(fake_mode, placeholder_vals=[static_tensor(fake_mode)], n_conv=0, n_filler=5)
    pass_(gm.graph)
    _assert_injected(False)


def test_auto_skips_when_graph_not_conv_heavy(fake_mode):
    """``nnodes >= 300 * nconv`` (conv-sparse graph): low conv ratio, ND-tiling gives little, so skip."""
    pass_ = ND_TilingWorkaroundPass()
    gm = build_graph_module(fake_mode, placeholder_vals=[dynamic_tensor(fake_mode)], n_conv=1, n_filler=320)
    pass_(gm.graph)
    _assert_injected(False)


# ---------------------------------------------------------------------------
# GPU integration tests: reproduce the max_tiles=3 Inductor codegen bug and
# verify the max_tiles=2 fix.
#
# PT 2.12 Inductor generates a 3D-grid reduction kernel (program_id(2)) when
# max_tiles=3 + tile_reductions=True, but launches with a 2D grid, causing
# Triton to crash with AttributeError("'NoneType' ... 'type'").
# Triton itself supports program_id(2) fine; the bug is in Inductor codegen.
# ---------------------------------------------------------------------------


class _ConvGroupNorm(torch.nn.Module):
    """Minimal model that triggers a fused convolution + group_norm reduction kernel.

    48 channels + GroupNorm(8) produces a ``triton_red_fused_convolution_native_group_norm``
    kernel whose pointwise + reduction split, combined with max_tiles=3, makes Inductor
    generate ``tl.program_id(2)`` for a grid dim that doesn't exist.
    """

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(48, 48, 3, padding=1)
        self.gn = torch.nn.GroupNorm(8, 48)

    def forward(self, x):
        return self.gn(self.conv(x))


_BUG_INPUT_SHAPE = (1, 48, 7, 34, 60)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(not IS_PT_212, reason="bug only manifests on PT >= 2.12")
def test_max_tiles_3_crashes_on_pt212():
    """Reproduce: max_tiles=3 + tile_reductions generates invalid 3D-grid kernel on PT 2.12."""
    torch._inductor.config.triton.prefer_nd_tiling = True
    torch._inductor.config.triton.max_tiles = 3
    torch._inductor.config.triton.tile_reductions = True

    model = _ConvGroupNorm().cuda().bfloat16().eval()
    x = torch.randn(*_BUG_INPUT_SHAPE, device="cuda", dtype=torch.bfloat16)

    compiled = torch.compile(model, backend="inductor")
    with pytest.raises(Exception, match="NoneType|program_id|InductorError"):
        with torch.no_grad():
            compiled(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_max_tiles_2_compiles_successfully():
    """Verify: max_tiles=2 avoids the 3D-grid bug on all PT versions."""
    torch._inductor.config.triton.prefer_nd_tiling = True
    torch._inductor.config.triton.max_tiles = 2
    torch._inductor.config.triton.tile_reductions = True

    model = _ConvGroupNorm().cuda().bfloat16().eval()
    x = torch.randn(*_BUG_INPUT_SHAPE, device="cuda", dtype=torch.bfloat16)

    compiled = torch.compile(model, backend="inductor")
    with torch.no_grad():
        out = compiled(x)
    assert out.shape == _BUG_INPUT_SHAPE


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(not IS_PT_212, reason="version-aware logic only differs on PT >= 2.12")
def test_nd_tiling_pass_uses_safe_max_tiles_on_pt212(fake_mode):
    """End-to-end: the pass itself picks max_tiles=2 on PT 2.12."""
    pass_ = ND_TilingWorkaroundPass()
    gm = _auto_eligible_graph(fake_mode)
    pass_(gm.graph)

    assert torch._inductor.config.triton.prefer_nd_tiling is True
    assert torch._inductor.config.triton.max_tiles == 2
    assert torch._inductor.config.triton.tile_reductions is True
