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

"""Logic tests for the dynamic-shape Triton ND-tiling workaround.

All the decision logic lives in
``MagiBackend._configure_custom_passes_by_graph_info``, which (a) probes
``is_dynamic`` from the graph's free symbols and (b) injects the ND-tiling
Inductor config based on the tri-state ``enable_dynamic_nd_tiling`` config:

  * ``True``  -> force on
  * ``False`` -> force off
  * ``None``  -> auto: dynamic shapes AND PyTorch < 2.11.0 AND ``nnodes > 300 * nconv``

The end-to-end speedup is validated separately in
``tests/perf_tests/test_dynamic_nd_tiling_perf.py``.
"""

import pytest
import torch
import torch.fx as fx
from torch.torch_version import TorchVersion

from magi_compiler.config import CompileConfig, get_compile_config
from magi_compiler.magi_backend import magi_backend as mb
from magi_compiler.magi_backend.magi_backend import MagiBackend

ND_TILING_KEYS = ("triton.prefer_nd_tiling", "triton.max_tiles", "triton.tile_reductions")


def _set_torch_version(monkeypatch, version):
    """Patch the module-level parsed torch version used by the gating logic.

    ``TORCH_VERSION`` is parsed once at import time, so tests override the
    parsed constant directly instead of patching ``torch.__version__``.
    """
    monkeypatch.setattr(mb, "TORCH_VERSION", TorchVersion(version))


def _make_backend(*, enable_dynamic_nd_tiling=None):
    """Build a MagiBackend without running the heavy __init__.

    We bypass __init__ (which would spin up a CompilerManager) and set only the
    attributes that ``_configure_custom_passes_by_graph_info`` reads.
    """
    backend = MagiBackend.__new__(MagiBackend)
    backend.compile_config = get_compile_config().model_copy(update={"enable_dynamic_nd_tiling": enable_dynamic_nd_tiling})
    backend.inductor_compile_config = {}
    return backend


@pytest.fixture
def fake_mode():
    """A FakeTensorMode backed by a fresh ShapeEnv for symbolic shapes."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    return FakeTensorMode(shape_env=ShapeEnv())


def _static_tensor(fake_mode):
    """A FakeTensor with fully concrete (non-symbolic) dims."""
    with fake_mode:
        return torch.empty(4, 8)


def _dynamic_tensor(fake_mode):
    """A FakeTensor whose first dim is a free symbol (mimics a dynamic batch)."""
    sym = fake_mode.shape_env.create_unbacked_symint()
    with fake_mode:
        return torch.empty(sym, 8)


def _symint(fake_mode):
    """A bare (scalar) ``torch.SymInt`` carrying a free symbol."""
    return fake_mode.shape_env.create_unbacked_symint()


def _build_graph(fake_mode, *, placeholder_vals=(), n_conv=0, n_filler=0):
    """Build a tiny fx.GraphModule for the decision logic.

    ``placeholder_vals`` populate placeholder ``meta["example_value"]`` (drive
    ``is_dynamic``). ``n_conv`` adds ``aten.convolution.default`` call nodes
    (each fed a weight placeholder carrying a 5-D ``meta["val"]`` so the
    conv-dim bookkeeping works), and ``n_filler`` adds plain call nodes to
    inflate the node count (drives the ``nnodes > 300 * nconv`` heuristic).
    """
    graph = fx.Graph()
    inputs = []
    for i, ev in enumerate(placeholder_vals):
        node = graph.placeholder(f"arg_{i}")
        node.meta["example_value"] = ev
        inputs.append(node)
    if not inputs:
        inputs.append(graph.placeholder("arg_0"))

    with fake_mode:
        weight_val = torch.empty(8, 8, 3, 3, 3)  # 5-D weight => conv3d (dim()-2 == 3)

    x = inputs[0]
    for c in range(n_conv):
        weight = graph.placeholder(f"weight_{c}")
        weight.meta["val"] = weight_val
        x = graph.call_function(torch.ops.aten.convolution.default, args=(x, weight))
    for _ in range(n_filler):
        x = graph.call_function(torch.ops.aten.relu.default, args=(x,))
    graph.output((x,))
    return fx.GraphModule(torch.nn.Module(), graph)


# ── is_dynamic detection ─────────────────────────────────────────────────


def _is_dynamic(fake_mode, *, placeholder_vals=(), example_inputs=()):
    backend = _make_backend()
    gm = _build_graph(fake_mode, placeholder_vals=placeholder_vals)
    backend._configure_custom_passes_by_graph_info(gm, list(example_inputs))
    return backend.is_dynamic


def test_is_dynamic_static(fake_mode):
    static = _static_tensor(fake_mode)
    assert _is_dynamic(fake_mode, placeholder_vals=[static], example_inputs=[static]) is False


def test_is_dynamic_via_placeholder_symint(fake_mode):
    """A symbolic dim on a placeholder example_value marks the compilation dynamic."""
    dynamic = _dynamic_tensor(fake_mode)
    assert _is_dynamic(fake_mode, placeholder_vals=[dynamic], example_inputs=[]) is True


def test_is_dynamic_via_example_inputs(fake_mode):
    """A symbolic dim on an example input marks the compilation dynamic."""
    dynamic = _dynamic_tensor(fake_mode)
    assert _is_dynamic(fake_mode, placeholder_vals=[], example_inputs=[dynamic]) is True


def test_is_dynamic_ignores_none_placeholder(fake_mode):
    """A missing/None placeholder example_value must not crash has_free_symbols."""
    static = _static_tensor(fake_mode)
    assert _is_dynamic(fake_mode, placeholder_vals=[None, static], example_inputs=[static]) is False


def test_is_dynamic_ignores_plain_int(fake_mode):
    """Non-symbolic scalar inputs (plain ints) are treated as static."""
    assert _is_dynamic(fake_mode, placeholder_vals=[], example_inputs=[3, 8]) is False


def test_is_dynamic_via_bare_symint_input(fake_mode):
    """A bare SymInt scalar input (no .shape) is correctly detected as dynamic."""
    assert _is_dynamic(fake_mode, placeholder_vals=[], example_inputs=[_symint(fake_mode)]) is True


# ── env var -> config field ──────────────────────────────────────────────


@pytest.mark.parametrize("env_val, expected", [("1", True), ("0", False)])
def test_env_var_drives_config_field(monkeypatch, env_val, expected):
    """MAGI_COMPILE_ENABLE_DYNAMIC_ND_TILING populates the config field directly."""
    monkeypatch.setenv("MAGI_COMPILE_ENABLE_DYNAMIC_ND_TILING", env_val)
    assert CompileConfig().enable_dynamic_nd_tiling is expected


# ── ND-tiling injection decision ─────────────────────────────────────────
#
# Injection requires either ``enable_dynamic_nd_tiling is True`` OR
# (``is None`` AND dynamic AND torch < 2.11.0 AND ``nnodes > 300 * nconv``).
# We build an auto-eligible graph (dynamic input + one conv + enough filler
# nodes so ``nnodes > 300 * nconv``) and then flip one condition per test.


def _assert_injected(backend, injected):
    if injected:
        assert backend.inductor_compile_config["triton.prefer_nd_tiling"] is True
        assert backend.inductor_compile_config["triton.max_tiles"] == 3
        assert backend.inductor_compile_config["triton.tile_reductions"] is True
    else:
        for key in ND_TILING_KEYS:
            assert key not in backend.inductor_compile_config


def _auto_eligible_graph(fake_mode):
    """Dynamic graph with 1 conv + plenty of filler nodes (nnodes > 300 * nconv)."""
    return _build_graph(fake_mode, placeholder_vals=[_dynamic_tensor(fake_mode)], n_conv=1, n_filler=320)


def test_force_on_injects_even_when_static(monkeypatch, fake_mode):
    """enable_dynamic_nd_tiling=True forces injection regardless of graph/version."""
    _set_torch_version(monkeypatch, "2.11.0")  # a "fixed" version the auto path would skip
    backend = _make_backend(enable_dynamic_nd_tiling=True)
    gm = _build_graph(fake_mode, placeholder_vals=[_static_tensor(fake_mode)], n_conv=1, n_filler=0)
    backend._configure_custom_passes_by_graph_info(gm, [])
    _assert_injected(backend, True)


def test_force_off_skips_even_when_auto_eligible(monkeypatch, fake_mode):
    """enable_dynamic_nd_tiling=False skips injection even when auto would enable."""
    _set_torch_version(monkeypatch, "2.9.1")
    backend = _make_backend(enable_dynamic_nd_tiling=False)
    backend._configure_custom_passes_by_graph_info(_auto_eligible_graph(fake_mode), [])
    _assert_injected(backend, False)


def test_auto_injects_when_all_conditions_met(monkeypatch, fake_mode):
    _set_torch_version(monkeypatch, "2.9.1")
    backend = _make_backend(enable_dynamic_nd_tiling=None)
    backend._configure_custom_passes_by_graph_info(_auto_eligible_graph(fake_mode), [])
    _assert_injected(backend, True)


def test_auto_skips_on_static_shapes(monkeypatch, fake_mode):
    _set_torch_version(monkeypatch, "2.9.1")
    backend = _make_backend(enable_dynamic_nd_tiling=None)
    gm = _build_graph(fake_mode, placeholder_vals=[_static_tensor(fake_mode)], n_conv=0, n_filler=5)
    backend._configure_custom_passes_by_graph_info(gm, [])
    _assert_injected(backend, False)


def test_auto_skips_on_fixed_version(monkeypatch, fake_mode):
    """Dynamic shapes but PyTorch >= 2.11.0: native coalesce path handles it."""
    _set_torch_version(monkeypatch, "2.11.0")
    backend = _make_backend(enable_dynamic_nd_tiling=None)
    backend._configure_custom_passes_by_graph_info(_auto_eligible_graph(fake_mode), [])
    _assert_injected(backend, False)


def test_auto_skips_when_graph_too_conv_dense(monkeypatch, fake_mode):
    """``nnodes <= 300 * nconv`` (conv-dense graph): the heuristic bails out."""
    _set_torch_version(monkeypatch, "2.9.1")
    backend = _make_backend(enable_dynamic_nd_tiling=None)
    # 1 conv + few filler nodes => nnodes well under 300 * 1 = 300.
    gm = _build_graph(fake_mode, placeholder_vals=[_dynamic_tensor(fake_mode)], n_conv=1, n_filler=5)
    backend._configure_custom_passes_by_graph_info(gm, [])
    _assert_injected(backend, False)
