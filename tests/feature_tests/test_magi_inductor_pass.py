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

"""Unit tests for the base class MagiInductorPass and its helper utilities."""

import pytest
import torch
from pydantic import ValidationError

from magi_compiler.config import CompileConfig
from magi_compiler.passes.pass_base import MagiInductorPass, snapshot_original_inductor_configs
from tests.feature_tests.conftest import build_graph_module, dynamic_tensor, static_tensor

_ND_TILING_ENV = "MAGI_COMPILE_PASS_CONFIG__ENABLE_ND_TILING_WORKAROUND"


class DummyPass(MagiInductorPass):
    inductor_config_keys_potentially_mutated_by_this_pass = ("triton.prefer_nd_tiling", "triton.max_tiles")

    def __call__(self, graph: torch.fx.Graph):
        # Mutate the configs during execution
        torch._inductor.config.triton.prefer_nd_tiling = True
        torch._inductor.config.triton.max_tiles = 3


class UndeclaredPass(MagiInductorPass):
    def __call__(self, graph: torch.fx.Graph):
        pass


def test_is_dynamic(fake_mode):
    """is_dynamic flags graphs whose placeholders carry free symbols."""
    pass_ = DummyPass()

    # Static graph
    gm_static = build_graph_module(fake_mode, placeholder_vals=[static_tensor(fake_mode)])
    assert not pass_.is_dynamic(gm_static.graph)

    # Dynamic graph
    gm_dynamic = build_graph_module(fake_mode, placeholder_vals=[dynamic_tensor(fake_mode)])
    assert pass_.is_dynamic(gm_dynamic.graph)


def test_is_dynamic_reads_val_meta_key(fake_mode):
    """is_dynamic prefers meta["val"], falling back to meta["example_value"]."""
    pass_ = DummyPass()

    # Symbol carried under meta["val"] (the preferred key) must be detected.
    gm_val = build_graph_module(fake_mode, placeholder_vals=[dynamic_tensor(fake_mode)], placeholder_meta_key="val")
    assert pass_.is_dynamic(gm_val.graph)

    # A placeholder with neither meta key set is treated as static (not dynamic).
    gm_no_meta = build_graph_module(fake_mode)
    assert not pass_.is_dynamic(gm_no_meta.graph)


def test_is_conv_heavy(fake_mode):
    """is_conv_heavy flags graphs whose node count is dense relative to convs."""
    pass_ = DummyPass()

    # 1 conv, 50 filler nodes -> nnodes = 53 (1 input + 1 weight + 1 conv + 50 relu).
    # threshold = 300, threshold * nconv = 300. nnodes < 300 -> is_conv_heavy is True.
    gm_heavy = build_graph_module(fake_mode, n_conv=1, n_filler=50)
    assert pass_.is_conv_heavy(gm_heavy.graph, threshold=300)

    # 1 conv, 320 filler nodes -> nnodes = 323.
    # threshold = 300, threshold * nconv = 300. nnodes >= 300 -> is_conv_heavy is False.
    gm_light = build_graph_module(fake_mode, n_conv=1, n_filler=320)
    assert not pass_.is_conv_heavy(gm_light.graph, threshold=300)


def test_is_conv_heavy_zero_conv(fake_mode):
    """A graph with no convolutions is never conv-heavy (threshold * 0 == 0)."""
    pass_ = DummyPass()
    gm_no_conv = build_graph_module(fake_mode, n_conv=0, n_filler=10)
    assert not pass_.is_conv_heavy(gm_no_conv.graph, threshold=300)


def test_snapshot_original_inductor_configs():
    """Verify that snapshot_original_inductor_configs snapshots declared keys correctly."""
    cfg = {}
    pass_ = DummyPass()
    snapshot_original_inductor_configs([pass_], cfg)

    # Check that declared keys are snapshotted
    for key in pass_.inductor_config_keys_potentially_mutated_by_this_pass:
        assert key in cfg

    # setdefault: a value already set by the user/upstream is preserved
    cfg_with_preset = {"triton.prefer_nd_tiling": True}
    snapshot_original_inductor_configs([pass_], cfg_with_preset)
    assert cfg_with_preset["triton.prefer_nd_tiling"] is True

    # A pass declaring an empty tuple contributes no anchors
    cfg_empty = {}
    snapshot_original_inductor_configs([UndeclaredPass()], cfg_empty)
    assert cfg_empty == {}


def test_snapshot_prevents_global_leakage(fake_mode):
    """Verify that snapshot_original_inductor_configs prevents global config leakage when used with config.patch."""
    cfg = torch._inductor.config

    # Snapshot original states to verify restoration later
    orig_prefer_nd_tiling = cfg.triton.prefer_nd_tiling
    orig_max_tiles = cfg.triton.max_tiles

    # Ensure they are currently at their default values (or at least we know what they are)
    assert orig_prefer_nd_tiling is False
    assert orig_max_tiles is None

    pass_ = DummyPass()
    inductor_compile_config = {}
    snapshot_original_inductor_configs([pass_], inductor_compile_config)

    # Simulate the compilation scope with config.patch
    with cfg.patch(inductor_compile_config):
        gm = build_graph_module(fake_mode)
        pass_(gm.graph)

        # Inside the scope, the config should be mutated by the pass
        assert cfg.triton.prefer_nd_tiling is True
        assert cfg.triton.max_tiles == 3

    # Outside the scope, the config must be restored to its original values, preventing leakage
    assert cfg.triton.prefer_nd_tiling == orig_prefer_nd_tiling
    assert cfg.triton.max_tiles == orig_max_tiles


@pytest.mark.parametrize("env_value, expected", [("1", True), ("true", True), ("0", False), ("false", False)])
def test_env_nested_delimiter_config_parsing(monkeypatch, env_value, expected):
    """A nested sub-config field is overridable via the MAGI_COMPILE_<SUBCONFIG>__<FIELD> env var.

    pydantic parses the bool field here, so only truthy/falsy literals
    (1/0/true/false) are accepted as on/off.
    """
    monkeypatch.setenv(_ND_TILING_ENV, env_value)
    config = CompileConfig()
    assert config.pass_config.enable_nd_tiling_workaround is expected


def test_env_unset_defaults_to_true(monkeypatch):
    """When the env var is unset, the binary field defaults to True."""
    monkeypatch.delenv(_ND_TILING_ENV, raising=False)
    config = CompileConfig()
    assert config.pass_config.enable_nd_tiling_workaround is True


@pytest.mark.parametrize("env_value", ["none", "null", "maybe", ""])
def test_env_rejects_non_bool_strings(monkeypatch, env_value):
    """The field is a bool, so non-bool strings raise a ValidationError.

    Only 1/0/true/false round-trip; everything else is rejected.
    """
    monkeypatch.setenv(_ND_TILING_ENV, env_value)
    with pytest.raises(ValidationError):
        CompileConfig()
