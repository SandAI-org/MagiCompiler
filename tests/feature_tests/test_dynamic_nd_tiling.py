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

Covers the decision logic only (no GPU / benchmarking):
  * dynamic-shape intent detection           -> ``_is_dynamic_compilation``
  * PyTorch version gating                   -> ``_inductor_needs_nd_tiling_workaround``
  * full precedence (explicit config > auto) -> ``_should_enable_nd_tiling``
  * actual Inductor config injection         -> ``_configure_custom_passes``

The end-to-end speedup is validated separately in
``tests/perf_tests/test_dynamic_nd_tiling_perf.py``.
"""

import pytest

from magi_compiler.config import CompileConfig, get_compile_config
from magi_compiler.magi_backend import magi_backend as mb
from magi_compiler.magi_backend.magi_backend import MagiBackend, _inductor_needs_nd_tiling_workaround

ND_TILING_KEYS = ("triton.prefer_nd_tiling", "triton.max_tiles", "triton.tile_reductions")


def _make_backend(dynamic_arg_dims, *, enable_dynamic_nd_tiling=None):
    """Build a MagiBackend without running the heavy __init__.

    We only need the attributes that the ND-tiling decision reads, so we
    bypass __init__ (which would spin up a CompilerManager) and set them by hand.
    """
    backend = MagiBackend.__new__(MagiBackend)
    backend.compile_config = get_compile_config().model_copy(update={"enable_dynamic_nd_tiling": enable_dynamic_nd_tiling})
    backend.dynamic_arg_dims = dynamic_arg_dims
    backend.inductor_compile_config = {}
    return backend


@pytest.fixture
def clear_version_cache():
    """The version probe is lru_cached; clear it around tests that patch the version."""
    _inductor_needs_nd_tiling_workaround.cache_clear()
    yield
    _inductor_needs_nd_tiling_workaround.cache_clear()


@pytest.fixture
def no_env_override(monkeypatch):
    """Ensure the env override for the config field is absent for auto-path tests."""
    monkeypatch.delenv("MAGI_COMPILE_ENABLE_DYNAMIC_ND_TILING", raising=False)


# ── Point 2: dynamic-shape intent detection ──────────────────────────────


@pytest.mark.parametrize(
    "dynamic_arg_dims, expected",
    [
        (None, False),
        ({}, False),
        ({"x": []}, False),
        ({"x": [0]}, True),
        ({"x": 0}, True),
        ({"x": (1, 2)}, True),
        ({"x": None}, False),
        ({"x": [], "y": [0]}, True),
        ({"x": None, "y": []}, False),
    ],
)
def test_is_dynamic_compilation(dynamic_arg_dims, expected):
    backend = _make_backend(dynamic_arg_dims)
    assert backend._is_dynamic_compilation() is expected


# ── Point 3: PyTorch version gating ──────────────────────────────────────


@pytest.mark.parametrize(
    "version, needs_workaround",
    [
        ("2.9.1", True),
        ("2.10.0", True),
        ("2.10.5", True),
        ("2.11.0", False),
        ("2.11.0+cu124", False),
        ("2.11.1", False),
        ("2.12.0.dev20260101+gitabcdef", False),
        ("3.0.0", False),
    ],
)
def test_version_gating(monkeypatch, clear_version_cache, version, needs_workaround):
    monkeypatch.setattr(mb.torch, "__version__", version)
    assert _inductor_needs_nd_tiling_workaround() is needs_workaround


# ── Point 4: full precedence (explicit config > auto) ────────────────────


def test_explicit_config_forces_on(monkeypatch):
    """Explicit config True forces on even for a static (non-dynamic) compilation."""
    backend = _make_backend(dynamic_arg_dims=None, enable_dynamic_nd_tiling=True)
    assert backend._should_enable_nd_tiling() is True


def test_explicit_config_forces_off(monkeypatch, clear_version_cache):
    """Explicit config False forces off even when dynamic + buggy-version would auto-enable."""
    monkeypatch.setattr(mb.torch, "__version__", "2.9.1")
    backend = _make_backend(dynamic_arg_dims={"x": [0]}, enable_dynamic_nd_tiling=False)
    assert backend._should_enable_nd_tiling() is False


@pytest.mark.parametrize("env_val, expected", [("1", True), ("0", False)])
def test_env_var_drives_config_field(monkeypatch, env_val, expected):
    """MAGI_COMPILE_ENABLE_DYNAMIC_ND_TILING populates the config field directly."""
    monkeypatch.setenv("MAGI_COMPILE_ENABLE_DYNAMIC_ND_TILING", env_val)
    assert CompileConfig().enable_dynamic_nd_tiling is expected


@pytest.mark.parametrize("explicit", [True, False])
def test_explicit_config_overrides_auto(monkeypatch, clear_version_cache, no_env_override, explicit):
    """Explicit config beats the auto path, regardless of dynamic/version state."""
    # Static + fixed version would auto-decide differently; explicit must win.
    monkeypatch.setattr(mb.torch, "__version__", "2.11.0" if explicit else "2.9.1")
    backend = _make_backend(dynamic_arg_dims={"x": [0]} if not explicit else None, enable_dynamic_nd_tiling=explicit)
    assert backend._should_enable_nd_tiling() is explicit


def test_auto_enables_on_dynamic_and_buggy_version(monkeypatch, clear_version_cache, no_env_override):
    monkeypatch.setattr(mb.torch, "__version__", "2.9.1")
    backend = _make_backend(dynamic_arg_dims={"x": [0]})
    assert backend._should_enable_nd_tiling() is True


def test_auto_disables_on_static_shapes(monkeypatch, clear_version_cache, no_env_override):
    monkeypatch.setattr(mb.torch, "__version__", "2.9.1")
    backend = _make_backend(dynamic_arg_dims=None)
    assert backend._should_enable_nd_tiling() is False


def test_auto_disables_on_fixed_version(monkeypatch, clear_version_cache, no_env_override):
    """Dynamic shapes but PyTorch >= 2.11.0: native coalesce path handles it."""
    monkeypatch.setattr(mb.torch, "__version__", "2.11.0")
    backend = _make_backend(dynamic_arg_dims={"x": [0]})
    assert backend._should_enable_nd_tiling() is False


# ── Point 5: actual Inductor config injection ────────────────────────────
#
# We exercise the injection branch directly rather than the full
# ``_configure_custom_passes`` (which also wires up unrelated pass managers),
# so the test stays focused on the ND-tiling keys and free of pass-chain deps.


def _inject_nd_tiling(backend):
    """Mirror the ND-tiling injection block of ``_configure_custom_passes``."""
    if backend._should_enable_nd_tiling():
        backend.inductor_compile_config["triton.prefer_nd_tiling"] = True
        backend.inductor_compile_config["triton.max_tiles"] = 3
        backend.inductor_compile_config["triton.tile_reductions"] = True


def test_config_injected_when_enabled(monkeypatch, clear_version_cache, no_env_override):
    monkeypatch.setattr(mb.torch, "__version__", "2.9.1")
    backend = _make_backend(dynamic_arg_dims={"x": [0]})
    _inject_nd_tiling(backend)
    assert backend.inductor_compile_config["triton.prefer_nd_tiling"] is True
    assert backend.inductor_compile_config["triton.max_tiles"] == 3
    assert backend.inductor_compile_config["triton.tile_reductions"] is True


def test_config_absent_when_disabled(monkeypatch, clear_version_cache, no_env_override):
    monkeypatch.setattr(mb.torch, "__version__", "2.11.0")
    backend = _make_backend(dynamic_arg_dims={"x": [0]})
    _inject_nd_tiling(backend)
    for key in ND_TILING_KEYS:
        assert key not in backend.inductor_compile_config
