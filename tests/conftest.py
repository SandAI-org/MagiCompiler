# Copyright (c) 2025 SandAI. All Rights Reserved.
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

import shutil

import pytest
import torch
from torch._inductor import cpu_vec_isa as _vec_isa

from magi_compiler.config import get_compile_config

from .model_definition import MLPConfig, RMSNormConfig

# ---------------------------------------------------------------------------
# Backport of PyTorch upstream PR #181617: cache check_build() results
# in-memory so the filelock + subprocess probe runs at most once per code
# snippet per process.  PT 2.12 lacks the on-disk .load_ok marker that
# newer PyTorch uses, causing FileNotFoundError races in containers.
# ---------------------------------------------------------------------------
_check_build_cache: dict[str, bool] = {}
_original_check_build = _vec_isa.VecISA.check_build


def _cached_check_build(self, code: str) -> bool:  # noqa: ANN001
    if code not in _check_build_cache:
        _check_build_cache[code] = _original_check_build(self, code)
    return _check_build_cache[code]


_vec_isa.VecISA.check_build = _cached_check_build

# Warm up: trigger all ISA probes once at collection time.
_vec_isa.pick_vec_isa()


@pytest.fixture(scope="function")
def device():
    """Device fixture"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="function")
def mlp_config():
    """MLP configuration fixture"""
    return MLPConfig(hidden_size=8, intermediate_size=32, params_dtype=torch.bfloat16)


@pytest.fixture(scope="function")
def rms_norm_config():
    """RMSNorm configuration fixture"""
    return RMSNormConfig(hidden_size=8, eps=1e-6)


@pytest.fixture(scope="function", autouse=True)
def cleanup_cache():
    """Auto cleanup cache fixture, executed before and after each test"""
    shutil.rmtree(get_compile_config().cache_root_dir, ignore_errors=True)
    yield
    shutil.rmtree(get_compile_config().cache_root_dir, ignore_errors=True)
