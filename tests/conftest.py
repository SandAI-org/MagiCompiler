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

import os
import shutil
from pathlib import Path

import filelock._unix as _fl_unix
import pytest
import torch

from magi_compiler.config import get_compile_config

from .model_definition import MLPConfig, RMSNormConfig

# ---------------------------------------------------------------------------
# Workaround: Docker overlayfs can lose unlinked inodes, causing
# fcntl.flock(fd, LOCK_UN) to raise FileNotFoundError.  This affects
# every Inductor filelock path (ISA probe, code cache, async compile).
# Patch _release to catch and safely handle the error.
# ---------------------------------------------------------------------------
_original_unix_release = _fl_unix.UnixFileLock._release


def _resilient_release(self):  # noqa: ANN001
    try:
        _original_unix_release(self)
    except FileNotFoundError:
        fd = self._context.lock_file_fd
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
            self._context.lock_file_fd = None


_fl_unix.UnixFileLock._release = _resilient_release

# Subdirectories owned by MagiCompiler that are safe to delete between tests.
# inductor_cache/ is managed by PyTorch Inductor (with async FileLock);
# deleting it mid-process causes FileNotFoundError on lock release.
_MAGI_OWNED_SUBDIRS = ("magi_cache", "magi_depyf")


def _cleanup_magi_cache() -> None:
    root = Path(get_compile_config().cache_root_dir)
    for name in _MAGI_OWNED_SUBDIRS:
        shutil.rmtree(root / name, ignore_errors=True)


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
    """Auto cleanup MagiCompiler cache between tests (preserves inductor_cache/)."""
    _cleanup_magi_cache()
    yield
    _cleanup_magi_cache()
