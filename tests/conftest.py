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

import filelock._unix as _fl_unix
import pytest
import torch

from magi_compiler.config import get_compile_config

from .model_definition import MLPConfig, RMSNormConfig

# ---------------------------------------------------------------------------
# Workaround for filelock >= 3.21 on Docker overlayfs / FUSE filesystems.
#
# Since filelock 3.21.0 (tox-dev/filelock PR #408), UnixFileLock._release()
# calls Path(lock_file).unlink() BEFORE fcntl.flock(fd, LOCK_UN).  On local
# ext4 this is safe (the inode stays alive while any fd is open), but on
# overlayfs (Docker's default) and FUSE/NFS the unlinked fd becomes stale,
# causing flock(LOCK_UN) to raise FileNotFoundError.
#
# This affects every Inductor filelock path: ISA probe (cpu_vec_isa),
# code cache (codecache.load_async), and async C++ compilation.
#
# Upstream references:
#   - https://github.com/tox-dev/filelock/issues/494  (_acquire ENOENT)
#   - https://github.com/tox-dev/filelock/pull/495    (fix for _acquire)
#   - https://github.com/tox-dev/py-filelock/issues/513 (EIO on close)
#   - https://github.com/pytorch/pytorch/issues/134384 (Inductor FNFE)
#
# filelock 3.24.3 fixed _acquire but NOT _release.  Pinning to 3.20.4
# (which never calls unlink in _release) is another option, but risks
# being overridden by PyTorch's transitive deps.  This patch is scoped
# to test execution only (conftest.py) and is the safest workaround.
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
