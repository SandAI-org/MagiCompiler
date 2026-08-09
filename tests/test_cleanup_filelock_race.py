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

"""Tests for the filelock resilience patch applied in conftest.py.

Docker overlayfs can cause fcntl.flock(fd, LOCK_UN) to raise FileNotFoundError
on unlinked inodes.  The conftest patches filelock._unix.UnixFileLock._release
to catch this error and close the fd safely.  These tests verify the patch.
"""

import os
import tempfile
from unittest.mock import patch

import filelock
import filelock._unix as _fl_unix

from tests.conftest import _resilient_release


class TestFilelockResiliencePatch:
    def test_patch_is_applied(self):
        """Verify conftest applied the resilient _release to UnixFileLock."""
        assert _fl_unix.UnixFileLock._release is _resilient_release

    def test_normal_release_still_works(self):
        """Normal lock/release cycle must not be affected by the patch."""
        tmpdir = tempfile.mkdtemp(prefix="magi_flock_normal_")
        lock_path = os.path.join(tmpdir, "test.lock")
        lock = filelock.FileLock(lock_path, timeout=5)
        lock.acquire()
        lock.release()
        assert not lock.is_locked
        os.unlink(lock_path)
        os.rmdir(tmpdir)

    def test_release_survives_simulated_overlayfs_error(self):
        """Simulate the overlayfs FileNotFoundError on LOCK_UN and verify
        the patch catches it instead of propagating."""
        tmpdir = tempfile.mkdtemp(prefix="magi_flock_overlay_")
        lock_path = os.path.join(tmpdir, "test.lock")
        lock = filelock.FileLock(lock_path, timeout=5)
        lock.acquire()

        import fcntl

        _real_flock = fcntl.flock

        def _flock_that_fails_on_unlock(fd, operation):
            if operation == fcntl.LOCK_UN:
                raise FileNotFoundError(2, "No such file or directory")
            return _real_flock(fd, operation)

        with patch.object(fcntl, "flock", side_effect=_flock_that_fails_on_unlock):
            lock.release()

        assert not lock.is_locked
        os.unlink(lock_path)
        os.rmdir(tmpdir)
