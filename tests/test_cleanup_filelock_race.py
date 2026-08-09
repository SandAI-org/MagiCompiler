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

"""Minimal reproduction + fix verification for the cleanup_cache vs Inductor filelock race.

Root cause (PT 2.12 CI #55):
  The autouse cleanup_cache fixture called shutil.rmtree(cache_root_dir) which
  wipes ~/.cache/magi_compiler/ INCLUDING inductor_cache/.  Inductor's async
  compiler holds FileLock objects inside that directory; when the lock file is
  deleted while still held, _release() -> fcntl.flock(fd, LOCK_UN) raises
  FileNotFoundError on Linux overlayfs (GitHub Actions container).

Fix:
  cleanup_cache now only deletes MagiCompiler-owned subdirs (magi_cache/,
  magi_depyf/), leaving inductor_cache/ untouched.
"""

import os
import shutil
import tempfile
import threading

from filelock import FileLock


def _hold_lock_then_release(lock_path: str, barrier: threading.Barrier, errors: list):
    """Simulate Inductor async compile: acquire lock, wait, then release."""
    lock = FileLock(lock_path, timeout=10)
    lock.acquire()
    barrier.wait()  # signal: lock is held
    barrier.wait()  # wait for main thread action
    try:
        lock.release()
    except Exception as e:
        errors.append(e)


class TestCleanupFilelockRace:
    def test_rmtree_deletes_held_lock_file(self):
        """REPRODUCE: shutil.rmtree on cache_root_dir deletes the lock file
        while Inductor's async thread still holds it -- the scenario that
        triggers FileNotFoundError on overlayfs."""
        tmpdir = tempfile.mkdtemp(prefix="magi_race_repro_")
        inductor_dir = os.path.join(tmpdir, "inductor_cache", "locks")
        os.makedirs(inductor_dir)
        lock_path = os.path.join(inductor_dir, "kernel.lock")

        barrier = threading.Barrier(2, timeout=5)
        errors: list[Exception] = []

        t = threading.Thread(target=_hold_lock_then_release, args=(lock_path, barrier, errors))
        t.start()

        barrier.wait()  # lock is now held by thread
        shutil.rmtree(tmpdir, ignore_errors=True)
        assert not os.path.exists(lock_path), "Lock file should be gone after rmtree"
        barrier.wait()  # let thread try to release
        t.join(timeout=5)

        # On macOS native fs, flock(fd, LOCK_UN) on a deleted-but-open fd
        # succeeds (inode kept alive).  On Linux overlayfs (CI containers),
        # this raises FileNotFoundError -- the exact CI failure we see.
        # Regardless of OS, the lock FILE is deleted, which is the root cause.

    def test_selective_cleanup_preserves_lock(self):
        """FIX: only deleting magi-owned subdirs keeps inductor_cache/ intact,
        so the async thread can safely release the lock."""
        tmpdir = tempfile.mkdtemp(prefix="magi_race_fix_")
        magi_cache = os.path.join(tmpdir, "magi_cache")
        magi_depyf = os.path.join(tmpdir, "magi_depyf")
        inductor_dir = os.path.join(tmpdir, "inductor_cache", "locks")
        os.makedirs(magi_cache)
        os.makedirs(magi_depyf)
        os.makedirs(inductor_dir)
        lock_path = os.path.join(inductor_dir, "kernel.lock")

        barrier = threading.Barrier(2, timeout=5)
        errors: list[Exception] = []

        t = threading.Thread(target=_hold_lock_then_release, args=(lock_path, barrier, errors))
        t.start()

        barrier.wait()  # lock is now held
        for subdir in ("magi_cache", "magi_depyf"):
            shutil.rmtree(os.path.join(tmpdir, subdir), ignore_errors=True)
        barrier.wait()  # let thread release
        t.join(timeout=5)

        assert len(errors) == 0, f"Lock release must succeed, got: {errors}"
        assert os.path.exists(inductor_dir), "inductor_cache/ must be preserved"
        assert not os.path.exists(magi_cache), "magi_cache should be deleted"
        assert not os.path.exists(magi_depyf), "magi_depyf should be deleted"

        shutil.rmtree(tmpdir, ignore_errors=True)
