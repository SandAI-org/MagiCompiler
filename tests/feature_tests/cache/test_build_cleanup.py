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

"""Tests for the build-directory cleanup mechanism in evt_runtime.py.

The _track_build / _untrack_build + signal-handler machinery ensures that
interrupted cpp_extension.load calls leave no stale lock files or partial
build artifacts on disk. These tests exercise that mechanism directly
(no GPU needed).
"""

import os
import signal
import subprocess
import sys
import textwrap

import pytest


@pytest.fixture(autouse=True)
def _isolate_pending_set():
    """Reset _PENDING_BUILD_DIRS before and after each test."""
    from magi_compiler.passes.piecewise_graph.fusion import evt_runtime as rt

    saved = rt._PENDING_BUILD_DIRS.copy()
    rt._PENDING_BUILD_DIRS.clear()
    yield
    rt._PENDING_BUILD_DIRS.clear()
    rt._PENDING_BUILD_DIRS.update(saved)


def test_track_untrack_basic(tmp_path):
    """Normal success path: track → present → untrack → absent."""
    from magi_compiler.passes.piecewise_graph.fusion import evt_runtime as rt

    build_dir = str(tmp_path / "build_ok")
    os.makedirs(build_dir)

    rt._track_build(build_dir)
    assert build_dir in rt._PENDING_BUILD_DIRS

    rt._untrack_build(build_dir)
    assert build_dir not in rt._PENDING_BUILD_DIRS


def test_cleanup_pending_removes_tracked_dirs(tmp_path):
    """_cleanup_pending_build_dirs wipes every tracked directory."""
    from magi_compiler.passes.piecewise_graph.fusion import evt_runtime as rt

    build_dir = str(tmp_path / "build_interrupted")
    os.makedirs(build_dir)
    # Simulate partial build artifacts
    (tmp_path / "build_interrupted" / "lock").touch()
    (tmp_path / "build_interrupted" / "kernel.cuda.o").touch()
    (tmp_path / "build_interrupted" / "build.ninja").touch()

    rt._track_build(build_dir)
    assert os.path.isdir(build_dir)

    rt._cleanup_pending_build_dirs()

    assert not os.path.exists(build_dir)
    assert len(rt._PENDING_BUILD_DIRS) == 0


def test_untracked_build_not_cleaned(tmp_path):
    """A directory that was tracked then untracked must survive cleanup."""
    from magi_compiler.passes.piecewise_graph.fusion import evt_runtime as rt

    build_dir = str(tmp_path / "build_completed")
    os.makedirs(build_dir)
    (tmp_path / "build_completed" / "module.so").touch()

    rt._track_build(build_dir)
    rt._untrack_build(build_dir)

    rt._cleanup_pending_build_dirs()

    assert os.path.isdir(build_dir)
    assert (tmp_path / "build_completed" / "module.so").exists()


def test_cleanup_on_signal_in_subprocess(tmp_path):
    """A subprocess that tracks a build_dir and receives SIGTERM must clean it up."""
    build_dir = str(tmp_path / "build_signal")

    script = textwrap.dedent(
        f"""\
        import os, sys, time
        sys.path.insert(0, {str((tmp_path / '..').resolve().parent)!r})

        build_dir = {build_dir!r}
        os.makedirs(build_dir, exist_ok=True)
        with open(os.path.join(build_dir, "lock"), "w") as f:
            f.write("locked")
        with open(os.path.join(build_dir, "partial.o"), "w") as f:
            f.write("junk")

        from magi_compiler.passes.piecewise_graph.fusion import evt_runtime as rt
        rt._track_build(build_dir)

        # Signal parent that we're ready
        sys.stdout.write("READY\\n")
        sys.stdout.flush()
        # Sleep long enough for parent to send signal
        time.sleep(60)
    """
    )

    proc = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(tmp_path)
    )

    try:
        line = proc.stdout.readline()
        assert line.strip() == "READY", f"Subprocess didn't become ready, got: {line!r}"

        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
        proc.wait()
        raise

    assert not os.path.exists(build_dir), f"build_dir {build_dir} should have been cleaned up by SIGTERM handler"


def test_cleanup_idempotent(tmp_path):
    """Calling _cleanup_pending_build_dirs twice is harmless."""
    from magi_compiler.passes.piecewise_graph.fusion import evt_runtime as rt

    build_dir = str(tmp_path / "build_double")
    os.makedirs(build_dir)
    rt._track_build(build_dir)

    rt._cleanup_pending_build_dirs()
    assert not os.path.exists(build_dir)

    # Second call: no-op, no exception.
    rt._cleanup_pending_build_dirs()
