# Copyright (c) 2026 SandAI. All Rights Reserved.

import io
from pathlib import Path

import pytest

from magi_compiler.utils import host_memory


def _mock_proc(monkeypatch, *, cgroup: str, mountinfo: str, available_kib: int = 1024) -> None:
    proc_files = {
        '/proc/meminfo': f'MemAvailable: {available_kib} kB\n',
        '/proc/self/cgroup': cgroup,
        '/proc/self/mountinfo': mountinfo,
    }
    open_file = open

    def open_proc(path, *args, **kwargs):
        if str(path) in proc_files:
            return io.StringIO(proc_files[str(path)])
        return open_file(path, *args, **kwargs)

    monkeypatch.setattr('builtins.open', open_proc)


def test_uncapped_host_uses_memavailable(monkeypatch) -> None:
    _mock_proc(monkeypatch, cgroup='', mountinfo='', available_kib=1024)
    assert host_memory.get_available_host_memory_bytes() == 1024 * 1024


@pytest.mark.parametrize('version', ['v1', 'v2'])
def test_cgroup_ancestors_cap_available_memory(tmp_path: Path, monkeypatch, version: str) -> None:
    child = tmp_path / 'worker'
    child.mkdir()
    if version == 'v2':
        limit_name, usage_name = 'memory.max', 'memory.current'
        cgroup = '0::/worker\n'
        filesystem = 'cgroup2 cgroup rw'
    else:
        limit_name, usage_name = 'memory.limit_in_bytes', 'memory.usage_in_bytes'
        cgroup = '4:memory:/worker\n'
        filesystem = 'cgroup cgroup rw,memory'
    for directory, limit, usage in [(tmp_path, 1024, 1000), (child, 1024, 512)]:
        (directory / limit_name).write_text(str(limit))
        (directory / usage_name).write_text(str(usage))
    _mock_proc(monkeypatch, cgroup=cgroup, mountinfo=f'1 0 0:1 / {tmp_path} rw - {filesystem}\n')
    assert host_memory.get_available_host_memory_bytes() == 24


@pytest.mark.parametrize('limit, usage, expected', [('max', '100', 1048576), ('100', '101', 0)])
def test_unlimited_and_exhausted_cgroups(tmp_path: Path, monkeypatch, limit: str, usage: str, expected: int) -> None:
    (tmp_path / 'memory.max').write_text(limit)
    (tmp_path / 'memory.current').write_text(usage)
    _mock_proc(monkeypatch, cgroup='0::/\n', mountinfo=f'1 0 0:1 / {tmp_path} rw - cgroup2 cgroup rw\n')
    assert host_memory.get_available_host_memory_bytes() == expected


@pytest.mark.parametrize('membership', ['/host/pod/worker', '/worker'])
def test_cgroup_mount_root_translation(tmp_path: Path, monkeypatch, membership: str) -> None:
    child = tmp_path / 'worker'
    child.mkdir()
    (child / 'memory.max').write_text('100')
    (child / 'memory.current').write_text('60')
    _mock_proc(monkeypatch, cgroup=f'0::{membership}\n', mountinfo=f'1 0 0:1 /host/pod {tmp_path} rw - cgroup2 cgroup rw\n')
    assert host_memory.get_available_host_memory_bytes() == 40
