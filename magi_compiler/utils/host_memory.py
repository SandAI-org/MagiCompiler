# Copyright (c) 2026 SandAI. All Rights Reserved.
"""Lightweight host (CPU) memory introspection for Linux."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Iterator


def get_host_mem_gb() -> dict[str, float]:
    """Read /proc/self/status and return key memory metrics in GiB.

    Returns a dict with keys:
      vm_peak  – peak virtual memory (VmPeak)
      vm_rss   – current resident set (VmRSS)
      vm_hwm   – high-water mark of RSS (VmHWM)
      rss_anon – anonymous (heap/stack) resident pages (RssAnon)
      rss_file – file-backed (mmap) resident pages (RssFile)
      rss_shmem – shared-memory resident pages (RssShmem)
    Missing fields default to 0.
    """
    fields = {
        "VmPeak": "vm_peak",
        "VmRSS": "vm_rss",
        "VmHWM": "vm_hwm",
        "RssAnon": "rss_anon",
        "RssFile": "rss_file",
        "RssShmem": "rss_shmem",
    }
    result: dict[str, float] = {v: 0.0 for v in fields.values()}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                key = line.split(":")[0]
                if key in fields:
                    kb = int(line.split()[1])
                    result[fields[key]] = kb / (1024 * 1024)
    except (OSError, ValueError):
        pass
    return result


def get_total_ram_gb() -> float:
    """Return total host RAM in GiB from /proc/meminfo."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except (OSError, ValueError):
        pass
    return 0.0


def fmt_host_mem(mem: dict[str, float] | None = None) -> str:
    """One-line human-readable summary of current host memory."""
    if mem is None:
        mem = get_host_mem_gb()
    return (
        f"VmHWM={mem['vm_hwm']:.1f}G "
        f"VmRSS={mem['vm_rss']:.1f}G "
        f"(anon={mem['rss_anon']:.1f}G file={mem['rss_file']:.1f}G shm={mem['rss_shmem']:.1f}G)"
    )


def get_available_host_memory_bytes() -> int:
    """Available RAM capped by every visible memory-cgroup ancestor's remaining budget."""
    with open('/proc/meminfo') as memory_file:
        available_bytes = next(int(line.split()[1]) * 1024 for line in memory_file if line.startswith('MemAvailable:'))
    for directory, mountpoint, limit_name, usage_name in _memory_cgroup_directories():
        for ancestor in (directory, *directory.parents):
            limit_path = ancestor / limit_name
            if limit_path.is_file():
                limit = limit_path.read_text().strip()
                if limit != 'max':
                    remaining_bytes = max(0, int(limit) - int((ancestor / usage_name).read_text()))
                    available_bytes = min(available_bytes, remaining_bytes)
            if ancestor == mountpoint:
                break
    return available_bytes


def _memory_cgroup_directories() -> Iterator[tuple[Path, Path, str, str]]:
    memberships = {}
    with open('/proc/self/cgroup') as membership_file:
        for line in membership_file:
            _, controllers, hierarchy_path = line.strip().split(':', 2)
            if not controllers:
                memberships['cgroup2'] = PurePosixPath(hierarchy_path)
            elif 'memory' in controllers.split(','):
                memberships['cgroup'] = PurePosixPath(hierarchy_path)
    with open('/proc/self/mountinfo') as mount_file:
        for line in mount_file:
            mount_fields, filesystem_fields = line.rstrip().split(' - ', 1)
            filesystem_type, _, mount_options = filesystem_fields.split()[:3]
            if filesystem_type not in memberships:
                continue
            if filesystem_type == 'cgroup' and 'memory' not in mount_options.split(','):
                continue
            fields = mount_fields.split()
            mount_root = PurePosixPath(fields[3])
            mountpoint = Path(fields[4])
            hierarchy_path = memberships[filesystem_type]
            if hierarchy_path.is_relative_to(mount_root):
                relative_path = hierarchy_path.relative_to(mount_root)
            else:
                # A cgroup namespace can make the membership path relative to the mounted root.
                relative_path = hierarchy_path.relative_to('/')
            directory = mountpoint / relative_path
            if filesystem_type == 'cgroup2':
                yield directory, mountpoint, 'memory.max', 'memory.current'
            else:
                yield directory, mountpoint, 'memory.limit_in_bytes', 'memory.usage_in_bytes'
