# Copyright (c) 2026 SandAI. All Rights Reserved.
"""Lightweight host (CPU) memory introspection for Linux."""

from __future__ import annotations


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
