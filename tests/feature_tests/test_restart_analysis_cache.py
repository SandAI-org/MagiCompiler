# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_restart_analysis_cache_handle_marks_and_skips_artifact_load(tmp_path: Path):
    """Two-process integration test:
    process1 warms cache and should hit RestartAnalysis;
    process2 reuses cache and should complete without hard failure.
    """

    helper_path = Path(__file__).parent / "cache_reuse_helper" / "restart_analysis_cache_helper.py"
    cache_root = tmp_path / "cache"
    out1 = tmp_path / "run1.json"
    out2 = tmp_path / "run2.json"

    env = os.environ.copy()
    env["MAGI_LOGGING_LEVEL"] = "info"

    cmd1 = [sys.executable, str(helper_path), "--cache-root", str(cache_root), "--output", str(out1)]
    p1 = subprocess.run(cmd1, env=env, capture_output=True, text=True)
    assert p1.returncode == 0, f"worker1 failed\nstdout:\n{p1.stdout}\nstderr:\n{p1.stderr}"
    assert "standalone_compile raised RestartAnalysis" in p1.stderr

    cmd2 = [sys.executable, str(helper_path), "--cache-root", str(cache_root), "--output", str(out2)]
    p2 = subprocess.run(cmd2, env=env, capture_output=True, text=True)
    assert p2.returncode == 0, f"worker2 failed\nstdout:\n{p2.stdout}\nstderr:\n{p2.stderr}"
    assert "too many values to unpack" not in p2.stderr

    cache_files = list(cache_root.rglob("magi_cache_indices.py"))
    assert cache_files, "no cache file generated"
    any_marked = False
    for cache_file in cache_files:
        raw = ast.literal_eval(cache_file.read_text())
        for _, handle in raw.items():
            if len(handle) >= 3 and int(handle[2]) > 0:
                any_marked = True
                break
        if any_marked:
            break
    assert any_marked, "expected at least one cache handle with restart_analysis_count>0"

    payload1 = json.loads(out1.read_text())
    payload2 = json.loads(out2.read_text())
    expected_shape = payload1["shape"]
    assert payload2["shape"] == expected_shape
    assert abs(payload1["sum"] - payload2["sum"]) < 1e-2

    # Validate the second process produced a new magi_depyf run under cache root.
    assert payload2["new_run_dirs"], "expected process2 to generate a new magi_depyf run directory"

    # Inspect process2 run timeline events from cache_root/magi_depyf/run_xxx.
    merged_events = {
        "fullgraph_before_compiler_manager_load": [],
        "fullgraph_after_compiler_manager_load": [],
        "fullgraph_before_compiler_compile": [],
        "fullgraph_failed_compiler_compile": [],
    }
    for _, ev_map in payload2["subgraph0_events_by_run"].items():
        for name, event_records in ev_map.items():
            if name in merged_events:
                merged_events[name].extend(event_records)

    before_load = merged_events["fullgraph_before_compiler_manager_load"]
    after_load = merged_events["fullgraph_after_compiler_manager_load"]
    before_compile = merged_events["fullgraph_before_compiler_compile"]
    failed_compile = merged_events["fullgraph_failed_compiler_compile"]

    assert before_load, "expected process2 timeline to record compiler_manager_load for graph_index=0"
    assert after_load, "expected process2 timeline to record compiler_manager_load completion for graph_index=0"
    assert before_compile, "expected process2 timeline to record compiler_compile for graph_index=0"
    assert failed_compile, "expected process2 timeline to record failed compiler_compile (RestartAnalysis) for graph_index=0"

    load_results = [(record.get("attributes") or {}).get("load_result") for record in after_load if isinstance(record, dict)]
    assert "hit" in load_results, "expected process2 timeline to show subgraph_0 cache load hit after one RestartAnalysis"

    hit_record = next(
        (
            record
            for record in after_load
            if isinstance(record, dict) and (record.get("attributes") or {}).get("load_result") == "hit"
        ),
        None,
    )
    assert hit_record is not None

    hit_index = int(hit_record["index"])
    compile_after_hit = [
        record for record in before_compile if isinstance(record, dict) and int(record.get("index", -1)) > hit_index
    ]
    assert not compile_after_hit, "expected no further compiler_compile for subgraph_0 after cache load hit in process2"
