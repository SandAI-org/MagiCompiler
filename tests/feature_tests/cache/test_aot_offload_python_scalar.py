# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_HELPER = Path(__file__).parent / "cache_reuse_helper" / "aot_offload_scalar_helper.py"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_aot_offload_keeps_python_scalar_inputs_on_cpu(tmp_path: Path):
    """An AOT compile reaches the backend after Dynamo deleted its ``grapharg`` meta, and AOT artifacts drop node meta.
    Lifted Python scalars must still stay on CPU there: on CUDA the compile failed (``aten._local_scalar_dense``) and
    fell back to JIT with a warning, so nothing was saved and every process compiled from scratch."""
    env = {
        **os.environ,
        "MAGI_COMPILE_CACHE_ROOT_DIR": str(tmp_path / "cache"),
        "MAGI_COMPILE_OFFLOAD_CONFIG__MODEL_CPU_OFFLOAD": "1",
    }
    runs = []
    for name in ("compile", "load"):
        output = tmp_path / f"{name}.json"
        proc = subprocess.run([sys.executable, str(_HELPER), "--output", str(output)], env=env, capture_output=True, text=True)
        assert proc.returncode == 0, f"{name} process failed\nstderr:\n{proc.stderr}"
        runs.append(json.loads(output.read_text()))

    compiled, loaded = runs
    assert compiled == {"aot_compile_error": None, "aot_loaded": False, "correct": True}
    assert loaded == {"aot_compile_error": None, "aot_loaded": True, "correct": True}
