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

"""Test magi_compile backend introspection: verify inductor artifact extraction.

This test uses a minimal model compiled through MagiBackend (with Inductor)
and verifies that the introspection pipeline can:
  1. Detect the magi_compile backend
  2. Extract full-graph and split-graph info
  3. Read Inductor-generated kernel source from saved artifacts
  4. Write inductor_output.py files for each compiled subgraph
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Inductor compilation")


def _make_config(cache_root_dir: str, cudagraph_mode: str = "NONE"):
    """Create a minimal CompileConfig suitable for testing."""
    from magi_compiler.config import CompileConfig, CompileMode, CudaGraphMode

    mode = CudaGraphMode[cudagraph_mode]
    with patch.object(sys, "argv", ["test"]):
        return CompileConfig(
            compile_mode=CompileMode.MAGI_COMPILE,
            backend="inductor",
            cache_root_dir=cache_root_dir,
            splitting_ops=[],
            cudagraph_mode=mode,
            compile_sizes=[],
        )


def _cudagraph_passthrough(self, func, *args, layer_number=None, **kwargs):
    """Replacement for CudaGraphMgr.run that skips capture/replay."""
    return func(*args, **kwargs)


def _compile_simple_model(tmpdir: str, cudagraph_mode: str = "NONE"):
    """Compile a simple model via MagiBackend and return the original function.

    The model is: Linear(32→64) → ReLU → Linear(64→16).

    For cudagraph modes (PIECEWISE/FULL), we mock ``CudaGraphMgr.run`` to
    skip the actual CUDA graph capture/replay while preserving the wrapping
    structure — this is sufficient to verify introspection correctness.
    """
    from magi_compiler.magi_backend.cuda_graph_mgr import CudaGraphMgr
    from magi_compiler.magi_backend.magi_backend import MagiBackend
    from magi_compiler.utils import OrderedSet

    w1 = torch.randn(64, 32, device="cuda")
    b1 = torch.randn(64, device="cuda")
    w2 = torch.randn(16, 64, device="cuda")
    b2 = torch.randn(16, device="cuda")

    def fn(x):
        h = torch.nn.functional.linear(x, w1, b1)
        h = torch.nn.functional.relu(h)
        return torch.nn.functional.linear(h, w2, b2)

    config = _make_config(tmpdir, cudagraph_mode=cudagraph_mode)
    backend = MagiBackend(config, model_tag="test", traced_files=OrderedSet(), inductor_compile_config={})
    compiled = torch.compile(fn, backend=backend)

    ctx = patch.object(CudaGraphMgr, "run", _cudagraph_passthrough) if cudagraph_mode != "NONE" else _nullcontext()
    with torch.no_grad(), ctx:
        compiled(torch.randn(4, 32, device="cuda"))

    return fn


class _nullcontext:
    """Minimal no-op context manager (avoid importing contextlib)."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _assert_magi_backend_detected(fn, expected_cudagraph_mode: str = "NONE"):
    from magi_compiler.magi_depyf.inspect.introspect import Introspector

    info = Introspector.build_function_info(fn)
    assert len(info.entries) >= 1
    entry = info.entries[0]
    magi_fns = [cf for cf in entry.compiled_fns if cf.backend == "magi_compile"]
    assert len(magi_fns) >= 1, (
        f"Should detect magi_compile backend, got backends: " f"{[cf.backend for cf in entry.compiled_fns]}"
    )
    cf = magi_fns[0]
    assert (
        cf.cudagraph_mode == expected_cudagraph_mode
    ), f"Expected cudagraph_mode={expected_cudagraph_mode}, got {cf.cudagraph_mode}"
    return cf


def _assert_inductor_source(cf):
    compiled_sgs = [sg for sg in cf.subgraph_infos if not sg.is_splitting_graph]
    assert len(compiled_sgs) > 0, "Should have compiled (non-splitting) subgraphs"
    for sg in compiled_sgs:
        assert sg.inductor_code is not None, f"Subgraph {sg.name} should have inductor source code"
        assert len(sg.inductor_code) > 100, f"Subgraph {sg.name} inductor code seems too short ({len(sg.inductor_code)} chars)"


class TestMagiBackendInductorSource:
    """Verify Inductor kernel source extraction — no cudagraph."""

    def setup_method(self):
        torch._dynamo.reset()
        self.tmpdir = tempfile.mkdtemp(prefix="magi_depyf_test_magi_")

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        torch._dynamo.reset()

    def test_detects_magi_compile_backend(self):
        fn = _compile_simple_model(self.tmpdir)
        _assert_magi_backend_detected(fn)

    def test_has_full_graph_and_split_graph(self):
        fn = _compile_simple_model(self.tmpdir)
        cf = _assert_magi_backend_detected(fn)
        assert cf.readable_code is not None, "Should have full graph readable code"
        assert len(cf.subgraph_infos) > 0, "Should have subgraph infos"

    def test_inductor_source_extracted(self):
        fn = _compile_simple_model(self.tmpdir)
        cf = _assert_magi_backend_detected(fn)
        _assert_inductor_source(cf)

    def test_write_inductor_output_files(self):
        from magi_compiler.magi_depyf.inspect import write_function
        from magi_compiler.magi_depyf.inspect.introspect import Introspector

        fn = _compile_simple_model(self.tmpdir)
        info = Introspector.build_function_info(fn)
        output_dir = Path(self.tmpdir) / "write_output"
        root = write_function(info, output_dir)

        assert (root / "overview.md").exists()

        inductor_files = list(root.rglob("inductor_output.py"))
        assert len(inductor_files) > 0, f"Should generate inductor_output.py files. " f"Files found: {list(root.rglob('*'))}"
        for f in inductor_files:
            content = f.read_text()
            assert len(content) > 100, f"{f} seems too short"


class TestMagiBackendPiecewiseCudaGraph:
    """Verify introspection works with PIECEWISE cudagraph mode.

    In PIECEWISE mode, each PiecewiseBackend submodule is wrapped by
    gen_wrap_func_for_cudagraph.  The wrapper copies __dict__ from
    PiecewiseBackend, so attribute-based detection should still work.
    """

    def setup_method(self):
        torch._dynamo.reset()
        self.tmpdir = tempfile.mkdtemp(prefix="magi_depyf_test_pw_cg_")

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        torch._dynamo.reset()

    def test_detects_backend(self):
        fn = _compile_simple_model(self.tmpdir, cudagraph_mode="PIECEWISE")
        _assert_magi_backend_detected(fn, expected_cudagraph_mode="PIECEWISE")

    def test_has_subgraph_info(self):
        fn = _compile_simple_model(self.tmpdir, cudagraph_mode="PIECEWISE")
        cf = _assert_magi_backend_detected(fn, expected_cudagraph_mode="PIECEWISE")
        assert cf.readable_code is not None
        assert len(cf.subgraph_infos) > 0

    def test_inductor_source_extracted(self):
        fn = _compile_simple_model(self.tmpdir, cudagraph_mode="PIECEWISE")
        cf = _assert_magi_backend_detected(fn, expected_cudagraph_mode="PIECEWISE")
        _assert_inductor_source(cf)


class TestMagiBackendFullCudaGraph:
    """Verify introspection works with FULL cudagraph mode.

    In FULL mode, the entire split_gm is wrapped by
    gen_wrap_func_for_cudagraph, so MSF.optimized_call is a function
    rather than a GraphModule.  The introspector must unwrap the closure
    chain to find the actual GraphModule.
    """

    def setup_method(self):
        torch._dynamo.reset()
        self.tmpdir = tempfile.mkdtemp(prefix="magi_depyf_test_full_cg_")

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        torch._dynamo.reset()

    def test_detects_backend(self):
        fn = _compile_simple_model(self.tmpdir, cudagraph_mode="FULL")
        _assert_magi_backend_detected(fn, expected_cudagraph_mode="FULL")

    def test_has_subgraph_info(self):
        fn = _compile_simple_model(self.tmpdir, cudagraph_mode="FULL")
        cf = _assert_magi_backend_detected(fn, expected_cudagraph_mode="FULL")
        assert cf.readable_code is not None
        assert len(cf.subgraph_infos) > 0

    def test_inductor_source_extracted(self):
        fn = _compile_simple_model(self.tmpdir, cudagraph_mode="FULL")
        cf = _assert_magi_backend_detected(fn, expected_cudagraph_mode="FULL")
        _assert_inductor_source(cf)


class TestDumpSrcEndToEnd:
    """Integration test: dump_src context-manager across all cudagraph modes.

    Verifies the full pipeline (CaptureSession → Introspector → writer) produces
    consistent output regardless of cudagraph wrapping.
    """

    def setup_method(self):
        torch._dynamo.reset()
        self.tmpdir = tempfile.mkdtemp(prefix="magi_depyf_dump_src_")
        self.cache_dir = tempfile.mkdtemp(prefix="magi_depyf_dump_cache_")

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        torch._dynamo.reset()

    def _run_dump_src(self, cudagraph_mode: str):
        from magi_compiler.magi_backend.cuda_graph_mgr import CudaGraphMgr
        from magi_compiler.magi_backend.magi_backend import MagiBackend
        from magi_compiler.magi_depyf.inspect import explain_compilation
        from magi_compiler.utils import OrderedSet

        w = torch.randn(16, 8, device="cuda")
        b = torch.randn(16, device="cuda")

        def fn(x):
            return torch.nn.functional.relu(torch.nn.functional.linear(x, w, b))

        cache = tempfile.mkdtemp(dir=self.cache_dir)
        config = _make_config(cache, cudagraph_mode=cudagraph_mode)
        backend = MagiBackend(config, model_tag="test", traced_files=OrderedSet(), inductor_compile_config={})
        compiled = torch.compile(fn, backend=backend)

        output_dir = Path(self.tmpdir) / cudagraph_mode
        ctx = patch.object(CudaGraphMgr, "run", _cudagraph_passthrough) if cudagraph_mode != "NONE" else _nullcontext()
        with torch.no_grad(), ctx:
            with explain_compilation(str(output_dir)):
                compiled(torch.randn(2, 8, device="cuda"))

        return output_dir

    def _assert_output(self, output_dir: Path):
        compiled_dir = output_dir / "compiled_functions"
        timeline_dir = output_dir / "timeline_events"
        assert compiled_dir.exists(), f"compiled_functions missing under {output_dir}"
        assert timeline_dir.exists(), f"timeline_events missing under {output_dir}"

        fn_dir = [p for p in compiled_dir.iterdir() if p.is_dir()]
        assert len(fn_dir) == 1, f"Expected one function dir under compiled_functions, got {fn_dir}"
        root = fn_dir[0]
        assert (root / "overview.md").exists()
        assert (root / "decompiled_code.py").exists()
        assert (root / "entry_0" / "decompiled_code.py").exists()

        inductor_files = list(root.rglob("inductor_output.py"))
        assert len(inductor_files) > 0, f"Should have inductor_output.py, files: {list(root.rglob('*'))}"
        for f in inductor_files:
            assert f.stat().st_size > 100

    def test_dump_src_none(self):
        out = self._run_dump_src("NONE")
        self._assert_output(out)

    def test_dump_src_piecewise(self):
        out = self._run_dump_src("PIECEWISE")
        self._assert_output(out)

    def test_dump_src_full(self):
        out = self._run_dump_src("FULL")
        self._assert_output(out)

    def test_structure_identical_across_modes(self):
        """All three modes should produce the same file tree (ignoring hashed names)."""
        import re

        trees = {}
        for mode in ("NONE", "PIECEWISE", "FULL"):
            out = self._run_dump_src(mode)
            torch._dynamo.reset()
            files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
            normalized = [re.sub(r"__compiled_fn_\d+_[0-9a-f_]+", "COMPILED_FN", f) for f in files]
            normalized = [re.sub(r"timeline_events/files/\d+_", "timeline_events/files/EVENT_", f) for f in normalized]
            normalized = [f for f in normalized if "_cudagraph_wrap/" not in f]
            normalized = [f for f in normalized if "_postcleanuppass/" not in f]
            normalized = [f for f in normalized if "_fixfunctionalizationpass/" not in f]
            trees[mode] = normalized

        assert (
            trees["NONE"] == trees["PIECEWISE"]
        ), f"NONE vs PIECEWISE structure differs:\n{trees['NONE']}\nvs\n{trees['PIECEWISE']}"
        assert trees["NONE"] == trees["FULL"], f"NONE vs FULL structure differs:\n{trees['NONE']}\nvs\n{trees['FULL']}"

    def test_timeline_isolation_across_output_dirs(self):
        out_none = self._run_dump_src("NONE")
        torch._dynamo.reset()
        out_full = self._run_dump_src("FULL")

        timeline_none = [
            json.loads(line)
            for line in (out_none / "timeline_events" / "timeline.jsonl").read_text().splitlines()
            if line.strip()
        ]
        timeline_full = [
            json.loads(line)
            for line in (out_full / "timeline_events" / "timeline.jsonl").read_text().splitlines()
            if line.strip()
        ]

        assert timeline_none, "NONE mode timeline should not be empty"
        assert timeline_full, "FULL mode timeline should not be empty"
        assert timeline_none[0]["index"] == 0
        assert timeline_full[0]["index"] == 0

        none_dirs = sorted(p.name for p in (out_none / "timeline_events" / "files").iterdir() if p.is_dir())
        full_dirs = sorted(p.name for p in (out_full / "timeline_events" / "files").iterdir() if p.is_dir())
        assert none_dirs, "NONE mode should produce timeline event directories"
        assert full_dirs, "FULL mode should produce timeline event directories"

        import re

        none_dir_shapes = {re.sub(r"^\d+_", "", name) for name in none_dirs}
        full_dir_shapes = {re.sub(r"^\d+_", "", name) for name in full_dirs}
        none_dir_shapes = {name for name in none_dir_shapes if "cudagraph_wrap" not in name}
        full_dir_shapes = {name for name in full_dir_shapes if "cudagraph_wrap" not in name}
        assert none_dir_shapes == full_dir_shapes, "Both runs should have independent but structurally similar timelines"
