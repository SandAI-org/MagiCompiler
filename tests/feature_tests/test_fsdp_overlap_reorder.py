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

"""FsdpOverlapReorder integration test.

The reorder pass is an Inductor scheduler callback that needs a process group + a
real ``torch.compile``, so we drive it via a ``torchrun`` subprocess helper
(fsdp_overlap_helper/reorder_helper.py) and assert on its stdout markers -- same
subprocess pattern as test_autograd_function_cache_flag.py.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

_HELPER = Path(__file__).parent / "fsdp_overlap_helper" / "reorder_helper.py"

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
requires_torchrun = pytest.mark.skipif(shutil.which("torchrun") is None, reason="requires torchrun")


def _run(nproc: int, *extra: str, port: str = "29631") -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["MAGI_LOGGING_LEVEL"] = env.get("MAGI_LOGGING_LEVEL", "info")
    return subprocess.run(
        ["torchrun", f"--nproc_per_node={nproc}", f"--master_port={port}", str(_HELPER), *extra],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


@requires_cuda
@requires_torchrun
def test_reorder_single_rank():
    """world=1: the reorder pass runs inside a real compile, sees a weight gather,
    reorders without error, and the compiled output matches eager."""
    p = _run(1)
    out = p.stdout + p.stderr
    assert p.returncode == 0, f"helper failed:\n{out[-3000:]}"
    assert "REORDER_CALLED gathers=1" in p.stdout, out[-3000:]
    assert "REORDER_OK ran=True" in p.stdout, out[-3000:]
    assert "REORDER_PASS" in p.stdout, out[-3000:]


@requires_cuda
@requires_torchrun
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires >=2 GPUs")
def test_reorder_multi_rank():
    """world=2: same, with a real 2-rank all-gather group."""
    p = _run(2)
    out = p.stdout + p.stderr
    assert p.returncode == 0, f"helper failed:\n{out[-3000:]}"
    assert "REORDER_PASS" in p.stdout, out[-3000:]


class _FakeIR:
    """Sizes are REAL sympy expressions, as ``node.get_size()`` returns: the
    canonicalization must survive sympy's StrPrinter, which orders commutative
    terms by symbol NAME -- a hand-written repr string would bypass exactly the
    layer the bug lives in."""

    def __init__(self, size):
        self.op_overload = "fake.op"
        self._size = size
        self.origins = None

    def get_size(self):
        return self._size


class _FakeSnode:
    snodes = None

    def __init__(self, size):
        self.node = _FakeIR(size)


def _graph(*sizes):
    """Build a fake snode list; each size is a list of sympy exprs / ints."""
    return [_FakeSnode(size) for size in sizes]


def test_fingerprint_canonicalizes_shape_symbols():
    """Dynamic-shape symbol NAMES are per-rank numbering noise (the same
    logical dim is s82 on rank 0 and s74 on rank 1 for the same graph): the
    fingerprint must be invariant under symbol renaming, while still
    distinguishing genuinely different symbolic structure.

    Modeled like the real wan graph: the local-seq symbol first appears alone
    in a placeholder-like dim, then the CP all_to_all output sums it with the
    fresh cross-rank symbol, then downstream nodes use the fresh symbol alone.
    """
    import sympy

    from magi_compiler.passes.fsdp_overlap.reorder import _graph_fingerprint

    def syms(*names):
        return [sympy.Symbol(n, positive=True, integer=True) for n in names]

    # Same digit count (the case the string-level rename happened to handle).
    (a0, b0), (a1, b1) = syms("s27", "s82"), syms("s27", "s74")
    rank0 = _graph([a0, 3072], [a0 + b0, 6, 64], [b0, 64])
    rank1 = _graph([a1, 3072], [a1 + b1, 6, 64], [b1, 64])
    assert _graph_fingerprint(rank0) == _graph_fingerprint(rank1)

    # Digit-count crossing: sympy prints ``s27 + s174`` as ``s174 + s27``
    # (StrPrinter sorts terms by name), so any rename applied AFTER printing
    # sees a different first-appearance order and diverges.
    (a2, b2) = syms("s27", "s174")
    rank2 = _graph([a2, 3072], [a2 + b2, 6, 64], [b2, 64])
    assert _graph_fingerprint(rank0) == _graph_fingerprint(rank2)

    # Same digit count but flipped relative order: fresh symbol sorts BEFORE
    # the shared one on one rank (s34 < s50) and AFTER it on the other.
    (a3, b3), (a4, b4) = syms("s50", "s82"), syms("s50", "s34")
    rank3 = _graph([a3, 3072], [a3 + b3, 6, 64], [b3, 64])
    rank4 = _graph([a4, 3072], [a4 + b4, 6, 64], [b4, 64])
    assert _graph_fingerprint(rank3) == _graph_fingerprint(rank4)

    # Genuinely different LINKAGE must still differ: downstream node reuses the
    # local-seq symbol instead of the cross-rank one.
    linked_other = _graph([a0, 3072], [a0 + b0, 6, 64], [a0, 64])
    assert _graph_fingerprint(rank0) != _graph_fingerprint(linked_other)

    # Genuinely different EXPRESSION structure must differ: 2*s vs s + s'.
    doubled = _graph([a0, 3072], [2 * a0, 6, 64], [b0, 64])
    assert _graph_fingerprint(rank0) != _graph_fingerprint(doubled)


@requires_cuda
@requires_torchrun
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires >=2 GPUs")
def test_reorder_graph_mismatch_fail_fast():
    """world=2 with rank1 compiling a structurally DIFFERENT graph: the cross-rank
    graph-fingerprint check must fire on both ranks (warning), leave the schedule
    unchanged, and complete without deadlock."""
    p = _run(2, "--mismatch", port="29632")
    out = p.stdout + p.stderr
    assert p.returncode == 0, f"helper failed:\n{out[-3000:]}"
    assert "REORDER_MISMATCH unchanged=True" in p.stdout, out[-3000:]
    assert "REORDER_WARNED" in p.stdout, out[-3000:]  # the fail-fast warning fired
    assert "REORDER_PASS" in p.stdout, out[-3000:]
