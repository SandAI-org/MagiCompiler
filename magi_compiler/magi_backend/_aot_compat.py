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

"""AOT compilation compatibility shim for PyTorch 2.9 and 2.12.

PyTorch 2.12 introduced AOTCompiledFunction with a new deserialize/save API,
replacing the older CompileArtifacts class from 2.9.  This module provides
a unified interface that works across both versions.
"""

from __future__ import annotations

try:
    from torch._dynamo.aot_compile import AOTCompiledFunction as _AOTCompiledFunction

    _HAS_AOT_COMPILED_FUNCTION = True
except ImportError:
    _AOTCompiledFunction = None
    _HAS_AOT_COMPILED_FUNCTION = False

try:
    from torch._dynamo.aot_compile import CompileArtifacts as _CompileArtifacts

    _HAS_COMPILE_ARTIFACTS = True
except ImportError:
    _CompileArtifacts = None
    _HAS_COMPILE_ARTIFACTS = False


def load_aot_artifacts(aot_path: str, f_globals: dict | None = None):
    """Load AOT-compiled artifacts from disk.

    Returns the callable compiled function on success.

    PyTorch >= 2.12: uses AOTCompiledFunction.deserialize(data, f_globals=...)
    PyTorch < 2.12:  uses CompileArtifacts.deserialize(data).compiled_function()
    """
    with open(aot_path, "rb") as f:
        data = f.read()

    if _HAS_AOT_COMPILED_FUNCTION:
        return _AOTCompiledFunction.deserialize(data, f_globals=f_globals)

    assert _HAS_COMPILE_ARTIFACTS, "Neither AOTCompiledFunction (torch>=2.12) nor CompileArtifacts (torch<2.12) is available"
    artifacts = _CompileArtifacts.deserialize(data)
    return artifacts.compiled_function()


def save_aot_artifacts(aot_compiled_fn, aot_path: str, aot_compile_artifacts=None) -> None:
    """Save AOT-compiled artifacts to disk.

    PyTorch >= 2.12: calls aot_compiled_fn.save_compiled_function(path)
    PyTorch < 2.12:  serializes CompileArtifacts via CompileArtifacts.serialize()
    """
    if _HAS_AOT_COMPILED_FUNCTION:
        aot_compiled_fn.save_compiled_function(aot_path)
        return

    assert (
        _HAS_COMPILE_ARTIFACTS and aot_compile_artifacts is not None
    ), "CompileArtifacts required for saving on PyTorch < 2.12"
    with open(aot_path, "wb") as f:
        f.write(_CompileArtifacts.serialize(aot_compile_artifacts))


def extract_aot_artifacts_from_fn(aot_compiled_fn):
    """Extract CompileArtifacts from the compiled function (PyTorch < 2.12 only).

    In PyTorch 2.12+, artifacts are managed internally by AOTCompiledFunction
    and this function returns None.
    """
    if _HAS_AOT_COMPILED_FUNCTION:
        return None

    save_fn = aot_compiled_fn.save_compiled_function
    idx = save_fn.__code__.co_freevars.index("self")
    return save_fn.__closure__[idx].cell_contents
