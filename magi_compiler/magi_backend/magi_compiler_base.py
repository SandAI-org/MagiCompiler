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

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import json
import os
import sys
from contextlib import contextmanager
from pyclbr import Class
from types import CodeType
from typing import Callable, Literal

import torch

from magi_compiler.config import CompileConfig, model_rank_dir_name
from magi_compiler.magi_backend.magi_backend import init_backend
from magi_compiler.magi_depyf.timeline import emit_after_dynamo_bytecode_transform, observe_lifecycle
from magi_compiler.utils import OrderedSet, compute_code_hash, compute_hash, magi_logger


def _source_meta_path(aot_path: str) -> str:
    """Return the path to the source-meta JSON file next to *aot_path*."""
    return os.path.join(os.path.dirname(aot_path), "source_meta.json")


def _save_source_checksum(aot_path: str, traced_files: set[str] | OrderedSet) -> None:
    """Persist a source-file checksum next to the AOT artifact for later validation.

    Called from ``api.py`` after ``aot_compile()`` returns.  ``traced_files`` is
    excluded from ``compile_config.hash`` so the AOT path is stable; this function
    is called before ``traced_files.clear()``.
    """
    if not traced_files:
        magi_logger.warning("AOT serialize: no traced files to save source checksum for")
        return
    checksum = compute_code_hash(traced_files)
    meta_path = _source_meta_path(aot_path)
    with open(meta_path, "w") as f:
        json.dump({"traced_files": list(traced_files), "checksum": checksum}, f)
    magi_logger.info("AOT serialize: source_meta.json saved (%d files, checksum=%s)", len(traced_files), checksum[:16])


def _verify_source_unchanged(aot_path: str) -> list[str] | None:
    """Verify traced source files haven't changed since the AOT artifact was saved.

    Returns the list of traced files if meta exists, else ``None``.
    """
    meta_path = _source_meta_path(aot_path)
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    traced_files = meta["traced_files"]
    actual_checksum = compute_code_hash(set(traced_files))
    if meta["checksum"] != actual_checksum:
        raise RuntimeError("Source code has changed since the last AOT compilation. Please recompile.")
    return traced_files


class MagiCompileState:
    """Companion object holding all compilation state and utilities for a ``@magi_compile``'d target.

    Created by the ``@magi_compile`` decorator and stored as ``obj._magi`` (for modules)
    or captured in a closure (for functions).
    Keeps compilation concerns (torch.compile wrapper, bytecode hooks, AOT artifacts)
    separated from the target's own attributes.
    """

    def __init__(
        self,
        obj: Callable | Class,
        compile_config: CompileConfig,
        model_idx: int,
        model_tag: str,
        dynamic_arg_dims: dict[str, int | list[int]],
        target_method_name: str | None = None,
    ):
        self.obj = obj
        self.compile_config = compile_config
        self.model_idx = model_idx
        self.model_tag = model_tag
        self.dynamic_arg_dims = dynamic_arg_dims
        self.traced_files: OrderedSet = OrderedSet()
        self.inductor_compile_config: dict = {}
        self.target_method_name: str | None = None
        self.target_function: Callable | None = None

        if target_method_name:
            self.target_method_name = target_method_name
            self.target_function = getattr(obj.__class__, self.target_method_name)
            self.original_code_for_hook: CodeType = self.target_function.__code__
            self.original_entry = self.target_function.__get__(obj, obj.__class__)
        elif callable(obj):
            self.original_code_for_hook: CodeType = inspect.unwrap(obj).__code__
            self.original_entry = obj
        else:
            raise TypeError(f"Unsupported object type for MagiCompileState: {type(obj)}")

        self.compiled_entry: Callable | None = None
        self.jit_compiled_code: CodeType | None = None
        self.aot_compiled_fn: Callable | None = None
        self.aot_compile_artifacts: object | None = None

    def _ensure_compiled(self):
        """Lazy initialization of the ``torch.compile`` wrapper.

        Called on first actual compilation (JIT or AOT cache miss).
        On AOT cache hits, this is never called — avoiding ``torch.compile`` overhead entirely.
        """
        if self.compiled_entry is not None:
            return
        backend = init_backend(
            self.compile_config, self.model_idx, self.model_tag, self.traced_files, self.inductor_compile_config
        )
        options = None
        if isinstance(backend, str) and backend == "inductor":
            options = self.inductor_compile_config
        if self.compile_config.aot:
            from magi_compiler.config import CompileMode

            assert self.compile_config.compile_mode == CompileMode.MAGI_COMPILE, (
                "AOT compile (MAGI_COMPILE_AOT=1) requires compile_mode=MAGI_COMPILE. "
                f"Got compile_mode={self.compile_config.compile_mode.value}. "
                "The standard inductor backend does not implement SerializableCallable."
            )
            options = options or {}
            # Drop all the guards in the AOT compile mode as bytecode hook is not used anymore.
            options["guard_filter_fn"] = lambda guards: [False for _ in guards]
            assert hasattr(torch._dynamo.config, "enable_aot_compile"), "enable_aot_compile config not available"
            torch._dynamo.config.enable_aot_compile = True

        # torch.compile returns a ``compile_wrapper`` closure defined in
        # torch/_dynamo/eval_frame.py.  When ``enable_aot_compile=True`` it
        # gets an extra ``.aot_compile`` attribute (eval_frame.py:880) that
        # delegates to ``torch._dynamo.aot_compile.aot_compile_fullgraph``
        # (aot_compile.py:108).
        self.compiled_entry = torch.compile(
            self.original_entry, fullgraph=True, dynamic=True, backend=backend, options=options
        )

    @property
    def aot_compilation_path(self) -> str:
        """
        AOT artifact path, following the same ``model_{idx}_{tag}_rank_{rank}``
        layout as the JIT cache (``cache_dump_path``):

            cache_root_dir/torch_aot_compile/model_{idx}_{tag}_rank_{rank}/{hash}/model

        The ``{hash}`` contains all factors *except* the source files being
        traced through (unknown before Dynamo runs).  On loading we verify
        traced-file checksums separately via ``_verify_source_unchanged``.
        """
        hash_key = compute_hash([self.original_entry, self.model_idx, self.compile_config.hash, self.dynamic_arg_dims])
        cache_dir = os.path.join(
            self.compile_config.cache_root_dir,
            "torch_aot_compile",
            model_rank_dir_name(self.model_idx, self.model_tag),
            hash_key,
        )
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, "model")

    @observe_lifecycle("aot_cache_load")
    def load_aot_compile_artifacts(self):
        aot_path = self.aot_compilation_path
        if not os.path.exists(aot_path):
            magi_logger.info("AOT cache miss: file not found at %s", aot_path)
            return
        saved_traced_files = _verify_source_unchanged(aot_path)
        if saved_traced_files is None:
            magi_logger.info("AOT cache miss: no source_meta.json at %s", aot_path)
            return

        magi_logger.info("AOT cache hit: loading compiled artifacts from %s", aot_path)

        from torch._dynamo.aot_compile import CompileArtifacts

        with open(aot_path, "rb") as f:
            self.aot_compile_artifacts = CompileArtifacts.deserialize(f.read())
        self.aot_compiled_fn = self.aot_compile_artifacts.compiled_function()
        magi_logger.info("AOT cache loaded successfully from %s", aot_path)
        return True

    @observe_lifecycle("aot_artifact_save")
    def save_aot_compile_artifacts(self) -> None:
        """Save the AOT-compiled function and source checksum to disk."""
        from torch._dynamo.aot_compile import CompileArtifacts

        aot_path = self.aot_compilation_path

        with open(aot_path, "wb") as f:
            f.write(CompileArtifacts.serialize(self.aot_compile_artifacts))
        _save_source_checksum(self.aot_compilation_path, self.traced_files)
        magi_logger.info("AOT path: artifacts saved to %s", aot_path)

    _AOT_MAX_RETRIES = 3

    @observe_lifecycle("aot_compile")
    def aot_compile(self, *args, **kwargs):
        """
        Run the model in AOT (Ahead-Of-Time) compile mode.

        All compilation work is completed before execution, suitable for production environment.
        This results in longer compilation time but superior runtime performance.

        The compiled function is stored in ``self._aot_compiled_fn``
        for later serialization / dispatch.

        In the standard JIT path, ``TensorifyScalarRestartAnalysis`` raised by
        ``tensorify_python_scalars`` is caught by Dynamo's ``compile_frame``
        retry loop.  In the AOT path, however, the backend is invoked *outside*
        that loop (``aot_compile_fullgraph`` line 184), so the exception
        propagates uncaught.  We therefore add our own retry here: on each
        attempt ``TensorifyState`` has accumulated more specializations, so the
        next ``fullgraph_capture`` produces a graph that ``tensorify_python_scalars``
        can accept.
        """
        from torch._dynamo.exc import TensorifyScalarRestartAnalysis

        self._aot_retry_count = 0
        for attempt in range(self._AOT_MAX_RETRIES):
            try:
                self.aot_compiled_fn = self.compiled_entry.aot_compile((args, kwargs))
                save_fn = self.aot_compiled_fn.save_compiled_function
                idx = save_fn.__code__.co_freevars.index("self")
                self.aot_compile_artifacts = save_fn.__closure__[idx].cell_contents
                return
            except TensorifyScalarRestartAnalysis:
                if attempt >= self._AOT_MAX_RETRIES - 1:
                    raise
                self._aot_retry_count = attempt + 1
                magi_logger.info(
                    "TensorifyScalarRestartAnalysis during AOT compilation "
                    "(attempt %d/%d), retrying with updated TensorifyState",
                    attempt + 1,
                    self._AOT_MAX_RETRIES,
                )
                self.compiled_entry = None
                self._ensure_compiled()

    @contextmanager
    def _jit_capture_compiled_bytecode(self):
        """Register a Dynamo bytecode hook to capture compiled bytecode.

        Each time Dynamo completes compilation of a frame (e.g., the ``forward``
        method), this hook is triggered by ``torch._dynamo``. We use it to
        extract the newly generated bytecode (``new_code``) and store it in
        ``self.compiled_code``.

        This allows future calls to bypass Dynamo's tracing entirely by
        directly swapping the method's bytecode.

        The hook implementation includes logic to ensure we only capture the
        compiled code for the specific instance and method we are interested
        in, avoiding accidental capture from other parts of the system.
        """

        def _bytecode_hook(old_code: CodeType, new_code: CodeType):
            """Hook to save the compiled bytecode for direct execution."""
            if old_code is not self.original_code_for_hook:
                return
            frame = sys._getframe()
            while frame and frame.f_back:
                frame = frame.f_back
                code_name = frame.f_code.co_name
                file_name = frame.f_code.co_filename.split(os.path.sep)[-1]
                if code_name == "_compile" and file_name == "convert_frame.py":
                    break
            frame = frame.f_locals["frame"]
            assert frame.f_code == old_code

            if self.target_method_name is not None:
                if "self" in frame.f_locals and frame.f_locals["self"] is not self.obj:
                    return

            emit_after_dynamo_bytecode_transform()
            # Save the compiled bytecode
            self.jit_compiled_code = new_code

        handle = torch._dynamo.convert_frame.register_bytecode_hook(_bytecode_hook)
        try:
            yield
        finally:
            handle.remove()

    @contextmanager
    def dispatch_to_compiled_fwd(self, mode: Literal["jit", "aot"] = "jit"):
        """Temporarily swap in compiled code and yield a callable invoker.

        For JIT mode the original ``__code__`` is swapped with the compiled
        bytecode and restored in ``finally``.  For AOT mode the pre-compiled
        function is used directly with no cleanup needed.
        """
        dispatch_via_method = self.target_method_name is not None

        if mode == "jit":
            assert self.jit_compiled_code is not None
            if dispatch_via_method:
                assert self.target_function is not None
                original_code = self.target_function.__code__
                self.target_function.__code__ = self.jit_compiled_code
                try:
                    yield self.target_function.__get__(self.obj, self.obj.__class__)
                finally:
                    self.target_function.__code__ = original_code
            else:
                target = inspect.unwrap(self.obj)
                original_code = target.__code__
                target.__code__ = self.jit_compiled_code
                try:
                    yield target
                finally:
                    target.__code__ = original_code

        elif mode == "aot":
            assert self.aot_compiled_fn is not None
            if dispatch_via_method:
                yield lambda *args, **kwargs: self.aot_compiled_fn(self.obj, *args, **kwargs)
            else:
                yield self.aot_compiled_fn

        else:
            raise ValueError(f"Invalid mode: {mode}")
