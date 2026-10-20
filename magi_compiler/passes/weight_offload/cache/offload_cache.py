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

"""Sidecar next to the piecewise compile cache for host-offload slot remapping.

The compiled artifact bakes process-local slot integers into ``magi::h2d_load``.
This object matches ``host_slots.py`` against the weights this process bound,
installs a remap for replay, and refuses to mix bake-time and replay-time slot
spaces under one sidecar.
"""

from __future__ import annotations

import ast
import pprint
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch.fx as fx

from magi_compiler.utils import magi_logger

from .slot_table import collect_host_slot_table, match_host_slot_tables, refresh_resident_flags


class OffloadCache:
    """Host-slot sidecar: match, remap, replay residents, wrap a loaded graph.

    ``_remap`` is the one piece of session state.  ``None`` means bake:
    ``load`` misses while offload is on, and ``save`` may write a sidecar.
    A dict means replay: ``load`` wraps the artifact, ``save`` / ``store``
    must not rewrite slot integers.  Offload off is a no-op: load and store
    stay allowed, remap stays ``None``, and ``bind`` is never called.
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.host_slots_path: Path | None = None
        self._slot_table: dict[int, dict[str, Any]] = {}
        self._remap: dict[int, int] | None = None

    def initialize(self, cache_dir: Path) -> None:
        self.host_slots_path = cache_dir / "host_slots.py"
        self._slot_table = {}
        self._remap = None

    def bind(self, graph: fx.GraphModule) -> str | None:
        """Match the sidecar to ``graph``'s bound weights.

        Returns None when the compile indices are safe to replay, and otherwise
        why not, so the caller can discard indices whose artifacts bake another
        pool's slots.
        """
        from ..runtime import host_weights

        self._slot_table = collect_host_slot_table(graph)
        self._remap = None
        sidecar = self._load_sidecar()
        remap = None if sidecar is None else match_host_slot_tables(sidecar, self._slot_table)
        if remap is None:
            reason = (
                "offload host_slots sidecar missing"
                if sidecar is None
                else "offload host_slots sidecar does not match this process's bound weights"
            )
            magi_logger.info("host offload: %s; compile cache will miss and bake a new one", reason)
            return reason
        current_slots = [remap[baked] for baked, info in sidecar.items() if info.get("resident")]
        if current_slots:
            host_weights.make_resident_many(current_slots)
        self._remap = remap
        magi_logger.info("host offload: compile cache reusable (%d slot(s) remapped)", len(remap))
        return None

    def allows_load(self) -> bool:
        """False only when offload is on and the sidecar has not matched."""
        return (not self.enabled) or self._remap is not None

    def allows_store(self) -> bool:
        """False on replay: a freshly compiled graph would bake this process's slots."""
        return self._remap is None

    def wrap_loaded(self, compiled: Callable) -> Callable:
        remap = self._remap
        if remap is None:
            return compiled

        def wrapped(*args, __fn=compiled, __remap=remap):
            from ..runtime.slot_remap import using_slot_remap

            with using_slot_remap(__remap):
                return __fn(*args)

        return wrapped

    def save(self) -> None:
        if not self.enabled or self.host_slots_path is None:
            return
        if self._remap is not None:
            # Replay: artifacts on disk still use the baked slot space.  Rewriting
            # the sidecar with this process's integers would desync the next load.
            return
        table = refresh_resident_flags(self._slot_table)
        printer = pprint.PrettyPrinter(indent=4)
        self.host_slots_path.write_text(printer.pformat(table))

    def _load_sidecar(self) -> dict[int, dict] | None:
        if self.host_slots_path is None or not self.host_slots_path.exists():
            return None
        try:
            raw = ast.literal_eval(self.host_slots_path.read_text())
        except (OSError, SyntaxError, ValueError) as exc:
            magi_logger.warning("host offload: failed to parse host_slots sidecar (%s); treating as cache miss", exc)
            return None
        if not isinstance(raw, dict):
            magi_logger.warning("host offload: host_slots sidecar is not a dict; treating as cache miss")
            return None
        try:
            return {int(k): dict(v) for k, v in raw.items()}
        except (TypeError, ValueError) as exc:
            magi_logger.warning("host offload: host_slots sidecar has invalid entries (%s); treating as cache miss", exc)
            return None
