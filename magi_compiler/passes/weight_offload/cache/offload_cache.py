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
import os
import pprint
from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch.fx as fx

from magi_compiler.utils import magi_logger

from .slot_table import (
    HostSlotMetadata,
    collect_host_slot_table,
    match_host_slot_tables,
    parse_host_slot_table,
    refresh_resident_flags,
)


class OffloadCache:
    """Host-slot sidecar: match, remap, replay residents, wrap a loaded graph.

    ``_cached_to_current_slots`` is None until a sidecar matches.
    Replay wraps the artifact with that mapping and forbids rewriting its slot space.
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.host_slots_path: Path | None = None
        self._graph_slot_table: dict[int, HostSlotMetadata] = {}
        self._cached_to_current_slots: dict[int, int] | None = None

    def initialize(self, cache_dir: Path) -> None:
        self.host_slots_path = cache_dir / "host_slots.py"
        self._graph_slot_table = {}
        self._cached_to_current_slots = None

    def prepare_graph_replay(self, graph: fx.GraphModule, *, sync_across_ranks: bool = False) -> str | None:
        """Match the sidecar to ``graph``'s bound weights.

        Returns None when the compile indices are safe to replay, and otherwise
        why not, so the caller can discard indices whose artifacts bake another
        pool's slots.
        """
        from ..runtime.host_weights import get_global_host_weight_store

        self._graph_slot_table = collect_host_slot_table(graph)
        self._cached_to_current_slots = None
        cached_slots = self._load_sidecar()
        cached_to_current_slots = (
            None if cached_slots is None else match_host_slot_tables(cached_slots, self._graph_slot_table)
        )
        cache_miss_reason = None
        if cached_to_current_slots is None:
            cache_miss_reason = (
                "offload host_slots sidecar missing"
                if cached_slots is None
                else "offload host_slots sidecar does not match this process's bound weights"
            )
        if sync_across_ranks:
            from ...transfer_overlap import rank_sync

            # Decide before restoring residents or installing a remap: a peer may need fresh scheduling.
            if rank_sync.any_rank_has_flag(cache_miss_reason is not None):
                cache_miss_reason = cache_miss_reason or "offload host_slots sidecar missed on another rank"
        if cache_miss_reason is not None:
            magi_logger.info("host offload: %s; compile cache will miss and bake a new one", cache_miss_reason)
            return cache_miss_reason
        resident_slots_to_restore = [
            cached_to_current_slots[cached_slot]
            for cached_slot, slot_metadata in cached_slots.items()
            if slot_metadata["resident"]
        ]
        if resident_slots_to_restore:
            get_global_host_weight_store().make_weights_resident(resident_slots_to_restore)
        self._cached_to_current_slots = cached_to_current_slots
        magi_logger.info("host offload: compile cache reusable (%d slot(s) remapped)", len(cached_to_current_slots))
        return None

    def can_replay_artifact(self) -> bool:
        """False only when offload is on and the sidecar has not matched."""
        return (not self.enabled) or self._cached_to_current_slots is not None

    def can_store_artifact(self) -> bool:
        """False on replay: a freshly compiled graph would bake this process's slots."""
        return self._cached_to_current_slots is None

    def wrap_with_slot_remap(self, compiled_fn: Callable) -> Callable:
        cached_to_current_slots = self._cached_to_current_slots
        if cached_to_current_slots is None:
            return compiled_fn

        def run_with_slot_remap(*args, __compiled_fn=compiled_fn, __slot_remap=cached_to_current_slots):
            from ..runtime.slot_remap import using_slot_remap

            with using_slot_remap(__slot_remap):
                return __compiled_fn(*args)

        return run_with_slot_remap

    def save_slot_sidecar(self) -> None:
        if not self.enabled or self.host_slots_path is None:
            return
        if self._cached_to_current_slots is not None:
            # Replay: artifacts on disk still use the baked slot space.  Rewriting
            # the sidecar with this process's integers would desync the next load.
            return
        slot_table = refresh_resident_flags(self._graph_slot_table)
        printer = pprint.PrettyPrinter(indent=4)
        # Same-directory replacement keeps readers from observing a partially written sidecar.
        temporary_file = NamedTemporaryFile(mode="w", dir=self.host_slots_path.parent, prefix=".host_slots.", delete=False)
        temporary_path = Path(temporary_file.name)
        try:
            with temporary_file:
                temporary_file.write(printer.pformat(slot_table))
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            temporary_path.replace(self.host_slots_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _load_sidecar(self) -> dict[int, HostSlotMetadata] | None:
        if self.host_slots_path is None or not self.host_slots_path.exists():
            return None
        try:
            sidecar_data = ast.literal_eval(self.host_slots_path.read_text())
        except (OSError, SyntaxError, ValueError) as exc:
            magi_logger.warning("host offload: failed to parse host_slots sidecar (%s); treating as cache miss", exc)
            return None
        if not isinstance(sidecar_data, dict):
            magi_logger.warning("host offload: host_slots sidecar is not a dict; treating as cache miss")
            return None
        try:
            return parse_host_slot_table(sidecar_data)
        except (KeyError, TypeError, ValueError) as exc:
            magi_logger.warning("host offload: host_slots sidecar has invalid entries (%s); treating as cache miss", exc)
            return None
