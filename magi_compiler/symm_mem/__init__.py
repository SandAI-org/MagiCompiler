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

"""Symmetric-memory weight storage and the copy-engine all-gather built on it.

``all_gather`` is deliberately not re-exported here: importing it defines the
``magi::symm_all_gather`` ops, and callers use the import itself as the probe for
whether the copy-engine transport is available.  Import that module by path.
"""

from .arena import (
    ShardEntry,
    SymmArena,
    barrier_after_load,
    find_shard_by_layout,
    lookup_shard,
    materialize_into_arenas,
    migrate_to_arenas,
    patch_symm_arena_apply,
    register_shard,
    registered_arenas,
    reset_registry,
)

__all__ = [
    "ShardEntry",
    "SymmArena",
    "barrier_after_load",
    "find_shard_by_layout",
    "lookup_shard",
    "materialize_into_arenas",
    "migrate_to_arenas",
    "patch_symm_arena_apply",
    "register_shard",
    "registered_arenas",
    "reset_registry",
]
