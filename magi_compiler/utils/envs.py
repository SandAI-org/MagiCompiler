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

import contextlib
import os
from typing import Iterator


def _env_to_bool(env_name: str, default: bool) -> bool:
    env_value = str(os.environ.get(env_name, default))
    if env_value.lower() in {"1", "true", "yes", "y", "on", "enabled"}:
        return True
    if env_value.lower() in {"0", "false", "no", "n", "off", "disabled"}:
        return False
    return default


@contextlib.contextmanager
def set_env_var(key: str, value: str) -> Iterator[None]:
    """Temporarily set an environment variable."""
    old = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


# Enable FX graph visualization, which actually controls `enable_fx_graph_viz` in the Inductor config.
MAGI_ENABLE_FX_GRAPH_VIZ: bool = _env_to_bool("MAGI_ENABLE_FX_GRAPH_VIZ", default=False)

# FX graph visualization node description mode: simple or detailed.
MAGI_FX_GRAPH_VIZ_NODE_DESC: str = os.getenv("MAGI_FX_GRAPH_VIZ_NODE_DESC", "simple")

# Equal to TORCHINDUCTOR_PATTERN_MATCH_DEBUG environment.
MAGI_PATTERN_MATCH_DEBUG: str | None = os.getenv("MAGI_PATTERN_MATCH_DEBUG")

# Default path for shared memory binaries.
MAGI_SHARED_BIN_PATH = "/dev/shm"

# Logging level for MagiCompiler (DEBUG / INFO / WARNING / ERROR). Read once at import time.
MAGI_LOGGING_LEVEL: str = os.getenv("MAGI_LOGGING_LEVEL", "WARNING").upper()
