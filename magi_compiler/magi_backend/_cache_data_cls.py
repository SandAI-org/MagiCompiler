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

import dataclasses


@dataclasses.dataclass
class CacheHandle:
    key: str | None
    path: str
    restart_analysis_count: int = 0


@dataclasses.dataclass(frozen=True)
class CacheEntry:
    runtime_shape: int | None
    graph_index: int
    backend_name: str
    # A digest of the sizes dynamo turned into constants for this trace. runtime_shape
    # alone does not identify an artifact: it is None for every dynamically compiled
    # graph, while the dimensions nobody marked dynamic are baked in and differ between
    # traces. Without this, a graph traced for one input size is silently handed to
    # another and fails inside the kernel on a stride assertion.
    static_sizes: str = ""
