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


from ._utils import *
from .compile_counter import compilation_counter
from .device import device_capability, device_capability_major
from .envs import set_env_var
from .hash import compute_code_hash, compute_code_hash_with_content, compute_hash
from .logger import logger, magi_logger
from .nvtx import add_nvtx_event, instrument_nvtx
from .ordered_set import OrderedSet
from .singleton_meta import SingletonMeta

__all__ = [
    "compilation_counter",
    "set_env_var",
    "compute_code_hash",
    "compute_code_hash_with_content",
    "compute_hash",
    "logger",
    "magi_logger",
    "OrderedSet",
    "SingletonMeta",
    "instrument_nvtx",
    "add_nvtx_event",
    "device_capability",
    "device_capability_major",
]
