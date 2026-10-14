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

import hashlib
import os
from collections.abc import Sequence
from functools import reduce
from typing import Any, List, Union

import torch
import torch.fx as fx
from torch.distributed.tensor import DTensor

HASH_LENGTH = 10


def _fn_hash_key(fn) -> str:
    sha256_hash = hashlib.sha256()
    sha256_hash.update(fn.__qualname__.encode())
    sha256_hash.update(str(fn.__code__.co_firstlineno).encode())
    return sha256_hash.hexdigest()[:HASH_LENGTH]


def compute_hash(obj: Union[Any, List[Any]]) -> str:
    if isinstance(obj, list):
        return reduce(lambda x, y: compute_hash(x + y), [compute_hash(_item) for _item in obj], "")

    elif isinstance(obj, dict):
        return reduce(lambda x, y: compute_hash(x + y), [compute_hash(_k) + compute_hash(_v) for _k, _v in obj.items()], "")

    elif callable(obj):
        return _fn_hash_key(obj)

    return hashlib.md5(str(obj).encode(), usedforsecurity=False).hexdigest()[:HASH_LENGTH]


def compute_code_hash_with_content(file_contents: dict[str, str]) -> str:
    items = list(sorted(file_contents.items(), key=lambda x: x[0]))
    hash_content = []
    for filepath, content in items:
        hash_content.append(filepath)
        if filepath == "<string>":
            # This means the function was dynamically generated, with e.g. exec(). We can't actually check these.
            continue
        hash_content.append(content)
    return compute_hash("\n".join(hash_content))


def _input_signature(x: Any) -> str:
    if isinstance(x, DTensor):
        return f"DTensor({x.dtype}, {x.placements}, {tuple(x.device_mesh.shape)})"
    if isinstance(x, torch.Tensor):
        return f"Tensor({x.dtype})"
    return type(x).__name__


def compute_graph_hash(graph: fx.GraphModule, example_inputs: Sequence[Any]) -> str:
    # DTensor placements live in the inputs, not in the traced source files (e.g. SimpleFSDP fully_shard vs replicate).
    return compute_hash([graph.code, [_input_signature(x) for x in example_inputs]])


def compute_code_hash(files: set[str]) -> str:
    file_contents = {}
    for filepath in files:
        # Skip files that don't exist (e.g., <string>, <frozen modules>, etc.)
        if not os.path.isfile(filepath):
            file_contents[filepath] = ""
        else:
            with open(filepath) as f:
                file_contents[filepath] = f.read()
    return compute_code_hash_with_content(file_contents)
