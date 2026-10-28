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
import itertools
import os
import re
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


# Dynamo names a lifted module parameter or buffer placeholder ``..._parameters_<name>_`` / ``..._buffers_<name>_``.
_MODULE_STATE = re.compile(r"_(parameters|buffers)_")


def _state_layout(x: torch.Tensor) -> str:
    layout = f"{x.dtype}{list(x.shape)}"
    if isinstance(x, DTensor):
        layout += f" {x.placements} {tuple(x.device_mesh.shape)}"
    return layout


def _input_signature(name: str, x: torch.Tensor) -> str | None:
    if _MODULE_STATE.search(name):
        return f"{name}: {_state_layout(x)}"
    if isinstance(x, DTensor):
        return f"{name}: {x.dtype} {x.placements} {tuple(x.device_mesh.shape)}"
    return None


def compute_graph_hash(graph: fx.GraphModule, example_inputs: Sequence[Any]) -> str:
    """Every parameter and buffer by name, dtype and shape, so another layer count or weight layout (PP stage,
    quantization) gets its own key, plus DTensor placements, which do not appear in traced files.

    Only these stay the same in the retrace after TensorifyScalarRestartAnalysis and between an AOT compile and load:
    the retrace renames FX nodes and drops the 0-d tensors that carried Python floats, and the AOT compile sees
    concrete sizes for the dims that are symbolic on load, so activations and graph.code stay out.
    """
    names = [node.name for node in graph.graph.nodes if node.op == "placeholder"]
    sigs = [_input_signature(name, x) for name, x in zip(names, example_inputs) if isinstance(x, torch.Tensor)]
    return compute_hash([sig for sig in sigs if sig is not None])


def compute_module_state_hash(obj: object) -> str:
    """Every parameter and buffer of ``obj`` by name, dtype and shape, or nothing unless ``obj`` is a module.

    The AOT key needs it: an AOT load skips Dynamo and drops every guard, so nothing else checks the weight layout.
    """
    if not isinstance(obj, torch.nn.Module):
        return compute_hash([])
    state = itertools.chain(obj.named_parameters(), obj.named_buffers())
    return compute_hash([f"{name}: {_state_layout(x)}" for name, x in state])


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
