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

"""EVT (Epilogue Visitor Tree) intermediate representation.

Dataclass IR built by the FX pass from ``aten.mm`` consumers, consumed by
``evt_codegen.py`` to render a CUTLASS .cu. Canonicalised to deterministic
JSON for the JIT module cache key. Adding a new op requires updating both
this file and the codegen.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import List, Optional, Union

UNARY_OPS = frozenset(
    {"neg", "sigmoid", "silu", "gelu_erf", "gelu_tanh", "tanh", "relu", "square", "erf", "exp", "log", "sqrt", "rsqrt", "abs"}
)

BINARY_OPS = frozenset({"add", "sub", "mul", "div", "max", "min"})

SCALAR_UNARY_OPS = frozenset(
    {
        "add_scalar",  # x + c
        "sub_scalar",  # x - c
        "mul_scalar",  # x * c
        "div_scalar",  # x / c
        "rsub_scalar",  # c - x
        "clamp_min_c",  # max(x, c)
        "clamp_max_c",  # min(x, c)
        "scaled_silu_alpha",  # x * sigmoid(alpha * x), used by gelu7
        "pow_scalar",  # x ** c (only sensible for small integer c)
    }
)

ALL_OPS = UNARY_OPS | BINARY_OPS | SCALAR_UNARY_OPS

# Strings (not torch.dtype) so the IR is JSON-serialisable.
DTYPES = frozenset({"bfloat16", "float16", "float32"})

# Hardware-native ALU compute types supported by the EVT epilogue.
#
# Floating-point: FP32, FP16, BF16 are full-speed on both H100 (sm_90) and
# RTX 5090 (sm_120). FP64 is full-speed on H100 but extremely slow on 5090,
# so we exclude it from the EVT path.
#
# Integer: INT64, INT32, INT16, INT8 are ALU-supported on both architectures,
# but CUTLASS VisitorCompute / Sm90Compute templates are only instantiated
# for floating-point types, so integer compute_dtype is not valid here.
COMPUTE_DTYPES = frozenset({"bfloat16", "float16", "float32"})


@dataclass(frozen=True)
class Accum:
    """The fp32 GEMM accumulator. Always the unique starting leaf of the IR."""

    kind: str = "accum"


@dataclass(frozen=True)
class RowBroadcast:
    """1-D (N,) tensor broadcast along M. ``input_idx`` indexes the runtime extras list."""

    input_idx: int
    dtype: str
    kind: str = "row_bcast"


@dataclass(frozen=True)
class ColBroadcast:
    """1-D (M,) tensor broadcast along N."""

    input_idx: int
    dtype: str
    kind: str = "col_bcast"


@dataclass(frozen=True)
class AuxLoad:
    """2-D (M, N) row-major aux tensor. stride[1] must be 1, stride[0] 16-byte aligned."""

    input_idx: int
    dtype: str
    kind: str = "aux_load"


@dataclass(frozen=True)
class Compute:
    """An interior elementwise op over EVT subtrees.

    ``compute_dtype`` controls the precision of this node's VisitorCompute /
    Sm90Compute template instantiation. Defaults to ``"float32"`` (the GEMM
    accumulator's native precision). A preceding ``to(bf16)`` in the FX
    chain sets it to ``"bfloat16"`` so the kernel runs that op in bf16.
    """

    op: str
    children: tuple
    scalar: Optional[float] = None
    compute_dtype: str = "float32"
    kind: str = "compute"

    def __post_init__(self):
        if self.op not in ALL_OPS:
            raise ValueError(f"Unknown EVT op: {self.op!r}")
        if self.compute_dtype not in COMPUTE_DTYPES:
            raise ValueError(f"Unsupported compute_dtype {self.compute_dtype!r} for EVT. " f"Valid: {sorted(COMPUTE_DTYPES)}")
        if self.op in UNARY_OPS:
            if len(self.children) != 1 or self.scalar is not None:
                raise ValueError(f"UNARY op {self.op!r} requires 1 child, no scalar")
        elif self.op in BINARY_OPS:
            if len(self.children) != 2 or self.scalar is not None:
                raise ValueError(f"BINARY op {self.op!r} requires 2 children, no scalar")
        elif self.op in SCALAR_UNARY_OPS:
            if len(self.children) != 1 or self.scalar is None:
                raise ValueError(f"SCALAR_UNARY op {self.op!r} requires 1 child + scalar")


@dataclass(frozen=True)
class Store:
    """Root of the IR. Casts the fp32 result to ``out_dtype`` and writes D."""

    child: object  # any IR node
    out_dtype: str
    kind: str = "store"

    def __post_init__(self):
        if self.out_dtype not in DTYPES:
            raise ValueError(f"Unknown out_dtype {self.out_dtype!r}")


IRNode = Union[Accum, RowBroadcast, ColBroadcast, AuxLoad, Compute, Store]


def to_dict(node) -> dict:
    """Recursively convert an IR tree into a JSON-friendly dict for stable hashing."""
    if isinstance(node, Accum):
        return {"kind": "accum"}
    if isinstance(node, RowBroadcast):
        return {"kind": "row_bcast", "input_idx": node.input_idx, "dtype": node.dtype}
    if isinstance(node, ColBroadcast):
        return {"kind": "col_bcast", "input_idx": node.input_idx, "dtype": node.dtype}
    if isinstance(node, AuxLoad):
        return {"kind": "aux_load", "input_idx": node.input_idx, "dtype": node.dtype}
    if isinstance(node, Compute):
        d = {"kind": "compute", "op": node.op, "children": [to_dict(c) for c in node.children]}
        if node.scalar is not None:
            d["scalar"] = repr(float(node.scalar))
        if node.compute_dtype != "float32":
            d["compute_dtype"] = node.compute_dtype
        return d
    if isinstance(node, Store):
        return {"kind": "store", "out_dtype": node.out_dtype, "child": to_dict(node.child)}
    raise TypeError(f"Unknown IR node type: {type(node).__name__}")


def to_canonical_json(node) -> str:
    """Deterministic JSON string for an IR tree. Same IR ⇒ same string."""
    return json.dumps(to_dict(node), sort_keys=True, separators=(",", ":"))


def cache_key(node, a_dtype: str, b_dtype: str) -> str:
    """SHA-256 hash of (IR JSON, A dtype, B dtype). Used as the JIT module key."""
    payload = {"ir": to_dict(node), "a": a_dtype, "b": b_dtype, "version": 1}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def walk_leaves(node) -> List:
    """Return all leaf nodes in left-to-right pre-order."""
    out: list = []

    def _go(n):
        if isinstance(n, (Accum, RowBroadcast, ColBroadcast, AuxLoad)):
            out.append(n)
        elif isinstance(n, Compute):
            for c in n.children:
                _go(c)
        elif isinstance(n, Store):
            _go(n.child)
        else:
            raise TypeError(f"Unknown IR node type: {type(n).__name__}")

    _go(node)
    return out


def is_trivial(node) -> bool:
    """Store(Accum) — no compute; FX pass should refuse to emit these."""
    return isinstance(node, Store) and isinstance(node.child, Accum)


def num_extras(node) -> int:
    """Maximum input_idx + 1 across all non-Accum leaves, or 0 if none."""
    indices: list = [leaf.input_idx for leaf in walk_leaves(node) if not isinstance(leaf, Accum)]
    return max(indices) + 1 if indices else 0
