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

A small dataclass IR that the FX pass builds while walking the consumers of an
``aten.mm`` node, and that ``evt_codegen.py`` consumes to render a CUTLASS .cu
source. The IR is canonicalised to a deterministic JSON string used as the
cache key for the JIT'd kernel module.

The IR is rooted at a single ``Store`` node and forms a DAG of compute nodes
over leaves (``Accum``, ``RowBroadcast``, ``ColBroadcast``, ``AuxLoad``).

Op naming: every name in ``UNARY_OPS`` / ``BINARY_OPS`` corresponds to a
CUTLASS visitor template that ``evt_codegen.py`` knows how to emit. Adding a
new op requires updating both this file and the codegen.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import List, Optional, Union

# Ops that take a single child tensor and produce a tensor of the same shape.
# All run in fp32 inside the EVT epilogue.
UNARY_OPS = frozenset(
    {"neg", "sigmoid", "silu", "gelu_erf", "gelu_tanh", "tanh", "relu", "square", "erf", "exp", "log", "sqrt", "rsqrt", "abs"}
)

# Ops that take two child tensors. Both children must be EVT subtrees.
BINARY_OPS = frozenset({"add", "sub", "mul", "div", "max", "min"})

# Unary ops that bake a single fp32 scalar into the functor at codegen time.
# Used to fold scalar literals out of the IR so they don't bloat the cache key.
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

# Output dtype tags propagated from FakeTensor metadata into Store and leaves.
# Kept as strings (not torch.dtype) so the IR is JSON-serialisable.
DTYPES = frozenset({"bfloat16", "float16", "float32"})


# ── Leaf nodes ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Accum:
    """The fp32 GEMM accumulator. Always the unique starting leaf of the IR."""

    kind: str = "accum"


@dataclass(frozen=True)
class RowBroadcast:
    """1-D (N,) tensor broadcast along the M axis. Maps to VisitorRowBroadcast.

    ``input_idx`` is the position of this tensor in the runtime ``extras`` list.
    ``dtype`` is the storage dtype; the visitor casts to fp32 internally.
    """

    input_idx: int
    dtype: str
    kind: str = "row_bcast"


@dataclass(frozen=True)
class ColBroadcast:
    """1-D (M,) tensor broadcast along the N axis. Maps to VisitorColBroadcast."""

    input_idx: int
    dtype: str
    kind: str = "col_bcast"


@dataclass(frozen=True)
class AuxLoad:
    """2-D (M, N) row-major aux tensor. Maps to VisitorAuxLoad.

    Caller must guarantee ``stride[1] == 1`` and that ``stride[0]`` is 16-byte
    aligned (cp.async requirement).
    """

    input_idx: int
    dtype: str
    kind: str = "aux_load"


# ── Compute nodes ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Compute:
    """An interior fp32 elementwise op.

    Children are EVT subtrees (any of the leaf or compute types).
    For SCALAR_UNARY_OPS, ``children`` has length 1 and ``scalar`` carries the
    baked constant.
    For UNARY_OPS, ``children`` has length 1, ``scalar`` is None.
    For BINARY_OPS, ``children`` has length 2, ``scalar`` is None.
    """

    op: str
    children: tuple
    scalar: Optional[float] = None
    kind: str = "compute"

    def __post_init__(self):
        # Validate at construction time so codegen never sees a malformed IR.
        if self.op not in ALL_OPS:
            raise ValueError(f"Unknown EVT op: {self.op!r}")
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


# Union type alias for type hints.
IRNode = Union[Accum, RowBroadcast, ColBroadcast, AuxLoad, Compute, Store]


# ── Canonicalisation + serialisation ──────────────────────────────────────────


def to_dict(node) -> dict:
    """Recursively convert an IR node tree into a JSON-friendly dict.

    The dict layout is designed for stable hashing: keys appear in a fixed
    order and floats are formatted with ``repr`` so 1.702 vs 1.7020000001
    never collide.
    """
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
            # repr of a float is round-trip-safe; explicitly stringify so JSON
            # never serialises 1.7000000000000002.
            d["scalar"] = repr(float(node.scalar))
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


# ── Tree walkers ──────────────────────────────────────────────────────────────


def walk_leaves(node) -> List:
    """Return all leaf nodes (Accum / RowBroadcast / ColBroadcast / AuxLoad)
    in left-to-right pre-order. Used by codegen to enumerate kernel inputs."""
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
    """An IR is trivial when ``Store(Accum)`` — no compute on the accumulator.

    Trivial IRs would replace cuBLAS with a more expensive kernel for no
    benefit, so the FX pass should refuse to emit them.
    """
    return isinstance(node, Store) and isinstance(node.child, Accum)


def num_extras(node) -> int:
    """Maximum input_idx + 1 across all non-Accum leaves, or 0 if none."""
    indices: list = [leaf.input_idx for leaf in walk_leaves(node) if not isinstance(leaf, Accum)]
    return max(indices) + 1 if indices else 0
