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

"""FX pass that fuses aten.mm + elementwise epilogue into a CUTLASS EVT call.

Two backends:
  * Generic EVT — for the 6 non-swiglu activations and 1-D bias/scale variants.
    Builds an IR tree (see ``evt_ir.py``), serialises to JSON, replaces the
    matched chain with a single ``torch.ops.magi_epilogue.matmul_fused_epilogue``
    call. The runtime renders + JIT-compiles a CUTLASS Sm80EVT kernel keyed by
    the IR hash (see ``evt_runtime.py``).
  * swiglu — pattern-matches the canonical recipe (slice-stride-2 + dual
    clamps + scaled SiLU) and dispatches to a vendored DualGemm one-stage
    kernel that writes (M, N/2) directly.

Eligibility gates (alignment, B layout, dtype) are checked up-front. Anything
not eligible stays as ``aten.mm`` for cuBLAS to handle. We do NOT fall back to
the Triton fusion path on sm120; per user decision, EVT replaces it entirely.
"""

from __future__ import annotations

import json
import operator
from typing import List, Optional, Tuple

import torch
import torch.fx as fx

from magi_compiler.passes.pass_base import MagiInductorPass
from magi_compiler.utils.device import device_capability_major

from . import evt_runtime  # ensures torch.library op + fake impl are registered
from .evt_ir import Accum, AuxLoad, ColBroadcast, Compute, RowBroadcast, Store, is_trivial, num_extras, to_canonical_json
from .evt_runtime import GREEDY_ALIGN_BITS

# ── Op tables ────────────────────────────────────────────────────────────────
# Pure passthrough — no value or dtype change; alias the same IR node.
_PASSTHROUGH_OPS = frozenset({torch.ops.aten.clone.default, torch.ops.aten.contiguous.default, torch.ops.aten.alias.default})

# Dtype-conversion ops update current_compute_dtype so downstream Compute nodes
# use the target precision (e.g. to(bf16) → subsequent ops run in bf16).
_TYPE_CONV_OPS = frozenset({torch.ops.prims.convert_element_type.default, torch.ops.aten._to_copy.default})

# Unary ops with a direct EVT IR equivalent.
_UNARY_OPS = {
    torch.ops.aten.neg.default: "neg",
    torch.ops.aten.sigmoid.default: "sigmoid",
    torch.ops.aten.tanh.default: "tanh",
    torch.ops.aten.silu.default: "silu",
    torch.ops.aten.relu.default: "relu",
    torch.ops.aten.square.default: "square",
    torch.ops.aten.erf.default: "erf",
    torch.ops.aten.exp.default: "exp",
    torch.ops.aten.log.default: "log",
    torch.ops.aten.sqrt.default: "sqrt",
    torch.ops.aten.rsqrt.default: "rsqrt",
    torch.ops.aten.abs.default: "abs",
}

_BINARY_OPS = {
    torch.ops.aten.add.Tensor: "add",
    torch.ops.aten.sub.Tensor: "sub",
    torch.ops.aten.mul.Tensor: "mul",
    torch.ops.aten.div.Tensor: "div",
    torch.ops.aten.maximum.default: "max",
    torch.ops.aten.minimum.default: "min",
    operator.add: "add",
    operator.sub: "sub",
    operator.mul: "mul",
    operator.truediv: "div",
}

# Scalar binary ops → SCALAR_UNARY_OPS in IR.
_SCALAR_BINARY_TO_SCALAR_UNARY = {
    torch.ops.aten.add.Scalar: "add_scalar",
    torch.ops.aten.sub.Scalar: "sub_scalar",
    torch.ops.aten.mul.Scalar: "mul_scalar",
    torch.ops.aten.div.Scalar: "div_scalar",
}


# Output-dtype encode helper (mirrors evt_runtime).
_DTYPE_TO_STR = {torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32"}


def _val_dtype(node) -> Optional[torch.dtype]:
    val = node.meta.get("val") if isinstance(node, fx.Node) else None
    return val.dtype if val is not None else None


def _val_shape(node) -> Optional[Tuple]:
    val = node.meta.get("val") if isinstance(node, fx.Node) else None
    return tuple(val.shape) if val is not None else None


def _val_stride(node) -> Optional[Tuple]:
    val = node.meta.get("val") if isinstance(node, fx.Node) else None
    try:
        return tuple(val.stride()) if val is not None else None
    except Exception:
        return None


def _is_static_int(x) -> bool:
    return type(x) is int


# Greedy alignment: try 128-bit first, fall back to 64-bit. CUTLASS only needs
# the leading dim divisible by AlignmentX, so picking the largest power-of-2
# that fits gets us vectorised loads when shapes allow but doesn't lock out
# 64-bit-only shapes (e.g. K=12 for bf16 → 4-elem-aligned).
def _largest_pow2_align_bits(n, dtype: torch.dtype) -> Optional[int]:
    """Return the largest bit-width in (128, 64) that divides ``n * itemsize_bits``.

    For dynamic ``n`` (SymInt) we conservatively return the smallest candidate
    (64) — runtime is the authoritative gate; we just need to admit the fusion
    here. Returns None when even the smallest candidate doesn't fit, in which
    case the caller must abort fusion.
    """
    if not _is_static_int(n):
        return GREEDY_ALIGN_BITS[-1]
    n_int = int(n)
    for bits in GREEDY_ALIGN_BITS:
        align_elems = max(1, bits // (dtype.itemsize * 8))
        if n_int % align_elems == 0:
            return bits
    return None


def _is_transpose_node(n) -> bool:
    """True iff ``n`` is a 2-D transpose (aten.t / transpose(0,1) / permute([1,0]))."""
    if not isinstance(n, fx.Node) or n.op != "call_function":
        return False
    if n.target is torch.ops.aten.t.default:
        return True
    if n.target is torch.ops.aten.transpose.int:
        # transpose(x, dim0, dim1) — accept (0, 1) on a 2D tensor.
        if len(n.args) >= 3:
            d0, d1 = n.args[1], n.args[2]
            return {d0, d1} == {0, 1}
        return False
    if n.target is torch.ops.aten.permute.default:
        # permute(x, [1, 0]) on a 2D tensor.
        if len(n.args) >= 2:
            perm = n.args[1]
            return list(perm) == [1, 0]
        return False
    return False


def _b_layout_kind(B_node):
    """Classify B for the EVT generic path.

    Returns (b_layout, underlying_b_node, n_dim) where:
      * b_layout = "row" : B is (K, N) row-major contiguous; pass B as-is.
      * b_layout = "col" : B is a stride-transpose of a contiguous (N, K)
                            tensor; pass the underlying tensor; kernel uses
                            LayoutB=ColumnMajor.
      * (None, None, None) : B is not in a supported layout.
    """
    shape = _val_shape(B_node)
    stride = _val_stride(B_node)
    if shape is None or stride is None or len(shape) != 2:
        return None, None, None
    K_or_N0, N_or_K1 = shape[0], shape[1]
    # Contiguous (K, N): row layout. N = shape[1].
    if stride == (N_or_K1, 1):
        return "row", B_node, N_or_K1
    # Stride-transposed (K, N) view of a contig (N, K) weight: stride == (1, K).
    # Only accept an explicit t/transpose/permute([1,0]) so we can pass the
    # underlying (N, K) row-major weight to the runtime.  A bare stride-only
    # view would keep the (K, N) logical shape, causing the runtime to swap
    # N_w and K_w (it assumes B.size(0)=N for evt_col).
    if _is_transpose_node(B_node):
        weight = B_node.args[0]
        w_shape = _val_shape(weight) if isinstance(weight, fx.Node) else None
        w_stride = _val_stride(weight) if isinstance(weight, fx.Node) else None
        if w_shape is not None and len(w_shape) == 2 and w_stride == (w_shape[1], 1):
            # weight is (N, K) row-major contig; N = w_shape[0].
            return "col", weight, w_shape[0]
    return None, None, None


# ── swiglu structural validation ───────────────────────────────────────────
def _validate_swiglu_structure(chain_nodes: List[fx.Node], mm_node: fx.Node) -> Optional[Tuple[float, float, float]]:
    """Strictly validate the decomposed swiglu pattern and extract constants.

    The canonical decomposition is::

        mm → _to_copy(fp32)
          → slice(dim=1, start=0, step=2)  [gate]
          → slice(dim=1, start=1, step=2)  [linear]
          → clamp(gate, None, limit)
          → clamp(linear, -limit, limit)
          → mul(gate_clamp, alpha) → sigmoid → mul(gate_clamp, sigmoid)
          → add(linear_clamp, one) → mul(gate_silu, linear_offset)
          → _to_copy(out_dtype)

    Returns ``(alpha, limit, one)`` on match, ``None`` on structural mismatch.
    """

    # ── Phase 1: classify nodes into roles ──────────────────────────────────
    gate_slice: Optional[fx.Node] = None
    linear_slice: Optional[fx.Node] = None
    gate_clamp: Optional[fx.Node] = None
    linear_clamp: Optional[fx.Node] = None
    alpha_mul: Optional[fx.Node] = None
    sigmoid_node: Optional[fx.Node] = None
    gate_silu: Optional[fx.Node] = None
    linear_add: Optional[fx.Node] = None
    final_mul: Optional[fx.Node] = None

    limit: Optional[float] = None
    alpha: Optional[float] = None
    one: Optional[float] = None

    _clamp_targets = {torch.ops.aten.clamp.default, torch.ops.aten.clamp_max.default, torch.ops.aten.clamp_min.default}
    _mul_targets = {torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar}
    _add_targets = {torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar}

    linear_clamp_min: Optional[fx.Node] = None
    linear_clamp_min_val: Optional[float] = None

    for n in chain_nodes:
        t = n.target

        # ── stride-2 slices ─────────────────────────────────────────────
        if t == torch.ops.aten.slice.Tensor:
            if len(n.args) >= 4 and n.args[1] == 1 and (len(n.args) < 5 or n.args[4] == 2):
                step = n.args[4] if len(n.args) >= 5 else 1
                if step != 2:
                    continue
                start = n.args[2]
                if start == 0 and gate_slice is None:
                    gate_slice = n
                elif start == 1 and linear_slice is None:
                    linear_slice = n

        # ── clamp ───────────────────────────────────────────────────────
        elif t in _clamp_targets:
            if t == torch.ops.aten.clamp_min.default:
                # clamp_min(linear_slice, -limit) — first half of decomposed
                # linear clamp: clamp(x, -limit, limit) → clamp_min → clamp_max
                if (
                    len(n.args) >= 2
                    and isinstance(n.args[0], fx.Node)
                    and isinstance(n.args[1], (int, float))
                    and n.args[0] is linear_slice
                    and linear_clamp_min is None
                ):
                    linear_clamp_min = n
                    linear_clamp_min_val = float(n.args[1])

            elif t == torch.ops.aten.clamp_max.default:
                if len(n.args) >= 2 and isinstance(n.args[0], fx.Node) and isinstance(n.args[1], (int, float)):
                    if n.args[0] is gate_slice and gate_clamp is None:
                        gate_clamp = n
                        limit = float(n.args[1])
                    elif n.args[0] is linear_clamp_min and linear_clamp is None:
                        linear_clamp = n
            else:
                # clamp.default(x, min_val, max_val)
                if len(n.args) >= 3 and isinstance(n.args[0], fx.Node):
                    min_val = n.args[1]
                    max_val = n.args[2]
                    if (
                        isinstance(max_val, (int, float))
                        and n.args[0] is gate_slice
                        and (min_val is None)
                        and gate_clamp is None
                    ):
                        gate_clamp = n
                        limit = float(max_val)
                    elif (
                        isinstance(min_val, (int, float))
                        and isinstance(max_val, (int, float))
                        and n.args[0] is linear_slice
                        and linear_clamp is None
                    ):
                        linear_clamp = n

        # ── sigmoid ─────────────────────────────────────────────────────
        elif t == torch.ops.aten.sigmoid.default:
            if sigmoid_node is None:
                sigmoid_node = n

        # ── mul / add ───────────────────────────────────────────────────
        elif t in _mul_targets:
            if (
                len(n.args) >= 2
                and isinstance(n.args[1], (int, float))
                and any(u.target == torch.ops.aten.sigmoid.default for u in n.users)
            ):
                alpha_mul = n
                alpha = float(n.args[1])
            # Other muls are classified in Phase 2 (need sigmoid_node first).

        elif t in _add_targets:
            if len(n.args) >= 2 and isinstance(n.args[0], fx.Node) and isinstance(n.args[1], (int, float)):
                if n.args[0] is linear_clamp and linear_add is None:
                    linear_add = n
                    one = float(n.args[1])

    # ── Phase 2: classify mul nodes that depend on sigmoid ──────────────────
    for n in chain_nodes:
        t = n.target
        if t not in _mul_targets:
            continue
        if n is alpha_mul:
            continue
        if len(n.args) < 2:
            continue
        a0, a1 = n.args[0], n.args[1]
        if not (isinstance(a0, fx.Node) and isinstance(a1, fx.Node)):
            continue
        # gate_silu = mul(gate_clamp, sigmoid)
        if (
            gate_silu is None
            and {a0, a1} == {gate_clamp, sigmoid_node}
            and gate_clamp is not None
            and sigmoid_node is not None
        ):
            gate_silu = n
        # final_mul = mul(gate_silu, linear_add)
        elif final_mul is None and gate_silu is not None and linear_add is not None and {a0, a1} == {gate_silu, linear_add}:
            final_mul = n

    # ── Phase 3: validate all required roles are present ────────────────────
    if any(
        x is None
        for x in (
            gate_slice,
            linear_slice,
            gate_clamp,
            linear_clamp,
            alpha_mul,
            sigmoid_node,
            gate_silu,
            linear_add,
            final_mul,
        )
    ):
        return None

    if alpha is None or limit is None or one is None:
        return None

    # ── Phase 4: cross-validate data-flow edges ─────────────────────────────

    # Both slices must share the same source (the _to_copy of mm).
    if gate_slice.args[0] is not linear_slice.args[0]:
        return None

    # Linear clamp limits must be consistent: min == -limit, max == limit.
    # Two forms:
    #   (a) clamp.default(x, -limit, limit)  — single op
    #   (b) clamp_min(x, -limit) → clamp_max(_, limit)  — decomposed pair
    if linear_clamp.target == torch.ops.aten.clamp.default:
        lin_min = linear_clamp.args[1]
        lin_max = linear_clamp.args[2]
        if not (isinstance(lin_min, (int, float)) and float(lin_min) == -limit):
            return None
        if not (isinstance(lin_max, (int, float)) and float(lin_max) == limit):
            return None
    elif linear_clamp.target == torch.ops.aten.clamp_max.default and linear_clamp_min is not None:
        if not (isinstance(linear_clamp_min_val, (int, float)) and float(linear_clamp_min_val) == -limit):
            return None
        lin_max_val = linear_clamp.args[1]
        if not (isinstance(lin_max_val, (int, float)) and float(lin_max_val) == limit):
            return None
    else:
        return None

    # sigmoid input must be alpha_mul.
    if sigmoid_node.args[0] is not alpha_mul:
        return None

    # alpha_mul input must be gate_clamp.
    if alpha_mul.args[0] is not gate_clamp:
        return None

    return (alpha, limit, one)


# ── swiglu weight / chain validation ──────────────────────────────────────


_SWIGLU7_CHAIN_OPS = frozenset(
    {
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.clamp.default,
        torch.ops.aten.clamp_min.default,
        torch.ops.aten.clamp_max.default,
        torch.ops.aten.sigmoid.default,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.add.Scalar,
        torch.ops.aten.mul.Scalar,
        torch.ops.prims.convert_element_type.default,
        torch.ops.aten._to_copy.default,
        torch.ops.aten.clone.default,
        torch.ops.aten.contiguous.default,
        torch.ops.aten.alias.default,
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten._unsafe_view.default,
    }
)


def _validate_swiglu_weight(mm_node: fx.Node) -> Optional[Tuple[fx.Node, fx.Node, int, int]]:
    """Check B's underlying data is contiguous (N, K) bf16 with N even.

    K alignment and A/B dtype-compatibility are guaranteed by the caller
    (``_try_fuse_evt``).  This validates swiglu-specific constraints only.

    Requires an explicit transpose node (``t(weight)``) so we can extract the
    underlying ``weight`` with shape (N, K).  The runtime reads ``B.size(0)``
    as N, so the tensor passed to the kernel must be (N, K)-shaped.

    Returns ``(B_node, weight_node, N, K)`` on success, ``None`` on failure.
    """
    B_node = mm_node.args[1]
    if not isinstance(B_node, fx.Node) or not _is_transpose_node(B_node):
        return None
    weight_node = B_node.args[0]
    if not isinstance(weight_node, fx.Node):
        return None
    w_shape = _val_shape(weight_node)
    w_stride = _val_stride(weight_node)
    if w_shape is None or len(w_shape) != 2 or w_stride is None:
        return None
    N, K = w_shape
    if w_stride != (K, 1):
        return None
    if not (_is_static_int(N) and N % 2 == 0):
        return None
    if _val_dtype(mm_node.args[0]) != torch.bfloat16 or _val_dtype(weight_node) != torch.bfloat16:
        return None
    # SM90 TMA requires K * sizeof(elem) % 16 == 0.
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9:
        elem_bytes = torch.bfloat16.itemsize
        if _is_static_int(K) and (int(K) * elem_bytes) % 16 != 0:
            return None
    return B_node, weight_node, N, K


def _validate_swiglu_chain(mm_node: fx.Node, N: int) -> Optional[Tuple[List[fx.Node], fx.Node, torch.dtype, str]]:
    """Collect the epilogue chain, validate shape/escape/structure, extract constants.

    Returns ``(chain_nodes, last_chain_node, out_dt, sw7_json)`` on success,
    ``None`` on failure.
    """
    chain_nodes: List[fx.Node] = []
    chain_set: set = {mm_node}
    last_chain_node: Optional[fx.Node] = None
    curr = mm_node.next
    while curr is not None and curr.op != "output":
        uses_chain = any(isinstance(a, fx.Node) and a in chain_set for a in curr.args)
        if not uses_chain:
            curr = curr.next
            continue
        if curr.target not in _SWIGLU7_CHAIN_OPS:
            break
        chain_nodes.append(curr)
        chain_set.add(curr)
        last_chain_node = curr
        curr = curr.next

    if last_chain_node is None:
        return None
    out_dt = _val_dtype(last_chain_node) or torch.bfloat16
    out_shape = _val_shape(last_chain_node)
    if out_shape is None or len(out_shape) != 2:
        return None
    if not _is_static_int(out_shape[1]) or out_shape[1] != N // 2:
        return None
    # Refuse if any intermediate escapes the fused region.
    for n in chain_nodes[:-1]:
        for u in n.users:
            if u not in chain_set:
                return None
    constants = _validate_swiglu_structure(chain_nodes, mm_node)
    if constants is None:
        return None
    sw7_alpha, sw7_limit, sw7_one = constants
    sw7_json = json.dumps({"alpha": sw7_alpha, "limit": sw7_limit, "one": sw7_one}, sort_keys=True)
    return chain_nodes, last_chain_node, out_dt, sw7_json


# ── Shared graph-rewrite helper ────────────────────────────────────────────


def _emit_and_replace(
    graph: fx.Graph,
    last_node: fx.Node,
    op_args: tuple,
    nodes_to_erase: List[fx.Node],
    extra_dead: Optional[List[fx.Node]] = None,
) -> fx.Node:
    """Insert ``matmul_fused_epilogue``, propagate meta, replace uses, erase dead nodes."""
    with graph.inserting_after(last_node):
        new_node = graph.call_function(torch.ops.magi_epilogue.matmul_fused_epilogue.default, args=op_args)
    val_last = last_node.meta.get("val")
    if val_last is not None:
        try:
            n_pad = evt_runtime._aligned_n_stride(int(val_last.shape[-1]), val_last.dtype)
        except (TypeError, ValueError):
            n_pad = None
        if n_pad is not None:
            new_node.meta["val"] = val_last.new_empty_strided(val_last.shape, (n_pad, 1))

    last_node.replace_all_uses_with(new_node)
    for n in reversed(nodes_to_erase):
        if len(n.users) == 0 and n is not new_node:
            graph.erase_node(n)
    if extra_dead:
        for n in extra_dead:
            if isinstance(n, fx.Node) and len(n.users) == 0:
                graph.erase_node(n)
    return new_node


# ── Pass ─────────────────────────────────────────────────────────────────────


class MatmulEvtEpilogueFusionPass(MagiInductorPass):
    """Fuse aten.mm + elementwise chain into a CUTLASS EVT call.

    Active on:
      * sm_90 (Hopper / H100)        — lowers via CUTLASS 3.x Sm90EVT codegen.
      * sm_120+ (Blackwell consumer) — lowers via CUTLASS 2.x Sm80EVT codegen.

    The codegen renderer is picked inside ``evt_runtime._compile_evt_module``
    based on the live device's arch tag.
    """

    def __init__(self, allow_extras: bool = True) -> None:
        major = device_capability_major()
        self._enabled = major == 9 or major >= 12
        self.allow_extras = allow_extras

    def __call__(self, graph: fx.Graph) -> bool:
        if not self._enabled:
            return False
        fused = 0
        for node in list(graph.nodes):
            if node.op != "call_function":
                continue
            if node.target not in (torch.ops.aten.mm.default, torch.ops.aten.mm):
                continue
            if self._try_fuse_evt(graph, node):
                fused += 1
        if fused:
            graph.eliminate_dead_code()
        return fused > 0

    # ── Generic EVT chain walker ──────────────────────────────────────────────

    def _try_fuse_evt(self, graph: fx.Graph, mm_node: fx.Node) -> bool:
        A, B = mm_node.args[0], mm_node.args[1]
        if not isinstance(A, fx.Node) or not isinstance(B, fx.Node):
            return False
        a_dtype = _val_dtype(A)
        b_dtype = _val_dtype(B)
        if a_dtype not in (torch.bfloat16, torch.float16) or a_dtype != b_dtype:
            return False
        a_shape = _val_shape(A)
        b_shape = _val_shape(B)
        if a_shape is None or b_shape is None or len(a_shape) != 2 or len(b_shape) != 2:
            return False
        K = a_shape[1]
        if _largest_pow2_align_bits(K, a_dtype) is None:
            return False
        # SM90 TMA requires globalStride to be 16-byte aligned.  A is
        # RowMajor (M, K) so stride_A[0] = K; need K * elem_bytes % 16 == 0.
        # (For bf16 this reduces to K % 8 == 0.)
        if (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability() == (9, 0)
            and _is_static_int(K)
            and (int(K) * a_dtype.itemsize) % 16 != 0
        ):
            return False
        a_stride = _val_stride(A)
        if a_stride is None or a_stride != (a_shape[1], 1):
            return False

        node_to_ir: dict = {mm_node: Accum()}
        fused_nodes: List[fx.Node] = [mm_node]
        walk_seen: List[fx.Node] = [mm_node]
        extras_nodes: List[fx.Node] = []
        saw_slice = False
        current_compute_dtype = "float32"
        last_node = mm_node
        last_ir = node_to_ir[mm_node]

        # ── Walker-local helpers ──
        curr = mm_node.next

        def _absorb(ir):
            nonlocal last_node, last_ir, curr
            node_to_ir[curr] = ir
            fused_nodes.append(curr)
            walk_seen.append(curr)
            last_node = curr
            last_ir = ir
            curr = curr.next

        def _alias(existing_ir):
            nonlocal last_node, last_ir, curr
            node_to_ir[curr] = existing_ir
            walk_seen.append(curr)
            last_node = curr
            last_ir = existing_ir
            curr = curr.next

        # Walk consumers in source order, greedily absorbing supported ops.
        while curr is not None and curr.op != "output":
            if not any(isinstance(a, fx.Node) and a in node_to_ir for a in curr.args):
                curr = curr.next
                continue

            target = curr.target

            if target in _PASSTHROUGH_OPS:
                _alias(node_to_ir[curr.args[0]])
                continue

            if target in _TYPE_CONV_OPS:
                target_dtype = _val_dtype(curr)
                if target_dtype is not None and target_dtype in _DTYPE_TO_STR:
                    current_compute_dtype = _DTYPE_TO_STR[target_dtype]
                _alias(node_to_ir[curr.args[0]])
                continue

            if target in (torch.ops.aten.view.default, torch.ops.aten.reshape.default, torch.ops.aten._unsafe_view.default):
                if _val_shape(curr.args[0]) == _val_shape(curr):
                    _alias(node_to_ir[curr.args[0]])
                    continue
                break

            if target is torch.ops.aten.slice.Tensor:
                step = curr.args[4] if len(curr.args) > 4 else curr.kwargs.get("step", 1)
                if step == 2:
                    saw_slice = True
                break

            if target in _UNARY_OPS:
                _absorb(Compute(_UNARY_OPS[target], (node_to_ir[curr.args[0]],), compute_dtype=current_compute_dtype))
                continue

            if target is torch.ops.aten.gelu.default:
                op_name = "gelu_tanh" if curr.kwargs.get("approximate", "none") == "tanh" else "gelu_erf"
                _absorb(Compute(op_name, (node_to_ir[curr.args[0]],), compute_dtype=current_compute_dtype))
                continue

            if target in _SCALAR_BINARY_TO_SCALAR_UNARY:
                if not isinstance(curr.args[1], (int, float)):
                    break
                scalar_val = float(curr.args[1])
                if target in (torch.ops.aten.add.Scalar, torch.ops.aten.sub.Scalar):
                    alpha = curr.kwargs.get("alpha", 1)
                    if not isinstance(alpha, (int, float)):
                        break
                    scalar_val = float(alpha) * scalar_val
                _absorb(
                    Compute(
                        _SCALAR_BINARY_TO_SCALAR_UNARY[target],
                        (node_to_ir[curr.args[0]],),
                        scalar=scalar_val,
                        compute_dtype=current_compute_dtype,
                    )
                )
                continue

            if target in (torch.ops.aten.clamp.default, torch.ops.aten.clamp_min.default, torch.ops.aten.clamp_max.default):
                child_ir = node_to_ir[curr.args[0]]
                if target is torch.ops.aten.clamp_min.default:
                    lo = curr.args[1] if len(curr.args) > 1 else curr.kwargs.get("min")
                    hi = None
                elif target is torch.ops.aten.clamp_max.default:
                    lo = None
                    hi = curr.args[1] if len(curr.args) > 1 else curr.kwargs.get("max")
                else:
                    lo = curr.args[1] if len(curr.args) > 1 else curr.kwargs.get("min")
                    hi = curr.args[2] if len(curr.args) > 2 else curr.kwargs.get("max")
                if (lo is not None and not isinstance(lo, (int, float))) or (
                    hi is not None and not isinstance(hi, (int, float))
                ):
                    break
                ir_now = child_ir
                if lo is not None:
                    ir_now = Compute("clamp_min_c", (ir_now,), scalar=float(lo), compute_dtype=current_compute_dtype)
                if hi is not None:
                    ir_now = Compute("clamp_max_c", (ir_now,), scalar=float(hi), compute_dtype=current_compute_dtype)
                _absorb(ir_now)
                continue

            if target is torch.ops.aten.pow.Tensor_Scalar:
                exp = curr.args[1] if len(curr.args) > 1 else None
                child_ir = node_to_ir[curr.args[0]]
                if exp == 2 or exp == 2.0:
                    _absorb(Compute("square", (child_ir,), compute_dtype=current_compute_dtype))
                elif isinstance(exp, (int, float)):
                    _absorb(Compute("pow_scalar", (child_ir,), scalar=float(exp), compute_dtype=current_compute_dtype))
                else:
                    break
                continue

            if target in _BINARY_OPS:
                ir = self._try_lower_binary(curr, target, node_to_ir, extras_nodes, A, B, current_compute_dtype)
                if ir is None:
                    break
                _absorb(ir)
                continue

            break

        if saw_slice:
            return self._try_fuse_swiglu(graph, mm_node)

        result = self._validate_evt_epilogue(
            B, b_dtype, mm_node, node_to_ir, fused_nodes, walk_seen, last_node, last_ir, extras_nodes
        )
        if result is None:
            return False
        ir_json, b_underlying, n_out, out_dt_id, kind = result

        _emit_and_replace(graph, last_node, (A, b_underlying, extras_nodes, ir_json, kind, n_out, out_dt_id), walk_seen)
        return True

    # ── Post-walk EVT validation ──────────────────────────────────────────────

    def _validate_evt_epilogue(
        self, B, b_dtype, mm_node, node_to_ir, fused_nodes, walk_seen, last_node, last_ir, extras_nodes
    ):
        """Post-walk eligibility gates for the generic EVT path.

        Returns ``(ir_json, b_underlying, n_out, out_dt_id, kind)`` on success,
        ``None`` on any gate failure.
        """
        if last_ir is node_to_ir[mm_node]:
            return None

        fused_set = set(fused_nodes) | set(walk_seen)
        for n in walk_seen[:-1]:
            for u in n.users:
                if u not in fused_set:
                    return None

        b_layout, b_underlying, n_dim = _b_layout_kind(B)
        if b_layout is None:
            return None
        if b_layout == "row" and _largest_pow2_align_bits(n_dim, b_dtype) is None:
            return None

        out_dt = _val_dtype(last_node) or torch.bfloat16
        if out_dt not in _DTYPE_TO_STR:
            return None

        if torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0) and _is_static_int(n_dim):
            n_int = int(n_dim)
            if (n_int * out_dt.itemsize) % 16 != 0:
                return None

        ir_root = Store(child=last_ir, out_dtype=_DTYPE_TO_STR[out_dt])
        if is_trivial(ir_root):
            return None
        if not self.allow_extras and num_extras(ir_root) > 0:
            return None
        ir_json = to_canonical_json(ir_root)
        kind = "evt_row" if b_layout == "row" else "evt_col"
        return ir_json, b_underlying, n_dim, evt_runtime.out_dtype_id(out_dt), kind

    # ── Binary op lowering ────────────────────────────────────────────────────

    def _try_lower_binary(self, curr, target, node_to_ir, extras_nodes, A, B, compute_dtype):
        """Try to lower a binary op to IR. Returns an IR node or None (caller breaks)."""
        op_name = _BINARY_OPS[target]
        lhs_raw, rhs_raw = curr.args[0], curr.args[1]

        # aten.add.Tensor / aten.sub.Tensor carry an ``alpha`` kwarg:
        #   add(self, other, alpha=a) → self + a * other
        #   sub(self, other, alpha=a) → self - a * other
        # operator.add/sub and mul/div/max/min have no alpha.
        has_alpha = target in (torch.ops.aten.add.Tensor, torch.ops.aten.sub.Tensor)
        alpha = 1
        if has_alpha:
            alpha = curr.kwargs.get("alpha", 1)
            if not isinstance(alpha, (int, float)):
                return None

        if isinstance(rhs_raw, (int, float)) and isinstance(lhs_raw, fx.Node) and lhs_raw in node_to_ir:
            scalar_op = {"add": "add_scalar", "sub": "sub_scalar", "mul": "mul_scalar", "div": "div_scalar"}.get(op_name)
            if scalar_op is None:
                return None
            scalar_val = float(alpha) * float(rhs_raw) if has_alpha else float(rhs_raw)
            return Compute(scalar_op, (node_to_ir[lhs_raw],), scalar=scalar_val, compute_dtype=compute_dtype)

        if isinstance(lhs_raw, (int, float)) and isinstance(rhs_raw, fx.Node) and rhs_raw in node_to_ir:
            rhs_ir = node_to_ir[rhs_raw]
            if has_alpha and alpha != 1:
                rhs_ir = Compute("mul_scalar", (rhs_ir,), scalar=float(alpha), compute_dtype=compute_dtype)
            if op_name in ("add", "mul"):
                scalar_op = "add_scalar" if op_name == "add" else "mul_scalar"
                return Compute(scalar_op, (rhs_ir,), scalar=float(lhs_raw), compute_dtype=compute_dtype)
            if op_name == "sub":
                return Compute("rsub_scalar", (rhs_ir,), scalar=float(lhs_raw), compute_dtype=compute_dtype)
            return None

        lhs_ir = self._ir_for_arg(lhs_raw, node_to_ir, extras_nodes, A, B)
        rhs_ir = self._ir_for_arg(rhs_raw, node_to_ir, extras_nodes, A, B)
        if lhs_ir is None or rhs_ir is None:
            return None
        if has_alpha and alpha != 1:
            rhs_ir = Compute("mul_scalar", (rhs_ir,), scalar=float(alpha), compute_dtype=compute_dtype)
        return Compute(op_name, (lhs_ir, rhs_ir), compute_dtype=compute_dtype)

    # ── External operand classification ───────────────────────────────────────

    def _ir_for_arg(self, arg, node_to_ir, extras_nodes, A_node, B_node):
        """Classify operand: internal → existing IR; external → leaf node; None ⇒ abort."""
        if not isinstance(arg, fx.Node):
            return None
        if arg in node_to_ir:
            return node_to_ir[arg]
        if not self.allow_extras:
            return None
        a_shape = _val_shape(A_node)
        b_shape = _val_shape(B_node)
        if a_shape is None or b_shape is None:
            return None
        M = a_shape[0]
        N = b_shape[1]
        shape = _val_shape(arg)
        stride = _val_stride(arg)
        dt = _val_dtype(arg)
        if shape is None or dt is None:
            return None
        dt_str = _DTYPE_TO_STR.get(dt)
        if dt_str is None:
            return None
        if len(shape) == 1:
            n0 = shape[0]
            m_is_static = _is_static_int(M)
            n_is_static = _is_static_int(N)
            if n_is_static and n0 == N:
                if m_is_static and n0 == M:
                    return None
                idx = self._add_extra(extras_nodes, arg)
                return RowBroadcast(input_idx=idx, dtype=dt_str)
            if m_is_static and n0 == M:
                idx = self._add_extra(extras_nodes, arg)
                return ColBroadcast(input_idx=idx, dtype=dt_str)
            return None
        if len(shape) == 2:
            if shape[0] == 1 and shape[1] == N:
                idx = self._add_extra(extras_nodes, arg)
                return RowBroadcast(input_idx=idx, dtype=dt_str)
            if shape[1] == 1 and shape[0] == M:
                idx = self._add_extra(extras_nodes, arg)
                return ColBroadcast(input_idx=idx, dtype=dt_str)
            if shape[0] == M and shape[1] == N and stride is not None and stride[1] == 1:
                idx = self._add_extra(extras_nodes, arg)
                return AuxLoad(input_idx=idx, dtype=dt_str)
        return None

    def _add_extra(self, extras_nodes, arg) -> int:
        for i, e in enumerate(extras_nodes):
            if e is arg:
                return i
        extras_nodes.append(arg)
        return len(extras_nodes) - 1

    # ── swiglu special-case ──────────────────────────────────────────────────

    def _try_fuse_swiglu(self, graph: fx.Graph, mm_node: fx.Node) -> bool:
        """Match the canonical swiglu epilogue and dispatch to DualGemm."""
        wt = _validate_swiglu_weight(mm_node)
        if wt is None:
            return False
        B_node, weight_node, N, K = wt

        ch = _validate_swiglu_chain(mm_node, N)
        if ch is None:
            return False
        chain_nodes, last_chain_node, out_dt, sw7_json = ch

        out_dt_id = evt_runtime.out_dtype_id(out_dt)
        n_out = N // 2
        _emit_and_replace(
            graph,
            last_chain_node,
            (mm_node.args[0], weight_node, [], sw7_json, "swiglu_dual", n_out, out_dt_id),
            chain_nodes,
            extra_dead=[mm_node, B_node],
        )
        return True
