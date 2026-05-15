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
    matched chain with a single ``torch.ops.magi_epilogue.matmul_custom_evt``
    call. The runtime renders + JIT-compiles a CUTLASS Sm80EVT kernel keyed by
    the IR hash (see ``evt_runtime.py``).
  * swiglu7 — pattern-matches the canonical recipe (slice-stride-2 + dual
    clamps + scaled SiLU) and dispatches to a vendored DualGemm one-stage
    kernel that writes (M, N/2) directly.

Eligibility gates (alignment, B layout, dtype) are checked up-front. Anything
not eligible stays as ``aten.mm`` for cuBLAS to handle. We do NOT fall back to
the Triton fusion path on sm120; per user decision, EVT replaces it entirely.
"""

from __future__ import annotations

import operator
from typing import List, Optional, Tuple

import torch
import torch.fx as fx

from magi_compiler.cuda.device import device_capability_major
from magi_compiler.passes.pass_base import MagiInductorPass

from . import evt_runtime  # ensures torch.library op + fake impl are registered
from .evt_ir import Accum, AuxLoad, ColBroadcast, Compute, RowBroadcast, Store, is_trivial, num_extras, to_canonical_json

# ── Op tables ────────────────────────────────────────────────────────────────
# Pure passthrough — no value or dtype change; alias the same IR node.
_PASSTHROUGH_OPS = frozenset({torch.ops.aten.clone.default, torch.ops.aten.contiguous.default, torch.ops.aten.alias.default})

# Dtype-conversion ops; the EVT compute is always fp32 internally so these are
# absorbed as no-ops as long as the start/end of the chain reach the same final
# precision (we capture that via the Store node's out_dtype).
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

# Binary tensor ops.
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
_GREEDY_ALIGN_BITS = (128, 64)


def _largest_pow2_align_bits(n, dtype: torch.dtype) -> Optional[int]:
    """Return the largest bit-width in (128, 64) that divides ``n * itemsize_bits``.

    For dynamic ``n`` (SymInt) we conservatively return the smallest candidate
    (64) — runtime is the authoritative gate; we just need to admit the fusion
    here. Returns None when even the smallest candidate doesn't fit, in which
    case the caller must abort fusion.
    """
    if not _is_static_int(n):
        return _GREEDY_ALIGN_BITS[-1]
    n_int = int(n)
    for bits in _GREEDY_ALIGN_BITS:
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
    # The underlying tensor is the transpose-producer's input when the FX
    # graph models the view explicitly via t/transpose/permute([1,0]); fall
    # back to using B itself (its data_ptr is the same).
    if _is_transpose_node(B_node):
        weight = B_node.args[0]
        w_shape = _val_shape(weight) if isinstance(weight, fx.Node) else None
        w_stride = _val_stride(weight) if isinstance(weight, fx.Node) else None
        if w_shape is not None and len(w_shape) == 2 and w_stride == (w_shape[1], 1):
            # weight is (N, K) row-major contig; N = w_shape[0].
            return "col", weight, w_shape[0]
    # Generic stride-transposed view (no explicit transpose node) — also OK:
    # we read the same memory bytes as a (N, K) row-major buffer at B itself.
    if stride == (1, K_or_N0):
        # B is (K, N) col-major == underlying (N, K) row-major. We don't have
        # an explicit weight node so we pass B directly; the kernel reads
        # (N, K) with N = shape[1], K = shape[0]. Detection via stride alone.
        return "col", B_node, N_or_K1
    return None, None, None


# ── Pass ─────────────────────────────────────────────────────────────────────


# Sentinel returned by _try_fuse_evt to communicate "abort, leave mm intact".
_ABORT = object()


class MatmulEvtEpilogueFusionPass(MagiInductorPass):
    """Fuse aten.mm + elementwise chain into a CUTLASS EVT call (sm_120)."""

    def __init__(self, allow_extras: bool = True) -> None:
        # On non-sm120 we degrade to a no-op; the manager wires us only on
        # sm120 anyway, but defending against misuse is cheap.
        self._enabled = device_capability_major() >= 12
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
            r = self._try_fuse_evt(graph, node)
            if r:
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
        # Alignment gates — A is RowMajor (M, K) so ldA = K must divide
        # AlignmentA. We greedy-pick AlignmentA at runtime (128 → 64 bits),
        # so the FX gate only refuses K not even 64-bit-aligned (= K%4 for
        # bf16/fp16). B's N-side gate is path-specific and checked after
        # b_layout is resolved. D's N is unconstrained here: the runtime
        # allocates a padded buffer and returns a strided view, so any n_out
        # divides into AlignmentC.
        a_shape = _val_shape(A)
        b_shape = _val_shape(B)
        if a_shape is None or b_shape is None or len(a_shape) != 2 or len(b_shape) != 2:
            return False
        K = a_shape[1]
        if _largest_pow2_align_bits(K, a_dtype) is None:
            return False

        # node_to_ir: each fused fx.Node → its IR subtree. mm_node maps to Accum.
        node_to_ir: dict = {mm_node: Accum()}
        # In-order list of fused fx nodes (for erase + escape detection).
        fused_nodes: List[fx.Node] = [mm_node]
        # Walked-and-removed nodes including type-conv/passthrough that don't
        # appear in node_to_ir as new IR nodes (they alias their input).
        walk_seen: List[fx.Node] = [mm_node]
        # External tensors injected as RowBroadcast/ColBroadcast/AuxLoad leaves.
        # extras_nodes[i] is the fx.Node passed at runtime as extras[i].
        extras_nodes: List[fx.Node] = []
        # Tracks whether the IR has any swiglu7-style slice. If so we abort
        # generic EVT and try the swiglu7 matcher instead.
        saw_slice = False

        last_node = mm_node
        last_ir = node_to_ir[mm_node]

        # Walk consumers in source order, greedily absorbing supported ops.
        curr = mm_node.next
        while curr is not None and curr.op != "output":
            uses_fused = any(isinstance(a, fx.Node) and a in node_to_ir for a in curr.args)
            if not uses_fused:
                curr = curr.next
                continue

            target = curr.target

            # ── Pass-through (clone / contiguous / alias) ─────────────────────
            if target in _PASSTHROUGH_OPS:
                node_to_ir[curr] = node_to_ir[curr.args[0]]
                walk_seen.append(curr)
                last_node = curr
                last_ir = node_to_ir[curr]
                curr = curr.next
                continue

            # ── Type conversion (no-op in fp32 EVT) ───────────────────────────
            if target in _TYPE_CONV_OPS:
                node_to_ir[curr] = node_to_ir[curr.args[0]]
                walk_seen.append(curr)
                last_node = curr
                last_ir = node_to_ir[curr]
                curr = curr.next
                continue

            # ── Pure view ops (only if shape unchanged) ───────────────────────
            if target in (torch.ops.aten.view.default, torch.ops.aten.reshape.default, torch.ops.aten._unsafe_view.default):
                in_shape = _val_shape(curr.args[0])
                out_shape = _val_shape(curr)
                if in_shape == out_shape:
                    node_to_ir[curr] = node_to_ir[curr.args[0]]
                    walk_seen.append(curr)
                    last_node = curr
                    last_ir = node_to_ir[curr]
                    curr = curr.next
                    continue
                break

            # ── Slice stride-2 (swiglu marker) ────────────────────────────────
            if target is torch.ops.aten.slice.Tensor:
                step = curr.args[4] if len(curr.args) > 4 else curr.kwargs.get("step", 1)
                if step == 2:
                    saw_slice = True
                break

            # ── Unary ops ─────────────────────────────────────────────────────
            if target in _UNARY_OPS:
                op_name = _UNARY_OPS[target]
                child_ir = node_to_ir[curr.args[0]]
                ir = Compute(op_name, (child_ir,))
                node_to_ir[curr] = ir
                fused_nodes.append(curr)
                walk_seen.append(curr)
                last_node = curr
                last_ir = ir
                curr = curr.next
                continue

            # ── GELU (default = erf, alternative = tanh) ──────────────────────
            if target is torch.ops.aten.gelu.default:
                approx = curr.kwargs.get("approximate", "none")
                op_name = "gelu_tanh" if approx == "tanh" else "gelu_erf"
                child_ir = node_to_ir[curr.args[0]]
                ir = Compute(op_name, (child_ir,))
                node_to_ir[curr] = ir
                fused_nodes.append(curr)
                walk_seen.append(curr)
                last_node = curr
                last_ir = ir
                curr = curr.next
                continue

            # ── Scalar variants of add/sub/mul/div ────────────────────────────
            if target in _SCALAR_BINARY_TO_SCALAR_UNARY:
                op_name = _SCALAR_BINARY_TO_SCALAR_UNARY[target]
                child_ir = node_to_ir[curr.args[0]]
                if not isinstance(curr.args[1], (int, float)):
                    break
                scalar = float(curr.args[1])
                ir = Compute(op_name, (child_ir,), scalar=scalar)
                node_to_ir[curr] = ir
                fused_nodes.append(curr)
                walk_seen.append(curr)
                last_node = curr
                last_ir = ir
                curr = curr.next
                continue

            # ── Clamp family ──────────────────────────────────────────────────
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
                    ir_now = Compute("clamp_min_c", (ir_now,), scalar=float(lo))
                if hi is not None:
                    ir_now = Compute("clamp_max_c", (ir_now,), scalar=float(hi))
                node_to_ir[curr] = ir_now
                fused_nodes.append(curr)
                walk_seen.append(curr)
                last_node = curr
                last_ir = ir_now
                curr = curr.next
                continue

            # ── pow.Tensor_Scalar — only the small-int special-cases ──────────
            if target is torch.ops.aten.pow.Tensor_Scalar:
                exp = curr.args[1] if len(curr.args) > 1 else None
                child_ir = node_to_ir[curr.args[0]]
                if exp == 2 or exp == 2.0:
                    ir = Compute("square", (child_ir,))
                elif isinstance(exp, (int, float)):
                    ir = Compute("pow_scalar", (child_ir,), scalar=float(exp))
                else:
                    break
                node_to_ir[curr] = ir
                fused_nodes.append(curr)
                walk_seen.append(curr)
                last_node = curr
                last_ir = ir
                curr = curr.next
                continue

            # ── Binary tensor ops ─────────────────────────────────────────────
            if target in _BINARY_OPS:
                op_name = _BINARY_OPS[target]
                lhs_raw = curr.args[0]
                rhs_raw = curr.args[1]
                # Fold int/float scalars on the RHS to scalar variants.
                if isinstance(rhs_raw, (int, float)) and isinstance(lhs_raw, fx.Node) and lhs_raw in node_to_ir:
                    scalar_op = {"add": "add_scalar", "sub": "sub_scalar", "mul": "mul_scalar", "div": "div_scalar"}.get(
                        op_name
                    )
                    if scalar_op is None:
                        break
                    ir = Compute(scalar_op, (node_to_ir[lhs_raw],), scalar=float(rhs_raw))
                    node_to_ir[curr] = ir
                    fused_nodes.append(curr)
                    walk_seen.append(curr)
                    last_node = curr
                    last_ir = ir
                    curr = curr.next
                    continue
                # Fold scalar-on-LHS for commutative ops; for sub/div we need rsub/rdiv.
                if isinstance(lhs_raw, (int, float)) and isinstance(rhs_raw, fx.Node) and rhs_raw in node_to_ir:
                    if op_name in ("add", "mul"):
                        scalar_op = "add_scalar" if op_name == "add" else "mul_scalar"
                        ir = Compute(scalar_op, (node_to_ir[rhs_raw],), scalar=float(lhs_raw))
                    elif op_name == "sub":
                        ir = Compute("rsub_scalar", (node_to_ir[rhs_raw],), scalar=float(lhs_raw))
                    else:
                        break
                    node_to_ir[curr] = ir
                    fused_nodes.append(curr)
                    walk_seen.append(curr)
                    last_node = curr
                    last_ir = ir
                    curr = curr.next
                    continue
                # Both tensor — either internal (already in IR) or external.
                lhs_ir = self._ir_for_arg(lhs_raw, node_to_ir, extras_nodes, A, B)
                rhs_ir = self._ir_for_arg(rhs_raw, node_to_ir, extras_nodes, A, B)
                if lhs_ir is None or rhs_ir is None:
                    break
                ir = Compute(op_name, (lhs_ir, rhs_ir))
                node_to_ir[curr] = ir
                fused_nodes.append(curr)
                walk_seen.append(curr)
                last_node = curr
                last_ir = ir
                curr = curr.next
                continue

            # Unsupported op — stop greedy walk.
            break

        # If we saw a stride-2 slice and the chain is plausibly swiglu7, try
        # the dedicated matcher. It rebuilds independently from mm_node.
        if saw_slice:
            return self._try_fuse_swiglu7(graph, mm_node)

        # Verify we made progress.
        if last_ir is node_to_ir[mm_node]:
            return False  # only Accum — replacing cuBLAS with EVT is no win

        # Refuse if any escape: an intermediate fused node is consumed outside
        # the fused region. (EVT has no "extra outputs"; the user explicitly
        # opted out of cross-domain fan-out.)
        #
        # The exclusion ``n is not last_node`` is intentional — the last node
        # in the fused chain becomes the EVT op's output and is allowed to
        # have downstream consumers (that's the whole point of fusion).
        # Earlier writes ([:-1] explicitly skips the last position) must not
        # have any external user, otherwise the fused chain would silently
        # drop their value. This previously read ``walk_seen[:-0]`` which is
        # ``walk_seen[:0]`` (an empty slice!) so escape detection was a no-op
        # and trivially-fusable chains like ``mm → add(residual) → square``
        # were emitted even when ``add(residual)`` was reused downstream.
        fused_set = set(fused_nodes) | set(walk_seen)
        for n in walk_seen[:-1]:
            for u in n.users:
                if u not in fused_set:
                    return False

        # Final eligibility check: A contiguous, B in a supported layout.
        a_stride = _val_stride(A)
        if a_stride is None:
            return False
        a_shape_now = _val_shape(A)
        if a_stride != (a_shape_now[1], 1):
            return False
        b_layout, b_underlying, n_dim = _b_layout_kind(B)
        if b_layout is None:
            return False

        # Path-specific B-side alignment gate. evt_row: B is (K, N) row-major,
        # ldB = N — must divide AlignmentB. We greedy-pick (128 → 64 bits) at
        # runtime, so the FX gate only refuses N not even 64-bit-aligned.
        # evt_col: B is (N, K) row-major (read as (K, N) col-major), ldB = K,
        # already covered by the entry K-gate. D's N stays unconstrained —
        # runtime pads.
        if b_layout == "row":
            if _largest_pow2_align_bits(n_dim, b_dtype) is None:
                return False

        # Determine output dtype from the last fused node's FakeTensor metadata.
        out_dt = _val_dtype(last_node) or torch.bfloat16
        if out_dt not in _DTYPE_TO_STR:
            return False

        # Output-side (D) alignment gate. The runtime allocates D as
        # (M, n_pad) where n_pad = _aligned_n_stride(n_out, out_dt) and the
        # CUTLASS AlignmentC is greedy-picked from that ldd at compile time
        # (128 → 64 bits). The FX gate only refuses if even the smallest
        # candidate (64 bits) can't divide n_pad — that catches future
        # configurations where the host padding is reduced or disabled.
        # SymInt n_dim defers to the runtime gate (returns the small candidate).
        if _is_static_int(n_dim):
            n_pad_static = evt_runtime._aligned_n_stride(int(n_dim), out_dt)
            if _largest_pow2_align_bits(n_pad_static, out_dt) is None:
                return False

        ir_root = Store(child=last_ir, out_dtype=_DTYPE_TO_STR[out_dt])
        if is_trivial(ir_root):
            return False
        # If extras are disabled, refuse any IR that needs them.
        if not self.allow_extras and num_extras(ir_root) > 0:
            return False

        # SM90 (H100) uses a CUTLASS 3.x EVT codegen that has slightly tighter
        # constraints than the SM80 path — most notably it supports at most
        # one AuxLoad (the C-operand TMA path is the only aux load CUTLASS
        # 3.x's standard CollectiveBuilder exposes). If this IR isn't
        # renderable on sm_90 we'd rather have torch.compile lower the chain
        # than fall back to SM80-on-Hopper, which runs ~2× slower than cuBLAS
        # in backward-compat mode.
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9:
            from .sm90.evt_codegen import can_render as _sm90_can_render

            if not _sm90_can_render(ir_root):
                return False

        ir_json = to_canonical_json(ir_root)
        n_out = n_dim
        out_dt_id = evt_runtime.out_dtype_id(out_dt)
        kind = "evt_row" if b_layout == "row" else "evt_col"

        with graph.inserting_after(last_node):
            new_node = graph.call_function(
                torch.ops.magi_epilogue.matmul_custom_evt.default,
                args=(A, b_underlying, extras_nodes, ir_json, kind, n_out, out_dt_id),
            )
        # Propagate FakeTensor meta with 128-bit-aligned row stride matching
        # what the CUDA impl actually returns. Narrow the exception to the
        # int(SymInt) cast for dynamic-N graphs — meta propagation is best-
        # effort there; the runtime still returns a correct strided tensor.
        val_last = last_node.meta.get("val")
        if val_last is not None:
            try:
                n_pad = evt_runtime._aligned_n_stride(int(val_last.shape[-1]), val_last.dtype)
            except (TypeError, ValueError):
                n_pad = None
            if n_pad is not None:
                new_node.meta["val"] = val_last.new_empty_strided(val_last.shape, (n_pad, 1))

        last_node.replace_all_uses_with(new_node)
        for n in reversed(walk_seen):
            if len(n.users) == 0 and n is not new_node:
                graph.erase_node(n)
        return True

    def _ir_for_arg(self, arg, node_to_ir, extras_nodes, A_node, B_node):
        """Return an IR subtree for a binary-op operand. Internal → IR; external
        → leaf (RowBroadcast / ColBroadcast / AuxLoad). None ⇒ abort."""
        if not isinstance(arg, fx.Node):
            return None
        if arg in node_to_ir:
            return node_to_ir[arg]
        if not self.allow_extras:
            return None
        # Classify external tensor by shape relative to (M, N).
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
        # 1-D case: must distinguish (N,) vs (M,). Compare ints directly.
        # When M is SymInt (dynamic batch dim) the M==N collision can't happen
        # at compile time, so trust the (N,) match for RowBroadcast. Only the
        # "both static + equal" case is ambiguous and we abort.
        if len(shape) == 1:
            n0 = shape[0]
            m_is_static = _is_static_int(M)
            n_is_static = _is_static_int(N)
            if n_is_static and n0 == N:
                # Could still collide with a (M,) col-broadcast iff M is also
                # static and equal — abort in that ambiguous case.
                if m_is_static and n0 == M:
                    return None
                idx = self._add_extra(extras_nodes, arg)
                return RowBroadcast(input_idx=idx, dtype=dt_str)
            if m_is_static and n0 == M:
                idx = self._add_extra(extras_nodes, arg)
                return ColBroadcast(input_idx=idx, dtype=dt_str)
            return None
        if len(shape) == 2:
            # (1, N) row-broadcast view.
            if shape[0] == 1 and shape[1] == N:
                idx = self._add_extra(extras_nodes, arg)
                return RowBroadcast(input_idx=idx, dtype=dt_str)
            # (M, 1) col-broadcast view.
            if shape[1] == 1 and shape[0] == M:
                idx = self._add_extra(extras_nodes, arg)
                return ColBroadcast(input_idx=idx, dtype=dt_str)
            # Full (M, N) aux load — require row-major contiguous.
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

    # ── swiglu7 special-case ──────────────────────────────────────────────────

    def _try_fuse_swiglu7(self, graph: fx.Graph, mm_node: fx.Node) -> bool:
        """Match the canonical swiglu7 epilogue and dispatch to DualGemm.

        We do not attempt to encode swiglu7 in the EVT IR (the dual GEMM is a
        whole different kernel structure). Instead we walk forward from mm_node
        looking for the exact pattern produced by ``athena.activation.swiglu7``
        after Inductor decomposition.

        On a successful match we emit the magi_epilogue.matmul_custom_evt op
        with kind="swiglu7_dual". The ``B`` argument must be the underlying
        weight tensor of shape (N, K) — typically the predecessor of an
        ``aten.t`` node feeding the mm.
        """
        # Recover the underlying weight: B should be a 2-D transpose
        # (aten.t / transpose(0,1) / permute([1,0])) of a contiguous (N, K)
        # weight. Otherwise bail (no two-stage fallback).
        B_node = mm_node.args[1]
        if not isinstance(B_node, fx.Node) or not _is_transpose_node(B_node):
            return False
        weight_node = B_node.args[0]
        if not isinstance(weight_node, fx.Node):
            return False
        w_shape = _val_shape(weight_node)
        w_stride = _val_stride(weight_node)
        if w_shape is None or len(w_shape) != 2 or w_stride is None:
            return False
        N, K = w_shape
        # N must be even (gate/linear interleaved split). The output
        # n_out = N // 2 is padded by the runtime to AlignmentC, so no
        # further N divisibility is needed. K-side alignment is the same
        # greedy 128 → 64 bit gate as the EVT path: the vendored .cu now
        # accepts AlignmentA / AlignmentB via -D macros (see
        # ``_compile_swiglu7_dual``), so K only needs to divide 64 bits.
        if not (_is_static_int(N) and N % 2 == 0):
            return False
        if w_stride != (K, 1):
            return False  # not contiguous (N, K) — abort
        a_dtype = _val_dtype(mm_node.args[0])
        if a_dtype != torch.bfloat16 or _val_dtype(weight_node) != torch.bfloat16:
            return False
        if _largest_pow2_align_bits(K, a_dtype) is None:
            return False
        # SM90 (H100) swiglu7 path uses Sm90DualGemm with TMA — TMA requires
        # the innermost stride **in bytes** to be a multiple of 16. For A's
        # K-contiguous load that means K * sizeof(elem) % 16 == 0. CUTLASS
        # encodes this in sm90_dual_gemm.h's can_implement as
        #   constexpr int min_k_align = 128 / cutlass::sizeof_bits<ElementA>;
        #   if (problem_size.k() % min_k_align != 0) return kErrorInvalidProblem;
        # which is the same condition expressed in elements. Express it in
        # bytes here so future fp8 / fp32 swiglu7 paths inherit the gate
        # without a one-line dtype fix. On sm_120 / Ada the SM80 multistage
        # path supports 64-bit alignment, so this gate only fires on Hopper.
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9:
            elem_bytes = a_dtype.itemsize
            if _is_static_int(K) and (int(K) * elem_bytes) % 16 != 0:
                return False

        # We walk the chain in source order and collect every node belonging to
        # the swiglu7 epilogue — anything else aborts. We don't need to verify
        # the exact structure (the kernel does that intrinsically); we just need
        # to find the final tensor that becomes the chain's only output, plus
        # the set of nodes to erase.
        chain_nodes: List[fx.Node] = []
        chain_set: set = {mm_node}
        last_chain_node: Optional[fx.Node] = None
        curr = mm_node.next
        while curr is not None and curr.op != "output":
            uses_chain = any(isinstance(a, fx.Node) and a in chain_set for a in curr.args)
            if not uses_chain:
                curr = curr.next
                continue
            if curr.target not in (
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
            ):
                # Non-whitelist op consuming the chain → it's the boundary.
                # Finalise last_chain_node as the previous node and stop.
                # The output-shape check below verifies we actually saw the
                # swiglu7 pattern (chain output's last dim must equal N//2).
                break
            chain_nodes.append(curr)
            chain_set.add(curr)
            last_chain_node = curr
            curr = curr.next

        if last_chain_node is None:
            return False
        # Output dtype from the final node.
        out_dt = _val_dtype(last_chain_node) or torch.bfloat16
        out_shape = _val_shape(last_chain_node)
        if out_shape is None or len(out_shape) != 2:
            return False
        if not _is_static_int(out_shape[1]) or out_shape[1] != N // 2:
            # The swiglu7 output's last dim must be N/2.
            return False

        # Output-side (D) alignment gate. Same logic as the EVT path —
        # require that the host-padded ldd satisfies at least the 64-bit
        # AlignmentC fallback (it always does under the current cache-line
        # padding, but the gate future-proofs against a smaller-pad mode).
        n_pad_static = evt_runtime._aligned_n_stride(int(N) // 2, out_dt)
        if _largest_pow2_align_bits(n_pad_static, out_dt) is None:
            return False

        # No escape: every chain node's external uses must funnel through the
        # final node (otherwise the DualGemm kernel produces only D and we'd
        # lose the intermediate consumer).
        for n in chain_nodes[:-1]:
            for u in n.users:
                if u not in chain_set:
                    return False

        # Emit the call. We do NOT pass IR JSON — the swiglu7 path ignores it.
        out_dt_id = evt_runtime.out_dtype_id(out_dt)
        n_out = N // 2
        with graph.inserting_after(last_chain_node):
            new_node = graph.call_function(
                torch.ops.magi_epilogue.matmul_custom_evt.default,
                args=(mm_node.args[0], weight_node, [], "", "swiglu7_dual", n_out, out_dt_id),
            )
        # Propagate FakeTensor meta with 128-bit-aligned row stride matching
        # what the CUDA impl actually returns.
        val_last = last_chain_node.meta.get("val")
        if val_last is not None:
            try:
                n_pad = evt_runtime._aligned_n_stride(int(val_last.shape[-1]), val_last.dtype)
            except (TypeError, ValueError):
                n_pad = None
            if n_pad is not None:
                new_node.meta["val"] = val_last.new_empty_strided(val_last.shape, (n_pad, 1))

        last_chain_node.replace_all_uses_with(new_node)
        for n in reversed(chain_nodes):
            if len(n.users) == 0 and n is not new_node:
                graph.erase_node(n)
        # Erase mm and the t() node if no longer used.
        if len(mm_node.users) == 0:
            graph.erase_node(mm_node)
        if isinstance(B_node, fx.Node) and len(B_node.users) == 0:
            graph.erase_node(B_node)
        return True
