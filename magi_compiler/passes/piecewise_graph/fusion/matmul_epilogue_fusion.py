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

import json
import operator

import torch
import torch.fx as fx
from torch.fx.node import Node

from magi_compiler.passes.pass_base import MagiInductorPass

from .triton_kernels import matmul_custom_epilogue

_LIB = torch.library.Library("magi_epilogue", "DEF")
_LIB.define("matmul_custom(Tensor A, Tensor B, Tensor[] extras, str epilogue_code, bool reduce_n_by_2) -> Tensor")


@torch.library.impl(_LIB, "matmul_custom", "CUDA")
def _matmul_custom_cuda(A, B, extras, epilogue_code, reduce_n_by_2):
    return matmul_custom_epilogue(A, B, extras, epilogue_code, reduce_n_by_2)


@torch.library.register_fake("magi_epilogue::matmul_custom")
def _matmul_custom_abstract(A, B, extras, epilogue_code, reduce_n_by_2):
    N_out = B.shape[1] // 2 if reduce_n_by_2 else B.shape[1]
    # Mirror the 128-byte-aligned row stride used by the real kernel so that
    # Inductor's assert_size_stride matches what we actually return.
    # Keep the logical shape as (M, N_out) — changing it would interfere with
    # Inductor's own K-dimension padding for the downstream mm.
    align_elems = 128 // A.element_size()
    N_stride = (N_out + align_elems - 1) // align_elems * align_elems
    return A.new_empty_strided((A.shape[0], N_out), (N_stride, 1))


# ── Triton expression templates ────────────────────────────────────────────────
# Unary elementwise ops: {x} = operand expression string
_UNARY_EXPRS = {
    # Arithmetic
    torch.ops.aten.neg.default: "-({x})",
    torch.ops.aten.abs.default: "tl.abs({x})",
    torch.ops.aten.sign.default: "tl.math.sign({x})",
    torch.ops.aten.reciprocal.default: "1.0 / ({x})",
    torch.ops.aten.square.default: "({x}) * ({x})",
    # Exponential / logarithm
    torch.ops.aten.exp.default: "tl.exp({x})",
    torch.ops.aten.exp2.default: "tl.exp2({x})",
    torch.ops.aten.expm1.default: "tl.exp({x}) - 1.0",
    torch.ops.aten.log.default: "tl.log({x})",
    torch.ops.aten.log2.default: "tl.log2({x})",
    torch.ops.aten.log10.default: "tl.log({x}) * 0.4342944819032518",
    torch.ops.aten.log1p.default: "tl.log(1.0 + ({x}))",
    # Square-root family
    torch.ops.aten.sqrt.default: "tl.sqrt({x})",
    torch.ops.aten.rsqrt.default: "1.0 / tl.sqrt({x})",
    # Trigonometric
    torch.ops.aten.sin.default: "tl.sin({x})",
    torch.ops.aten.cos.default: "tl.cos({x})",
    torch.ops.aten.tan.default: "tl.math.tan({x})",
    torch.ops.aten.asin.default: "tl.math.asin({x})",
    torch.ops.aten.acos.default: "tl.math.acos({x})",
    torch.ops.aten.atan.default: "tl.math.atan({x})",
    # Hyperbolic
    torch.ops.aten.tanh.default: "tl.tanh({x})",
    torch.ops.aten.sinh.default: "tl.math.sinh({x})",
    torch.ops.aten.cosh.default: "tl.math.cosh({x})",
    # Activations
    torch.ops.aten.sigmoid.default: "tl.sigmoid({x})",
    torch.ops.aten.relu.default: "tl.maximum({x}, 0.0)",
    # Error function
    torch.ops.aten.erf.default: "tl.math.erf({x})",
    torch.ops.aten.erfinv.default: "tl.math.erfinv({x})",
    torch.ops.aten.erfc.default: "tl.math.erfc({x})",
    # Rounding
    torch.ops.aten.floor.default: "tl.math.floor({x})",
    torch.ops.aten.ceil.default: "tl.math.ceil({x})",
    torch.ops.aten.trunc.default: "tl.math.trunc({x})",
    torch.ops.aten.round.default: "tl.math.round({x})",
    torch.ops.aten.frac.default: "({x}) - tl.math.trunc({x})",
    # Bitwise / logical
    torch.ops.aten.logical_not.default: "~({x})",
    torch.ops.aten.bitwise_not.default: "~({x})",
    # Predicates
    torch.ops.aten.isnan.default: "tl.math.isnan({x})",
    torch.ops.aten.isinf.default: "tl.math.isinf({x})",
    torch.ops.aten.isfinite.default: "~tl.math.isinf({x}) & ~tl.math.isnan({x})",
}

# Binary elementwise ops: {x} = left, {y} = right
_BINARY_EXPRS = {
    # Addition / subtraction (alpha handled separately)
    torch.ops.aten.add.Tensor: "({x}) + ({y})",
    torch.ops.aten.add.Scalar: "({x}) + ({y})",
    operator.add: "({x}) + ({y})",
    torch.ops.aten.sub.Tensor: "({x}) - ({y})",
    torch.ops.aten.sub.Scalar: "({x}) - ({y})",
    operator.sub: "({x}) - ({y})",
    # Multiplication / division
    torch.ops.aten.mul.Tensor: "({x}) * ({y})",
    torch.ops.aten.mul.Scalar: "({x}) * ({y})",
    operator.mul: "({x}) * ({y})",
    torch.ops.aten.div.Tensor: "({x}) / ({y})",
    torch.ops.aten.div.Scalar: "({x}) / ({y})",
    operator.truediv: "({x}) / ({y})",
    torch.ops.aten.remainder.Tensor: "({x}) % ({y})",
    torch.ops.aten.remainder.Scalar: "({x}) % ({y})",
    operator.mod: "({x}) % ({y})",
    # Min / max
    torch.ops.aten.maximum.default: "tl.maximum({x}, {y})",
    torch.ops.aten.minimum.default: "tl.minimum({x}, {y})",
    # Trigonometric binary
    torch.ops.aten.atan2.default: "tl.math.atan2({x}, {y})",
    # Bitwise / logical binary
    torch.ops.aten.bitwise_and.Tensor: "({x}) & ({y})",
    torch.ops.aten.bitwise_and.Scalar: "({x}) & ({y})",
    operator.and_: "({x}) & ({y})",
    torch.ops.aten.bitwise_or.Tensor: "({x}) | ({y})",
    torch.ops.aten.bitwise_or.Scalar: "({x}) | ({y})",
    operator.or_: "({x}) | ({y})",
    torch.ops.aten.bitwise_xor.Tensor: "({x}) ^ ({y})",
    torch.ops.aten.bitwise_xor.Scalar: "({x}) ^ ({y})",
    operator.xor: "({x}) ^ ({y})",
    torch.ops.aten.logical_and.default: "({x}) & ({y})",
    torch.ops.aten.logical_or.default: "({x}) | ({y})",
    torch.ops.aten.logical_xor.default: "({x}) ^ ({y})",
}

# Ops that pass through without any value transformation
_PASSTHROUGH_OPS = frozenset(
    {
        torch.ops.prims.convert_element_type.default,
        torch.ops.aten._to_copy.default,
        torch.ops.aten.clone.default,
        torch.ops.aten.contiguous.default,
        torch.ops.aten.alias.default,
    }
)


def _get_static_dims(mm_node: fx.Node) -> dict:
    """Return {name: value} for mm dimensions that are compile-time-constant.

    FX shapes carry plain Python ``int`` for static dims and ``torch.SymInt``
    for symbolic (dynamic) ones.  ``type(d) is int`` excludes SymInt even in
    PyTorch versions where SymInt happens to subclass int.
    """
    static: dict = {}
    A, B = mm_node.args
    try:
        val_a = A.meta.get("val") if isinstance(A, fx.Node) else None
        if val_a is not None and val_a.dim() == 2:
            for name, idx in (("M", 0), ("K", 1)):
                d = val_a.shape[idx]
                if type(d) is int:
                    static[name] = d
        val_b = B.meta.get("val") if isinstance(B, fx.Node) else None
        if val_b is not None and val_b.dim() == 2:
            d = val_b.shape[1]
            if type(d) is int:
                static["N"] = d
    except Exception:
        pass
    return static


class MatmulCustomEpilogueFusionPass(MagiInductorPass):
    def __call__(self, graph: fx.Graph) -> bool:
        fused = 0
        for node in list(graph.nodes):
            if node.op == "call_function" and node.target in (torch.ops.aten.mm.default, torch.ops.aten.mm):
                fused += self._try_fuse_custom_chain(graph, node)

        if fused:
            graph.eliminate_dead_code()
        return fused > 0

    def _try_fuse_custom_chain(self, graph: fx.Graph, mm_node: fx.Node) -> int:
        A, B = mm_node.args

        fused_nodes = {mm_node: "acc"}
        nodes_to_remove = []
        epilogue_lines = []
        extras = []
        is_swiglu = False

        def get_val(arg):
            if isinstance(arg, Node):
                if arg in fused_nodes:
                    return fused_nodes[arg]
                # External tensor — inject a load
                idx = len(extras)
                extras.append(arg)
                name = f"ext_{idx}"
                val = arg.meta.get("val")
                if val is not None and val.dim() == 1:
                    epilogue_lines.append(f"{name}_ptrs = Extra_{idx}_ptr + offs_dn[None, :]")
                    epilogue_lines.append(f"{name} = tl.load({name}_ptrs, mask=offs_dn[None, :] < N, other=0.0)")
                else:
                    epilogue_lines.append(
                        f"{name}_ptrs = Extra_{idx}_ptr + stride_dm * offs_dm[:, None] + stride_dn * offs_dn[None, :]"
                    )
                    epilogue_lines.append(f"{name} = tl.load({name}_ptrs, mask=mask, other=0.0)")
                fused_nodes[arg] = name
                return name
            return str(arg)

        curr = mm_node.next
        last_fused_node = mm_node

        while curr.op != "output":
            uses_fused = any(isinstance(a, Node) and a in fused_nodes for a in curr.args)
            if not uses_fused:
                curr = curr.next
                continue

            var_name = f"v_{curr.name}"
            target = curr.target
            code = None

            # ── 1. Pass-through (type conversion / clone / alias) ─────────────
            if target in _PASSTHROUGH_OPS:
                fused_nodes[curr] = fused_nodes[curr.args[0]]
                nodes_to_remove.append(curr)
                last_fused_node = curr
                curr = curr.next
                continue

            # ── 2. Unary elementwise ops (from dispatch table) ────────────────
            elif target in _UNARY_EXPRS:
                x = get_val(curr.args[0])
                code = f"{var_name} = " + _UNARY_EXPRS[target].format(x=x)

            # ── 3. Compound activation functions ──────────────────────────────
            elif target in (torch.ops.aten.silu.default, torch.ops.aten.silu):
                x = get_val(curr.args[0])
                code = f"{var_name} = ({x}) * tl.sigmoid({x})"

            elif target in (torch.ops.aten.gelu.default, torch.ops.aten.gelu):
                x = get_val(curr.args[0])
                approx = curr.kwargs.get("approximate", "none")
                if approx == "tanh":
                    code = (
                        f"{var_name} = ({x}) * 0.5 * "
                        f"(1.0 + tl.tanh(0.7978845608 * (({x}) + 0.044715 * ({x}) * ({x}) * ({x}))))"
                    )
                else:
                    code = f"{var_name} = 0.5 * ({x}) * (1.0 + tl.math.erf(({x}) * 0.7071067811865476))"

            elif target == torch.ops.aten.leaky_relu.default:
                x = get_val(curr.args[0])
                slope = curr.args[1] if len(curr.args) > 1 else curr.kwargs.get("negative_slope", 0.01)
                code = f"{var_name} = tl.where({x} >= 0.0, {x}, {slope} * ({x}))"

            elif target == torch.ops.aten.hardtanh.default:
                x = get_val(curr.args[0])
                lo = curr.args[1] if len(curr.args) > 1 else curr.kwargs.get("min_val", -1.0)
                hi = curr.args[2] if len(curr.args) > 2 else curr.kwargs.get("max_val", 1.0)
                code = f"{var_name} = tl.minimum(tl.maximum({x}, {lo}), {hi})"

            elif target == torch.ops.aten.hardsigmoid.default:
                x = get_val(curr.args[0])
                code = f"{var_name} = tl.minimum(tl.maximum(({x}) / 6.0 + 0.5, 0.0), 1.0)"

            elif target == torch.ops.aten.hardswish.default:
                x = get_val(curr.args[0])
                code = f"{var_name} = ({x}) * tl.minimum(tl.maximum(({x}) / 6.0 + 0.5, 0.0), 1.0)"

            elif target == torch.ops.aten.mish.default:
                x = get_val(curr.args[0])
                code = f"{var_name} = ({x}) * tl.tanh(tl.log(1.0 + tl.exp({x})))"

            # ── 4. Clamp family ───────────────────────────────────────────────
            elif target in (
                torch.ops.aten.clamp.default,
                torch.ops.aten.clamp.Tensor,
                torch.ops.aten.clamp_max.default,
                torch.ops.aten.clamp_min.default,
            ):
                x = get_val(curr.args[0])
                if target is torch.ops.aten.clamp_max.default:
                    lo, hi = None, curr.args[1] if len(curr.args) > 1 else curr.kwargs.get("max", None)
                elif target is torch.ops.aten.clamp_min.default:
                    lo, hi = curr.args[1] if len(curr.args) > 1 else curr.kwargs.get("min", None), None
                else:
                    lo = curr.args[1] if len(curr.args) > 1 else curr.kwargs.get("min", None)
                    hi = curr.args[2] if len(curr.args) > 2 else curr.kwargs.get("max", None)
                expr = x
                if lo is not None:
                    expr = f"tl.maximum({expr}, {get_val(lo)})"
                if hi is not None:
                    expr = f"tl.minimum({expr}, {get_val(hi)})"
                code = f"{var_name} = {expr}"

            # ── 5. Ternary select ─────────────────────────────────────────────
            elif target in (torch.ops.aten.where.self, torch.ops.aten.where.ScalarSelf, torch.ops.aten.where.ScalarOther):
                cond = get_val(curr.args[0])
                t = get_val(curr.args[1])
                f_ = get_val(curr.args[2])
                code = f"{var_name} = tl.where({cond}, {t}, {f_})"

            # ── 6. pow (special-cased exponents) ─────────────────────────────
            elif target in (torch.ops.aten.pow.Tensor_Scalar, torch.ops.aten.pow.Tensor_Tensor):
                x = get_val(curr.args[0])
                y = get_val(curr.args[1])
                if str(y) in ("2", "2.0"):
                    code = f"{var_name} = ({x}) * ({x})"
                elif str(y) in ("0.5",):
                    code = f"{var_name} = tl.sqrt({x})"
                elif str(y) in ("-0.5",):
                    code = f"{var_name} = 1.0 / tl.sqrt({x})"
                elif str(y) in ("-1", "-1.0"):
                    code = f"{var_name} = 1.0 / ({x})"
                else:
                    code = f"{var_name} = tl.math.pow({x}, {y})"

            # ── 7. div with rounding_mode ─────────────────────────────────────
            elif target is torch.ops.aten.div.Tensor_mode:
                x = get_val(curr.args[0])
                y = get_val(curr.args[1])
                rounding_mode = curr.kwargs.get("rounding_mode", None) or (curr.args[2] if len(curr.args) > 2 else None)
                if rounding_mode == "floor":
                    code = f"{var_name} = tl.math.floor(({x}) / ({y}))"
                elif rounding_mode == "trunc":
                    code = f"{var_name} = tl.math.trunc(({x}) / ({y}))"
                else:
                    code = f"{var_name} = ({x}) / ({y})"

            # ── 8. Binary elementwise ops (from dispatch table) ───────────────
            elif target in _BINARY_EXPRS:
                x = get_val(curr.args[0])
                y_raw = curr.args[1]
                y = get_val(y_raw)
                # Handle optional alpha scalar for add/sub (aten convention)
                alpha = (curr.args[2] if len(curr.args) > 2 else None) or curr.kwargs.get("alpha", None)
                if alpha is not None and alpha != 1:
                    y = f"{alpha} * ({y})"
                code = f"{var_name} = " + _BINARY_EXPRS[target].format(x=x, y=y)

            # ── 9. Slice: SwiGLU (stride-2 along last dim) ───────────────────
            elif target is torch.ops.aten.slice.Tensor:
                dim = curr.args[1] if len(curr.args) > 1 else curr.kwargs.get("dim", 0)
                start = curr.args[2] if len(curr.args) > 2 else curr.kwargs.get("start", None)
                step = curr.args[4] if len(curr.args) > 4 else curr.kwargs.get("step", 1)

                src = curr.args[0]
                if isinstance(src, fx.Node) and "val" in src.meta:
                    rank = src.meta["val"].dim()
                    is_last_dim = (dim % rank) == (rank - 1)
                else:
                    is_last_dim = dim == -1

                if is_last_dim and step == 2:
                    is_swiglu = True
                    x = get_val(curr.args[0])
                    if not x.endswith("_reshaped"):
                        epilogue_lines.append(f"{x}_reshaped = tl.reshape({x}, (BLOCK_M, BLOCK_N // 2, 2))")
                        epilogue_lines.append(f"{x}_split_0, {x}_split_1 = tl.split({x}_reshaped)")
                        fused_nodes[curr.args[0]] = f"{x}_reshaped"
                        base_x = x
                    else:
                        base_x = x[:-9]  # strip '_reshaped'

                    idx = 0 if (start == 0 or start is None) else 1
                    code = f"{var_name} = {base_x}_split_{idx}"
                else:
                    break  # non-strided / non-trailing slice — stop fusion

            # ── Unsupported op — stop greedy fusion ────────────────────────────
            else:
                break

            if code:
                epilogue_lines.append(code)
                fused_nodes[curr] = var_name
                nodes_to_remove.append(curr)
                last_fused_node = curr

            curr = curr.next

        # Validate: intermediate nodes must not escape the fused set
        if not nodes_to_remove:
            return 0
        for node in nodes_to_remove[:-1]:
            for user in node.users:
                if user not in nodes_to_remove:
                    return 0

        final_var = fused_nodes[last_fused_node]

        # Skip fusion if the epilogue is a no-op (only passthrough ops were
        # collected — e.g. a bare _to_copy after mm).  Replacing cuBLAS with
        # a Triton GEMM that does the exact same work is strictly slower.
        if final_var == "acc":
            return 0

        epilogue_lines.append(f"acc = {final_var}")

        epilogue_code = "\n".join(epilogue_lines)

        # Prepend a comment that encodes which mm dimensions are statically
        # known at trace time.  triton_kernels.py parses this header and
        # annotates the corresponding kernel parameters as tl.constexpr so
        # Triton can specialise (and optimise) the compiled kernel per value.
        static_dims = _get_static_dims(mm_node)
        if static_dims:
            epilogue_code = f"# @static:{json.dumps(static_dims, separators=(',', ':'))}\n" + epilogue_code

        with graph.inserting_after(last_fused_node):
            fused_node = graph.call_function(
                torch.ops.magi_epilogue.matmul_custom.default, args=(A, B, extras, epilogue_code, is_swiglu)
            )
            if "val" in last_fused_node.meta:
                val = last_fused_node.meta["val"]
                # Propagate the 128-byte-aligned row stride so downstream
                # assert_size_stride checks match what we actually return.
                try:
                    N_out = int(val.shape[-1])
                    elem_size = val.element_size()
                    align_elems = 128 // elem_size
                    N_stride = (N_out + align_elems - 1) // align_elems * align_elems
                    new_stride = val.stride()[:-2] + (N_stride, 1)
                    fused_node.meta["val"] = val.new_empty_strided(val.shape, new_stride)
                except Exception:
                    fused_node.meta["val"] = val

        last_fused_node.replace_all_uses_with(fused_node)

        for n in reversed(nodes_to_remove):
            graph.erase_node(n)
        graph.erase_node(mm_node)

        return 1
