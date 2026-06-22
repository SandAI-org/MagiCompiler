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

"""Conv2d/Conv3d channels-last layout pass for the post-grad ATen graph.

Forces channels-last (NHWC for 4D, NDHWC for 5D) at every ``aten.convolution``
boundary by graph rewriting only -- no patching of PyTorch internals.

Mechanism: insert ``aten.clone(memory_format=channels_last(_3d))`` before each
conv input/weight and set the clone's ``meta["val"]`` to a channels-last
FakeTensor. The clone *lowering* ignores ``memory_format`` (a TODO in
``lowering.py``), so the channels-last signal lives purely in the FX meta
strides. With ``layout_optimization=False`` (set by this pass), the
pre-registered ``constrain_conv_to_fx_strides`` reads those conv-input FX meta
strides -- now channels-last -- and applies ``require_stride_order`` at the conv
boundary. The clone lowers to a FlexibleLayout Pointwise, so that freeze is
zero-cost (the buffer is allocated channels-last directly, no extra copy) and
``conv_layout()`` then infers a channels-last output.

In auto mode the pass only fires on static, conv-dense graphs; dynamic-shape
graphs are skipped. ``force_on=True`` applies it unconditionally.
"""

from collections import Counter

import torch
from torch import fx
from torch.fx.experimental.symbolic_shapes import has_free_symbols

from ...magi_depyf.timeline import emit_pass_lifecycle
from ...utils import magi_logger
from ..pass_base import MagiInductorPass

aten = torch.ops.aten


def _meta_val(node: fx.Node) -> torch.Tensor | None:
    val = node.meta.get("val") if hasattr(node, "meta") else None
    return val if isinstance(val, torch.Tensor) else None


# Single-input, layout-transparent ops the conv stride constraint can hoist through.
_HOISTABLE_OPS = (aten.constant_pad_nd.default,)


class ConvChannelsLastPass(MagiInductorPass):
    """
    Make conv2d/conv3d inputs channels-last on the post-grad ATen graph.

    For every ``aten.convolution`` node, clone x and weight with their FX
    ``meta["val"]`` set channels-last (the clone lowering itself ignores
    ``memory_format``). With layout_optimization=False,
    ``constrain_conv_to_fx_strides`` reads those meta strides and enforces
    channels-last at the conv boundary, and ``conv_layout`` infers a
    channels-last output.

    If the conv input comes from a single-consumer layout-transparent op
    (``constant_pad_nd``), the clone is hoisted above it and its FX meta
    rewritten to channels-last, so the pad kernel stays coalesced instead of
    becoming an NC(D)HW->N(D)HWC transpose (which Inductor tiles badly under
    dynamic shapes).
    """

    def __init__(self, force_on: bool = False):
        # force_on=True (config enable_conv_channels_last is True) applies the
        # rewrite unconditionally; force_on=False (auto / None) lets __call__
        # decide from the graph (static + conv-dense graphs only).
        self.force_on = force_on

    @emit_pass_lifecycle
    def __call__(self, graph: fx.Graph) -> bool:
        if not self.force_on:
            # Decide dynamic-ness from the graph's own placeholders: a graph is
            # dynamic if any placeholder's fake/example value carries free symbols.
            placeholder_vals = (n.meta.get("val", n.meta.get("example_value")) for n in graph.nodes if n.op == "placeholder")
            is_dynamic = any(v is not None and has_free_symbols(v) for v in placeholder_vals)

            # Count number of nodes
            nnodes = len(list(graph.nodes))
            conv_nodes = [n for n in graph.nodes if n.target == torch.ops.aten.convolution.default]
            nconv = len(conv_nodes)
            Counter(n.args[1].meta["val"].dim() - 2 for n in conv_nodes)
            # dim_counts[1] / dim_counts[2] / dim_counts[3] means number of conv1d/2d/3d

            # TODO: If tiling optimization is upgraded to support conv layout opt
            # under dynamic shapes, we can remove `is_dynamic` check.
            if is_dynamic or nnodes < 300 * nconv:
                return False

        torch._inductor.config.layout_optimization = False

        # (input node, memory_format) -> clone node, so a weight shared by
        # several convs (or a tensor feeding several convs) is cloned once.
        clone_cache: dict[tuple[fx.Node, torch.memory_format], fx.Node] = {}
        num_hoisted = 0

        def channels_last_clone(inp: fx.Node, memory_format, insert_point) -> fx.Node | None:
            key = (inp, memory_format)
            cached = clone_cache.get(key)
            if cached is not None:
                return cached
            inp_val = _meta_val(inp)
            if inp_val is None:
                return None
            if inp_val.is_contiguous(memory_format=memory_format):
                return None  # already channels-last per meta
            with graph.inserting_before(insert_point):
                cl = graph.call_function(aten.clone.default, (inp,), {"memory_format": memory_format})
            cl.meta = {**inp.meta}
            cl.meta["val"] = inp_val.clone(memory_format=memory_format)
            clone_cache[key] = cl
            return cl

        def make_channels_last(node: fx.Node, memory_format, depth: int = 0) -> bool:
            """Make ``node``'s FX meta channels-last; return True on success."""
            nonlocal num_hoisted
            node_val = _meta_val(node)
            if node_val is None:
                return False
            if node_val.is_contiguous(memory_format=memory_format):
                return True  # already channels-last per meta

            # Hoist through single-consumer layout-transparent ops: rewrite this
            # op's meta to channels-last and recurse on its input, so the
            # transpose fuses with the upstream producer instead of the pad kernel.
            if depth < 8 and node.op == "call_function" and node.target in _HOISTABLE_OPS and len(node.users) == 1:
                src = node.args[0]
                if isinstance(src, fx.Node):
                    if not make_channels_last(src, memory_format, depth + 1):
                        # Chain top: materialise the layout change here, above
                        # the hoistable op.
                        cl = channels_last_clone(src, memory_format, node)
                        if cl is None:
                            return False
                        node.replace_input_with(src, cl)
                    node.meta["val"] = node_val.clone(memory_format=memory_format)
                    num_hoisted += 1
                    return True
            return False

        num_converted = 0
        for conv in list(graph.nodes):
            if conv.op != "call_function" or conv.target != aten.convolution.default:
                continue
            x_val = _meta_val(conv.args[0])
            if x_val is None:
                continue
            if x_val.ndim == 4:
                memory_format = torch.channels_last
            elif x_val.ndim == 5:
                memory_format = torch.channels_last_3d
            else:
                continue  # conv1d etc.: leave untouched

            new_args = list(conv.args)
            changed = False
            for idx in (0, 1):  # x, weight
                inp = new_args[idx]
                if not isinstance(inp, fx.Node):
                    continue
                # Try hoisting first (rewrites pad metas upstream in place).
                if idx == 0 and make_channels_last(inp, memory_format):
                    inp_val = _meta_val(inp)
                    if inp_val is not None and inp_val.is_contiguous(memory_format=memory_format):
                        changed = True
                        continue
                cl = channels_last_clone(inp, memory_format, conv)
                if cl is not None:
                    new_args[idx] = cl
                    changed = True
            if changed:
                conv.args = tuple(new_args)
                num_converted += 1

        if num_converted:
            graph.lint()
            magi_logger.info(
                "ConvChannelsLastPass: routed %d forward conv(s) through channels-last clones "
                "(%d clone node(s) inserted, %d pad meta(s) hoisted to channels-last)",
                num_converted,
                len(clone_cache),
                num_hoisted,
            )
        return (num_converted) > 0
