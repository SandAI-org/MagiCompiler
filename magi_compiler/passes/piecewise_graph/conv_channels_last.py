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

Mechanism Under the Hood:
1. **FX-Meta Stride Injection**: The pass inserts ``aten.clone`` nodes before
   each conv input/weight and manually configures their ``node.meta["val"]``
   to carry channels-last FakeTensors. Because Inductor's clone lowering
   ignores ``memory_format`` (a known PyTorch upstream TODO), the channels-last
   signal is carried purely within the FX meta strides.
2. **Inductor Constraint Co-design**: With ``layout_optimization=False`` set
   by this pass, Inductor's pre-registered ``constrain_conv_to_fx_strides``
   (in ``torch/_inductor/kernel/conv.py``) fires. It reads our modified FX input
   meta strides and triggers ``require_stride_order`` at the conv boundary.
3. **Zero-Cost Strided Allocation**: The clone lowers to a Pointwise kernel
   with ``FlexibleLayout``. Consequently, the ``require_stride_order`` constraint
   is zero-cost: the upstream buffer is allocated directly in channels-last
   layout without generating an extra transpose copy kernel.
4. **cuDNN Memory-Format Probe**: With channels-last inputs safely matching
   stride constraints, ``conv_layout()`` naturally infers a channels-last
   cuDNN output ``FixedLayout``.

The pass only fires on static, conv-heavy graphs; dynamic-shape or conv-sparse
graphs are skipped (their channels-last transpose tiles badly / gains little).
"""

import torch
from torch import fx

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
    """Make conv2d/conv3d inputs channels-last on the post-grad ATen graph.

    If the conv input comes from a single-consumer, layout-transparent op
    (e.g., ``constant_pad_nd``), the clone is hoisted above it and its meta
    rewritten to channels-last. This keeps the pad kernel coalesced instead of
    triggering an extra memory-bound NC(D)HW -> N(D)HWC transpose.
    """

    inductor_config_keys_potentially_mutated_by_this_pass = ("layout_optimization",)

    @emit_pass_lifecycle
    def __call__(self, graph: fx.Graph) -> bool:
        # Only rewrite static, conv-heavy graphs. Channels-last inserts an
        # NC(D)HW->N(D)HWC transpose; under dynamic shapes Inductor tiles it
        # badly, and on conv-sparse graphs the few cuDNN channels-last kernels
        # don't pay for the extra copies.
        # TODO: If tiling optimization is upgraded to support conv layout opt
        # under dynamic shapes, we can remove the ``is_dynamic`` check.
        if self.is_dynamic(graph) or not self.is_conv_heavy(graph):
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
