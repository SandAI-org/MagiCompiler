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

from __future__ import annotations

import math

import torch
import torch.fx as fx

from magi_compiler.utils import magi_logger

# The functional collectives we lower a SimpleFSDP weight gather into.
_ALL_GATHER = torch.ops._c10d_functional.all_gather_into_tensor.default
_WAIT = torch.ops._c10d_functional.wait_tensor.default
_PAD = torch.ops.aten.constant_pad_nd.default
_SLICE = torch.ops.aten.slice.Tensor
_TO_COPY = torch.ops.aten._to_copy.default


def _is_prim(node: fx.Node, name: str) -> bool:
    return node.op == "call_function" and getattr(node.target, "__name__", None) == name


def _input_is_weight(node: fx.Node) -> bool:
    """The redistribute input is a SimpleFSDP weight/bias param placeholder."""
    src = node.args[0] if node.args else None
    if not isinstance(src, fx.Node) or src.op not in ("placeholder", "get_attr"):
        return False
    name = f"{src.name} {src.target}".lower()
    return any(t in name for t in ("parameter", "parameters", "weight", "bias"))


def _dtensor_meta(node: fx.Node):
    """Return the DTensor example_value carried on a node, or None."""
    ev = node.meta.get("example_value")
    return ev if getattr(ev, "_spec", None) is not None else None


def lower_prim_redistribute_to_collectives(graph: fx.GraphModule) -> int:
    """Rewrite SimpleFSDP weight ``prim_redistribute`` + ``prim_to_local`` pairs
    into explicit functional collectives, so the launch and the wait become two
    distinct FX nodes that a later graph-split pass can place in *different*
    submods (enabling cross-boundary overlap with the MoE op).

    For one ``Shard(0)`` weight (full dim0 ``F``, world ``W``, local shard the
    input placeholder's ``_local_tensor``), this emits the exact sequence Inductor
    itself produces for a ``redistribute(Replicate()).to_local()``:

        [ _to_copy(local, fwd_dtype) ]?          # only if forward_dtype set
        [ constant_pad_nd(local, dim0 -> chunk) ]?   # only if local < chunk (uneven)
        all_gather_into_tensor(local, W, group_name)  # -> (W*chunk, ...)
        wait_tensor(ag)
        [ slice(wait, 0, 0, F) ]?                # only if W*chunk != F (uneven)

    where ``chunk = ceil(F / W)``.  ``all_gather_into_tensor`` and ``wait_tensor``
    are kept as separate nodes.  Only ``Shard(0)`` is handled; anything else is
    left as the original prim path and logged.

    Returns the number of redistribute pairs lowered.
    """
    lowered = 0
    skipped = 0

    for node in list(graph.graph.nodes):
        if not _is_prim(node, "prim_redistribute"):
            continue
        if not _input_is_weight(node):
            continue

        # prim_to_local consumer (the node whose output the rest of the graph uses).
        to_local = next((u for u in node.users if _is_prim(u, "prim_to_local")), None)
        if to_local is None:
            skipped += 1
            continue

        src = node.args[0]
        out_dt = _dtensor_meta(node)
        in_dt = _dtensor_meta(src)
        if out_dt is None or in_dt is None:
            skipped += 1
            continue

        spec = in_dt._spec
        placement = spec.placements[0] if len(spec.placements) == 1 else None
        if placement is None or not getattr(placement, "is_shard", lambda: False)() or placement.dim != 0:
            # Only single-mesh Shard(0) is supported; leave others on the prim path.
            skipped += 1
            continue

        mesh = spec.mesh
        try:
            group_name = mesh._dim_group_names[0]
            world = int(mesh.size(0))
        except Exception as exc:  # pragma: no cover - defensive
            magi_logger.warning("FSDP lowering skip %s: cannot resolve group/world (%s)", node.name, exc)
            skipped += 1
            continue

        full = out_dt._local_tensor  # FakeTensor with the global (replicated) shape
        local = in_dt._local_tensor  # FakeTensor with this rank's shard shape
        F = int(full.shape[0])
        L = int(local.shape[0])
        chunk = math.ceil(F / world)

        # forward_dtype: cast the local shard before the gather (matches torchtitan).
        fwd_dtype = None
        try:
            import inspect

            fwd_dtype = inspect.getclosurevars(node.target).nonlocals.get("kwargs_as_value", {}).get("forward_dtype")
        except Exception:
            fwd_dtype = None

        with graph.graph.inserting_before(node):
            # The weight placeholder is still a Shard(0) DTensor; the functional
            # collective needs a PLAIN local tensor (DTensor has no sharding
            # strategy for all_gather_into_tensor).  Extract the local shard first
            # -- the same `param.to_local()` step the eager _materialize_param uses.
            cur = graph.graph.call_method("to_local", (src,))
            cur.meta["example_value"] = local
            cur_dtype = local.dtype

            if fwd_dtype is not None and fwd_dtype != cur_dtype:
                cur = graph.graph.call_function(_TO_COPY, (cur,), {"dtype": fwd_dtype})
                cur.meta["example_value"] = local.to(fwd_dtype)
                cur_dtype = fwd_dtype

            # Pad the local shard up to `chunk` rows when this rank owns fewer
            # (uneven Shard(0): trailing ranks get remainder/empty).
            #
            # meta example_values MUST be created via FakeTensor ops (``local.new_empty``),
            # NOT ``torch.empty(..., device=cuda)``: a FakeTensor's ``.device`` is a real
            # ``cuda:N`` so torch.empty allocates a REAL full-size buffer per weight
            # (~the whole unsharded model across all weights) and OOMs.
            if L < chunk:
                pad = [0, 0] * (local.dim() - 1) + [0, chunk - L]
                padded = graph.graph.call_function(_PAD, (cur, pad, 0.0))
                padded.meta["example_value"] = local.new_empty((chunk, *local.shape[1:]), dtype=cur_dtype)
                cur = padded

            ag = graph.graph.call_function(_ALL_GATHER, (cur, world, group_name))
            ag.meta["example_value"] = local.new_empty((world * chunk, *local.shape[1:]), dtype=cur_dtype)
            # Mark as a SimpleFSDP weight gather so the per-submod bucketing pass
            # can coalesce these into a single all_gather_into_tensor_coalesced.
            ag.meta["magi_fsdp_weight_ag"] = True

            wait = graph.graph.call_function(_WAIT, (ag,))
            wait.meta["example_value"] = local.new_empty((world * chunk, *local.shape[1:]), dtype=cur_dtype)

            result = wait
            if world * chunk != F:
                sliced = graph.graph.call_function(_SLICE, (wait, 0, 0, F))
                sliced.meta["example_value"] = local.new_empty((F, *local.shape[1:]), dtype=cur_dtype)
                result = sliced

        # Re-point every user of prim_to_local at the explicit-collective result,
        # then drop the two prim nodes.
        to_local.replace_all_uses_with(result)
        graph.graph.erase_node(to_local)
        graph.graph.erase_node(node)
        lowered += 1

    if lowered:
        graph.graph.lint()
        graph.recompile()
    magi_logger.info(
        "FSDP redistribute lowering: lowered %d weight prim_redistribute -> explicit collectives (skipped %d)",
        lowered,
        skipped,
    )
    return lowered
