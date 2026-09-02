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

"""Unit tests for the SimpleFSDP weight redistribute lowering pass
(``magi_compiler.passes.fsdp_overlap.lower_prim_redistribute_to_collectives``).

The pass matches ``prim_redistribute`` + ``prim_to_local`` fx nodes (by
``target.__name__``) whose input is a weight placeholder carrying a ``Shard(0)``
DTensor ``example_value``, and rewrites them into explicit
``all_gather_into_tensor`` + ``wait_tensor``.  ``prim_redistribute`` is a
torch.compile-internal prim that can't be constructed directly, so we build a
minimal synthetic graph with plain functions named ``prim_redistribute`` /
``prim_to_local`` and REAL 1-rank DTensor metas (this drives the exact matching /
rewrite logic without depending on Dynamo capturing the prim).

Uses a 1-rank process group + device mesh (GPU required).
"""

import os

import pytest
import torch
import torch.fx as fx

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture(scope="module")
def dist_1rank():
    """A single-rank process group + cuda device mesh (module-scoped)."""
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29661")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    created = False
    if not dist.is_initialized():
        dist.init_process_group("gloo")
        created = True
    torch.cuda.set_device(0)
    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh("cuda", (1,))
    yield mesh
    if created:
        dist.destroy_process_group()


# named functions whose __name__ is EXACTLY "prim_redistribute" / "prim_to_local"
# (the pass matches node.target.__name__ against those strings).
def _make_prim(name):
    def f(x):
        return x

    f.__name__ = name
    return f


prim_redistribute = _make_prim("prim_redistribute")
prim_to_local = _make_prim("prim_to_local")


def _build_redistribute_graph(mesh, weight_name, dtype=torch.bfloat16, rows=8, cols=4):
    """weight(Shard0 DTensor) -> prim_redistribute(Replicate) -> prim_to_local -> out."""
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor

    full = torch.randn(rows, cols, device="cuda", dtype=dtype)
    sharded = distribute_tensor(full, mesh, [Shard(0)])
    replicated = distribute_tensor(full, mesh, [Replicate()])

    g = fx.Graph()
    w = g.placeholder(weight_name)
    w.meta["example_value"] = sharded
    rd = g.call_function(prim_redistribute, (w,))
    rd.meta["example_value"] = replicated
    tl = g.call_function(prim_to_local, (rd,))
    tl.meta["example_value"] = replicated._local_tensor
    g.output((tl,))
    return fx.GraphModule(torch.nn.Module(), g)


_AG = torch.ops._c10d_functional.all_gather_into_tensor.default
_WAIT = torch.ops._c10d_functional.wait_tensor.default


def _targets(gm):
    return [n.target for n in gm.graph.nodes if n.op == "call_function"]


@requires_cuda
def test_lowering_rewrites_shard0_weight(dist_1rank):
    from magi_compiler.passes.fsdp_overlap import lower_prim_redistribute_to_collectives

    gm = _build_redistribute_graph(dist_1rank, "model_fc1_weight_parameter")
    n = lower_prim_redistribute_to_collectives(gm)

    assert n == 1, "one Shard(0) weight redistribute should be lowered"
    targets = _targets(gm)
    assert _AG in targets
    assert _WAIT in targets
    # the prims are gone, replaced by explicit collectives
    assert prim_redistribute not in targets and prim_to_local not in targets
    # the gather is tagged so the bucketing pass can find it
    tagged = [x for x in gm.graph.nodes if x.meta.get("magi_fsdp_weight_ag")]
    assert len(tagged) == 1


@requires_cuda
def test_lowering_skips_non_weight_input(dist_1rank):
    """A redistribute whose input placeholder is NOT weight-named is left untouched."""
    from magi_compiler.passes.fsdp_overlap import lower_prim_redistribute_to_collectives

    gm = _build_redistribute_graph(dist_1rank, "some_activation")  # not weight/param/bias
    n = lower_prim_redistribute_to_collectives(gm)
    assert n == 0
    targets = _targets(gm)
    assert _AG not in targets
    assert prim_redistribute in targets  # untouched


@requires_cuda
def test_lowering_gather_and_wait_are_separate_nodes(dist_1rank):
    """Launch and wait must be DISTINCT nodes (so the reorder can move the launch
    independently of the wait) -- the whole point of lowering."""
    from magi_compiler.passes.fsdp_overlap import lower_prim_redistribute_to_collectives

    gm = _build_redistribute_graph(dist_1rank, "layer_weight")
    lower_prim_redistribute_to_collectives(gm)
    ag = [n for n in gm.graph.nodes if n.op == "call_function" and n.target is _AG]
    wait = [n for n in gm.graph.nodes if n.op == "call_function" and n.target is _WAIT]
    assert len(ag) == 1 and len(wait) == 1
    assert wait[0].args[0] is ag[0]  # wait consumes the gather's output
