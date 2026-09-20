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

"""Host offload of a model that is NOT sharded.

Everything here runs with no process group, no DTensor and no all-gather: the
point of the source abstraction is that offload no longer needs any of them.
What differs from the FSDP flavour is only how a weight is recognized, and two
consequences of that worth pinning down -- the loaded bytes are the whole weight
rather than a shard, and they stay live until the last reader rather than dying
at a gather.
"""

import pytest
import torch
import torch.fx as fx

from magi_compiler.passes.weight_offload import PlainParamSource, bind_weights_to_host, insert_h2d_loads

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

_WAIT = torch.ops._c10d_functional.wait_tensor.default


@pytest.fixture(autouse=True)
def clean_pool():
    from magi_compiler.passes.weight_offload import host_pool

    host_pool.reset()
    yield
    host_pool.reset()


def _linear_graph(*params):
    """``x`` and some weights in, one matmul per weight out.

    Shaped like what Dynamo hands the backend: every parameter is lifted to a
    placeholder at the top, which is exactly why grouping cannot go by
    declaration order.
    """
    g = fx.Graph()
    x = g.placeholder("l_x_")
    x.meta["example_value"] = torch.empty(8, params[0].shape[0], device="meta", dtype=params[0].dtype)

    holders = []
    for i, p in enumerate(params):
        n = g.placeholder(f"L_self_modules_layers_{i}_parameters_weight_")
        n.meta["example_value"] = p
        holders.append(n)

    out = x
    for n, p in zip(holders, params):
        out = g.call_function(torch.matmul, (out, n))
        out.meta["example_value"] = torch.empty(8, p.shape[1], device="meta", dtype=p.dtype)
    g.output((out,))
    gm = fx.GraphModule(torch.nn.Module(), g)
    return gm, [None, *params]  # example_inputs: x has no live parameter behind it


def _nodes(gm, target):
    return [n for n in gm.graph.nodes if n.op == "call_function" and n.target is target]


@requires_cuda
def test_a_plain_parameter_is_offloaded_without_any_fsdp():
    from magi_compiler.passes.weight_offload import host_pool
    from magi_compiler.passes.weight_offload.h2d_op import H2D_LOAD

    w = torch.nn.Parameter(torch.randn(256, 256, device="cuda", dtype=torch.bfloat16))
    gm, examples = _linear_graph(w)

    source = PlainParamSource()
    assert bind_weights_to_host(gm, examples, source, min_bytes=0) == 1
    assert w.untyped_storage().nbytes() == 0, "the device storage must actually be released"
    assert insert_h2d_loads(gm, source) == 1

    (load,) = _nodes(gm, H2D_LOAD)
    assert load.args[1] == 0, "the load carries the host-pool slot"
    assert host_pool.name_of(0) == "layers.0.weight"

    # The matmul reads the loaded copy, not the freed placeholder.
    (mm,) = _nodes(gm, torch.matmul)
    assert any(getattr(a, "target", None) is _WAIT for a in mm.all_input_nodes)


@requires_cuda
def test_only_parameters_are_taken_not_activations():
    """Every graph input looks alike; only the ones that are the same every
    forward are worth moving."""
    w = torch.nn.Parameter(torch.randn(256, 256, device="cuda", dtype=torch.bfloat16))
    gm, examples = _linear_graph(w)
    # x's "live value" is a plain tensor, not a Parameter.
    examples[0] = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)

    assert bind_weights_to_host(gm, examples, PlainParamSource(), min_bytes=0) == 1


@requires_cuda
def test_weights_group_by_first_use_not_by_declaration():
    """Dynamo lifts every parameter to the top of the graph, so their placeholder
    order says nothing about when they run.  Grouping by it would put the first
    layer and the last in one submission -- a load that has to land before layer
    0 and is not needed until layer N."""
    from magi_compiler.passes.weight_offload.h2d_op import H2D_LOAD, H2D_LOAD_COALESCED

    params = [torch.nn.Parameter(torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)) for _ in range(4)]
    gm, examples = _linear_graph(*params)

    # Room for two weights per load (128 KiB each).
    source = PlainParamSource(group_bytes=2 * 256 * 256 * 2)
    assert bind_weights_to_host(gm, examples, source, min_bytes=0) == 4
    assert insert_h2d_loads(gm, source) == 2

    coalesced = _nodes(gm, H2D_LOAD_COALESCED)
    assert len(coalesced) == 2 and not _nodes(gm, H2D_LOAD)
    # Each group holds consecutive layers, never the first paired with the last.
    for node in coalesced:
        names = [h.name for h in node.args[0]]
        idx = sorted(int(n.split("layers_")[1].split("_")[0]) for n in names)
        assert idx[-1] - idx[0] == len(idx) - 1, f"group spans non-adjacent layers: {idx}"


@requires_cuda
def test_one_load_per_weight_when_no_group_cap_is_set():
    from magi_compiler.passes.weight_offload.h2d_op import H2D_LOAD

    params = [torch.nn.Parameter(torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)) for _ in range(3)]
    gm, examples = _linear_graph(*params)

    source = PlainParamSource()
    assert bind_weights_to_host(gm, examples, source, min_bytes=0) == 3
    assert insert_h2d_loads(gm, source) == 3
    assert len(_nodes(gm, H2D_LOAD)) == 3


@requires_cuda
def test_the_size_floor_still_applies():
    small = torch.nn.Parameter(torch.randn(16, 16, device="cuda", dtype=torch.bfloat16))
    gm, examples = _linear_graph(small)

    assert bind_weights_to_host(gm, examples, PlainParamSource(), min_bytes=4 << 20) == 0
    assert small.untyped_storage().nbytes() > 0


@requires_cuda
def test_offload_survives_a_real_inductor_compile_without_fsdp():
    """The whole point of the decoupling, end to end in one process.

    No process group is initialized here -- if anything in the offload path still
    reached for a mesh, a rank or a collective, this would raise rather than run.
    """
    import torch.distributed as dist

    assert not dist.is_initialized(), "this test exists to prove offload needs no process group"

    from magi_compiler.passes.weight_offload import host_pool
    from magi_compiler.passes.weight_offload.h2d_op import H2D_LOAD

    w = torch.nn.Parameter(torch.randn(512, 256, device="cuda", dtype=torch.bfloat16), requires_grad=False)
    x = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
    ref = x @ w.detach().clone()

    (slot,) = host_pool.bind_many([w.data], names=["w"])

    def f(weight, inp):
        return inp @ _WAIT(H2D_LOAD(weight, slot))

    # ``requires_grad=False`` is not incidental: ``magi::h2d_load`` has no
    # autograd kernel, so a grad-requiring weight would warn here and silently
    # drop the gradient in a training graph.  Inference runs under no_grad, which
    # is the only path this feature claims today.
    with torch.no_grad():
        out = torch.compile(f, backend="inductor", fullgraph=True)(w, x)
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)


# --------------------------------------------------- host-first materialize


class _Compiled(torch.nn.Module):
    """Stands in for a ``@magi_compile``'d submodule of a bigger unsharded model."""

    def __init__(self):
        super().__init__()
        self.big = torch.nn.Linear(256, 256, bias=False)
        self.small = torch.nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.big(x)


def _meta_root() -> torch.nn.Module:
    """A meta-built model: compiled inner, eager sibling the prologue still reads."""
    with torch.device("meta"):
        root = torch.nn.Module()
        root.eager = torch.nn.Linear(4, 4, bias=False)
        root.inner = _Compiled()
        root.to(torch.bfloat16)
    return root


def _patched(instance, *, min_shard_mib=0.0):
    from magi_compiler.config import get_compile_config
    from magi_compiler.passes.weight_offload import patch_materialize

    conf = get_compile_config().model_copy(deep=True)
    conf.offload_config.graph_weight_offload = True
    conf.offload_config.host_first_materialize = True
    conf.offload_config.offload_min_shard_mib = min_shard_mib
    patch_materialize(instance, conf)
    return instance


@requires_cuda
def test_to_empty_materializes_plain_parameters_in_host_memory():
    """The unsharded counterpart of the FSDP host-first test.

    Scope still comes from the module tree: only the compiled subtree is
    patched, so an embedding the eager prologue reads keeps its device storage.
    """
    root = _meta_root()
    _patched(root.inner)
    root.to_empty(device=torch.device("cuda"))

    assert root.inner.big.weight.device.type == "cpu"
    assert root.inner.big.weight.is_pinned()
    assert root.eager.weight.device.type == "cuda", "a weight the eager path reads must keep its storage"


@requires_cuda
def test_plain_parameter_device_moves_are_left_alone():
    """A dtype cast after to_empty must not quietly re-reserve host memory."""
    root = _meta_root()
    _patched(root.inner)
    root.to_empty(device=torch.device("cuda"))
    before = root.inner.big.weight.data_ptr()
    root.inner.float()

    weight = root.inner.big.weight
    assert weight.dtype is torch.float32
    assert weight.data_ptr() != before


@requires_cuda
def test_a_plain_parameter_below_the_floor_stays_on_the_device():
    root = _meta_root()
    _patched(root.inner, min_shard_mib=1.0)  # 256x256 bf16 = 128 KiB
    root.to_empty(device=torch.device("cuda"))

    assert root.inner.big.weight.device.type == "cuda"
    assert root.inner.small.weight.device.type == "cuda"


@requires_cuda
def test_plain_handoff_swaps_in_a_storage_free_device_stand_in():
    from magi_compiler.passes.weight_offload import host_pool
    from magi_compiler.passes.weight_offload.host_first import handoff_if_pending

    root = _meta_root()
    _patched(root.inner)
    root.to_empty(device=torch.device("cuda"))

    expected = torch.arange(root.inner.big.weight.numel(), dtype=torch.bfloat16).reshape(root.inner.big.weight.shape)
    with torch.no_grad():
        root.inner.big.weight.copy_(expected)

    assert handoff_if_pending(root.inner) == 2
    assert handoff_if_pending(root.inner) == 0, "handoff must not run twice"

    weight = root.inner.big.weight
    assert weight.device.type == "cuda", "the graph is lowered against the parameter's device"
    assert weight.untyped_storage().nbytes() == 0, "no device bytes until the graph loads them"
    assert weight.shape == expected.shape and weight.dtype == expected.dtype

    slot = host_pool.slot_of(weight)
    assert slot is not None, "PlainParamSource finds the pre-parked Parameter by identity"
    torch.testing.assert_close(host_pool.get(slot), expected)
    host_pool.make_resident(slot)
    torch.testing.assert_close(weight.cpu(), expected)


@requires_cuda
def test_a_pre_parked_plain_parameter_needs_no_binding():
    """The join between host-first and PlainParamSource.

    After handoff the Parameter is CUDA with empty storage.  ``unusable`` would
    reject it for that; ``slot_of`` has to recognize it as already bound.
    """
    from magi_compiler.passes.weight_offload import host_pool
    from magi_compiler.passes.weight_offload.h2d_op import H2D_LOAD
    from magi_compiler.passes.weight_offload.host_first import handoff_if_pending

    root = _meta_root()
    _patched(root.inner)
    root.to_empty(device=torch.device("cuda"))
    handoff_if_pending(root.inner)

    w = root.inner.big.weight
    gm, examples = _linear_graph(w)
    source = PlainParamSource()
    assert bind_weights_to_host(gm, examples, source, min_bytes=0) == 1
    assert host_pool.num_bound() == 2, "binding a pre-parked Parameter must not park it a second time"
    assert insert_h2d_loads(gm, source) == 1

    (load,) = _nodes(gm, H2D_LOAD)
    assert load.args[1] == host_pool.slot_of(w)
