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

"""``defer_host_materialization``: each weight is copied once, after the first compile picks the residents.

Single-process cases; a weight resident on some ranks of a sharing group is covered in
``test_rank_shared_host_weights.py``.
"""

import pytest
import torch

from magi_compiler.passes.weight_offload import get_global_host_weight_store, install_host_first_hooks

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

_ROWS = 256
_COLS = 64


@pytest.fixture(autouse=True)
def reset_host_weight_store():
    get_global_host_weight_store().reset()
    yield
    get_global_host_weight_store().reset()


@pytest.fixture
def no_process_group():
    import torch.distributed as dist

    assert not dist.is_initialized(), "deferred materialization without sharing must work without a process group"


class _ModelWithUnreadWeight(torch.nn.Module):
    """``unread`` is deferred with ``big`` but no compiled graph reads it."""

    def __init__(self):
        super().__init__()
        self.big = torch.nn.Linear(_COLS, _ROWS, bias=False)
        self.unread = torch.nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.big(x)


def _install_deferring_hooks(module: torch.nn.Module) -> torch.nn.Module:
    install_host_first_hooks(instance=module, min_weight_bytes=0, defer_host_materialization=True)
    return module


@requires_cuda
def test_first_cuda_move_neither_reads_nor_pins_the_weight():
    """Until compilation the CPU original stays the only copy, and loads read zeros instead of it."""
    module = _install_deferring_hooks(_ModelWithUnreadWeight())
    cpu_original = module.big.weight.detach()

    module.cuda()

    store = get_global_host_weight_store()
    slot = store.slot_for_tensor(module.big.weight)
    assert store.is_pending(slot)
    assert store.pending_source(slot).data_ptr() == cpu_original.data_ptr()
    assert module.big.weight.untyped_storage().nbytes() == 0
    assert not store.host_buffer(slot).any()


@requires_cuda
def test_a_meta_built_model_is_refused_before_anything_is_allocated():
    """to_empty would reserve host buffers before any checkpoint is read, defeating the deferral."""
    with torch.device("meta"):
        module = _install_deferring_hooks(_ModelWithUnreadWeight())

    with pytest.raises(RuntimeError, match="host-built models only"):
        module.to_empty(device="cuda")
    assert get_global_host_weight_store().weight_count() == 0


def _compile_with_deferral(module: torch.nn.Module, *, max_device_weight_mib: int) -> torch.nn.Module:
    from magi_compiler import magi_compile
    from magi_compiler.config import CompileMode, CudaGraphMode

    def patch(compile_config):
        compile_config.compile_mode = CompileMode.MAGI_COMPILE
        compile_config.cudagraph_mode = CudaGraphMode.NONE
        compile_config.disable_graph_split = True
        compile_config.fsdp_config.cost_mode = "analytical"
        compile_config.offload_config.graph_weight_offload = True
        compile_config.offload_config.defer_host_materialization = True
        compile_config.offload_config.offload_min_shard_mib = 0
        compile_config.offload_config.offload_h2d_bandwidth_gbps = 48.0
        compile_config.offload_config.offload_max_device_weight_mib = max_device_weight_mib
        return compile_config

    return magi_compile(module, config_patch=patch, dynamic_arg_dims={"x": 0})


@requires_cuda
@pytest.mark.usefixtures("no_process_group")
@pytest.mark.parametrize(
    "max_device_weight_mib,is_big_resident", [pytest.param(0, False, id="offloaded"), pytest.param(64, True, id="resident")]
)
def test_first_compile_places_each_deferred_weight_once(max_device_weight_mib, is_big_resident):
    """The first run already computes with the checkpoint, not the placeholder. Afterwards a resident
    weight lives only on the device, any other only on the host, the unread one included."""
    torch.manual_seed(0)
    reference = _ModelWithUnreadWeight().eval()
    module = _compile_with_deferral(_ModelWithUnreadWeight().eval(), max_device_weight_mib=max_device_weight_mib)
    module.load_state_dict(reference.state_dict())
    module.cuda()
    reference.cuda()
    inputs = torch.randn(8, _COLS, device="cuda")

    with torch.no_grad():
        expected = reference(inputs)
        torch.testing.assert_close(module(inputs), expected)
        torch.testing.assert_close(module(inputs), expected)

    store = get_global_host_weight_store()
    big_slot = store.slot_for_tensor(module.big.weight)
    unread_slot = store.slot_for_tensor(module.unread.weight)
    assert not store.is_pending(big_slot) and not store.is_pending(unread_slot)
    torch.testing.assert_close(store.host_buffer(unread_slot), reference.unread.weight.cpu())
    assert store.is_device_only(big_slot) == is_big_resident
    if is_big_resident:
        torch.testing.assert_close(module.big.weight.detach(), reference.big.weight.detach())
        with pytest.raises(RuntimeError, match="must stay on CUDA"):
            module.to("cpu")
    else:
        assert module.big.weight.untyped_storage().nbytes() == 0
        torch.testing.assert_close(store.host_buffer(big_slot), reference.big.weight.detach().cpu())
