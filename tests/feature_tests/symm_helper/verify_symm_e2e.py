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

"""Step-3 acceptance: the whole copy-engine path, from a meta-built model.

The other verification scripts each cut the chain somewhere.  This one runs it
end to end the way inference actually does, which is the only way to exercise
the piece that cannot be faked: the weights are never allocated as ordinary
tensors at all.  The model is built under ``torch.device("meta")``, sharded by
SimpleFSDP, and materialized with a single ``root.to_empty(cuda)`` -- and the
decorated block's patched ``_apply`` claims its own subtree on the way down.
The builder here is deliberately ignorant of symmetric memory, because the real
one is too.

Then a checkpoint load writes into the arena views (``copy_``, not ``assign``),
``@magi_compile`` compiles the block, the rewrite pass retargets the gathers,
and the reorder pass hoists them.  Checked at the end:

  1. **Placement** -- every block shard lives in a symmetric window, and the
     head, outside the decorated block, does not.
  2. **Rewrite** -- the gathers really became ``magi::symm_all_gather``.  A pass
     that silently no-ops leaves a correct, NCCL-transported model behind, so
     correctness alone cannot detect it.
  3. **Numerics** -- output matches an unsharded eager model holding the same
     checkpoint, and it must still match on the SECOND step, when the resident
     slots hold the previous step's weights.
  4. **Bytes moved** -- the resident-slot footprint, which is the memory this
     transport trades for SM occupancy.

Driven by ``tests/feature_tests/test_symm_e2e.py`` at two ranks; the checks are
asserted through the ``CHECK ...`` / ``E2E_PASS`` markers printed below.  Run it
directly for a bigger, more realistic shape (2+ NVLink-connected GPUs)::

    torchrun --nproc_per_node=8 tests/feature_tests/symm_helper/verify_symm_e2e.py
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("TORCH_SYMM_MEM_DISABLE_MULTICAST", "1")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.distributed.device_mesh import init_device_mesh  # noqa: E402

from magi_compiler import magi_compile  # noqa: E402
from magi_compiler.config import CompileMode, CudaGraphMode  # noqa: E402


def _block_cls(transport: str) -> type:
    """A decorated block, exactly as a model author would write one.

    Decoration happens at class-definition time, before any instance exists --
    which is what lets the transport choose where the weights are allocated.
    """

    class Block(nn.Module):
        def __init__(self, hidden: int, n_layers: int, dtype: torch.dtype):
            super().__init__()
            self.layers = nn.ModuleList(nn.Linear(hidden, hidden, bias=False, dtype=dtype) for _ in range(n_layers))

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    def patch(cfg):
        cfg.compile_mode = CompileMode.MAGI_COMPILE
        cfg.cudagraph_mode = CudaGraphMode.NONE
        cfg.disable_graph_split = True
        cfg.fsdp_config.enable_fsdp = True
        cfg.fsdp_config.transport = transport
        return cfg

    return magi_compile(Block, config_patch=patch, dynamic_arg_dims={"x": 0})


class Root(nn.Module):
    """``to_empty`` is called here, never on the block: the interception has to
    survive the recursion, or it never fires in a real model."""

    def __init__(self, block_cls: type, hidden: int, n_layers: int, dtype: torch.dtype):
        super().__init__()
        self.block = block_cls(hidden, n_layers, dtype)
        self.head = nn.Linear(hidden, hidden, bias=False, dtype=dtype)

    def forward(self, x):
        return self.head(self.block(x))


class _PlainBlock(nn.Module):
    """Undecorated twin of ``Block``: same parameter names, so one state_dict
    fits both, and no FSDP, so it gives an independent answer."""

    def __init__(self, hidden: int, n_layers: int, dtype: torch.dtype):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(hidden, hidden, bias=False, dtype=dtype) for _ in range(n_layers))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _load_into_shards(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Write each rank's slice of the checkpoint into the shard it already owns.

    ``copy_`` into the existing storage, never ``load_state_dict(assign=True)``:
    assigning would swap the arena views out for ordinary tensors and quietly
    demote every gather back to NCCL.
    """
    from torch.distributed.tensor import distribute_tensor

    with torch.no_grad():
        for name, p in _named_shards(model):
            full = state[name]
            want = distribute_tensor(full.to(p.device), p.device_mesh, p.placements)
            p._local_tensor.copy_(want._local_tensor)
    torch.cuda.synchronize()


def _named_shards(model: nn.Module):
    """``(state_dict name, DTensor)`` for every parameter, reaching through
    SimpleFSDP's parametrization the same way ``state_dict`` does."""
    from torch.distributed.tensor import DTensor

    for mod_name, mod in model.named_modules():
        for p_name, p in mod._parameters.items():
            if isinstance(p, DTensor):
                yield (f"{mod_name}.{p_name}" if mod_name else p_name), p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-tokens", type=int, default=4096)
    ap.add_argument("--transport", default="copy_engine", choices=["nccl", "copy_engine"])
    args = ap.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("cpu:gloo,cuda:nccl", device_id=torch.device("cuda", local_rank))
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", local_rank)
    dtype = torch.bfloat16
    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp",))
    os.environ.setdefault("MAGI_LOGGING_LEVEL", "INFO")

    def log(*a):
        if rank == 0:
            print(*a, flush=True)

    ce = args.transport == "copy_engine"

    # -- count the rewrites, without asking the compiled model to confess ----
    from magi_compiler.passes.fsdp_overlap import lower_and_bucket as _lb

    n_rewritten = 0
    orig_rewrite = _lb.rewrite_weight_ag_to_copy_engine

    def spy_rewrite(graph):
        nonlocal n_rewritten
        got = orig_rewrite(graph)
        n_rewritten += got
        return got

    _lb.rewrite_weight_ag_to_copy_engine = spy_rewrite

    # -- the reference: unsharded, eager, same checkpoint -------------------
    torch.manual_seed(1234)
    ref = Root(_PlainBlock, args.hidden, args.n_layers, dtype).to(device)
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(0.0, args.hidden**-0.5)
    state = {k: v.detach().clone() for k, v in ref.state_dict().items()}

    x = torch.randn(args.n_tokens, args.hidden, device=device, dtype=dtype).mul_(0.02)
    with torch.no_grad():
        truth = ref(x)
    del ref
    torch.cuda.empty_cache()

    # -- the model under test: meta build -> shard -> to_empty -> load ------
    from torchtitan.experiments.simple_fsdp.simple_fsdp import data_parallel

    with torch.device("meta"):
        model = Root(_block_cls(args.transport), args.hidden, args.n_layers, dtype)
    model = data_parallel(model, mesh, mode="fully_shard", ac_mode="full")
    model.to_empty(device=device)
    _load_into_shards(model, state)

    # (1) placement: the block's shards are in a window, the head's are not.
    from magi_compiler.symm_mem import lookup_shard, registered_arenas

    block_names = {n for n, _ in _named_shards(model) if n.startswith("block.")}
    in_arena = [n for n, p in _named_shards(model) if lookup_shard(p._local_tensor.data_ptr()) is not None]
    head_in_arena = [n for n in in_arena if not n.startswith("block.")]
    arena_mib = sum(a.nbytes for a in registered_arenas()) / 2**20
    placement_ok = (set(in_arena) == block_names) if ce else (in_arena == [])
    log(
        f"CHECK placement: {len(in_arena)}/{len(block_names)} block shards in {len(registered_arenas())} "
        f"window(s) ({arena_mib:.0f} MiB), {len(head_in_arena)} stray -> {'ok' if placement_ok else 'WRONG'}"
    )

    # (2) + (3): compile, then run twice.  The second step is the one that
    # would read a stale destination if the wait were missing.
    with torch.no_grad():
        out1 = model(x)
        torch.cuda.synchronize()
        out2 = model(x)
        torch.cuda.synchronize()

    _lb.rewrite_weight_ag_to_copy_engine = orig_rewrite

    expect_rewrites = args.n_layers if ce else 0
    rewrite_ok = n_rewritten == expect_rewrites
    log(
        f"CHECK rewrite: {n_rewritten}/{expect_rewrites} gathers on magi::symm_all_gather -> {'ok' if rewrite_ok else 'WRONG'}"
    )

    def rel(a, b):
        return ((a.float() - b.float()).norm() / (b.float().norm() + 1e-6)).item()

    r1, r2 = rel(out1, truth), rel(out2, truth)
    numeric_ok = bool(torch.isfinite(out1).all()) and max(r1, r2) < 5e-2
    log(f"CHECK numerics vs unsharded eager: step1 rel={r1:.6f} step2 rel={r2:.6f} -> {'ok' if numeric_ok else 'WRONG'}")

    log(
        f"\nCONFIG world={world} hidden={args.hidden} layers={args.n_layers} tokens={args.n_tokens} "
        f"transport={args.transport}\n"
        f"MEMORY arena={arena_mib:.0f}MiB "
        f"(one gathered layer = {args.hidden ** 2 * dtype.itemsize / 2**20:.0f}MiB)"
    )

    ok = placement_ok and rewrite_ok and numeric_ok
    t = torch.tensor([1 if ok else 0], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    all_ok = bool(t.item())
    log(f"\nE2E_{'PASS' if all_ok else 'FAIL'}")

    dist.barrier()
    dist.destroy_process_group()
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
