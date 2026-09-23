"""Helper script for test_aot_autograd_fallback.py.

Runs a real training step (forward + backward) through magi_compile, with a
spy on standalone_compile that records AOTConfig.force_non_lazy_backward_lowering
at the moment each subgraph is compiled.

This proves the _force_eager_backward_lowering() context manager is active
during the real compile path — not just in isolation.

Output JSON payload
-------------------
- backward_flag_during_compile: list of bool, one per standalone_compile call
- all_flags_true: True iff every call saw force_non_lazy_backward_lowering=True
- num_standalone_compile_calls: len(backward_flag_during_compile)
- num_compiled_artifacts_saved: from compilation_counter
- num_inductor_compiles: from compilation_counter
- loss: scalar training loss value
"""
from __future__ import annotations

import argparse
import json
from unittest.mock import patch

import torch
import torch._inductor as _inductor_mod
import torch.nn as nn

from magi_compiler import magi_compile
from magi_compiler.utils import compilation_counter

DEVICE = "cuda"
DTYPE = torch.bfloat16
HIDDEN = 16


@magi_compile(dynamic_arg_dims={"x": 0})
class TrainingModel(nn.Module):
    """Minimal training model whose backward triggers AOTAutograd lowering."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(HIDDEN, HIDDEN, dtype=DTYPE, device=DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).sum()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch._dynamo.reset()
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    _real_standalone_compile = _inductor_mod.standalone_compile
    backward_flag_during_compile: list[bool] = []

    def _spy_standalone_compile(graph, example_inputs, **kwargs):
        from torch._functorch._aot_autograd.schemas import AOTConfig

        # Walk the frame stack is fragile; instead, we create a throwaway
        # AOTConfig to observe whether __post_init__ has been patched.
        probe = AOTConfig(
            fw_compiler=lambda *a, **k: None,
            bw_compiler=lambda *a, **k: None,
            partition_fn=lambda *a, **k: None,
            decompositions={},
            num_params_buffers=0,
            aot_id=-1,
            keep_inference_input_mutations=False,
        )
        backward_flag_during_compile.append(probe.force_non_lazy_backward_lowering)
        return _real_standalone_compile(graph, example_inputs, **kwargs)

    with patch("torch._inductor.standalone_compile", side_effect=_spy_standalone_compile):
        model = TrainingModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        x = torch.randn(4, HIDDEN, device=DEVICE, dtype=DTYPE)

        optimizer.zero_grad()
        loss = model(x)
        loss.backward()
        optimizer.step()

    payload = {
        "backward_flag_during_compile": backward_flag_during_compile,
        "all_flags_true": all(backward_flag_during_compile),
        "num_standalone_compile_calls": len(backward_flag_during_compile),
        "num_compiled_artifacts_saved": compilation_counter.num_compiled_artifacts_saved,
        "num_inductor_compiles": compilation_counter.num_inductor_compiles,
        "loss": float(loss.float().item()),
    }
    with open(args.output, "w") as f:
        json.dump(payload, f)


if __name__ == "__main__":
    main()
