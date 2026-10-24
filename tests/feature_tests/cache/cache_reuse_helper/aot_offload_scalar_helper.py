# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""One AOT + CPU offload run of a module with a Python float attribute; writes what happened to ``--output``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from magi_compiler import magi_compile
from magi_compiler.magi_backend.magi_compiler_base import MagiCompileState


@torch.library.custom_op("magi_test::aot_add_scalar", mutates_args=())
def _add_scalar(x: torch.Tensor, value: float) -> torch.Tensor:
    return x + value


@_add_scalar.register_fake
def _(x, value):
    return torch.empty_like(x)


@magi_compile(dynamic_arg_dims={"x": 0}, config_patch=lambda c: c.model_copy(update={"aot": True}))
class FloatAttributeModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8))
        self.shift = 0.25

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _add_scalar(x * self.weight, self.shift)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    events = {"aot_compile_error": None, "aot_loaded": False}
    aot_compile, load_artifacts = MagiCompileState.aot_compile, MagiCompileState.load_aot_compile_artifacts

    def recording_aot_compile(self, *a, **k):
        try:
            return aot_compile(self, *a, **k)
        except Exception as e:
            events["aot_compile_error"] = f"{type(e).__name__}: {e}"
            raise

    def recording_load(self, *a, **k):
        loaded = load_artifacts(self, *a, **k)
        events["aot_loaded"] |= bool(loaded)
        return loaded

    MagiCompileState.aot_compile = recording_aot_compile
    MagiCompileState.load_aot_compile_artifacts = recording_load

    torch.manual_seed(0)
    model = FloatAttributeModule().cuda()
    x = torch.randn(4, 8, device="cuda")
    with torch.no_grad():
        out = model(x)
    events["correct"] = torch.allclose(out, x * model.weight.detach().cuda() + 0.25)
    args.output.write_text(json.dumps(events))


if __name__ == "__main__":
    main()
