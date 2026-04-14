# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import argparse
import json
import math

import torch
import torch.nn as nn

from magi_compiler import magi_compile, magi_register_custom_op
from magi_compiler.config import get_compile_config

DEVICE = "cuda"
DTYPE = torch.float16
SEQ_LEN = 4
HIDDEN = 4
HEADS = 2
LAYERS = 1


def _scaled_dot_product_attn_infer_output_meta(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.size(-1))
    score = torch.matmul(q, k.transpose(-2, -1)) * scale
    prob = torch.softmax(score, dim=-1)
    return torch.matmul(prob, v)


@magi_register_custom_op(
    "test::scaled_dot_product_attn_boundary",
    infer_output_meta_fn=_scaled_dot_product_attn_infer_output_meta,
    is_subgraph_boundary=True,
)
def _scaled_dot_product_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.size(-1))
    score = torch.matmul(q, k.transpose(-2, -1)) * scale
    prob = torch.softmax(score, dim=-1)
    return torch.matmul(prob, v)


class SimpleSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, HIDDEN, dtype=DTYPE, device=DEVICE)
        self.k_proj = nn.Linear(HIDDEN, HIDDEN, dtype=DTYPE, device=DEVICE)
        self.v_proj = nn.Linear(HIDDEN, HIDDEN, dtype=DTYPE, device=DEVICE)
        self.o_proj = nn.Linear(HIDDEN, HIDDEN, dtype=DTYPE, device=DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden = x.shape
        head_dim = hidden // HEADS
        q = self.q_proj(x).view(bsz, seq_len, HEADS, head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, HEADS, head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, HEADS, head_dim).transpose(1, 2)
        out = _scaled_dot_product_attn(q, k, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = SimpleSelfAttention()
        self.norm1 = nn.LayerNorm(HIDDEN, dtype=DTYPE, device=DEVICE)
        self.ffn = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN * 4, dtype=DTYPE, device=DEVICE),
            nn.GELU(),
            nn.Linear(HIDDEN * 4, HIDDEN, dtype=DTYPE, device=DEVICE),
        )
        self.norm2 = nn.LayerNorm(HIDDEN, dtype=DTYPE, device=DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


@magi_compile(dynamic_arg_dims={"x": 1})
class CompiledTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock() for _ in range(LAYERS)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def make_input() -> torch.Tensor:
    torch.manual_seed(2026)
    return torch.randn(1, SEQ_LEN, HIDDEN, device=DEVICE, dtype=DTYPE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-kind", choices=["baseline", "cache"], required=True)
    args = parser.parse_args()

    config = get_compile_config()
    config.splitting_ops = ["test::scaled_dot_product_attn_boundary"]

    torch._dynamo.reset()
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    model = CompiledTransformer().eval()
    x = make_input()

    with torch.no_grad():
        y = model(x)

    payload = {
        "mode": "aot" if config.aot else "jit",
        "run_kind": args.run_kind,
        "shape": list(y.shape),
        "sum": float(y.float().sum().item()),
        "mean": float(y.float().mean().item()),
    }
    with open(args.output, "w") as f:
        json.dump(payload, f)


if __name__ == "__main__":
    main()
