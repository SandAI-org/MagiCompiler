"""
FSDP Inference Example – MagiCompiler Full-Graph Capture (Phase 1).

This example demonstrates how to use MagiCompiler to capture the full
computation graph of an FSDP-wrapped model during inference, enabling
inter-layer fusion and communication-computation overlap.

Usage:
    # Single GPU (unit test mode):
    python examples/fsdp_inference_example.py

    # Multi-GPU (requires torchrun):
    torchrun --nproc_per_node=2 examples/fsdp_inference_example.py
"""

import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fsdp_inference_example")


# ── Model Definitions ────────────────────────────────────────────────


class TransformerBlock(nn.Module):
    """A single transformer block with self-attention and FFN."""

    def __init__(self, dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        h = self.norm2(x)
        return x + self.ffn(h)


class GPTLikeModel(nn.Module):
    """A GPT-like model with multiple transformer blocks for FSDP testing."""

    def __init__(self, dim: int = 64, num_blocks: int = 6, vocab_size: int = 1000):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.Sequential(
            *[TransformerBlock(dim) for _ in range(num_blocks)]
        )
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        h = self.blocks(h)
        h = self.ln_f(h)
        return self.head(h)


# ── Main ─────────────────────────────────────────────────────────────


def run_inference_example(args: argparse.Namespace) -> None:
    """Run the FSDP inference example with MagiCompiler."""

    # 1. Create model
    logger.info(f"Creating model (dim={args.dim}, blocks={args.blocks})...")
    model = GPTLikeModel(dim=args.dim, num_blocks=args.blocks)

    # 2. Optionally wrap with FSDP
    if args.fsdp:
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp.wrap import (
                transformer_auto_wrap_policy,
            )
            from torch.distributed.fsdp import ShardingStrategy

            torch.distributed.init_process_group(backend="nccl")
            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            model = model.cuda()

            model = FSDP(
                model,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                auto_wrap_policy=transformer_auto_wrap_policy(
                    TransformerBlock,
                ),
                device_id=local_rank,
            )
            logger.info("Model wrapped with FSDP.")
        except Exception as e:
            logger.warning(f"FSDP wrapping failed ({e}). Running without FSDP.")

    # 3. Create input
    batch_size, seq_len = args.batch_size, args.seq_len
    x = torch.randint(0, args.vocab_size, (batch_size, seq_len))
    if args.fsdp:
        x = x.cuda()

    logger.info(
        f"Input shape: {x.shape}, "
        f"Model parameters: {sum(p.numel() for p in model.parameters())}"
    )

    # 4. Compile with MagiCompiler
    from magicompiler import magicompile

    logger.info("Compiling model with MagiCompiler (FSDP full-graph capture)...")

    compiled_model = magicompile(
        model,
        mode="fsdp-inference",
        capture_full_graph=True,
        fuse_comm_computation=args.fuse_comm,
        optimization_level="level_2" if args.aggressive else "level_1",
        deterministic=args.deterministic,
    )

    # 5. Run inference
    logger.info("Running inference with MagiCompiler-optimized model...")
    with torch.no_grad():
        output = compiled_model(x)

    logger.info(f"Output shape: {output.shape}")

    # 6. Run a few more times to verify stability
    for i in range(3):
        x2 = torch.randint(0, args.vocab_size, (batch_size, seq_len))
        if args.fsdp:
            x2 = x2.cuda()
        output2 = compiled_model(x2)
        logger.info(f"  Run {i + 2}: output shape {output2.shape}")

    logger.info("✅ Inference example completed successfully!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MagiCompiler FSDP Inference Example"
    )
    parser.add_argument("--dim", type=int, default=64, help="Model dimension")
    parser.add_argument("--blocks", type=int, default=6, help="Number of transformer blocks")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence length")
    parser.add_argument("--vocab-size", type=int, default=1000, help="Vocabulary size")
    parser.add_argument("--fsdp", action="store_true", help="Wrap model with FSDP")
    parser.add_argument("--fuse-comm", action="store_true",
                        help="Fuse communication with computation")
    parser.add_argument("--aggressive", action="store_true",
                        help="Use aggressive optimization (level 2)")
    parser.add_argument("--deterministic", action="store_true",
                        help="Enforce deterministic execution")
    args = parser.parse_args()

    run_inference_example(args)


if __name__ == "__main__":
    main()
