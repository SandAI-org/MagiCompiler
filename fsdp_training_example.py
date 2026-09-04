"""
FSDP Training Example – MagiCompiler Full-Graph Capture (Phase 2).

This example demonstrates MagiCompiler's FSDP full-graph capture for training,
including forward + backward capture and auto-recompute for memory efficiency.

Usage:
    # Single GPU (unit test mode):
    python examples/fsdp_training_example.py

    # Multi-GPU (requires torchrun):
    torchrun --nproc_per_node=2 examples/fsdp_training_example.py
"""

import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fsdp_training_example")


class ResNetBlock(nn.Module):
    """A simple residual block."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = F.relu(self.linear1(x))
        h = self.linear2(h)
        return self.norm(F.relu(h + residual))


class TrainingModel(nn.Module):
    """A model with multiple residual blocks for training tests."""

    def __init__(self, dim: int = 64, num_blocks: int = 4, num_classes: int = 10):
        super().__init__()
        self.input_proj = nn.Linear(dim, dim)
        self.blocks = nn.Sequential(
            *[ResNetBlock(dim) for _ in range(num_blocks)]
        )
        self.output_proj = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.blocks(h)
        return self.output_proj(h)


def train_step(
    model: nn.Module,
    optimizer: optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    use_compiler: bool = False,
    compiled_fn=None,
) -> float:
    """Run a single training step, optionally using MagiCompiler."""
    optimizer.zero_grad()

    if use_compiler and compiled_fn is not None:
        output = compiled_fn(x)
    else:
        output = model(x)

    loss = F.cross_entropy(output, y)
    loss.backward()
    optimizer.step()

    return loss.item()


def run_training_example(args: argparse.Namespace) -> None:
    """Run the FSDP training example with MagiCompiler."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Create model and optimizer
    model = TrainingModel(
        dim=args.dim,
        num_blocks=args.blocks,
        num_classes=args.num_classes,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # 2. Optionally wrap with FSDP
    if args.fsdp:
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import ShardingStrategy
            torch.distributed.init_process_group(backend="nccl")
            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            model = model.cuda()

            model = FSDP(
                model,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                device_id=local_rank,
            )
            logger.info("Model wrapped with FSDP.")
        except Exception as e:
            logger.warning(f"FSDP wrapping failed ({e}).")

    # 3. Create training data
    x = torch.randn(args.batch_size, args.dim).to(device)
    y = torch.randint(0, args.num_classes, (args.batch_size,)).to(device)

    # 4. Optionally compile with MagiCompiler
    compiled_fn = None
    if args.compile:
        from magicompiler import magicompile

        logger.info("Compiling for training with MagiCompiler...")
        _, compiled_fn = magicompile(
            model,
            mode="fsdp-training",
            capture_full_graph=True,
            fuse_comm_computation=args.fuse_comm,
            auto_recompute=args.auto_recompute,
            optimization_level="level_2" if args.aggressive else "level_1",
        )

    # 5. Training loop
    logger.info("Starting training loop...")
    for step in range(args.steps):
        loss = train_step(
            model, optimizer, x, y,
            use_compiler=args.compile,
            compiled_fn=compiled_fn,
        )

        if step % 5 == 0 or step == args.steps - 1:
            logger.info(f"  Step {step + 1}/{args.steps}: loss = {loss:.4f}")

        # Generate new random data each step
        x = torch.randn(args.batch_size, args.dim).to(device)
        y = torch.randint(0, args.num_classes, (args.batch_size,)).to(device)

    logger.info("✅ Training example completed successfully!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MagiCompiler FSDP Training Example"
    )
    parser.add_argument("--dim", type=int, default=64, help="Model dimension")
    parser.add_argument("--blocks", type=int, default=4, help="Number of residual blocks")
    parser.add_argument("--num-classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--steps", type=int, default=20, help="Training steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--compile", action="store_true",
                        help="Use MagiCompiler compilation")
    parser.add_argument("--fsdp", action="store_true",
                        help="Wrap model with FSDP")
    parser.add_argument("--fuse-comm", action="store_true",
                        help="Fuse communication with computation")
    parser.add_argument("--auto-recompute", action="store_true",
                        help="Enable auto-recompute for memory efficiency")
    parser.add_argument("--aggressive", action="store_true",
                        help="Use aggressive optimization (level 2)")
    args = parser.parse_args()

    run_training_example(args)


if __name__ == "__main__":
    main()
