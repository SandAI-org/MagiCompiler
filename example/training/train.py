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

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import json
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from model import ModelArgs, Transformer, TransformerBlock
from tokenizer import ChatFormat, Tokenizer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import magi_compiler.utils.nvtx as nvtx


def setup_fsdp(model: nn.Module, device_id: int):
    """
    Wrap the given Llama3 model with PyTorch FSDP.
    We apply auto_wrap_policy to wrap each TransformerBlock individually.
    """
    llama_auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={TransformerBlock})

    fsdp_model = FSDP(
        model, auto_wrap_policy=llama_auto_wrap_policy, device_id=device_id, sync_module_states=True, use_orig_params=True
    )
    local_params = sum(p.numel() for p in fsdp_model.parameters())
    print(f"[Rank {device_id}] Local param count: {local_params:,}")
    return fsdp_model


class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
    ) -> "Llama":
        """
        Build a Llama instance by initializing and loading a model checkpoint.

        Args:
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.

        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory,
                or if the model parallel size does not match the number of checkpoint files.

        Note:
            This method initializes the distributed process group, sets the device to CUDA,
            and loads the pre-trained model and tokenizer.
        """
        assert 1 <= max_seq_len <= 8192, f"max_seq_len must be between 1 and 8192, got {max_seq_len}."
        assert os.path.isdir(ckpt_dir), f"Checkpoint directory '{ckpt_dir}' does not exist."
        assert os.path.isfile(tokenizer_path), f"Tokenizer file '{tokenizer_path}' does not exist."

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(seed)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params)
        tokenizer = Tokenizer(model_path=tokenizer_path)
        assert model_args.vocab_size == tokenizer.n_words
        if torch.cuda.is_bf16_supported():
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        else:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.formatter = ChatFormat(tokenizer)


def main():
    # Setup distributed environment
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        print("Not running in distributed mode. Set RANK and WORLD_SIZE to use FSDP.")
        # For demonstration purposes, we will mock the distributed setup if not provided.
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")

    # Initialize model parallel group (since model uses fairscale VocabParallelEmbedding)
    if not model_parallel_is_initialized():
        initialize_model_parallel(1)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    # Initialize a small config for testing
    config = ModelArgs(n_layers=10, max_batch_size=2, max_seq_len=1024)

    # Create Model
    if global_rank == 0:
        print(f"Initializing model on {world_size} devices...")
    model = Transformer(config).to(device)

    # Wrap with FSDP
    if global_rank == 0:
        print("Wrapping model with FSDP...")

    if torch.cuda.is_available():
        fsdp_model = setup_fsdp(model, device_id=local_rank)
    else:
        # For CPU testing, fallback to DDP or just standard model if FSDP CPU is not supported
        print(f"[Rank {global_rank}] CUDA not available. Running without FSDP for CPU fallback.")
        fsdp_model = model

    # Optimizer
    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-2)

    num_epochs = 5
    bsz, seq_len = config.max_batch_size, config.max_seq_len

    if global_rank == 0:
        print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # record start time
        start_time = time.time()

        # Dummy data for each epoch
        # Ensure different ranks get different data if needed, but here we just generate random
        torch.manual_seed(epoch * world_size + global_rank)
        input_ids = torch.randint(0, config.vocab_size, (bsz, seq_len), device=device)
        labels = torch.randint(0, config.vocab_size, (bsz, seq_len), device=device)

        optimizer.zero_grad()

        nvtx.switch_profile(epoch, 3, 4)
        # Forward pass
        logits = fsdp_model(input_ids, start_pos=0)

        # Loss
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))

        # Backward pass
        loss.backward()

        optimizer.step()

        # record end time
        end_time = time.time()

        if global_rank == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs} | Loss: {loss.item():.4f} | Time taken: {end_time - start_time:.4f} seconds"
            )

    if global_rank == 0:
        print("Training done!")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
