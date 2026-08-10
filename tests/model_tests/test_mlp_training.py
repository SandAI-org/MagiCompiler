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

import pytest
import torch
import torch.nn as nn

from magi_compiler.utils import add_nvtx_event
from tests.model_definition import (
    MLP,
    MLPConfig,
    Transformer,
    TransformerConfig,
    create_mlp_model_with_initial_params,
    create_transformer_model_with_initial_params,
)
from tests.utils import CleanupCacheContext, enable_remote_debug


def train_mlp_model(
    model: MLP,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_tokens: int,
    hidden_size: int,
    num_epochs: int,
    batches_per_epoch: int,
    gradient_accumulation_steps: int = 1,
) -> list[float]:
    """Execute training loop for MLP model (supports gradient accumulation)

    Args:
        model: MLP model to train
        optimizer: Optimizer
        device: Training device
        num_tokens: Number of tokens per batch
        hidden_size: Hidden layer dimension
        num_epochs: Number of training epochs
        batches_per_epoch: Number of batches per epoch
        gradient_accumulation_steps: Gradient accumulation steps, default is 1 (no accumulation)

    Returns:
        epoch_losses: List of average losses per epoch
    """
    epoch_losses = []

    print(f"Starting training: {num_epochs} epochs, {batches_per_epoch} batches per epoch")
    if gradient_accumulation_steps > 1:
        print(f"Using gradient accumulation, accumulation steps: {gradient_accumulation_steps}")

    for epoch in range(num_epochs):
        epoch_loss_sum = 0.0

        for batch_idx in range(batches_per_epoch):
            # Zero gradients at the start of each accumulation cycle
            if batch_idx % gradient_accumulation_steps == 0:
                optimizer.zero_grad()

            # Generate random input and target data
            input_tensor = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
            target_tensor = torch.ones(num_tokens, hidden_size, device=device, dtype=torch.float32)

            # Forward pass
            output = model(input_tensor)

            # Compute loss, divided by accumulation steps to maintain effective batch size consistency
            loss = nn.functional.mse_loss(output, target_tensor) / gradient_accumulation_steps

            # Backward pass (gradients are automatically accumulated)
            loss.backward()

            # Accumulate loss for logging (multiply by accumulation steps to restore original value)
            epoch_loss_sum += loss.item() * gradient_accumulation_steps

            # Update parameters after accumulating gradient_accumulation_steps batches
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()

        # Handle the last incomplete accumulation batch
        if batches_per_epoch % gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss_sum / batches_per_epoch
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.6f}")

    print("Training completed!")
    return epoch_losses


def train_transformer_model(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    num_epochs: int,
    batches_per_epoch: int,
    gradient_accumulation_steps: int = 1,
) -> list[float]:
    """Execute training loop for Transformer model (supports gradient accumulation)

    Args:
        model: Transformer model to train
        optimizer: Optimizer
        device: Training device
        batch_size: Number of sequences per batch
        seq_len: Length of each sequence
        vocab_size: Vocabulary size
        num_epochs: Number of training epochs
        batches_per_epoch: Number of batches per epoch
        gradient_accumulation_steps: Gradient accumulation steps, default is 1 (no accumulation)

    Returns:
        epoch_losses: List of average losses per epoch
    """
    epoch_losses = []

    print(f"Starting Transformer training: {num_epochs} epochs, {batches_per_epoch} batches per epoch")
    if gradient_accumulation_steps > 1:
        print(f"Using gradient accumulation, accumulation steps: {gradient_accumulation_steps}")

    for epoch in range(num_epochs):
        epoch_loss_sum = 0.0

        for batch_idx in range(batches_per_epoch):
            # Zero gradients at the start of each accumulation cycle
            if batch_idx % gradient_accumulation_steps == 0:
                optimizer.zero_grad()

            # Generate random input and target data
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)
            target_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)

            # Forward pass
            with add_nvtx_event("transformer_forward"):
                output = model(input_ids)

            # Compute loss, divided by accumulation steps to maintain effective batch size consistency
            loss = nn.functional.cross_entropy(output.view(-1, vocab_size), target_ids.view(-1)) / gradient_accumulation_steps

            # Backward pass (gradients are automatically accumulated)
            with add_nvtx_event("transformer_backward"):
                loss.backward()

            # Accumulate loss for logging (multiply by accumulation steps to restore original value)
            epoch_loss_sum += loss.item() * gradient_accumulation_steps

            # Update parameters after accumulating gradient_accumulation_steps batches
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()

        # Handle the last incomplete accumulation batch
        if batches_per_epoch % gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss_sum / batches_per_epoch
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.6f}")

    print("Training completed!")
    return epoch_losses


def verify_model_parameters_updated(
    initial_params: list[torch.Tensor], current_params: list[torch.Tensor], tolerance: float = 1e-6
) -> bool:
    """Verify whether model parameters have been updated after training

    Args:
        initial_params: Parameter snapshot before training
        current_params: Current parameters after training
        tolerance: Tolerance for determining if parameters are the same

    Returns:
        True if parameters have been updated, False otherwise
    """
    for initial_param, current_param in zip(initial_params, current_params):
        if not torch.allclose(initial_param, current_param, atol=tolerance):
            return True
    return False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available, skipping test")
def test_mlp_training_with_magi_compiler():
    """Test MLP training with magi_compiler in training scenario"""

    # Set device
    device = torch.device("cuda")

    # Create MLP configuration
    mlp_config = MLPConfig(hidden_size=8, intermediate_size=32, params_dtype=torch.bfloat16)

    # Create model and save initial parameters
    model, initial_params = create_mlp_model_with_initial_params(mlp_config, device)

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Training parameters
    num_tokens = 8
    hidden_size = mlp_config.hidden_size
    num_epochs = 2
    batches_per_epoch = 2

    # Execute training
    epoch_losses = train_mlp_model(
        model=model,
        optimizer=optimizer,
        device=device,
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        num_epochs=num_epochs,
        batches_per_epoch=batches_per_epoch,
    )

    # Verify model parameters have been updated
    params_updated = verify_model_parameters_updated(initial_params=initial_params, current_params=list(model.parameters()))

    assert params_updated, "Model parameters should change after training"

    print("Test passed: Model successfully completed multiple training epochs, parameters have been updated")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available, skipping test")
def test_transformer_training_with_magi_compiler():
    """Test Transformer training with magi_compiler in training scenario"""

    # Set device
    device = torch.device("cuda")

    # Tiny config — just enough to exercise the magi_compile training path.
    transformer_config = TransformerConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        params_dtype=torch.bfloat16,
    )

    # Create model and save initial parameters
    model, initial_params = create_transformer_model_with_initial_params(transformer_config, device)

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Training parameters
    batch_size = 2
    seq_len = 32
    vocab_size = transformer_config.vocab_size
    num_epochs = 2
    batches_per_epoch = 2

    # Execute training
    epoch_losses = train_transformer_model(
        model=model,
        optimizer=optimizer,
        device=device,
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        num_epochs=num_epochs,
        batches_per_epoch=batches_per_epoch,
    )

    # Verify model parameters have been updated
    params_updated = verify_model_parameters_updated(initial_params=initial_params, current_params=list(model.parameters()))

    assert params_updated, "Transformer model parameters should change after training"

    print("Test passed: Transformer model successfully completed multiple training epochs, parameters have been updated")


if __name__ == "__main__":
    # Usage:
    # ENABLE_REMOTE_DEBUG=true MAGI_ENABLE_FX_GRAPH_VIZ=true TORCH_LOGS=aot CUDA_VISIBLE_DEVICES=1 python pkgs/MagiCompiler/tests/test_mlp_training.py
    with CleanupCacheContext():
        enable_remote_debug()
        test_mlp_training_with_magi_compiler()
        test_transformer_training_with_magi_compiler()
