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

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from magi_compiler import magi_compile


@dataclass
class MLPConfig:
    """Configuration for the MLP module"""

    hidden_size: int
    intermediate_size: int
    params_dtype: torch.dtype = torch.bfloat16


@dataclass
class RMSNormConfig:
    """Configuration for the RMSNorm module"""

    hidden_size: int
    eps: float = 1e-6


class RMSNorm(nn.Module):
    """Simple RMSNorm implementation"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = x.to(self.weight.dtype) * self.weight
        return x.to(input_dtype)


class RawMLP(torch.nn.Module):
    """MLP module with traditional architecture (up-projection, activation, and down-projection).

    This is the uncompiled base class. Use ``MLP`` for the magi_compile-wrapped variant.
    """

    config: MLPConfig

    def __init__(self, config: MLPConfig):
        super().__init__()
        self.config = config
        self.pre_norm = RMSNorm(config.hidden_size)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, dtype=config.params_dtype)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False, dtype=config.params_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP module.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            output (torch.Tensor): Output tensor

        Shape:
            - x: (num_tokens, hidden_size)
            - output: (num_tokens, hidden_size)
        """
        x = self.pre_norm(x).to(torch.bfloat16)
        x = self.up_proj(x).to(torch.float32)
        x = F.silu(x).to(torch.bfloat16)
        x = self.down_proj(x).to(torch.float32)
        return x


@magi_compile(dynamic_arg_dims={"x": 0})
class MLP(RawMLP):
    """Compiled MLP module (magi_compile-wrapped ``RawMLP``)."""

    pass


class RawRMSNormModule(torch.nn.Module):
    """RMSNorm module for testing.

    This is the uncompiled base class. Use ``RMSNormModule`` for the magi_compile-wrapped variant.
    """

    config: RMSNormConfig

    def __init__(self, config: RMSNormConfig):
        super().__init__()
        self.config = config
        self.norm = RMSNorm(config.hidden_size, eps=config.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the RMSNorm module.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            output (torch.Tensor): Normalized output tensor

        Shape:
            - x: (num_tokens, hidden_size)
            - output: (num_tokens, hidden_size)
        """
        return self.norm(x)


@magi_compile(dynamic_arg_dims={"x": 0})
class RMSNormModule(RawRMSNormModule):
    """Compiled RMSNorm module (magi_compile-wrapped ``RawRMSNormModule``)."""

    pass


def create_rms_norm_model(config: RMSNormConfig, device: torch.device) -> RMSNormModule:
    """Create RMSNorm model

    Args:
        config: RMSNorm configuration
        device: Target device

    Returns:
        model: Created RMSNorm model
    """
    model = RMSNormModule(config).to(device)
    return model


def create_mlp_model(config: MLPConfig, device: torch.device) -> MLP:
    """Create MLP model

    Args:
        config: MLP configuration
        device: Target device

    Returns:
        model: Created MLP model
    """
    model = MLP(config).to(device)
    return model


def create_mlp_model_with_initial_params(config: MLPConfig, device: torch.device) -> tuple[MLP, list[torch.Tensor]]:
    """Create MLP model and return model with initial parameter snapshot

    Args:
        config: MLP configuration
        device: Target device

    Returns:
        model: Created MLP model
        initial_params: Initial snapshot of model parameters for verifying parameter updates
    """
    model = MLP(config).to(device)
    initial_params = [p.clone().detach() for p in model.parameters()]
    return model, initial_params


@dataclass
class TransformerConfig:
    """Configuration for the Transformer model"""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-6
    params_dtype: torch.dtype = torch.bfloat16


class Attention(nn.Module):
    """Multi-head attention module"""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False, dtype=config.params_dtype)
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, dtype=config.params_dtype
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, dtype=config.params_dtype
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False, dtype=config.params_dtype)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # GQA
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        flash_attn_out = torch.ops.aten._scaled_dot_product_flash_attention(
            q, k, v, dropout_p=0.0, is_causal=False, return_debug_mask=False
        )
        attn_output = flash_attn_out[0]

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        return self.o_proj(attn_output)


class TransformerMLP(nn.Module):
    """MLP module for Transformer"""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, dtype=config.params_dtype)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, dtype=config.params_dtype)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False, dtype=config.params_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


@magi_compile(dynamic_arg_dims={"x": 0})
class TransformerBlock(nn.Module):
    """A single Transformer block"""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = TransformerMLP(config)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x).to(torch.bfloat16)
        x = self.self_attn(x, attention_mask=attention_mask)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x).to(torch.bfloat16)
        x = self.mlp(x)
        x = residual + x
        return x


class Transformer(nn.Module):
    """A complete Transformer model"""

    config: TransformerConfig

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=config.params_dtype)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=config.params_dtype)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass of the Transformer model.

        Args:
            input_ids (torch.Tensor): Input token ids
            attention_mask (torch.Tensor): Attention mask

        Returns:
            output (torch.Tensor): Output logits

        Shape:
            - input_ids: (batch_size, seq_len)
            - attention_mask: (batch_size, 1, seq_len, seq_len)
            - output: (batch_size, seq_len, vocab_size)
        """
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        x = self.norm(x).to(torch.bfloat16)
        return self.lm_head(x)


def create_transformer_model(config: TransformerConfig, device: torch.device) -> Transformer:
    """Create Transformer model

    Args:
        config: Transformer configuration
        device: Target device

    Returns:
        model: Created Transformer model
    """
    model = Transformer(config).to(device)
    return model


def create_transformer_model_with_initial_params(
    config: TransformerConfig, device: torch.device
) -> tuple[Transformer, list[torch.Tensor]]:
    """Create Transformer model and return model with initial parameter snapshot

    Args:
        config: Transformer configuration
        device: Target device

    Returns:
        model: Created Transformer model
        initial_params: Initial snapshot of model parameters for verifying parameter updates
    """
    model = Transformer(config).to(device)
    initial_params = [p.clone().detach() for p in model.parameters()]
    return model, initial_params
