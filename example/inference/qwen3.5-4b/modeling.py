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

from __future__ import annotations

import copy
import itertools
import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

from magi_compiler import magi_register_custom_op

GridTHW = torch.Tensor | tuple[int, int, int]
logger = logging.getLogger(__name__)

# Official Qwen3.5-4B architecture used when no checkpoint (or only a config) is provided.
DEFAULT_QWEN35_4B_CONFIG: dict[str, Any] = {
    "image_token_id": 248056,
    "video_token_id": 248057,
    "vision_start_token_id": 248053,
    "vision_end_token_id": 248054,
    "tie_word_embeddings": True,
    "text_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "head_dim": 256,
        "hidden_act": "silu",
        "hidden_size": 2560,
        "intermediate_size": 9216,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 8,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_value_head_dim": 128,
        "max_position_embeddings": 262144,
        "num_attention_heads": 16,
        "num_hidden_layers": 32,
        "num_key_value_heads": 4,
        "pad_token_id": 0,
        "rms_norm_eps": 1e-6,
        "vocab_size": 248320,
        "rope_parameters": {
            "mrope_section": [11, 11, 10],
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
        },
    },
    "vision_config": {
        "depth": 24,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1024,
        "in_channels": 3,
        "intermediate_size": 4096,
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 2560,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
}


def _ns(value: dict[str, Any]) -> SimpleNamespace:
    out = SimpleNamespace(**value)
    for key, item in list(value.items()):
        if isinstance(item, dict):
            setattr(out, key, _ns(item))
    return out


def _normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    config.setdefault("video_token_id", config.get("video_token_id", 248057))
    config.setdefault("image_token_id", config.get("image_token_id", 248056))
    config.setdefault("vision_start_token_id", config.get("vision_start_token_id", 248053))
    config.setdefault("vision_end_token_id", config.get("vision_end_token_id", 248054))
    config["text_config"].setdefault("pad_token_id", config["text_config"].get("pad_token_id", 0))
    return config


def load_qwen35_config(model_path: str | Path | None = None) -> dict[str, Any]:
    if model_path is None:
        return _normalize_config(copy.deepcopy(DEFAULT_QWEN35_4B_CONFIG))
    config_path = Path(model_path) / "config.json"
    return _normalize_config(json.loads(config_path.read_text()))


def has_sharded_weights(model_path: str | Path | None) -> bool:
    return model_path is not None and (Path(model_path) / "model.safetensors.index.json").is_file()


def _act(name: str, x: torch.Tensor) -> torch.Tensor:
    if name == "silu":
        return F.silu(x)
    if name == "gelu_pytorch_tanh":
        return F.gelu(x, approximate="tanh")
    if name == "gelu":
        return F.gelu(x)
    raise ValueError(f"Unsupported activation: {name}")


class Qwen35RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (output * (1.0 + self.weight.float())).to(dtype=x.dtype)


class Qwen35RMSNormGated(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float() * torch.rsqrt(hidden_states.float().pow(2).mean(-1, keepdim=True) + self.eps)
        hidden_states = hidden_states * self.weight.float()
        if gate is not None:
            hidden_states = hidden_states * F.silu(gate.float())
        return hidden_states.to(dtype=input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def torch_causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    if activation == "silu":
        out = F.silu(out)
    return out[:, :, -seq_len:].to(hidden_states.dtype)


def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    return core_attn_out.transpose(1, 2).contiguous().to(initial_dtype), last_recurrent_state


def torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale
    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim, dtype=value.dtype, device=value.device)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)
        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    return core_attn_out.transpose(1, 2).contiguous().to(initial_dtype), last_recurrent_state


class Qwen35Cache:
    def __init__(self, layer_types: list[str], max_seq_len: int | None = None):
        self.layer_types = layer_types
        self.max_seq_len = max_seq_len
        self.full_key_states: list[torch.Tensor | None] = [None] * len(layer_types)
        self.full_value_states: list[torch.Tensor | None] = [None] * len(layer_types)
        self.linear_conv_states: list[torch.Tensor | None] = [None] * len(layer_types)
        self.linear_recurrent_states: list[torch.Tensor | None] = [None] * len(layer_types)
        self.seq_len = 0
        self.rope_deltas: torch.Tensor | None = None

    def reset(self) -> None:
        self.seq_len = 0
        self.rope_deltas = None
        if self.max_seq_len is None:
            self.full_key_states = [None] * len(self.layer_types)
            self.full_value_states = [None] * len(self.layer_types)
            self.linear_conv_states = [None] * len(self.layer_types)
            self.linear_recurrent_states = [None] * len(self.layer_types)

    def get_seq_length(self) -> int:
        return self.seq_len

    def has_previous_state(self, layer_idx: int) -> bool:
        return self.seq_len > 0 and (
            self.full_key_states[layer_idx] is not None or self.linear_conv_states[layer_idx] is not None
        )

    def update_full_attention(
        self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.max_seq_len is None:
            if self.full_key_states[layer_idx] is None:
                self.full_key_states[layer_idx] = key_states
                self.full_value_states[layer_idx] = value_states
            else:
                self.full_key_states[layer_idx] = torch.cat([self.full_key_states[layer_idx], key_states], dim=2)
                self.full_value_states[layer_idx] = torch.cat([self.full_value_states[layer_idx], value_states], dim=2)
            return self.full_key_states[layer_idx], self.full_value_states[layer_idx]

        full_key_states = self.full_key_states[layer_idx]
        full_value_states = self.full_value_states[layer_idx]
        start = self.seq_len
        end = start + key_states.shape[2]
        if end > self.max_seq_len:
            raise ValueError(f"Cache capacity {self.max_seq_len} is too small for sequence length {end}.")
        if full_key_states is None or full_value_states is None:
            full_key_states = torch.empty(
                (key_states.shape[0], key_states.shape[1], self.max_seq_len, key_states.shape[3]),
                dtype=key_states.dtype,
                device=key_states.device,
            )
            full_value_states = torch.empty(
                (value_states.shape[0], value_states.shape[1], self.max_seq_len, value_states.shape[3]),
                dtype=value_states.dtype,
                device=value_states.device,
            )
            self.full_key_states[layer_idx] = full_key_states
            self.full_value_states[layer_idx] = full_value_states
        full_key_states[:, :, start:end, :].copy_(key_states)
        full_value_states[:, :, start:end, :].copy_(value_states)
        return full_key_states[:, :, :end, :], full_value_states[:, :, :end, :]

    def update_linear_conv_state(self, layer_idx: int, conv_state: torch.Tensor) -> None:
        cached_state = self.linear_conv_states[layer_idx]
        if cached_state is not None and cached_state.shape == conv_state.shape:
            cached_state.copy_(conv_state)
        else:
            self.linear_conv_states[layer_idx] = conv_state

    def update_linear_recurrent_state(self, layer_idx: int, recurrent_state: torch.Tensor | None) -> None:
        if recurrent_state is None:
            self.linear_recurrent_states[layer_idx] = None
            return
        cached_state = self.linear_recurrent_states[layer_idx]
        if cached_state is not None and cached_state.shape == recurrent_state.shape:
            cached_state.copy_(recurrent_state)
        else:
            self.linear_recurrent_states[layer_idx] = recurrent_state


class Qwen35TextRotaryEmbedding(nn.Module):
    def __init__(self, config: SimpleNamespace, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_parameters = (
            vars(config.rope_parameters) if isinstance(config.rope_parameters, SimpleNamespace) else config.rope_parameters
        )
        self.rope_type = self.rope_parameters["rope_type"]
        self.mrope_section = self.rope_parameters.get("mrope_section", [11, 11, 10])
        self.attention_scaling = 1.0
        self.reset_inv_freq(device=device)

    def reset_inv_freq(self, device=None) -> None:
        base = self.rope_parameters["rope_theta"]
        partial_rotary_factor = self.rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(self.config, "head_dim", None) or self.config.hidden_size // self.config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, :, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        freqs_t = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        emb = torch.cat((freqs_t, freqs_t), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen35VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.reset_inv_freq()

    def reset_inv_freq(self, device=None) -> None:
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float, device=device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return (position_ids.unsqueeze(-1) * self.inv_freq).flatten(1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


def get_vision_position_ids(grid_thw: torch.Tensor, spatial_merge_size: int) -> torch.Tensor:
    position_ids = []
    for grid in grid_thw.tolist():
        t, h, w = grid
        h = h // spatial_merge_size
        w = w // spatial_merge_size
        temporal = torch.arange(t)
        height = torch.arange(h)
        width = torch.arange(w)
        t_grid, h_grid, w_grid = torch.meshgrid(temporal, height, width, indexing="ij")
        position_ids.append(torch.stack([t_grid, h_grid, w_grid], dim=0).reshape(3, -1))
    return torch.cat(position_ids, dim=1)


def get_vision_bilinear_indices_and_weights(
    grid_thw: GridTHW, num_grid_per_side: int, spatial_merge_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    t, h, w = _single_grid_tuple(grid_thw)
    if h == 0 or w == 0:
        raise ValueError(f"Invalid vision grid: {(t, h, w)}")
    hh = torch.linspace(0, num_grid_per_side - 1, h).round().to(torch.long)
    ww = torch.linspace(0, num_grid_per_side - 1, w).round().to(torch.long)
    h_grid, w_grid = torch.meshgrid(hh, ww, indexing="ij")
    idx = (h_grid * num_grid_per_side + w_grid).reshape(-1).repeat(t)
    return idx.unsqueeze(0), torch.ones_like(idx, dtype=torch.float).unsqueeze(0)


def _single_grid_tuple(grid_thw: GridTHW) -> tuple[int, int, int]:
    if isinstance(grid_thw, torch.Tensor):
        if grid_thw.shape[0] != 1:
            raise ValueError("This example expects one synthetic image per forward call.")
        return tuple(int(x) for x in grid_thw[0].tolist())
    if len(grid_thw) != 3:
        raise ValueError(f"Invalid vision grid: {grid_thw}")
    return tuple(int(x) for x in grid_thw)


def get_vision_cu_seqlens(grid_thw: GridTHW) -> torch.Tensor:
    if isinstance(grid_thw, torch.Tensor):
        lengths = grid_thw.prod(dim=-1).to(torch.int32)
    else:
        lengths = torch.tensor([math.prod(grid_thw)], dtype=torch.int32)
    cu = torch.zeros(lengths.numel() + 1, dtype=torch.int32)
    cu[1:] = lengths.cumsum(dim=0)
    return cu


@magi_register_custom_op("qwen35_4b::chunk_boundary", is_subgraph_boundary=True)
def qwen35_chunk_boundary(x: torch.Tensor) -> torch.Tensor:
    # ponytail: split the giant Qwen forward into small piecewise chunks so compile stays tractable.
    return x.clone()


class Qwen35VisionMLP(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = config.hidden_act

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(_act(self.act_fn, self.linear_fc1(hidden_state)))


class Qwen35VisionPatchEmbed(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class Qwen35VisionPatchMerger(nn.Module):
    def __init__(self, config: SimpleNamespace, use_postshuffle_norm: bool = False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class Qwen35VisionAttention(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)
        # ponytail: treat the packed vision tokens as one segment in this example; split by cu_seqlens if
        # multi-image isolation matters.
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)


class Qwen35VisionBlock(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen35VisionAttention(config=config)
        self.mlp = Qwen35VisionMLP(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens, position_embeddings, **kwargs)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen35VisionModel(nn.Module):
    input_modalities = ("image", "video")

    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.patch_embed = Qwen35VisionPatchEmbed(config=config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen35VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList([Qwen35VisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen35VisionPatchMerger(config=config, use_postshuffle_norm=False)

    def reset_buffers(self, device=None) -> None:
        self.rotary_pos_emb.reset_inv_freq(device=device)

    def forward(self, hidden_states: torch.Tensor, grid_thw: GridTHW, **kwargs) -> torch.Tensor:
        bilinear_indices, bilinear_weights = get_vision_bilinear_indices_and_weights(
            grid_thw, num_grid_per_side=self.num_grid_per_side, spatial_merge_size=self.config.spatial_merge_size
        )
        t, h, w = _single_grid_tuple(grid_thw)
        height = torch.arange(h, device=hidden_states.device)
        width = torch.arange(w, device=hidden_states.device)
        h_grid, w_grid = torch.meshgrid(height, width, indexing="ij")
        position_ids = torch.stack([h_grid, w_grid], dim=-1).reshape(-1, 2).repeat(t, 1)
        cu_seqlens = get_vision_cu_seqlens(grid_thw).to(hidden_states.device)
        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = (
            self.pos_embed(bilinear_indices.to(hidden_states.device)) * bilinear_weights[:, :, None].to(hidden_states.device)
        ).sum(0)
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)
        rotary_pos_emb = self.rotary_pos_emb(position_ids.to(hidden_states.device))
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        for block_idx, blk in enumerate(self.blocks, start=1):
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings, **kwargs)
            if block_idx < len(self.blocks):
                hidden_states = qwen35_chunk_boundary(hidden_states)
        return self.merger(hidden_states)


class Qwen35TextDecoderLayer(nn.Module):
    def __init__(self, config: SimpleNamespace, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.block_type = config.layer_types[layer_idx]
        if self.block_type == "linear_attention":
            self.linear_attn = Qwen35GatedDeltaNet(config, layer_idx)
        elif self.block_type == "full_attention":
            self.self_attn = Qwen35Attention(config, layer_idx)
        else:
            raise ValueError(f"Unsupported layer type: {self.block_type}")
        self.mlp = Qwen35MLP(config, config.intermediate_size)
        self.input_layernorm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: Qwen35Cache | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.block_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states=hidden_states, cache_params=past_key_values)
        else:
            hidden_states = self.self_attn(
                hidden_states=hidden_states, position_embeddings=position_embeddings, past_key_values=past_key_values
            )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen35GatedDeltaNet(nn.Module):
    def __init__(self, config: SimpleNamespace, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.act = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        a = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(a))
        self.norm = Qwen35RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

    def forward(self, hidden_states: torch.Tensor, cache_params: Qwen35Cache | None = None, **kwargs) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx)
        if use_precomputed_states:
            conv_state = cache_params.linear_conv_states[self.layer_idx]
            recurrent_state = cache_params.linear_recurrent_states[self.layer_idx]
        else:
            conv_state = None
            recurrent_state = None

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if use_precomputed_states and seq_len == 1:
            mixed_qkv = torch_causal_conv1d_update(
                mixed_qkv, conv_state, self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation
            )
        else:
            if use_precomputed_states:
                mixed_qkv = torch.cat([conv_state, mixed_qkv], dim=-1)
            if cache_params is not None:
                new_conv_state = mixed_qkv[:, :, -self.conv_kernel_size :].contiguous()
                if new_conv_state.shape[-1] < self.conv_kernel_size:
                    new_conv_state = F.pad(new_conv_state, (self.conv_kernel_size - new_conv_state.shape[-1], 0))
                cache_params.update_linear_conv_state(self.layer_idx, new_conv_state)
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, : mixed_qkv.shape[-1]])
            if use_precomputed_states:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        if use_precomputed_states and seq_len == 1:
            core_attn_out, last_recurrent_state = torch_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state if use_precomputed_states else None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
        if cache_params is not None:
            cache_params.update_linear_recurrent_state(self.layer_idx, last_recurrent_state)
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)


class Qwen35Attention(nn.Module):
    def __init__(self, config: SimpleNamespace, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim * 2, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Qwen35RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen35RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: Qwen35Cache | None = None,
        **kwargs,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states, gate = torch.chunk(self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            key_states, value_states = past_key_values.update_full_attention(self.layer_idx, key_states, value_states)
        past_len = 0 if past_key_values is None else key_states.shape[2] - query_states.shape[2]
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        if query_states.shape[2] == 1 and past_len == 0:
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=None,
                dropout_p=0.0 if not self.training else self.attention_dropout,
                is_causal=True,
            )
        else:
            q_len = query_states.shape[2]
            k_len = key_states.shape[2]
            mask = torch.full((q_len, k_len), float("-inf"), device=query_states.device, dtype=query_states.dtype)
            allowed = torch.arange(k_len, device=query_states.device)[None, :] <= (
                past_len + torch.arange(q_len, device=query_states.device)[:, None]
            )
            mask = mask.masked_fill(allowed, 0.0).view(1, 1, q_len, k_len)
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=mask,
                dropout_p=0.0 if not self.training else self.attention_dropout,
                is_causal=False,
            )
        attn_output = attn_output.transpose(1, 2).contiguous().view(*input_shape, -1)
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output)


class Qwen35MLP(nn.Module):
    def __init__(self, config: SimpleNamespace, intermediate_size: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = config.hidden_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(_act(self.act_fn, self.gate_proj(x)) * self.up_proj(x))


class Qwen35TextModel(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen35TextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen35TextRotaryEmbedding(config=config)

    def reset_buffers(self, device=None) -> None:
        self.rotary_emb.reset_inv_freq(device=device)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        past_key_values: Qwen35Cache | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer_idx, decoder_layer in enumerate(self.layers, start=1):
            hidden_states = decoder_layer(
                hidden_states, position_embeddings=position_embeddings, past_key_values=past_key_values
            )
            if layer_idx < len(self.layers):
                hidden_states = qwen35_chunk_boundary(hidden_states)
        hidden_states = self.norm(hidden_states)
        if use_cache and past_key_values is not None:
            past_key_values.seq_len += inputs_embeds.shape[1]
        return hidden_states


class Qwen35Model(nn.Module):
    base_model_prefix = "model"

    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.config = config
        self.visual = Qwen35VisionModel(config.vision_config)
        self.language_model = Qwen35TextModel(config.text_config)
        self.rope_deltas: torch.Tensor | None = None

    def reset_buffers(self, device=None) -> None:
        self.visual.reset_buffers(device=device)
        self.language_model.reset_buffers(device=device)
        self.rope_deltas = None

    def get_image_features(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor | None = None, **kwargs
    ) -> tuple[torch.Tensor, ...]:
        pixel_values = pixel_values.type(self.visual.patch_embed.proj.weight.dtype)
        vision_output = self.visual(pixel_values, grid_thw=image_grid_thw, **kwargs)
        image_embeds = vision_output
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        return tuple(torch.split(image_embeds, split_sizes))

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor | None = None,
        video_features: torch.FloatTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        special_image_mask = input_ids == self.config.image_token_id
        special_video_mask = input_ids == self.config.video_token_id
        n_image_tokens = special_image_mask.sum()
        if image_features is not None:
            if n_image_tokens * inputs_embeds.shape[-1] != image_features.numel():
                raise ValueError(
                    f"Image features and image tokens do not match, tokens: {n_image_tokens}, features: {image_features.shape[0]}"
                )
        n_video_tokens = special_video_mask.sum()
        if video_features is not None:
            if n_video_tokens * inputs_embeds.shape[-1] != video_features.numel():
                raise ValueError(
                    f"Video features and video tokens do not match, tokens: {n_video_tokens}, features: {video_features.shape[0]}"
                )
        return special_image_mask.unsqueeze(-1).to(inputs_embeds.device), special_video_mask.unsqueeze(-1).to(
            inputs_embeds.device
        )

    def get_vision_position_ids(
        self,
        start_position: int,
        grid_thw: list[int, int, int] | torch.Tensor,
        temp_merge_size: int = 1,
        spatial_merge_size: int = 1,
        time_interval: int = 1,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        llm_grid_t, llm_grid_h, llm_grid_w = (
            grid_thw[0].item() // temp_merge_size,
            grid_thw[1].item() // spatial_merge_size,
            grid_thw[2].item() // spatial_merge_size,
        )
        position_temporal = torch.arange(llm_grid_t, device=device) * time_interval
        position_height = torch.arange(llm_grid_h, device=device) + start_position
        position_width = torch.arange(llm_grid_w, device=device) + start_position
        t_grid, h_grid, w_grid = torch.meshgrid(position_temporal, position_height, position_width, indexing="ij")
        vision_position_ids = torch.stack([t_grid, h_grid, w_grid], dim=0).reshape(3, -1)
        vision_position_ids[0] += start_position
        return vision_position_ids

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        mm_token_type_ids: torch.IntTensor,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        mrope_position_deltas = []
        position_ids = torch.zeros(3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device)
        grid_iters = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(video_grid_thw) if video_grid_thw is not None else None,
        }
        for batch_idx, current_input_ids in enumerate(input_ids):
            input_token_type = mm_token_type_ids[batch_idx]
            if attention_mask is not None:
                current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]
                input_token_type = input_token_type[attention_mask[batch_idx].bool()]
            input_type_group = []
            for key, group in itertools.groupby(enumerate(input_token_type.tolist()), lambda x: x[1]):
                group = list(group)
                start_index = group[0][0]
                end_index = group[-1][0] + 1
                input_type_group.append((key, start_index, end_index))
            current_pos = 0
            llm_pos_ids_list = []
            for modality_type, start_idx, end_idx in input_type_group:
                if modality_type == 0:
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos
                    )
                    current_pos += text_len
                else:
                    grid_thw = next(grid_iters[modality_type])
                    vision_position_ids = self.get_vision_position_ids(
                        current_pos, grid_thw, 1, spatial_merge_size, device=input_ids.device
                    )
                    llm_pos_ids_list.append(vision_position_ids)
                    current_pos += int(max(grid_thw[1].item(), grid_thw[2].item()) // spatial_merge_size)
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            if attention_mask is not None:
                position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions.to(position_ids.device)
            else:
                position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas

    def compute_3d_position_ids(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Qwen35Cache | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
    ) -> torch.Tensor | None:
        past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()
        has_multimodal = image_grid_thw is not None or video_grid_thw is not None
        if has_multimodal and mm_token_type_ids is None and input_ids is not None:
            raise ValueError("Multimodal data requires mm_token_type_ids.")
        can_compute_mrope = input_ids is not None and mm_token_type_ids is not None and has_multimodal
        if can_compute_mrope and (self.rope_deltas is None or past_key_values_length == 0):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                mm_token_type_ids=mm_token_type_ids,
            )
            self.rope_deltas = rope_deltas
        elif self.rope_deltas is not None and (past_key_values_length > 0 or input_ids is None):
            batch_size, seq_length, _ = inputs_embeds.shape
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids = position_ids.masked_fill(attention_mask == 0, 0)
                position_ids = position_ids.view(1, batch_size, -1).repeat(3, 1, 1).to(inputs_embeds.device)
            else:
                position_ids = torch.arange(past_key_values_length, past_key_values_length + seq_length)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1).to(inputs_embeds.device)
            delta = self.rope_deltas.repeat_interleave(batch_size // self.rope_deltas.shape[0], dim=0)
            position_ids = position_ids + delta.to(device=inputs_embeds.device)
        else:
            position_ids = None
        return position_ids

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Qwen35Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.language_model.embed_tokens(input_ids)
        if pixel_values is not None:
            image_outputs = self.get_image_features(pixel_values, image_grid_thw, **kwargs)
            image_embeds = torch.cat(image_outputs, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[image_mask.squeeze(-1)] = image_embeds
        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )
        return self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )


class Qwen35ForConditionalGeneration(nn.Module):
    _keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]

    def __init__(
        self,
        model_path: str | Path | None = None,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
        load_weights: bool | None = None,
    ):
        super().__init__()
        self.model_path = Path(model_path) if model_path is not None else None
        self.dtype = dtype
        self.device = torch.device(device)
        self.config = _ns(load_qwen35_config(self.model_path))
        if load_weights is None:
            load_weights = has_sharded_weights(self.model_path)
        if load_weights and self.model_path is None:
            raise ValueError("load_weights=True requires a checkpoint directory.")
        if load_weights:
            self._init_modules(torch.device("meta"))
            self._load_sharded_weights()
        else:
            logger.warning("Using randomly initialized Qwen3.5-4B weights.")
            # Materialize on the target device/dtype so random init does not peak at fp32.
            prev_dtype = torch.get_default_dtype()
            torch.set_default_dtype(self.dtype)
            try:
                self._init_modules(self.device)
            finally:
                torch.set_default_dtype(prev_dtype)
        self._tie_weights()
        self.model.reset_buffers(device=self.device)
        self.eval().requires_grad_(False)
        self.reset_cache()

    def _init_modules(self, device: torch.device) -> None:
        with torch.device(device):
            self.model = Qwen35Model(self.config)
            self.lm_head = nn.Linear(self.config.text_config.hidden_size, self.config.text_config.vocab_size, bias=False)

    def _expected_keys(self) -> set[str]:
        keys = set(self.state_dict().keys())
        keys.discard("lm_head.weight")
        return keys

    def _load_sharded_weights(self) -> None:
        index_path = self.model_path / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        weight_map: dict[str, str] = index["weight_map"]
        expected = self._expected_keys()
        loaded: set[str] = set()
        unexpected: list[str] = []
        for shard_name in sorted(set(weight_map.values())):
            shard_path = self.model_path / shard_name
            shard = load_file(shard_path, device=str(self.device))
            shard = {k: v for k, v in shard.items() if not k.startswith("mtp.")}
            bad = set(shard) - expected
            if bad:
                unexpected.extend(sorted(bad))
            self.load_state_dict(shard, strict=False, assign=True)
            loaded.update(shard)
        missing = sorted(expected - loaded)
        if unexpected:
            raise RuntimeError(f"Unexpected weights in checkpoint: {unexpected[:10]}")
        if missing:
            raise RuntimeError(f"Missing weights in checkpoint: {missing[:10]}")

    def _tie_weights(self) -> None:
        self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def reset_cache(self, max_seq_len: int | None = None) -> None:
        if hasattr(self, "cache") and self.cache is not None and self.cache.max_seq_len == max_seq_len:
            self.cache.reset()
        else:
            self.cache = Qwen35Cache(self.config.text_config.layer_types, max_seq_len=max_seq_len)
        self.model.rope_deltas = None

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        use_cache: bool | None = True,
        logits_to_keep: int = 1,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            past_key_values=self.cache if use_cache else None,
            use_cache=use_cache,
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return self.lm_head(hidden_states[:, slice_indices, :])
