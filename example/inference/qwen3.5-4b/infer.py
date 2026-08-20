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

import math
import os
import time
from pathlib import Path

import torch
from modeling import Qwen35ForConditionalGeneration
from torch import nn

import magi_compiler.utils.nvtx as nvtx
from magi_compiler import magi_compile

MODEL_PATH = os.environ.get("MODEL_PATH")
SKIP_LOAD_MODEL = os.environ.get("SKIP_LOAD_MODEL", "false").lower() in {"1", "true", "yes"}
MODE = os.environ.get("MODE", "all")
SEQ_LEN = int(os.environ.get("SEQ_LEN", "128"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
IMAGE_GRID = tuple(int(x) for x in os.environ.get("IMAGE_GRID", "1,2,2").split(","))
PROFILE_CNT = int(os.environ.get("PROFILE_CNT", "3"))
DTYPE = torch.bfloat16
MODES = ("text_prefill", "text_decode", "image_prefill", "image_decode")
IMAGE_TOKEN_START = 2


def entrypoint_dynamic_arg_dims(mode: str) -> dict[str, int | list[int]]:
    if mode == "text_prefill":
        return {"input_ids": []}
    if mode == "text_decode":
        return {"input_ids": []}
    if mode == "image_prefill":
        return {"input_ids": [], "mm_token_type_ids": [], "pixel_values": [], "image_grid_thw": [], "position_ids": []}
    if mode == "image_decode":
        return {"input_ids": [], "position_ids": []}
    raise ValueError(f"Unsupported entrypoint mode: {mode!r}")


def entrypoint_model_tag(mode: str) -> str:
    if mode.startswith("image_"):
        grid = "x".join(str(x) for x in IMAGE_GRID)
        return f"qwen35_4b_{mode}_seq{SEQ_LEN}_grid{grid}"
    return f"qwen35_4b_{mode}_seq{SEQ_LEN}"


class Qwen35Entrypoints(nn.Module):
    def __init__(self, model: Qwen35ForConditionalGeneration, image_grid: tuple[int, int, int]):
        super().__init__()
        self.model = model
        self.image_grid = image_grid
        merge = model.config.vision_config.spatial_merge_size
        self.image_tokens = math.prod(image_grid) // (merge * merge)

    @nvtx.instrument_nvtx
    def text_prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, use_cache=True, logits_to_keep=1)

    @nvtx.instrument_nvtx
    def text_decode(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, use_cache=True, logits_to_keep=1)

    @nvtx.instrument_nvtx
    def image_prefill(
        self,
        input_ids: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        del image_grid_thw
        inputs_embeds = self.model.model.language_model.embed_tokens(input_ids)
        image_embeds = self.model.model.visual(pixel_values, grid_thw=self.image_grid)
        # The vision tower encodes a single synthetic image; broadcast it across the
        # batch so that every sample's image-token slots are populated.
        image_embeds = image_embeds.repeat(input_ids.shape[0], 1)
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[mm_token_type_ids.bool()] = image_embeds.view(-1, inputs_embeds.shape[-1]).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        hidden_states = self.model.model(
            input_ids=None,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=self.model.cache,
            inputs_embeds=inputs_embeds,
            use_cache=True,
        )
        return self.model.lm_head(hidden_states[:, -1:, :])

    @nvtx.instrument_nvtx
    def image_decode(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, position_ids=position_ids, use_cache=True, logits_to_keep=1)


def compile_entrypoints(runner: Qwen35Entrypoints) -> Qwen35Entrypoints:
    runner.text_prefill = magi_compile(
        runner.text_prefill,
        model_tag=entrypoint_model_tag("text_prefill"),
        dynamic_arg_dims=entrypoint_dynamic_arg_dims("text_prefill"),
    )
    runner.text_decode = magi_compile(
        runner.text_decode,
        model_tag=entrypoint_model_tag("text_decode"),
        dynamic_arg_dims=entrypoint_dynamic_arg_dims("text_decode"),
    )
    runner.image_prefill = magi_compile(
        runner.image_prefill,
        model_tag=entrypoint_model_tag("image_prefill"),
        dynamic_arg_dims=entrypoint_dynamic_arg_dims("image_prefill"),
    )
    runner.image_decode = magi_compile(
        runner.image_decode,
        model_tag=entrypoint_model_tag("image_decode"),
        dynamic_arg_dims=entrypoint_dynamic_arg_dims("image_decode"),
    )
    return runner


def build_runner(model: Qwen35ForConditionalGeneration, image_grid: tuple[int, int, int]) -> Qwen35Entrypoints:
    return compile_entrypoints(Qwen35Entrypoints(model, image_grid=image_grid))


def needs_image_inputs(modes: tuple[str, ...]) -> bool:
    return any(mode.startswith("image_") for mode in modes)


def sync_cuda() -> None:
    torch.cuda.synchronize()


def make_text_ids(model: Qwen35ForConditionalGeneration, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    vocab_limit = min(model.config.text_config.vocab_size, model.config.image_token_id) - 16
    return torch.randint(0, vocab_limit, (batch_size, seq_len), device=device, dtype=torch.long)


def make_image_inputs(
    model: Qwen35ForConditionalGeneration,
    batch_size: int,
    seq_len: int,
    image_grid: tuple[int, int, int],
    device: torch.device,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    if len(image_grid) != 3:
        raise ValueError("IMAGE_GRID must be three comma-separated ints, for example 1,2,2.")
    merge = model.config.vision_config.spatial_merge_size
    image_tokens = math.prod(image_grid) // (merge * merge)
    if math.prod(image_grid) % (merge * merge) != 0:
        raise ValueError(f"IMAGE_GRID={image_grid} must be divisible by spatial merge size {merge}.")
    min_seq_len = image_tokens + IMAGE_TOKEN_START + 1
    if seq_len < min_seq_len:
        raise ValueError(f"SEQ_LEN={seq_len} is too small for {image_tokens} image tokens.")

    input_ids = make_text_ids(model, batch_size, seq_len, device)
    mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
    image_start = IMAGE_TOKEN_START
    image_end = image_start + image_tokens
    # The single synthetic image is replicated verbatim across the batch so that
    # each sample carries the same image-token layout.
    input_ids[:, image_start - 1] = model.config.vision_start_token_id
    input_ids[:, image_start:image_end] = model.config.image_token_id
    input_ids[:, image_end] = model.config.vision_end_token_id
    mm_token_type_ids[:, image_start:image_end] = 1

    grid = torch.tensor([image_grid] * batch_size, dtype=torch.long, device=device)
    vc = model.config.vision_config
    patch_values = math.prod(image_grid)
    patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size
    pixel_values = torch.randn((patch_values, patch_dim), dtype=DTYPE, device=device)
    prefill_position_ids, rope_deltas = model.model.get_rope_index(input_ids, mm_token_type_ids, image_grid_thw=grid)
    decode_position_ids = torch.arange(seq_len, seq_len + 1, device=device, dtype=torch.long)
    decode_position_ids = decode_position_ids.view(1, 1, -1).expand(3, input_ids.shape[0], -1)
    decode_position_ids = decode_position_ids + rope_deltas.to(device=device, dtype=torch.long)
    return (input_ids, mm_token_type_ids, pixel_values, grid, prefill_position_ids), (decode_position_ids,)


@nvtx.instrument_nvtx
def run_prefill_mode(
    mode: str, runner: Qwen35Entrypoints, args: tuple[torch.Tensor, ...], label: str = "compiled"
) -> tuple[torch.Tensor, float]:
    fn = runner.text_prefill if mode == "text_prefill" else runner.image_prefill
    runner.model.reset_cache()
    with torch.inference_mode():
        outputs = fn(*args)
    sync_cuda()

    durations: list[float] = []
    for i in range(PROFILE_CNT + 1):
        runner.model.reset_cache()
        sync_cuda()
        start = time.perf_counter()
        with torch.inference_mode():
            outputs = fn(*args)
        sync_cuda()
        elapsed = time.perf_counter() - start
        durations.append(elapsed)
        print(f"{label} {mode} {i}-th forward: {elapsed:.4f}s logits={tuple(outputs.shape)}")
    return outputs, sum(durations) / len(durations)


@nvtx.instrument_nvtx
def run_decode_mode(
    mode: str,
    runner: Qwen35Entrypoints,
    prefill_args: tuple[torch.Tensor, ...],
    decode_args: tuple[torch.Tensor, ...],
    label: str = "compiled",
) -> tuple[torch.Tensor, float]:
    prefill_fn = runner.text_prefill if mode == "text_decode" else runner.image_prefill
    decode_fn = runner.text_decode if mode == "text_decode" else runner.image_decode

    def run_one_decode() -> torch.Tensor:
        runner.model.reset_cache()
        with torch.inference_mode():
            with nvtx.add_nvtx_event(f"{mode}_fake_prefill"):
                prefill_fn(*prefill_args)
        sync_cuda()
        with torch.inference_mode():
            with nvtx.add_nvtx_event(f"{mode}_real_decode"):
                return decode_fn(*decode_args)

    outputs = run_one_decode()
    sync_cuda()

    durations: list[float] = []
    for i in range(PROFILE_CNT + 1):
        runner.model.reset_cache()
        with torch.inference_mode():
            with nvtx.add_nvtx_event(f"{mode}_fake_prefill"):
                prefill_fn(*prefill_args)
        sync_cuda()
        start = time.perf_counter()
        with torch.inference_mode():
            with nvtx.add_nvtx_event(f"{mode}_real_decode"):
                outputs = decode_fn(*decode_args)
        sync_cuda()
        elapsed = time.perf_counter() - start
        durations.append(elapsed)
        print(f"{label} {mode} {i}-th token: {elapsed:.4f}s logits={tuple(outputs.shape)}")
    return outputs, sum(durations) / len(durations)


def resolve_model_path() -> Path | None:
    if not MODEL_PATH:
        return None
    return Path(MODEL_PATH)


def main() -> None:
    if MODE != "all" and MODE not in MODES:
        raise ValueError(f"Unsupported MODE={MODE!r}. Use one of {MODES} or all.")
    if BATCH_SIZE < 1:
        raise ValueError(f"BATCH_SIZE={BATCH_SIZE} must be >= 1.")
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3.5-4B inference example requires CUDA.")

    torch.random.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    model_path = resolve_model_path()
    model = Qwen35ForConditionalGeneration(
        model_path, dtype=DTYPE, device=device, load_weights=False if SKIP_LOAD_MODEL else None
    )
    runner = build_runner(model, image_grid=IMAGE_GRID)

    text_ids = make_text_ids(model, BATCH_SIZE, SEQ_LEN, device)
    decode_ids = make_text_ids(model, BATCH_SIZE, 1, device)
    modes = MODES if MODE == "all" else (MODE,)
    image_inputs = make_image_inputs(model, BATCH_SIZE, SEQ_LEN, IMAGE_GRID, device) if needs_image_inputs(modes) else None

    weight_source = "random" if SKIP_LOAD_MODEL or model_path is None else MODEL_PATH
    print(f"Model path: {MODEL_PATH or '(default Qwen3.5-4B config)'}")
    print(f"Weights: {weight_source}")
    print(f"Modes: {', '.join(modes)}")
    print(f"SEQ_LEN={SEQ_LEN} BATCH_SIZE={BATCH_SIZE} IMAGE_GRID={IMAGE_GRID} PROFILE_CNT={PROFILE_CNT}")

    outputs = None
    for i in range(PROFILE_CNT + 1):
        nvtx.switch_profile(i, 1, PROFILE_CNT + 1)
        for mode in modes:
            # Tag the whole per-mode execution in a single NVTX range so that, when all
            # modes run in one process, each mode shows up as a named region in nsys.
            with nvtx.add_nvtx_event(mode):
                if mode == "text_prefill":
                    outputs, _ = run_prefill_mode(mode, runner, (text_ids,))
                elif mode == "text_decode":
                    outputs, _ = run_decode_mode(mode, runner, (text_ids,), (decode_ids,))
                elif mode == "image_prefill":
                    assert image_inputs is not None
                    outputs, _ = run_prefill_mode(mode, runner, image_inputs[0])
                elif mode == "image_decode":
                    assert image_inputs is not None
                    outputs, _ = run_decode_mode(mode, runner, image_inputs[0], (decode_ids, *image_inputs[1]))
    print(f"Final logits: {tuple(outputs.shape)}")


if __name__ == "__main__":
    main()
