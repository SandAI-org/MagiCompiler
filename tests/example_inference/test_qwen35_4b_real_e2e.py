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

import os
import sys
from pathlib import Path

import pytest
import torch

from tests.example_inference.test_qwen35_4b_infer import EXAMPLE_DIR, load_infer_module

DEFAULT_TEXT = "\u8fd9\u662f\u4ec0\u4e48\u5b57\u6bcd\uff1f"


def _env_path(name: str) -> Path:
    value = os.environ.get(name)
    if value is None:
        pytest.skip(f"set {name} to run the Qwen3.5-4B real E2E test")
    return Path(value)


def _topk_ids(logits: torch.Tensor, k: int = 5) -> list[int]:
    return logits[0, -1].float().topk(k).indices.tolist()


def _assert_aligned(expected: torch.Tensor, actual: torch.Tensor) -> None:
    assert expected.shape == actual.shape
    assert torch.isfinite(expected).all().item()
    assert torch.isfinite(actual).all().item()

    expected_top5 = _topk_ids(expected)
    actual_top5 = _topk_ids(actual)
    assert expected_top5[0] in actual_top5, f"expected top1 {expected_top5[0]} not in actual top5 {actual_top5}"
    assert actual_top5[0] in expected_top5, f"actual top1 {actual_top5[0]} not in expected top5 {expected_top5}"


def test_real_image_prefill_matches_transformers():
    model_path = _env_path("MODEL_PATH")
    image_path = _env_path("QWEN35_E2E_IMAGE")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    transformers = pytest.importorskip("transformers")
    Image = pytest.importorskip("PIL.Image")
    official_cls = getattr(transformers, "Qwen3_5ForConditionalGeneration", None)
    if official_cls is None:
        pytest.skip("transformers does not provide Qwen3_5ForConditionalGeneration")

    module = load_infer_module()
    dtype = module.DTYPE
    processor = transformers.AutoProcessor.from_pretrained(model_path, local_files_only=True)
    text = os.environ.get("QWEN35_E2E_TEXT", DEFAULT_TEXT)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    with Image.open(image_path) as image:
        inputs = processor(text=[prompt], images=[image.convert("RGB")], return_tensors="pt")
    inputs = {name: value.to("cuda") for name, value in inputs.items()}
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)

    official_model = official_cls.from_pretrained(model_path, dtype=dtype, device_map="cuda", local_files_only=True).eval()
    with torch.inference_mode():
        expected = official_model(**inputs, use_cache=True, logits_to_keep=1).logits.detach()
    del official_model
    torch.cuda.empty_cache()

    local_model = module.Qwen35ForConditionalGeneration(model_path, dtype=dtype, device="cuda")
    image_grid = tuple(int(x) for x in inputs["image_grid_thw"][0].tolist())
    runner = module.Qwen35Entrypoints(local_model, image_grid=image_grid)
    added_example_dir = False
    if os.environ.get("QWEN35_E2E_COMPILED") == "1":
        sys.path.insert(0, str(EXAMPLE_DIR))
        added_example_dir = True
        runner = module.compile_entrypoints(runner)

    try:
        position_ids, _ = local_model.model.get_rope_index(
            inputs["input_ids"], inputs["mm_token_type_ids"], image_grid_thw=inputs["image_grid_thw"]
        )
        local_model.reset_cache()
        with torch.inference_mode():
            actual = runner.image_prefill(
                inputs["input_ids"],
                inputs["mm_token_type_ids"],
                inputs["pixel_values"],
                inputs["image_grid_thw"],
                position_ids,
            ).detach()
    finally:
        if added_example_dir:
            sys.path.remove(str(EXAMPLE_DIR))

    _assert_aligned(expected, actual)
