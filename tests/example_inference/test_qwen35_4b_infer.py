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

import importlib.util
import sys
import types
from pathlib import Path

import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "example" / "inference" / "qwen3.5-4b"


def load_infer_module():
    had_modeling = "modeling" in sys.modules
    old_modeling = sys.modules.pop("modeling", None)
    module_name = "qwen35_infer_test"
    had_module = module_name in sys.modules
    old_module = sys.modules.get(module_name)
    sys.path.insert(0, str(EXAMPLE_DIR))
    try:
        spec = importlib.util.spec_from_file_location(module_name, EXAMPLE_DIR / "infer.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXAMPLE_DIR))
        sys.modules.pop("modeling", None)
        if had_modeling:
            sys.modules["modeling"] = old_modeling
        sys.modules.pop(module_name, None)
        if had_module:
            sys.modules[module_name] = old_module


def test_entrypoint_helpers():
    module = load_infer_module()

    assert module.entrypoint_dynamic_arg_dims("text_prefill") == {"input_ids": []}
    assert module.entrypoint_dynamic_arg_dims("text_decode") == {"input_ids": []}
    assert module.entrypoint_dynamic_arg_dims("image_prefill") == {
        "input_ids": [],
        "mm_token_type_ids": [],
        "pixel_values": [],
        "image_grid_thw": [],
        "position_ids": [],
    }
    assert module.entrypoint_dynamic_arg_dims("image_decode") == {"input_ids": [], "position_ids": []}
    assert module.entrypoint_model_tag("text_prefill") == f"qwen35_4b_text_prefill_seq{module.SEQ_LEN}"
    assert module.entrypoint_model_tag("image_prefill").endswith("_grid1x2x2")
    assert module.needs_image_inputs(("text_prefill", "text_decode")) is False
    assert module.needs_image_inputs(("text_decode", "image_prefill")) is True


def test_load_infer_module_isolates_modeling_import(monkeypatch):
    old_modeling = types.ModuleType("modeling")
    monkeypatch.setitem(sys.modules, "modeling", old_modeling)

    module = load_infer_module()

    assert module.Qwen35ForConditionalGeneration.__module__ == "modeling"
    assert sys.modules["modeling"] is old_modeling


class FakeEmbeddings:
    def __call__(self, input_ids):
        return torch.zeros((*input_ids.shape, 4), dtype=torch.float32)


class FakeVisual:
    def __init__(self):
        self.calls = []

    def __call__(self, pixel_values, grid_thw):
        self.calls.append((pixel_values, grid_thw))
        return torch.full((1, 4), 5.0, dtype=torch.float32)


class FakeInnerModel:
    def __init__(self):
        self.language_model = types.SimpleNamespace(embed_tokens=FakeEmbeddings())
        self.visual = FakeVisual()
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs["inputs_embeds"] + 1


class FakeModel:
    def __init__(self):
        self.config = types.SimpleNamespace(vision_config=types.SimpleNamespace(spatial_merge_size=2))
        self.model = FakeInnerModel()
        self.cache = object()
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        return torch.ones((input_ids.shape[0], 1, 7), dtype=torch.float32)

    def lm_head(self, hidden_states):
        return torch.ones((*hidden_states.shape[:-1], 7), dtype=torch.float32)


def test_compiled_entrypoints_smoke_with_fake_model(monkeypatch):
    module = load_infer_module()
    compile_calls = []
    runtime_calls = []

    def fake_magi_compile(fn, *, model_tag, dynamic_arg_dims):
        compile_calls.append((model_tag, dynamic_arg_dims))

        def wrapped(*args, **kwargs):
            runtime_calls.append(model_tag)
            return fn(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(module, "magi_compile", fake_magi_compile)
    model = FakeModel()
    runner = module.build_runner(model, image_grid=(1, 2, 2))

    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    mm_token_type_ids = torch.tensor([[0, 0, 0, 1, 0]], dtype=torch.int32)
    pixel_values = torch.ones((4, 3), dtype=torch.float32)
    image_grid_thw = torch.tensor([[1, 2, 2]])
    prefill_position_ids = torch.arange(15).view(3, 1, 5)
    decode_ids = torch.tensor([[6]])
    decode_position_ids = torch.arange(3).view(3, 1, 1)

    assert runner.text_prefill(input_ids).shape == (1, 1, 7)
    assert runner.text_decode(decode_ids).shape == (1, 1, 7)
    assert runner.image_prefill(input_ids, mm_token_type_ids, pixel_values, image_grid_thw, prefill_position_ids).shape == (
        1,
        1,
        7,
    )
    assert runner.image_decode(decode_ids, decode_position_ids).shape == (1, 1, 7)

    assert compile_calls == [
        (module.entrypoint_model_tag("text_prefill"), module.entrypoint_dynamic_arg_dims("text_prefill")),
        (module.entrypoint_model_tag("text_decode"), module.entrypoint_dynamic_arg_dims("text_decode")),
        (module.entrypoint_model_tag("image_prefill"), module.entrypoint_dynamic_arg_dims("image_prefill")),
        (module.entrypoint_model_tag("image_decode"), module.entrypoint_dynamic_arg_dims("image_decode")),
    ]
    assert runtime_calls == [
        module.entrypoint_model_tag("text_prefill"),
        module.entrypoint_model_tag("text_decode"),
        module.entrypoint_model_tag("image_prefill"),
        module.entrypoint_model_tag("image_decode"),
    ]
    assert model.calls[0]["input_ids"] is input_ids
    assert model.calls[0]["use_cache"] is True
    assert model.calls[0]["logits_to_keep"] == 1
    assert model.calls[1]["input_ids"] is decode_ids
    assert model.calls[1]["use_cache"] is True
    assert model.calls[1]["logits_to_keep"] == 1
    assert model.model.visual.calls == [(pixel_values, (1, 2, 2))]
    image_call = model.model.calls[0]
    assert image_call["input_ids"] is None
    assert image_call["attention_mask"] is None
    assert image_call["position_ids"] is prefill_position_ids
    assert image_call["past_key_values"] is model.cache
    assert image_call["use_cache"] is True
    torch.testing.assert_close(image_call["inputs_embeds"][0, 3], torch.full((4,), 5.0))
    assert model.calls[2]["input_ids"] is decode_ids
    assert model.calls[2]["position_ids"] is decode_position_ids
    assert model.calls[2]["use_cache"] is True
    assert model.calls[2]["logits_to_keep"] == 1


def test_make_image_inputs_accepts_minimum_default_grid(monkeypatch):
    module = load_infer_module()
    monkeypatch.setattr(module, "DTYPE", torch.float32)

    class FakeQwenModel:
        config = types.SimpleNamespace(
            text_config=types.SimpleNamespace(vocab_size=1000),
            image_token_id=900,
            vision_start_token_id=901,
            vision_end_token_id=902,
            vision_config=types.SimpleNamespace(spatial_merge_size=2, in_channels=3, temporal_patch_size=1, patch_size=1),
        )

        class model:
            @staticmethod
            def get_rope_index(input_ids, mm_token_type_ids, image_grid_thw):
                return torch.zeros((3, 1, input_ids.shape[1]), dtype=torch.long), torch.zeros((1, 1), dtype=torch.long)

    prefill_args, decode_args = module.make_image_inputs(FakeQwenModel(), 4, (1, 2, 2), torch.device("cpu"))

    assert prefill_args[0].shape == (1, 4)
    assert decode_args[0].shape == (3, 1, 1)


def test_run_decode_mode_times_decode_after_prefill(monkeypatch):
    module = load_infer_module()
    events: list[tuple[str, bool, tuple[int, ...]]] = []
    state = {"timing_started": False}
    perf_values = iter([1.0, 2.0, 10.0, 13.0])

    def fake_perf_counter():
        state["timing_started"] = True
        return next(perf_values)

    monkeypatch.setattr(module.time, "perf_counter", fake_perf_counter)
    monkeypatch.setattr(module, "PROFILE_CNT", 1)
    monkeypatch.setattr(module, "switch_profile_if_enabled", lambda *_: None)
    monkeypatch.setattr(module, "sync_cuda", lambda: None)

    class FakeCacheModel:
        def reset_cache(self):
            events.append(("reset", state["timing_started"], ()))

    class FakeRunner:
        def __init__(self):
            self.model = FakeCacheModel()

        def text_prefill(self, input_ids):
            events.append(("prefill", state["timing_started"], tuple(input_ids.shape)))

        def text_decode(self, input_ids):
            events.append(("decode", state["timing_started"], tuple(input_ids.shape)))
            return torch.zeros((1, 1, 7), dtype=torch.float32)

    outputs, avg = module.run_decode_mode(
        "text_decode", FakeRunner(), (torch.ones((1, 4), dtype=torch.long),), (torch.ones((1, 1), dtype=torch.long),)
    )

    assert outputs.shape == (1, 1, 7)
    assert avg == 2.0
    assert events == [
        ("reset", False, ()),
        ("prefill", False, (1, 4)),
        ("decode", False, (1, 1)),
        ("reset", False, ()),
        ("prefill", False, (1, 4)),
        ("decode", True, (1, 1)),
        ("reset", True, ()),
        ("prefill", True, (1, 4)),
        ("decode", True, (1, 1)),
    ]


def test_no_machine_specific_defaults():
    source = "\n".join((EXAMPLE_DIR / name).read_text() for name in ("infer.py", "infer.sh", "README.md"))

    assert "/data1/" not in source
    assert "lyxu18" not in source
    assert "CUDA_VISIBLE_DEVICES=" not in source
    assert 'os.environ.get("MODEL_PATH",' not in source
