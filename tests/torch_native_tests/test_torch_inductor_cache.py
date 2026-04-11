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

"""Test TorchInductor cache."""

import os
import tempfile
from dataclasses import dataclass
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch._dynamo.utils import counters

from tests.model_definition import TransformerConfig, create_transformer_model_with_initial_params


@dataclass(frozen=True)
class CounterDelta:
    """Store cache counter deltas for one test run."""

    autograd_hit: int
    autograd_miss: int
    inductor_hit: int
    inductor_miss: int


EXPECTED = {
    "train": CounterDelta(autograd_hit=31, autograd_miss=2, inductor_hit=0, inductor_miss=0),
    "eval": CounterDelta(autograd_hit=31, autograd_miss=1, inductor_hit=0, inductor_miss=0),
}


@pytest.fixture(autouse=True)
def model_with_clean_cache():
    """Build a transformer model with an isolated Magi cache directory."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        with patch.dict(os.environ, {"MAGI_COMPILE_CACHE_ROOT_DIR": tmp_dir}):
            transformer_config = TransformerConfig(
                vocab_size=128256,
                hidden_size=1024,
                intermediate_size=4096,
                num_hidden_layers=32,
                num_attention_heads=32,
                num_key_value_heads=8,
                max_position_embeddings=8192,
                rms_norm_eps=1e-6,
                params_dtype=torch.bfloat16,
            )
            model, _ = create_transformer_model_with_initial_params(transformer_config, device="cuda")
            yield model, transformer_config


@pytest.fixture
def cache_snapshot():
    """Return a callable that snapshots current Dynamo cache counters."""

    def _snapshot() -> dict[str, dict[str, int]]:
        return {
            "autograd": {
                "hit": counters["aot_autograd"]["autograd_cache_hit"],
                "miss": counters["aot_autograd"]["autograd_cache_miss"],
            },
            "inductor": {
                "hit": counters["inductor"]["inductor_cache_hit"],
                "miss": counters["inductor"]["inductor_cache_miss"],
            },
        }

    return _snapshot


def _device() -> torch.device:
    """Select CUDA when available, otherwise CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _delta(before: dict, after: dict) -> CounterDelta:
    """Compute per-backend cache hit/miss deltas between two snapshots."""

    return CounterDelta(
        autograd_hit=after["autograd"]["hit"] - before["autograd"]["hit"],
        autograd_miss=after["autograd"]["miss"] - before["autograd"]["miss"],
        inductor_hit=after["inductor"]["hit"] - before["inductor"]["hit"],
        inductor_miss=after["inductor"]["miss"] - before["inductor"]["miss"],
    )


def _assert_delta(actual: CounterDelta, expected: CounterDelta):
    """Assert cache counter deltas against expected values."""

    assert actual == expected, f"counter delta mismatch, got={actual}, expected={expected}"


class TestTorchInductorCache:
    """Validate TorchInductor cache behavior in train/eval flows."""

    def test_training_mode(self, model_with_clean_cache, cache_snapshot):
        """Verify cache counter deltas for repeated training iterations."""
        model, model_config = model_with_clean_cache
        model = model.to(_device())
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        max_batch_size = 2
        num_epochs = 10

        before = cache_snapshot()
        for _ in range(num_epochs):
            dummy_input = torch.randint(
                0, model_config.vocab_size,
                (max_batch_size, model_config.max_position_embeddings),
                device=_device(),
            )
            dummy_label = torch.randint(
                0, model_config.vocab_size,
                (max_batch_size, model_config.max_position_embeddings),
                device=_device(),
            )

            optimizer.zero_grad()
            logits = model.forward(dummy_input)
            loss = F.cross_entropy(logits.view(-1, model_config.vocab_size), dummy_label.view(-1))
            loss.backward()
            optimizer.step()

        after = cache_snapshot()
        _assert_delta(_delta(before, after), EXPECTED["train"])

    def test_evaluation_mode(self, model_with_clean_cache, cache_snapshot):
        """Verify cache counter deltas for a no-grad evaluation forward."""

        model, model_config = model_with_clean_cache
        model = model.to(_device())
        model.eval()

        max_batch_size = 2
        before = cache_snapshot()
        with torch.no_grad():
            dummy_input = torch.randint(
                0, model_config.vocab_size,
                (max_batch_size, model_config.max_position_embeddings),
                device=_device(),
            )
            model.forward(dummy_input)
        after = cache_snapshot()

        _assert_delta(_delta(before, after), EXPECTED["eval"])