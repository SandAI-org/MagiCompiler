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

import os

import numpy as np
import soundfile as sf
from vllm.assets.video import VideoAsset, video_to_ndarrays
from vllm.multimodal.media.audio import load_audio
from vllm.sampling_params import SamplingParams

from vllm_omni.entrypoints.omni import Omni

__all__ = ["DEFAULT_MODEL", "Qwen2_5_OmniInference"]

SEED = 42
DEFAULT_MODEL = "Qwen/Qwen2.5-Omni-3B"
DEFAULT_SYSTEM = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
    "Group, capable of perceiving auditory and visual inputs, as well as "
    "generating text and speech."
)


def load_video_frames(video_path=None, num_frames=16):
    if video_path is None:
        return VideoAsset(name="baby_reading", num_frames=num_frames).np_ndarrays
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    return video_to_ndarrays(video_path, num_frames=num_frames)


def build_prompt(query_type, video_path=None, num_frames=16):
    def chat(user_body):
        return (
            f"<|im_start|>system\n{DEFAULT_SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{user_body}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    if query_type == "text":
        question = (
            "Explain the system architecture for a scalable audio "
            "generation pipeline. Answer in 15 words."
        )
        return {"prompt": chat(question)}

    if query_type == "use_video":
        return {
            "prompt": chat("<|vision_bos|><|VIDEO|><|vision_eos|>Why is this video funny?"),
            "multi_modal_data": {"video": load_video_frames(video_path, num_frames)},
        }

    if query_type == "use_audio_in_video":
        question = "Describe the content of the video, then convert what the baby say into text."
        if video_path is None:
            asset = VideoAsset(name="baby_reading", num_frames=num_frames)
            video = asset.np_ndarrays
            audio = asset.get_audio(sampling_rate=16000)
        else:
            video = load_video_frames(video_path, num_frames)
            audio_signal, sr = load_audio(video_path, sr=16000)
            audio = (audio_signal.astype(np.float32), sr)
        return {
            "prompt": chat(
                "<|vision_bos|><|VIDEO|><|vision_eos|>"
                "<|audio_bos|><|AUDIO|><|audio_eos|>"
                f"{question}"
            ),
            "multi_modal_data": {"video": video, "audio": audio},
            "mm_processor_kwargs": {"use_audio_in_video": True},
        }

    raise ValueError(
        f"Unsupported query_type={query_type!r}. "
        "Use QUERY_TYPE=text, use_video, or use_audio_in_video."
    )


def default_sampling_params():
    # thinker / talker / token2wav
    return [
        SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=2048,
            seed=SEED,
            detokenize=True,
            repetition_penalty=1.1,
        ),
        SamplingParams(
            temperature=0.9,
            top_p=0.8,
            top_k=40,
            max_tokens=2048,
            seed=SEED,
            detokenize=True,
            repetition_penalty=1.05,
            stop_token_ids=[8294],
        ),
        SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=2048,
            seed=SEED,
            detokenize=True,
            repetition_penalty=1.1,
        ),
    ]


class Qwen2_5_OmniInference:
    """Offline Qwen2.5-Omni via vLLM-Omni. Magi is toggled by VLLM_OMNI_MAGI_COMPILER."""

    def __init__(self, model_name=DEFAULT_MODEL, deploy_config=None):
        omni_kwargs = {
            "stage_init_timeout": 300,
            "init_timeout": 300,
            "worker_backend": "multi_process",
        }
        if deploy_config:
            omni_kwargs["deploy_config"] = deploy_config
        self.omni = Omni(model=model_name, **omni_kwargs)
        self.sampling_params = default_sampling_params()

    def infer(
        self,
        query_type="text",
        output_dir="output_audio",
        video_path=None,
        num_frames=16,
        write_outputs=True,
    ):
        prompts = [build_prompt(query_type, video_path=video_path, num_frames=num_frames)]
        if write_outputs:
            os.makedirs(output_dir, exist_ok=True)

        text_path = None
        audio_path = None
        for stage_outputs in self.omni.generate(prompts, self.sampling_params):
            output = stage_outputs.request_output
            request_id = output.request_id
            if stage_outputs.final_output_type == "text":
                if write_outputs:
                    text_path = os.path.join(output_dir, f"{request_id}.txt")
                    with open(text_path, "w", encoding="utf-8") as f:
                        f.write("Prompt:\n")
                        f.write(str(output.prompt) + "\n")
                        f.write("vllm_text_output:\n")
                        f.write(str(output.outputs[0].text).strip() + "\n")
                    print(f"Request ID: {request_id}, Text saved to {text_path}")
            elif stage_outputs.final_output_type == "audio" and write_outputs:
                audio_path = os.path.join(output_dir, f"output_{request_id}.wav")
                sf.write(
                    audio_path,
                    output.outputs[0].multimodal_output["audio"].detach().cpu().numpy(),
                    samplerate=24000,
                )
                print(f"Request ID: {request_id}, Saved audio to {audio_path}")

        return {"text_path": text_path, "audio_path": audio_path}

    def close(self):
        self.omni.close()
