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
import time

import torch
from modeling import DEFAULT_MODEL, Qwen2_5_OmniInference

import magi_compiler.utils.nvtx as nvtx

# Set VLLM_OMNI_MODEL=/path/to/Qwen2.5-Omni-3B for a local checkpoint.
MODEL = os.environ.get("VLLM_OMNI_MODEL", DEFAULT_MODEL)
QUERY_TYPE = os.environ.get("QUERY_TYPE", "text")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output_audio")
VIDEO_PATH = os.environ.get("VIDEO_PATH")
NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "16"))
DEPLOY_CONFIG = os.environ.get("DEPLOY_CONFIG")
PROFILE_CNT = int(os.environ.get("PROFILE_CNT", "3"))


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen2.5-Omni inference example requires CUDA.")

    print(f"Query type: {QUERY_TYPE}")
    print(f"Model: {MODEL}")
    print(f"VLLM_OMNI_MAGI_COMPILER: {os.environ.get('VLLM_OMNI_MAGI_COMPILER', '0')}")
    print(f"Deploy config: {DEPLOY_CONFIG}")
    print(f"Output dir: {OUTPUT_DIR}")

    pipeline = Qwen2_5_OmniInference(model_name=MODEL, deploy_config=DEPLOY_CONFIG)
    infer_kwargs = dict(
        query_type=QUERY_TYPE,
        output_dir=OUTPUT_DIR,
        video_path=VIDEO_PATH,
        num_frames=NUM_FRAMES,
    )

    try:
        # Warm up and trigger compilation.
        outputs = pipeline.infer(**infer_kwargs, write_outputs=True)

        for i in range(PROFILE_CNT + 1):
            nvtx.switch_profile(i, 0, PROFILE_CNT)
            start = time.perf_counter()
            pipeline.infer(**infer_kwargs, write_outputs=False)
            # Wall time for the full Omni pipeline (workers finish before generate returns).
            print(f"{QUERY_TYPE} {i}-th iter: {time.perf_counter() - start:.4f}s")

        print(f"outputs: {outputs}")
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
