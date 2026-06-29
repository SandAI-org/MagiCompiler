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
from modeling import Wan2_2_VAE

import magi_compiler.utils.nvtx as nvtx

# Set WAN2_2_VAE_PTH=/path/to/model/Wan2.2_VAE.pth to load real weights.
VAE_PTH = os.environ.get("WAN2_2_VAE_PTH")
LATENT_SHAPE = [int(x) for x in os.environ.get("LATENT_SHAPE", "48,7,34,60").split(",")]
MODE = os.environ.get("MODE", "decode")
PROFILE_CNT = int(os.environ.get("PROFILE_CNT", "3"))
DTYPE = torch.bfloat16


def get_video_shape(latent_shape):
    _, latent_t, latent_h, latent_w = latent_shape
    return [3, (latent_t - 1) * 4 + 1, latent_h * 16, latent_w * 16]


def run_vae(vae, latent, video):
    if MODE == "encode":
        return vae.encode([video])
    if MODE == "decode":
        return vae.decode([latent])
    raise ValueError(f"Unsupported MODE={MODE!r}. Use MODE=encode or MODE=decode.")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("WAN 2.2 VAE inference example requires CUDA.")

    torch.random.manual_seed(0)

    vae = Wan2_2_VAE(vae_pth=VAE_PTH, dtype=DTYPE, device="cuda")
    latent = torch.randn(LATENT_SHAPE, dtype=DTYPE, device=torch.device("cuda"))
    video_shape = get_video_shape(LATENT_SHAPE)
    video = torch.randn(video_shape, dtype=DTYPE, device=torch.device("cuda"))

    print(f"Mode: {MODE}")
    print(f"VAE checkpoint: {'WAN2_2_VAE_PTH' if VAE_PTH else None}")
    print(f"Latent shape: {LATENT_SHAPE}")
    print(f"Video shape: {video_shape}")

    # Warm up and trigger compilation.
    with torch.inference_mode():
        outputs = run_vae(vae, latent, video)
    torch.cuda.synchronize()

    for i in range(PROFILE_CNT + 1):
        nvtx.switch_profile(i, 0, PROFILE_CNT)
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.inference_mode():
            outputs = run_vae(vae, latent, video)

        torch.cuda.synchronize()
        print(f"{MODE} {i}-th image: {time.perf_counter() - start:.4f}s")

    print(f"outputs: {outputs[0].shape}")


if __name__ == "__main__":
    main()
