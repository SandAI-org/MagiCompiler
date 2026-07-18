# Qwen2.5-Omni Offline Inference

This example runs Qwen2.5-Omni offline inference through vLLM-Omni
(thinker, talker, token2wav). MagiCompiler is enabled inside vLLM-Omni when
`VLLM_OMNI_MAGI_COMPILER=1` (default in `infer.sh`); this example does not call
`magi_compile` directly.

Install MagiCompiler and vLLM-Omni first. Use two GPUs for the default deploy
layout (`CUDA_VISIBLE_DEVICES=0,1`). Set `VLLM_OMNI_MODEL` to a local checkpoint
if needed. Without `DEPLOY_CONFIG`, vLLM-Omni uses
`vllm_omni/deploy/qwen2_5_omni.yaml`.

## Usage

Text query:

```bash
VLLM_OMNI_MODEL=/path/to/Qwen2.5-Omni-3B \
CUDA_VISIBLE_DEVICES=0,1 \
bash example/inference/qwen2.5-omni/infer.sh
```

Video query:

```bash
VLLM_OMNI_MODEL=/path/to/Qwen2.5-Omni-3B \
CUDA_VISIBLE_DEVICES=0,1 \
QUERY_TYPE=use_video \
VIDEO_PATH=/path/to/video.mp4 \
bash example/inference/qwen2.5-omni/infer.sh
```

Skip nsys:

```bash
NSYS_PROFILE=false bash example/inference/qwen2.5-omni/infer.sh
```

`infer.py` warms up once (writes text/audio under `OUTPUT_DIR`), then runs the
NVTX profile loop without writing files. Printed times are end-to-end wall
clock for the multi-process Omni pipeline.

Useful env vars: `QUERY_TYPE` (`text` / `use_video` / `use_audio_in_video`),
`OUTPUT_DIR`, `VIDEO_PATH`, `NUM_FRAMES`, `PROFILE_CNT`,
`VLLM_OMNI_MAGI_COMPILER`, `DEPLOY_CONFIG`, `VLLM_OMNI_ROOT`.
