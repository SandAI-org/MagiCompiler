# Qwen3.5-4B Forward Inference Example

This example is a forward-level MagiCompiler benchmark/smoke test for a local Qwen3.5-4B checkpoint. It builds synthetic token and image tensors directly, so it does not run a tokenizer, chat template, natural-language decoder, `generate()`, or MTP path.

The model implementation in `modeling.py` uses PyTorch plus `safetensors` only. It intentionally does not import `transformers`, `tokenizers`, `PIL`, or `qwen_vl_utils`.

## Commands

Set `MODEL_PATH` to the checkpoint directory before running:

```bash
MODEL_PATH=/path/to/Qwen3.5-4B MODE=all NSYS_PROFILE=false \
bash example/inference/qwen3.5-4b/infer.sh
```

Run text prefill:

```bash
MODEL_PATH=/path/to/Qwen3.5-4B MODE=text_prefill NSYS_PROFILE=false \
bash example/inference/qwen3.5-4b/infer.sh
```

Run text decode:

```bash
MODEL_PATH=/path/to/Qwen3.5-4B MODE=text_decode NSYS_PROFILE=false \
bash example/inference/qwen3.5-4b/infer.sh
```

Run image prefill:

```bash
MODEL_PATH=/path/to/Qwen3.5-4B MODE=image_prefill NSYS_PROFILE=false \
bash example/inference/qwen3.5-4b/infer.sh
```

Run image decode:

```bash
MODEL_PATH=/path/to/Qwen3.5-4B MODE=image_decode NSYS_PROFILE=false \
bash example/inference/qwen3.5-4b/infer.sh
```

For the default `IMAGE_GRID=1,2,2`, the image modes need `SEQ_LEN>=4`.

Useful environment variables:

- `MODEL_PATH`: checkpoint directory, required
- `MODE`: `text_prefill`, `text_decode`, `image_prefill`, `image_decode`, or `all`, default `all`
- `SEQ_LEN`: synthetic sequence length, default `128`
- `IMAGE_GRID`: one synthetic image grid as `T,H,W`, default `1,2,2`
- `PROFILE_CNT`: timed loop count, default `3`
- `NSYS_PROFILE`: `true` or `false`, default `true`
- `PYTHON_BIN`: Python executable, default `python`
- `MAGI_COMPILE_CACHE_ROOT_DIR`: MagiCompiler cache root, default repo `.cache`
