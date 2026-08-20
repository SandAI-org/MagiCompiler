# Qwen3.5-4B Forward Inference Example

This example is a forward-level MagiCompiler benchmark/smoke test for Qwen3.5-4B. It builds synthetic token and image tensors directly, so it does not run a tokenizer, chat template, natural-language decoder, `generate()`, or MTP path.

The model implementation in `modeling.py` uses PyTorch plus `safetensors` only. It intentionally does not import `transformers`, `tokenizers`, `PIL`, or `qwen_vl_utils`.

`MODEL_PATH` is optional. Omit it (or set `SKIP_LOAD_MODEL=true`) to run with the built-in Qwen3.5-4B architecture and randomly initialized weights. That path is for compile/profile work only; logits are not meaningful.

## Commands

Run with randomly initialized weights:

```bash
MODE=all NSYS_PROFILE=false \
bash example/inference/qwen3.5-4b/infer.sh
```

Run the eager baseline without `magi_compile`:

```bash
COMPILE_MODE=eager MODE=all NSYS_PROFILE=false \
bash example/inference/qwen3.5-4b/infer.sh
```

Load a local checkpoint:

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

- `MODEL_PATH`: checkpoint directory, optional. When unset, the example uses the built-in Qwen3.5-4B config and random weights
- `SKIP_LOAD_MODEL`: `true` to skip loading checkpoint weights even if `MODEL_PATH` is set, default `false`
- `COMPILE_MODE`: `magi` to compile with MagiCompiler, or `eager` to run uncompiled PyTorch, default `magi`
- `MODE`: `text_prefill`, `text_decode`, `image_prefill`, `image_decode`, or `all`, default `all`
- `SEQ_LEN`: synthetic sequence length, default `128`
- `BATCH_SIZE`: synthetic batch size, default `1`
- `IMAGE_GRID`: one synthetic image grid as `T,H,W`, default `1,2,2`
- `PROFILE_CNT`: timed loop count, default `3`
- `NSYS_PROFILE`: `true` or `false`, default `true`
- `PYTHON_BIN`: Python executable, default `python`
- `MAGI_COMPILE_CACHE_ROOT_DIR`: MagiCompiler cache root, default repo `.cache`
