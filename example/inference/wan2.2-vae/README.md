# Conv-Heavy Model Inference Optimization

This example uses WAN 2.2 VAE encode/decode as a representative convolution-heavy inference workload. The scripts feed synthetic tensors with WAN 2.2-compatible shapes, so the case can be profiled without preparing input videos. A real checkpoint can be supplied through `WAN2_2_VAE_PTH`.

## Background

Many generative models contain convolution-heavy submodules, such as video VAEs, image/video decoders, and feature encoders. This example focuses on two common performance issues in PyTorch Inductor generated code:

1. **Convolution layout overhead**: cuDNN channels-last kernels, such as NHWC/NDHWC, are usually faster on Ampere and newer GPUs. If the compiler does not preserve that layout across convolution boundaries, cuDNN can pay repeated internal NC(D)HW to NHWC/NDHWC conversions.

2. **Dynamic-shape memory-kernel tiling overhead**: with dynamic H/W dimensions, the more severe bottleneck often shifts to Inductor's fused Triton kernels for memory-heavy operations around convolutions, such as transpose, permute, reshape, clone, and elementwise layout producers. Symbolic dimensions can make tiling analysis fall back to conservative 1D schedules, leaving these structured memory-operation kernels under-tiled.

WAN 2.2 VAE is used here as a representative benchmark because its encode/decode paths contain stacked 3D convolutions, residual blocks, temporal up/down-sampling, and spatial resampling.

## Optimization Principles

MagiCompiler applies post-grad Inductor graph passes for these two cases.

### Static Conv-Heavy Graphs: Channels-Last Layout

`ConvChannelsLastPass` handles static conv-heavy graphs by taking ownership of convolution layout in FX metadata.

The inserted `aten.clone(memory_format=channels_last)` or `aten.clone(memory_format=channels_last_3d)` is a metadata carrier, not a copy kernel that should survive in the final stream. It lets the pass attach channels-last fake-tensor strides to convolution input and weight nodes.

Inductor's `aten.convolution` lowering already reads input FX meta strides and applies `require_stride_order` to the producer buffers. After the pass rewrites those meta strides to channels-last, Inductor freezes the producer layout as NHWC/NDHWC before cuDNN sees the convolution.

Because the clone lowers as `Pointwise` with `FlexibleLayout`, the stride constraint is normally zero-copy: the producer buffer is allocated directly in channels-last layout instead of emitting an extra clone kernel. With a channels-last input, cuDNN's backend memory-format probe also lets `conv_layout()` infer a channels-last output layout.

### Dynamic H/W Graphs: Triton ND-Tiling Workaround

`ND_TilingWorkaroundPass` targets dynamic H/W graphs where Inductor emits fused Triton kernels for dense memory operations around convolutions.

The goal is to keep these kernels tiled along their natural multi-dimensional structure instead of flattening them into conservative Grid1D-style schedules. The pass enables:

- `prefer_nd_tiling=True`
- `max_tiles=3`
- `tile_reductions=True`

The pass is guarded so it only applies when the graph is both dynamic and conv-heavy. In this regime, improving Inductor-generated memory-kernel tiling is often more important than further reducing cuDNN layout conversions.

> This workaround is a targeted fix for the current dynamic-shape behavior. A more general heuristic tiling strategy for these Inductor-generated memory kernels will be added in future work.

## Test Script Usage

Run decode:

```bash
WAN2_2_VAE_PTH=/path/to/model/Wan2.2_VAE.pth MODE=decode \
bash example/inference/wan2.2-vae/infer.sh
```

Run encode:

```bash
WAN2_2_VAE_PTH=/path/to/model/Wan2.2_VAE.pth MODE=encode \
bash example/inference/wan2.2-vae/infer.sh
```

`infer.py` runs one unprofiled `encode`/`decode` call before the NVTX profiled loop to trigger compilation.

By default, `modeling.py` compiles encode/decode with dynamic H/W dimensions enabled through `dynamic_arg_dims={"x": [3, 4]}` and `dynamic_arg_dims={"z": [3, 4]}`. To run the static H/W variant, set those lists to `[]`.

## Performance Comparison

The following numbers are CUDA HW sum averages over profiled iterations on the WAN 2.2 VAE 540p workload, measured on an NVIDIA H100 80G HBM3 GPU. Parentheses show `MAGI_COMPILE` speedup over the corresponding baseline.

### Decode

| Shape mode | MAGI_COMPILE | TORCH_COMPILE | EAGER |
| --- | ---: | ---: | ---: |
| Static H/W | `457.943 ms` | `526.973 ms` (`1.15x`) | `855.131 ms` (`1.87x`) |
| Dynamic H/W | `553.543 ms` | `768.700 ms` (`1.39x`) | `855.131 ms` (`1.54x`) |

### Encode

| Shape mode | MAGI_COMPILE | TORCH_COMPILE | EAGER |
| --- | ---: | ---: | ---: |
| Static H/W | `134.444 ms` | `151.183 ms` (`1.12x`) | `269.702 ms` (`2.01x`) |
| Dynamic H/W | `179.025 ms` | `289.522 ms` (`1.62x`) | `269.702 ms` (`1.51x`) |
