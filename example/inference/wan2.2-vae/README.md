# Conv-Heavy Model Inference Optimization

This example uses WAN 2.2 VAE encode/decode as a representative convolution-heavy inference workload. The scripts feed synthetic tensors with WAN 2.2-compatible shapes, so the case can be profiled without preparing input videos. A real checkpoint can be supplied through `WAN2_2_VAE_PTH`.

## Background

Many generative models contain convolution-heavy submodules, such as video VAEs, image/video decoders, and feature encoders. After compilation, these models are usually not executed as convolution kernels alone. The generated workload is commonly a mix of:

- cuDNN convolution kernels for the main convolution computation.
- Triton fused kernels for memory-heavy operations around convolutions, such as layout changes, reshape, transpose, permute, clone, and elementwise producers.

For this reason, the main bottleneck of a conv-heavy model can change between static-shape and dynamic-shape inference. Static graphs are often limited by repeated cuDNN internal layout conversions, while graphs with dynamic dimensions can become dominated by under-tiled Triton fused memory kernels.

WAN 2.2 VAE is used here as a representative benchmark because its encode/decode paths contain stacked 3D convolutions, residual blocks, temporal up/down-sampling, and spatial resampling.

## Optimization Principles

MagiCompiler optimizes these two cases differently according to where the runtime cost comes from.

### Static Conv-Heavy Graphs: Channels-Last Layout

For static shapes, Inductor can usually optimize the surrounding Triton fused memory kernels well because the tensor sizes are known at compile time. In this case, a major remaining cost often comes from repeated layout conversions inside cuDNN convolution calls.

cuDNN commonly prefers channels-last layouts, such as NHWC or NDHWC, on Ampere and newer GPUs. If upstream graph nodes produce NC(D)HW tensors, cuDNN may need to convert the input layout internally before running convolution kernels.

MagiCompiler addresses this by arranging channels-last layout on the producer side of the graph. This lets convolution inputs reach cuDNN in the preferred layout, so multiple downstream convolutions can reuse that layout instead of paying repeated internal conversion overhead.

### Dynamic-Shape Graphs: Triton ND-Tiling Workaround

For graphs with dynamic dimensions, the bottleneck often shifts. Symbolic dimensions make it harder for the compiler to choose aggressive multi-dimensional tiling at compile time, so some structured memory operations around convolutions can fall back to more conservative 1D-style schedules.

In this regime, Triton fused kernels for reshape, transpose, permute, clone, and elementwise layout producers can become a large part of total runtime. Simply moving layout conversions outside cuDNN may increase pressure on these memory-heavy kernels, so the first priority is to improve their tiling behavior.

MagiCompiler addresses this by preserving ND tiling for the memory-heavy kernels around convolutions. This lets reshape, transpose, permute, clone, and elementwise producers execute closer to their natural multi-dimensional tensor structure instead of paying the cost of conservative 1D-style schedules.

This optimization is only applied to dynamic conv-heavy graphs, where improving Triton fused memory kernels is often more important than further reducing cuDNN layout conversion overhead.

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
