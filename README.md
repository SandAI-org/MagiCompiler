<div align="center">

# MagiCompiler

**Break the Boundaries of Local Compilation for Large Models**

<p align="center">
  <a href="https://github.com/SandAI-org/MagiCompiler/"><img src="https://img.shields.io/badge/github-repo-blue?logo=github" alt="GitHub Repo"></a>
  <a href="https://github.com/SandAI-org/MagiCompiler/releases"><img alt="license" src="https://img.shields.io/badge/Release-v1.0.0-blue"></a>
  <a href="https://github.com/SandAI-org/MagiCompiler/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-%3E%3D3.12-blue?logo=python" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-%3E%3D2.9-orange?logo=pytorch" alt="PyTorch"></a>
</p>

<p align="center">
    <a href="https://sand.ai"><img alt="blog" src="https://img.shields.io/badge/Sand%20AI-Homepage-333333.svg?logo=data:image/svg%2bxml;base64,PHN2ZyB3aWR0aD0iODAwIiBoZWlnaHQ9IjgwMCIgdmlld0JveD0iMCAwIDgwMCA4MDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGZpbGwtcnVsZT0iZXZlbm9kZCIgY2xpcC1ydWxlPSJldmVub2RkIiBkPSJNMjI3IDIyNS4wODVDMjI3IDIwMi4zMDMgMjI3IDE5MC45MTIgMjMxLjQzNyAxODIuMjExQzIzNS4zMzkgMTc0LjU1NyAyNDEuNTY2IDE2OC4zMzQgMjQ5LjIyNiAxNjQuNDM0QzI1Ny45MzMgMTYwIDI2OS4zMzIgMTYwIDI5Mi4xMjkgMTYwSDUwNy44NzFDNTA5LjI5NSAxNjAgNTEwLjY3NiAxNjAgNTEyLjAxNCAxNjAuMDAxQzUzMi4wODIgMTYwLjAxNyA1NDIuNjExIDE2MC4yNzcgNTUwLjc3NCAxNjQuNDM0QzU1OC40MzQgMTY4LjMzNCA1NjQuNjYxIDE3NC41NTcgNTY4LjU2MyAxODIuMjExQzU3MyAxOTAuOTEyIDU3MyAyMDIuMzAzIDU3MyAyMjUuMDg1VjI1Ni41NThDNTczIDI5MS4zMTkgNTczIDMwOC43IDU2NS4wMzUgMzIzLjI3OUM1NTguNzU2IDMzNC43NzIgNTQzLjU2NSAzNDYuMTEgNTIzLjA3OCAzNTkuNjA1QzUxNC42NzQgMzY1LjE0MSA1MTAuNDcyIDM2Ny45MDkgNTA1LjYzOSAzNjcuOTM2QzUwMC44MDYgMzY3Ljk2NCA0OTYuNTAzIDM2NS4yIDQ4Ny44OTYgMzU5LjY3MUw0ODcuODk2IDM1OS42N0w0NjYuNDY5IDM0NS45MDVDNDU2Ljg3NSAzMzkuNzQyIDQ1Mi4wNzggMzM2LjY2IDQ1Mi4wNzggMzMyLjIxOEM0NTIuMDc4IDMyNy43NzcgNDU2Ljg3NSAzMjQuNjk1IDQ2Ni40NjkgMzE4LjUzMUw1MjYuNzgyIDI3OS43ODVDNTM1LjI5MSAyNzQuMzE5IDU0MC40MzUgMjY0LjkwMyA1NDAuNDM1IDI1NC43OTRDNTQwLjQzNSAyMzguMzg2IDUyNy4xMjUgMjI1LjA4NSA1MTAuNzA1IDIyNS4wODVIMjg5LjI5NUMyNzIuODc1IDIyNS4wODUgMjU5LjU2NSAyMzguMzg2IDI1OS41NjUgMjU0Ljc5NEMyNTkuNTY1IDI2NC45MDMgMjY0LjcwOSAyNzQuMzE5IDI3My4yMTggMjc5Ljc4NUw1MTMuMTggNDMzLjk0MUM1NDIuNDQxIDQ1Mi43MzggNTU3LjA3MSA0NjIuMTM3IDU2NS4wMzUgNDc2LjcxNkM1NzMgNDkxLjI5NCA1NzMgNTA4LjY3NSA1NzMgNTQzLjQzNlY1NzQuOTE1QzU3MyA1OTcuNjk3IDU3MyA2MDkuMDg4IDU2OC41NjMgNjE3Ljc4OUM1NjQuNjYxIDYyNS40NDQgNTU4LjQzNCA2MzEuNjY2IDU1MC43NzQgNjM1LjU2NkM1NDIuMDY3IDY0MCA1MzAuNjY4IDY0MCA1MDcuODcxIDY0MEgyOTIuMTI5QzI2OS4zMzIgNjQwIDI1Ny45MzMgNjQwIDI0OS4yMjYgNjM1LjU2NkMyNDEuNTY2IDYzMS42NjYgMjM1LjMzOSA2MjUuNDQ0IDIzMS40MzcgNjE3Ljc4OUMyMjcgNjA5LjA4OCAyMjcgNTk3LjY5NyAyMjcgNTc0LjkxNVY1NDMuNDM2QzIyNyA1MDguNjc1IDIyNyA0OTEuMjk0IDIzNC45NjUgNDc2LjcxNkMyNDEuMjQ0IDQ2NS4yMjIgMjU2LjQzMyA0NTMuODg2IDI3Ni45MTggNDQwLjM5MkMyODUuMzIyIDQzNC44NTYgMjg5LjUyNSA0MzIuMDg4IDI5NC4zNTcgNDMyLjA2QzI5OS4xOSA0MzIuMDMyIDMwMy40OTQgNDM0Ljc5NyAzMTIuMSA0NDAuMzI2TDMzMy41MjcgNDU0LjA5MUMzNDMuMTIyIDQ2MC4yNTQgMzQ3LjkxOSA0NjMuMzM2IDM0Ny45MTkgNDY3Ljc3OEMzNDcuOTE5IDQ3Mi4yMiAzNDMuMTIyIDQ3NS4zMDEgMzMzLjUyOCA0ODEuNDY1TDMzMy41MjcgNDgxLjQ2NUwyNzMuMjIgNTIwLjIwOEMyNjQuNzA5IDUyNS42NzUgMjU5LjU2NSA1MzUuMDkxIDI1OS41NjUgNTQ1LjIwMkMyNTkuNTY1IDU2MS42MTIgMjcyLjg3NyA1NzQuOTE1IDI4OS4yOTkgNTc0LjkxNUg1MTAuNzAxQzUyNy4xMjMgNTc0LjkxNSA1NDAuNDM1IDU2MS42MTIgNTQwLjQzNSA1NDUuMjAyQzU0MC40MzUgNTM1LjA5MSA1MzUuMjkxIDUyNS42NzUgNTI2Ljc4IDUyMC4yMDhMMjg2LjgyIDM2Ni4wNTNDMjU3LjU2IDM0Ny4yNTYgMjQyLjkyOSAzMzcuODU3IDIzNC45NjUgMzIzLjI3OUMyMjcgMzA4LjcgMjI3IDI5MS4zMTkgMjI3IDI1Ni41NThWMjI1LjA4NVoiIGZpbGw9IiNGRkZGRkYiLz4KPC9zdmc+Cg=="></a>
    <a href="https://huggingface.co/sand-ai"><img alt="Hugging Face"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Sand AI-ffc107?color=ffc107&logoColor=white"/></a>
    <a href="https://x.com/SandAI_HQ"><img alt="Twitter Follow"
    src="https://img.shields.io/badge/Twitter-Sand%20AI-white?logo=x&logoColor=white"/></a>
    <a href="https://discord.gg/hgaZ86D7Wv"><img alt="Discord"
    src="https://img.shields.io/badge/Discord-Sand%20AI-7289da?logo=discord&logoColor=white&color=7289da"/></a>
</p>

<img src="docs/assets/magi_compiler_overview.jpeg" alt="MagiCompiler Overview" width="100%">

</div>

---

## 📢 Latest News

- **[03/25/2026]** ⚡️ **[LightX2V-MagiCompiler](https://github.com/SandAI-org/LightX2V-MagiCompiler) is now available!** This fork of [LightX2V](https://github.com/ModelTC/lightx2v) showcases how to seamlessly integrate MagiCompiler into a SOTA framework. With **minimal code changes**, it unlocks even greater acceleration! Try it out, check the [benchmark](https://github.com/SandAI-org/MagiCompiler#-6-benchmark) for details, and stay tuned for more integration demos!

- **[03/23/2026]** 🚀 **MagiCompiler is officially open-sourced!** Delivering whole-graph compilation for multi-modality inference and FSDP-aware whole-layer compilation for large model training.

---

## 📖 About

MagiCompiler is an advanced compiler and runtime augmentation framework built on top of `torch.compile`. Designed specifically for large-scale Transformer-like architectures, it addresses the critical bottlenecks of memory walls and operator overheads.

By stepping beyond traditional local operator optimization, MagiCompiler introduces system-level optimizations, seamlessly accelerating both **training** and **multi-modality inference** workloads with minimal code intrusion.

---

## 💡 Design Philosophy

### Compiler as Manager

> *"Reimagining the compiler: from generating kernels to orchestrating the entire dataflow."*

MagiCompiler's core philosophy is **Compiler as Manager**. We believe a modern deep learning compiler should not be restricted to mere kernel fusion. Instead, it acts as a global manager that owns the full lifecycle of execution. MagiCompiler actively manages subgraph dispatching, dynamically orchestrates dataflow (like offloading and prefetching), and controls memory allocation, ensuring optimal balance between compute efficiency and memory footprint.

### Key Features

#### 🎯 1. Unified Inference & Training
Tailored for Transformer-like architectures with scenario-specific strategies:
- **Inference**: Achieves **full-graph capture** across Transformer boundaries, maximizing kernel fusion scope.
- **Training**: Introduces **FSDP-aware layer-wise compilation**. Unlocks aggressive cross-op fusion while keeping distributed parameter sharding entirely transparent.

#### ⚡️ 2. Easy to Use, Free Gain, Plug and Play
No complex model refactoring needed. Just two decorators deliver up to **20%+ extra speedups** out-of-the-box, seamlessly integrating into SOTA multi-modality frameworks.

#### 🧠 3. Smart Asynchronous Offloading
For memory-constrained setups, our built-in **selective offloading policy** perfectly overlaps H2D transfers with computation, eliminating pipeline bubbles.

#### ♻️ 4. Heuristic Activation Recomputation
Say goodbye to manual `torch.utils.checkpoint`. MagiCompiler automatically saves compute-bound ops (e.g., MatMul, Attention) and recomputes memory-bound ones, slashing peak memory without sacrificing throughput.

#### 🛠 5. Magi Depyf Introspection
Meet **magi_depyf**, MagiCompiler’s native introspection toolkit. Compilation timelines, decompiled bytecode flows, split subgraphs, and backend artifacts are automatically dumped into the cache path as organized, human-readable files for easier debugging.

---

## ⚙️ Installation

**Requirements:**
- Python >= 3.12
- PyTorch >= 2.9
- CUDA Toolkit

> **Recommended for reproducibility:** start from the prebuilt Docker image first, then run examples inside the container.

```bash
# Option A (recommended) — Use prebuilt image
# Step 1 — Pull the image
docker pull sandai/magi-compiler:latest
# Step 2 - Start the container
docker run --name my-magi-compiler -it -d --privileged --gpus all --network host --ipc host \
  -v /path/on/host:/workspace sandai/magi-compiler:latest /bin/bash
# Step 3 - Attach the container
docker exec -it my-magi-compiler /bin/bash

# Option B — Local source installation
# Step 1 — Clone the repo
git clone https://github.com/SandAI-org/MagiCompiler.git
cd MagiCompiler

# Step 2 — System dependencies (optional, for FX graph visualization; Debian/Ubuntu)
sudo apt update && sudo apt install -y graphviz

# Step 3 — Python dependencies
pip install -r requirements.txt

# Step 4 — Install MagiCompiler (pick one)
pip install .   # End users (recommended)
# pip install -e . --no-build-isolation --config-settings editable_mode=compat  # Developer / editable
```

---

## 🚀 Quick Start

### 🧹 1. One Decorator to Rule Them All (`@magi_compile`)
Remove scattered `torch.compile` or `torch.compiler.disable` calls. Decorate your core Transformer block once for automatic full-graph capture and dynamic shape support (defaulting to dim 0).

```python
import torch
from torch import nn
from magi_compiler import magi_compile

# Decorate your core module once. No more scattered compile tweaks!
@magi_compile
class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = Attention(hidden_dim)
        self.mlp = MLP(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        x = x + self.attn(x, mask)
        x = x + self.mlp(x)
        return x

model = TransformerBlock(hidden_dim=1024).cuda()

# Execute normally - whole-graph compilation handles dynamic batches automatically!
out = model(torch.randn(4, 128, 1024, device="cuda"), None)
out = model(torch.randn(8, 128, 1024, device="cuda"), None)
```

### 🛠️ 2. Bridge Custom Kernels (`@magi_register_custom_op`)
Using custom kernels (FlashAttention, MoE routers) that break FX tracing? Don't disable compilation. Wrap them to teach the compiler how to handle them during graph partitioning and recomputation.

```python
from magi_compiler import magi_register_custom_op

@magi_register_custom_op(
    name="athena::flash_attn",
    infer_output_meta_fn=["q"],       # Output shape matches parameter 'q'
    is_subgraph_boundary=True,        # Split graph here for subgraph compilation
    is_compute_sensitive=True,        # Retain this output during recomputation
)
def flash_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    ... # Your custom kernel or C++ extension
```

### 🔧 3. Advanced Configurations
Explore `magi_compiler/config.py` for power-user features like custom backend toggles and fine-grained memory management. *(Comprehensive guides for popular training/inference frameworks are coming soon!)*

---

## 📊 Benchmark

### 🔥 H100 Extreme Acceleration

On a single NVIDIA H100, MagiCompiler outperforms current SOTA solutions (like LightX2V) by 9% to 26% across mainstream open-source video generation models.

<p align="center">
<img src="docs/assets/h100_inference.png" alt="H100 Inference Benchmark" width="85%">
</p>

### 💻 RTX 5090 Near Real-Time

Thanks to our underlying JIT offloading engine, [daVinci-MagiHuman](https://github.com/GAIR-NLP/daVinci-MagiHuman) achieves near real-time speeds, even on heavily VRAM-constrained consumer GPUs.

<p align="center">
<img src="docs/assets/rtx5090_inference.png" alt="RTX 5090 Inference Latency" width="85%">
</p>

---

## 🗺 Roadmap

We are actively developing MagiCompiler. Here is a glimpse into our upcoming milestones:

- [ ] **Ecosystem Integration**: Benchmarks and out-of-the-box integration guides for popular frameworks (e.g., `sglang-diffusion`, `vllm-omni`, and `LLaMA` training).
- [ ] **Official Hub & Tech Blog**: A dedicated website for advanced tutorials, documentation, and frontier engineering insights.
- [ ] **Hardware-Aware Auto-Scheduler**: An adaptive engine that dynamically orchestrates optimal strategies (auto-recomputation boundaries, offloading) based on your hardware constraints.
- [ ] **Next-Gen Custom Backend (v2.0)**: Pushing hardware limits with extreme kernel-level efficiency, native distributed communication and MegaKernels.

---

## 📝 Citation

If you find MagiCompiler useful in your research or production, please consider citing us:

```bibtex
@software{magi_compiler_2026,
  author = {Hongyu Jia and Zhiyao Cen and Taoran Wang and Yunbo Zhang},
  title = {MagiCompiler: Break the Boundaries of Local Compilation for Large Models},
  year = {2026},
  url = {https://github.com/SandAI-org/MagiCompiler}
}
```

---

## 🙏 Acknowledgement

MagiCompiler is deeply inspired by and builds upon the shoulders of giants. We extend our heartfelt gratitude to the [PyTorch](https://pytorch.org/) team for their foundational work on `torch.compile` and `torch.fx`, and to the [vLLM](https://github.com/vllm-project/vllm) community for their pioneering contributions to large model inference.

**We are moving fast, and we want you on board!** MagiCompiler is under rapid development. If you are passionate about pushing the limits of large model compilation, we'd love to have you with us. From opening issues and discussing architectures to submitting core PRs, every contribution matters. Let's engineer the future of AI infrastructure together!

---

## ⭐ Star History

<div align="center">
  <a href="https://star-history.com/#SandAI-org/MagiCompiler&Date">
    <img src="https://api.star-history.com/svg?repos=SandAI-org/MagiCompiler&type=Date" alt="Star History Chart" style="max-width: 60%; height: auto;"/>
  </a>
</div>

---

## 📜 License

This project is licensed under the [Apache License 2.0](LICENSE).
