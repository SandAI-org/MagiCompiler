% MagiCompiler documentation master file
% :github_url: https://github.com/SandAI-org/MagiCompiler

MagiCompiler documentation
===================================

**Overview :**

MagiCompiler is an advanced compiler and runtime augmentation framework built on top of `torch.compile`. Designed for large-scale Transformer-like architectures, it moves beyond traditional local operator optimization and introduces system-level optimizations—whole-graph capture, FSDP-aware layer-wise compilation, asynchronous offloading, and heuristic activation recomputation—to seamlessly accelerate both **training** and **multi-modality inference** with minimal code intrusion.

We are committed to continually improving the performance and generality of MagiCompiler for the broader research community. Stay tuned for exciting enhancements and new features on the horizon!

```{toctree}
:glob:
:maxdepth: 2
:caption: Contents

user_guide/toc
blog/toc

```
