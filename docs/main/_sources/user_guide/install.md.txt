# Installation

```{contents}
:local: true
```

## Requirements

- Python >= 3.12
- PyTorch >= 2.9
- CUDA Toolkit

:::{tip}
For reproducibility, we recommend starting from the prebuilt Docker image first,
then running examples inside the container.
:::

## Option A (recommended) — Prebuilt Docker Image

```bash
# Step 1 — Pull the image
docker pull sandai/magi-compiler:latest

# Step 2 — Start the container
docker run --name my-magi-compiler -it -d --privileged --gpus all --network host --ipc host \
  -v /path/on/host:/workspace sandai/magi-compiler:latest /bin/bash

# Step 3 — Attach the container
docker exec -it my-magi-compiler /bin/bash
```

## Option B — Local Source Installation

```bash
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

## Optional — Install CUTLASS for matmul epilogue fusion

Required for the CUTLASS-based matmul + epilogue fusion pass (`sm_90` / `sm_120`).
Without CUTLASS the compiler still works but skips this optimization.

```bash
git clone --depth 1 https://github.com/NVIDIA/cutlass.git /usr/local/cutlass
# Or specify a custom path:
#   git clone --depth 1 https://github.com/NVIDIA/cutlass.git /your/path
#   export MAGI_CUTLASS_ROOT=/your/path
export CUDACXX=${CUDA_INSTALL_PATH}/bin/nvcc
mkdir /usr/local/cutlass/build && cd /usr/local/cutlass/build
cmake .. -DCUTLASS_NVCC_ARCHS=90a   # NVIDIA Hopper GPU architecture
# cmake .. -DCUTLASS_NVCC_ARCHS=120a  # NVIDIA consumer Blackwell (RTX 50 series)
```
