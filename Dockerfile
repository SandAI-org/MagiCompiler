# syntax=docker/dockerfile:1.7
FROM nvcr.io/nvidia/pytorch:25.10-py3

ARG FLASH_ATTENTION_COMMIT_ID="b613d9e2c8475945baff3fd68f2030af1b890acf"

# CUTLASS — source is always cloned (the magi_compiler EVT-fusion path
# JIT-includes its headers and our /usr/local/cutlass tree is the readable
# reference checkout). The CMake-driven profiler/library is compiled
# only for supported targets; every other arch gets headers only.
#
# Supported NVCC arch strings (CUTLASS_NVCC_ARCHS):
#   90a  — Hopper (H100, compute_cap 9.x, WGMMA/TMA)
#   120a — consumer Blackwell (RTX 50 series, compute_cap 12.x)
#
# Override behaviour with build args:
#   --build-arg CUTLASS_BUILD=yes|no|auto
#     yes  — force cmake configure (requires CUTLASS_NVCC_ARCHS or a GPU)
#     no   — skip cmake even if a supported GPU is present
#     auto — (default) compile iff nvidia-smi reports 9.x or 12.x
#   --build-arg CUTLASS_NVCC_ARCHS=90a|120a
ARG CUTLASS_COMMIT_ID="f74fea9ce35868d3ae9f8d1dce1969d7250d3f90"
ARG CUTLASS_BUILD="auto"
ARG CUTLASS_NVCC_ARCHS=""

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace

RUN --mount=type=secret,id=http_proxy,required=false \
    --mount=type=secret,id=https_proxy,required=false \
    export http_proxy="$(cat /run/secrets/http_proxy 2>/dev/null || true)" && \
    export https_proxy="$(cat /run/secrets/https_proxy 2>/dev/null || true)" && \
    apt-get -qq update && \
    DEBIAN_FRONTEND=noninteractive apt-get -qq install -y --no-install-recommends \
    ca-certificates \
    git \
    build-essential \
    cmake \
    ninja-build && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

RUN pip install --upgrade pip setuptools wheel ninja

RUN --mount=type=secret,id=http_proxy,required=false \
    --mount=type=secret,id=https_proxy,required=false \
    export http_proxy="$(cat /run/secrets/http_proxy 2>/dev/null || true)" && \
    export https_proxy="$(cat /run/secrets/https_proxy 2>/dev/null || true)" && \
    mkdir -p /tmp/flash-attention && \
    cd /tmp/flash-attention && \
    git init && \
    git remote add origin https://github.com/Dao-AILab/flash-attention.git && \
    git fetch origin ${FLASH_ATTENTION_COMMIT_ID} --depth 1 && \
    git checkout ${FLASH_ATTENTION_COMMIT_ID} && \
    (git submodule update --init --recursive --depth 1 --jobs 8 || git submodule update --init --recursive --depth 1 --jobs 1) && \
    cd /tmp/flash-attention/hopper && \
    python setup.py install && \
    python_path=$(python -c "import site; print(site.getsitepackages()[0])") && \
    mkdir -p ${python_path}/flash_attn_3 && \
    cp /tmp/flash-attention/hopper/flash_attn_interface.py ${python_path}/flash_attn_3/ && \
    rm -rf /tmp/flash-attention


RUN --mount=type=secret,id=http_proxy,required=false \
    --mount=type=secret,id=https_proxy,required=false \
    export http_proxy="$(cat /run/secrets/http_proxy 2>/dev/null || true)" && \
    export https_proxy="$(cat /run/secrets/https_proxy 2>/dev/null || true)" && \
    mkdir -p /usr/local/cutlass && \
    cd /usr/local/cutlass && \
    git init -q && \
    git remote add origin https://github.com/NVIDIA/cutlass.git && \
    git fetch origin ${CUTLASS_COMMIT_ID} --depth 1 && \
    git checkout ${CUTLASS_COMMIT_ID} && \
    (git submodule update --init --recursive --depth 1 --jobs 8 || \
     git submodule update --init --recursive --depth 1 --jobs 1)


RUN set -eu; \
    _cutlass_arch_from_gpu() { \
        if ! command -v nvidia-smi >/dev/null 2>&1; then return 1; fi; \
        cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')"; \
        case "${cap}" in \
            9.*) echo "90a" ;; \
            12.*) echo "120a" ;; \
            *) return 1 ;; \
        esac; \
    }; \
    if [ -n "${CUTLASS_NVCC_ARCHS}" ]; then \
        NVCC_ARCHS="${CUTLASS_NVCC_ARCHS}"; \
        echo "[CUTLASS] Using CUTLASS_NVCC_ARCHS=${NVCC_ARCHS} (build-arg override)."; \
    elif arch="$(_cutlass_arch_from_gpu)"; then \
        NVCC_ARCHS="${arch}"; \
        echo "[CUTLASS] nvidia-smi → CUTLASS_NVCC_ARCHS=${NVCC_ARCHS}."; \
    else \
        NVCC_ARCHS=""; \
    fi; \
    case "${CUTLASS_BUILD}" in \
        no) echo "[CUTLASS] CUTLASS_BUILD=no — skipping cmake configure."; exit 0 ;; \
        yes) \
            if [ -z "${NVCC_ARCHS}" ]; then \
                echo "[CUTLASS] CUTLASS_BUILD=yes but no arch: set CUTLASS_NVCC_ARCHS=90a|120a or build on a 9.x/12.x GPU."; \
                exit 1; \
            fi; \
            DO_BUILD=1 ;; \
        auto) \
            if [ -z "${NVCC_ARCHS}" ]; then \
                echo "[CUTLASS] No sm_90/sm_120 GPU and no CUTLASS_NVCC_ARCHS — skipping cmake (headers still available)."; \
                exit 0; \
            fi; \
            DO_BUILD=1 ;; \
        *) echo "[CUTLASS] Unknown CUTLASS_BUILD=${CUTLASS_BUILD}"; exit 1 ;; \
    esac; \
    case "${NVCC_ARCHS}" in \
        90a|120a) ;; \
        *) echo "[CUTLASS] Unsupported CUTLASS_NVCC_ARCHS=${NVCC_ARCHS} (expected 90a or 120a)."; exit 1 ;; \
    esac; \
    [ -n "${DO_BUILD:-}" ] && cd /usr/local/cutlass && \
    export CUDACXX="${CUDA_INSTALL_PATH:-${CUDA_HOME:-/usr/local/cuda}}/bin/nvcc" && \
    mkdir -p build && cd build && \
    cmake .. -DCUTLASS_NVCC_ARCHS="${NVCC_ARCHS}"

RUN --mount=type=secret,id=http_proxy,required=false \
    --mount=type=secret,id=https_proxy,required=false \
    export http_proxy="$(cat /run/secrets/http_proxy 2>/dev/null || true)" && \
    export https_proxy="$(cat /run/secrets/https_proxy 2>/dev/null || true)" && \
    apt-get -qq update && \
    DEBIAN_FRONTEND=noninteractive apt-get -qq install -y --no-install-recommends \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

WORKDIR /app
