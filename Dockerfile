# syntax=docker/dockerfile:1.7
FROM nvcr.io/nvidia/pytorch:25.10-py3

ARG FLASH_ATTENTION_COMMIT_ID="b613d9e2c8475945baff3fd68f2030af1b890acf"

# CUTLASS — source is always cloned (the magi_compiler EVT-fusion path
# JIT-includes its headers and our /opt/cutlass tree is the readable
# reference checkout). The CMake-driven profiler/library is compiled
# *only* when the build host is an RTX 5090 (sm_120, Blackwell consumer);
# every other arch gets the source tree but no built artefacts.
#
# Override behaviour with a build arg:
#   --build-arg CUTLASS_BUILD=yes   force compile (e.g. on a build farm
#                                   without a GPU but targeting sm_120)
#   --build-arg CUTLASS_BUILD=no    force skip even if 5090 detected
#   --build-arg CUTLASS_BUILD=auto  (default) compile iff nvidia-smi
#                                   reports compute_cap == 12.x
ARG CUTLASS_COMMIT_ID="f74fea9ce35868d3ae9f8d1dce1969d7250d3f90"
ARG CUTLASS_BUILD="auto"

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
    mkdir -p /opt/cutlass && \
    cd /opt/cutlass && \
    git init -q && \
    git remote add origin https://github.com/NVIDIA/cutlass.git && \
    git fetch origin ${CUTLASS_COMMIT_ID} --depth 1 && \
    git checkout ${CUTLASS_COMMIT_ID} && \
    (git submodule update --init --recursive --depth 1 --jobs 8 || \
     git submodule update --init --recursive --depth 1 --jobs 1)


RUN set -eu; \
    case "${CUTLASS_BUILD}" in \
        no) echo "[CUTLASS] CUTLASS_BUILD=no — skipping cmake configure."; exit 0 ;; \
        yes) DO_BUILD=1 ;; \
        auto) \
            if command -v nvidia-smi >/dev/null 2>&1 && \
               nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
                 | head -n1 | grep -Eq '^12\.'; then \
                echo "[CUTLASS] nvidia-smi reports sm_120 — running cmake configure."; \
                DO_BUILD=1; \
            else \
                echo "[CUTLASS] No sm_120 detected at build time — skipping cmake (headers still available)."; \
                exit 0; \
            fi ;; \
        *) echo "[CUTLASS] Unknown CUTLASS_BUILD=${CUTLASS_BUILD}"; exit 1 ;; \
    esac; \
    [ -n "${DO_BUILD:-}" ] && cd /opt/cutlass && \
    export CUDACXX="${CUDA_INSTALL_PATH:-${CUDA_HOME:-/usr/local/cuda}}/bin/nvcc" && \
    mkdir -p build && cd build && \
    cmake .. -DCUTLASS_NVCC_ARCHS=120a

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
