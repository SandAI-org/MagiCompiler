# syntax=docker/dockerfile:1.7
FROM nvcr.io/nvidia/pytorch:25.10-py3

ARG FLASH_ATTENTION_COMMIT_ID="b613d9e2c8475945baff3fd68f2030af1b890acf"

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
    apt-get -qq update && \
    DEBIAN_FRONTEND=noninteractive apt-get -qq install -y --no-install-recommends \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

WORKDIR /app
