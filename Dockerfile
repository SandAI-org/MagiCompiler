ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.10-py3
FROM ${BASE_IMAGE}

ARG no_proxy
ARG http_proxy
ARG https_proxy
ARG CUTLASS_COMMIT_ID="f74fea9ce35868d3ae9f8d1dce1969d7250d3f90"

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace

# System packages.
RUN apt-get -qq update && \
    DEBIAN_FRONTEND=noninteractive apt-get -qq install -y --no-install-recommends \
    ca-certificates git graphviz && \
    rm -rf /var/lib/apt/lists/*

# CUTLASS headers only (EVT-fusion codegen needs the include path).
RUN git clone --depth 1 https://github.com/NVIDIA/cutlass.git /usr/local/cutlass && \
    cd /usr/local/cutlass && git fetch origin ${CUTLASS_COMMIT_ID} --depth 1 && \
    git checkout ${CUTLASS_COMMIT_ID}

# ── Dependency layer (cached unless requirements files change) ──────────
COPY requirements.txt requirements-test.txt /app/
RUN pip install --upgrade pip "setuptools<82" wheel && \
    grep -v "^triton" /app/requirements.txt > /tmp/req.txt && \
    grep -vE "^(torchvision|torchtitan)" /app/requirements-test.txt >> /tmp/req.txt && \
    pip install -r /tmp/req.txt && \
    pip install --no-deps torchtitan==0.2.0 && \
    rm -f /tmp/req.txt

# ── Source layer (only this and below re-run on code changes) ───────────
COPY . /app
WORKDIR /app
RUN pip install --no-build-isolation -e .
