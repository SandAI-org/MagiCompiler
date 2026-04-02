#!/bin/bash
# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." &> /dev/null && pwd)

GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NNODES=${NNODES:-1}

if [ "${NSYS_PROFILE:-true}" = "true" ]; then
    mkdir -p "$PROJECT_ROOT/nsys_reports"

    NSYS_OUTPUT="$PROJECT_ROOT/nsys_reports/nsys_llama3_ts_$(date +%Y%m%d_%H%M%S)_worldsize_$((GPUS_PER_NODE * NNODES))"
    [ -n "$COMPILE_MODE" ] && NSYS_OUTPUT="${NSYS_OUTPUT}_compile_${COMPILE_MODE}"
    [ -n "$CUDA_GRAPH_MODE" ] && NSYS_OUTPUT="${NSYS_OUTPUT}_cudagraph_${CUDA_GRAPH_MODE}"

    NSYS_CMD="nsys profile --force-overwrite true -o $NSYS_OUTPUT --trace=cuda,nvtx --capture-range=cudaProfilerApi"
fi

export MAGI_COMPILE_CACHE_ROOT_DIR=${MAGI_COMPILE_CACHE_ROOT_DIR:-"$PROJECT_ROOT/.cache"}

$NSYS_CMD torchrun \
    --nnodes=$NNODES \
    --node_rank=${NODE_RANK:-0} \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv-backend=c10d \
    --rdzv-endpoint=${MASTER_ADDR:-localhost}:${MASTER_PORT:-29500} \
    "$SCRIPT_DIR/train.py"
