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

### Distributed args ###
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
DISTRIBUTED_ARGS="--nnodes=$NNODES --node_rank=$NODE_RANK --nproc_per_node=$GPUS_PER_NODE --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT"

### Nsys args ###
NSYS_PROFILE=${NSYS_PROFILE:-true}

if [ "$NSYS_PROFILE" = true ]; then
    mkdir -p "$PROJECT_ROOT/nsys_reports"

    BASE_NAME="nsys_llama3"
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    NSYS_SUFFIX="ts_${TIMESTAMP}"

    [ -n "$WORLD_SIZE" ] && NSYS_SUFFIX="${NSYS_SUFFIX}_worldsize_${WORLD_SIZE}"
    [ -n "$COMPILE_MODE" ] && NSYS_SUFFIX="${NSYS_SUFFIX}_compile_${COMPILE_MODE}"
    [ -n "$CUDA_GRAPH_MODE" ] && NSYS_SUFFIX="${NSYS_SUFFIX}_cudagraph_${CUDA_GRAPH_MODE}"

    NSYS_OUTPUT="$PROJECT_ROOT/nsys_reports/${BASE_NAME}_${NSYS_SUFFIX}"

    NSYS_CMD="nsys profile --force-overwrite true -o $NSYS_OUTPUT --trace=cuda,nvtx --capture-range=cudaProfilerApi"
else
    NSYS_CMD=""
fi

### Environment Variables For Debugging ###
export ENABLE_REMOTE_DEBUG=${ENABLE_REMOTE_DEBUG:-false}
export MAGI_COMPILE_CACHE_ROOT_DIR=${MAGI_COMPILE_CACHE_ROOT_DIR:-"$PROJECT_ROOT/.cache"}
export MAGI_ENABLE_FX_GRAPH_VIZ=${MAGI_ENABLE_FX_GRAPH_VIZ:-false}

$NSYS_CMD torchrun $DISTRIBUTED_ARGS $SCRIPT_DIR/train.py \
    $NSYS_ARGS
