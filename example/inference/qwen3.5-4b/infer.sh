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
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." &> /dev/null && pwd)

export MODE=${MODE:-all}
PYTHON_BIN=${PYTHON_BIN:-python}

if [ "${NSYS_PROFILE:-true}" = "true" ]; then
    mkdir -p "$PROJECT_ROOT/nsys_reports"

    NSYS_OUTPUT="$PROJECT_ROOT/nsys_reports/nsys_qwen35_4b_${MODE}_$(date +%Y%m%d_%H%M%S)"
    echo "${MODE} nsys report: ${NSYS_OUTPUT}.nsys-rep"

    NSYS_CMD=(nsys profile --force-overwrite true -o "$NSYS_OUTPUT" --trace=cuda,nvtx --capture-range=cudaProfilerApi)
else
    NSYS_CMD=()
fi

export MAGI_COMPILE_CACHE_ROOT_DIR=${MAGI_COMPILE_CACHE_ROOT_DIR:-"$PROJECT_ROOT/.cache"}
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

"${NSYS_CMD[@]}" "$PYTHON_BIN" -u "$SCRIPT_DIR/infer.py" "$@"
