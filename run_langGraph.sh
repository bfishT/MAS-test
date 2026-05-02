#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p runs/langgraph/logs runs/langgraph/predictions runs/langgraph/plans

API_KEY="${OPENCODE_API_KEY:?OPENCODE_API_KEY is not set}"
MODEL="deepseek-v4-flash"
BASE_URL="https://opencode.ai/zen/go/v1"
INSTANCE="astropy__astropy-14309"

python run_with_langgraph.py \
    --dataset_name_or_path princeton-nlp/SWE-bench_Verified \
    --split test \
    --instance_ids "${INSTANCE}" \
    --model "${MODEL}" \
    --api-key "${API_KEY}" \
    --base-url "${BASE_URL}" \
    --output_dir runs/langgraph/predictions \
    --plan_dir runs/langgraph/plans \
    2>&1 | tee "runs/langgraph/logs/${INSTANCE}.log"
