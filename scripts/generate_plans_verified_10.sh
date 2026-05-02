#!/usr/bin/env bash
# Generate DAG plans for SWE-bench Verified (10 instances)
# This ONLY generates plans, does NOT run Docker or Agent inference.
# Usage: bash scripts/generate_plans_verified_10.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"

PLAN_DIR="runs/langgraph/plans"
LOG_DIR="runs/langgraph/logs"

mkdir -p "${PLAN_DIR}" "${LOG_DIR}"

API_KEY="${OPENCODE_API_KEY:?OPENCODE_API_KEY is not set}"
MODEL="deepseek-v4-flash"
BASE_URL="https://opencode.ai/zen/go/v1"

INSTANCE_IDS=(
  astropy__astropy-12907
  astropy__astropy-13033
  astropy__astropy-13236
  astropy__astropy-13398
  astropy__astropy-13453
  astropy__astropy-13579
  astropy__astropy-13977
  astropy__astropy-14096
  astropy__astropy-14182
  astropy__astropy-14309
)

echo "========================================"
echo "Generate DAG Plans Only (10 inst)"
echo "Model: ${MODEL}"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Instances: ${#INSTANCE_IDS[@]}"
echo "========================================"

python run_with_langgraph.py \
    --dataset_name_or_path princeton-nlp/SWE-bench_Verified \
    --split test \
    --instance_ids "${INSTANCE_IDS[@]}" \
    --model "${MODEL}" \
    --api-key "${API_KEY}" \
    --base-url "${BASE_URL}" \
    --plan_dir "${PLAN_DIR}" \
    --generate-plans-only \
    --delay 1.0 \
    2>&1 | tee "${LOG_DIR}/generate_plans.log"

echo "========================================"
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Plans saved to: ${PLAN_DIR}/"
echo ""
echo "Next step: run the agent pipeline with:"
echo "  bash scripts/run_langgraph_verified_10.sh"
echo "========================================"
