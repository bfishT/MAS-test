#!/usr/bin/env bash
# LangGraph Agent - SWE-bench Verified (10 instances smoke test)
# Model: deepseek-v4-flash via OpenCode API
# Usage: bash scripts/run_langgraph_verified_10.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"

RUNS_DIR="runs/langgraph"
PRED_DIR="${RUNS_DIR}/predictions"
LOG_DIR="${RUNS_DIR}/logs"
PLAN_DIR="${RUNS_DIR}/plans"

mkdir -p "${PRED_DIR}" "${LOG_DIR}" "${PLAN_DIR}"

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

PRED_FILE="${PRED_DIR}/${MODEL}__SWE-bench_Verified__test.jsonl"

echo "========================================"
echo "LangGraph Verified Smoke Test (10 inst)"
echo "Model: ${MODEL}"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Instances: ${#INSTANCE_IDS[@]}"
echo "Predictions: ${PRED_FILE}"
echo "Logs:        ${LOG_DIR}"
echo "Plans:       ${PLAN_DIR}"
echo "========================================"

# Ensure langgraph is installed
if ! python -c "import langgraph" 2>/dev/null; then
    echo "[INFO] langgraph not found, installing..."
    uv pip install langgraph
fi

for id in "${INSTANCE_IDS[@]}"; do
    PER_LOG="${LOG_DIR}/${id}.log"
    SUMMARY_LOG="${LOG_DIR}/_summary.log"

    if grep -q "\"${id}\"" "${PRED_FILE}" 2>/dev/null; then
        echo "[Skip] ${id} already in predictions, remove from jsonl to re-run" | tee -a "${SUMMARY_LOG}"
        continue
    fi

    echo "[$(date '+%H:%M:%S')] Running ${id} ..." | tee -a "${SUMMARY_LOG}"
    python run_with_langgraph.py \
        --dataset_name_or_path princeton-nlp/SWE-bench_Verified \
        --split test \
        --instance_ids "${id}" \
        --model "${MODEL}" \
        --api-key "${API_KEY}" \
        --base-url "${BASE_URL}" \
        --output_dir "${PRED_DIR}" \
        --plan_dir "${PLAN_DIR}" \
        2>&1 | tee "${PER_LOG}"
    echo "[$(date '+%H:%M:%S')] Done ${id}" | tee -a "${SUMMARY_LOG}"
    sleep 2
done

echo "========================================"
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Predictions: ${PRED_FILE}"
echo "Logs:        ${LOG_DIR}"
echo "========================================"
