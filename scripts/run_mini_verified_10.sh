#!/usr/bin/env bash
# Mini-SWE-Agent - SWE-bench Verified (10 instances smoke test)
# Model: deepseek-v4-flash via OpenCode API
# Usage: bash scripts/run_mini_verified_10.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"

RUNS_DIR="runs/mini-swe"
LOG_DIR="${RUNS_DIR}/logs"
PRED_DIR="${RUNS_DIR}/predictions"

mkdir -p "${LOG_DIR}" "${PRED_DIR}"

API_KEY="${OPENCODE_API_KEY:?OPENCODE_API_KEY is not set}"
MODEL="openai/deepseek-v4-flash"
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
echo "Mini-SWE Verified Smoke Test (10 inst)"
echo "Model: ${MODEL}"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Instances: ${#INSTANCE_IDS[@]}"
echo "Logs:        ${LOG_DIR}"
echo "Predictions: ${PRED_DIR}"
echo "========================================"

# Mini-SWE-Agent only supports --filter for a single instance,
# so we loop over them sequentially.

SKIPPED=0
RAN=0
FAILED=0

for id in "${INSTANCE_IDS[@]}"; do
  echo ""
  TRAJ_FILE="${id}/${id}.traj.json"

  # Skip if already completed
  if [ -f "${TRAJ_FILE}" ]; then
    echo "===== [Mini-SWE] SKIP ${id} (trajectory already exists) ====="
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  echo "===== [Mini-SWE] Running ${id} ====="
  if uv run mini-extra swebench \
      -m "${MODEL}" \
      -c .venv/lib/python3.12/site-packages/minisweagent/config/benchmarks/swebench.yaml \
      -c "model.model_kwargs.api_base=${BASE_URL}" \
      -c "model.model_kwargs.api_key=${API_KEY}" \
      -c "model.cost_tracking=ignore_errors" \
      --subset princeton-nlp/SWE-bench_Verified \
      --split test \
      --filter "${id}" \
      2>&1 | tee "${LOG_DIR}/${id}.log"; then
    echo "===== [Mini-SWE] Finished ${id} ====="
    RAN=$((RAN + 1))
  else
    echo "===== [Mini-SWE] FAILED ${id} ====="
    FAILED=$((FAILED + 1))
  fi
  sleep 2
done

echo ""
echo "========================================"
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results: ${RAN} ran, ${FAILED} failed, ${SKIPPED} skipped"
echo "Logs:        ${LOG_DIR}/"
echo "Trajectories: ${PROJECT_DIR}/{id}/{id}.traj.json"
echo ""
echo "To generate predictions for eval, run:"
echo "  python scripts/extract_predictions.py"
echo "========================================"
