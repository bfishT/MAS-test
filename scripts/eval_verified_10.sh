#!/usr/bin/env bash
# Evaluate SWE-bench predictions for the 10-instance smoke test.
#
# Usage:
#   bash scripts/eval_verified_10.sh --agent langgraph          # auto-select langgraph predictions
#   bash scripts/eval_verified_10.sh --agent mini-swe           # auto-select mini-swe predictions
#   bash scripts/eval_verified_10.sh --predictions path.jsonl   # custom predictions file
#   bash scripts/eval_verified_10.sh --predictions path.jsonl --run-id my_run --workers 4
#
# Results go to: runs/eval/<run_id>/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"

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

MODEL="deepseek-v4-flash"
AGENT=""
PREDICTIONS_PATH=""
RUN_ID=""
MAX_WORKERS="2"
EVAL_DIR="runs/eval"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --agent)
      AGENT="$2"; shift 2 ;;
    --predictions)
      PREDICTIONS_PATH="$2"; shift 2 ;;
    --run-id)
      RUN_ID="$2"; shift 2 ;;
    --workers)
      MAX_WORKERS="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1"
      echo "Usage: bash scripts/eval_verified_10.sh --agent langgraph|mini-swe [--predictions path] [--run-id id] [--workers N]"
      exit 1 ;;
  esac
done

# Determine predictions path
if [[ -z "${PREDICTIONS_PATH}" ]]; then
  if [[ -z "${AGENT}" ]]; then
    echo "[ERROR] Specify --agent langgraph|mini-swe or --predictions <path>"
    echo ""
    echo "Available prediction files:"
    find runs/ -name "*.jsonl" 2>/dev/null || echo "  (none)"
    exit 1
  fi
  PREDICTIONS_PATH="runs/${AGENT}/predictions/${MODEL}__SWE-bench_Verified__test.jsonl"
fi

# Determine run_id
if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="${AGENT:-eval}_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "${EVAL_DIR}"

RESULT_DIR="${EVAL_DIR}/${RUN_ID}"
REPORT_FILE="${EVAL_DIR}/${MODEL}.${RUN_ID}.json"

if [[ -d "${RESULT_DIR}" ]]; then
  echo "[ERROR] Run directory already exists: ${RESULT_DIR}"
  echo "Use --run-id <new_id> or remove it first: rm -rf ${RESULT_DIR}"
  exit 1
fi

if [[ ! -f "${PREDICTIONS_PATH}" ]]; then
  echo "[ERROR] Predictions file not found: ${PREDICTIONS_PATH}"
  echo ""
  echo "Looking for available prediction files:"
  find runs/ -name "*.jsonl" 2>/dev/null || echo "  (none found)"
  exit 1
fi

echo "========================================"
echo "Evaluating Verified Smoke Test (10 inst)"
echo "Agent:       ${AGENT:-custom}"
echo "Predictions: ${PREDICTIONS_PATH}"
echo "Run ID:      ${RUN_ID}"
echo "Max Workers: ${MAX_WORKERS}"
echo "Results:     ${RESULT_DIR}"
echo "Report:      ${REPORT_FILE}"
echo "Started at:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --predictions_path "${PREDICTIONS_PATH}" \
    --instance_ids "${INSTANCE_IDS[@]}" \
    --run_id "${RUN_ID}" \
    --max_workers "${MAX_WORKERS}" \
    --report_dir "${EVAL_DIR}"

echo ""
echo "========================================"
echo "Evaluation finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results: ${RESULT_DIR}"
echo "Report:  ${REPORT_FILE}"
echo "========================================"
