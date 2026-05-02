#!/usr/bin/env bash
set -euo pipefail

MODEL="deepseek-v4-flash"
RUN_ID="${1:-eval_$(date +%Y%m%d_%H%M%S)}"
EVAL_DIR="runs/eval"
RESULT_DIR="${EVAL_DIR}/${RUN_ID}"

if [[ -d "${RESULT_DIR}" ]]; then
  echo "[ERROR] Run directory already exists: ${RESULT_DIR}"
  echo "Pass a different run_id: bash eval.sh <run_id>"
  exit 1
fi

python -m swebench.harness.run_evaluation \
          --dataset_name princeton-nlp/SWE-bench_Verified  \
          --predictions_path runs/langgraph/predictions/${MODEL}__SWE-bench_Verified__test.jsonl \
          --instance_ids astropy__astropy-14309 \
          --run_id "${RUN_ID}" \
          --report_dir "${EVAL_DIR}" \
          --max_workers 1
