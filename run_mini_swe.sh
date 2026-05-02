#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

RUNS_DIR="runs/mini-swe"
LOG_DIR="${RUNS_DIR}/logs"
mkdir -p "${LOG_DIR}"

MODEL="openai/deepseek-v4-flash"
BASE_URL="https://opencode.ai/zen/go/v1"
API_KEY="${OPENCODE_API_KEY:?OPENCODE_API_KEY is not set}"
CONFIG=".venv/lib/python3.12/site-packages/minisweagent/config/benchmarks/swebench.yaml"
INSTANCE="astropy__astropy-14309"

# Skip if already completed
TRAJ_FILE="${INSTANCE}/${INSTANCE}.traj.json"
if [ -f "${TRAJ_FILE}" ]; then
  echo "SKIP: ${INSTANCE} already completed (${TRAJ_FILE} exists)"
  exit 0
fi

LOG_FILE="${LOG_DIR}/${INSTANCE}.log"
echo "Running ${INSTANCE} ... (log: ${LOG_FILE})"

if /usr/bin/time -v uv run mini-extra swebench \
    -m "${MODEL}" \
    -c "${CONFIG}" \
    -c "model.model_kwargs.api_base=${BASE_URL}" \
    -c "model.model_kwargs.api_key=${API_KEY}" \
    -c "model.cost_tracking=ignore_errors" \
    --subset princeton-nlp/SWE-bench_Verified \
    --split test \
    --filter "${INSTANCE}" \
    2>&1 | tee "${LOG_FILE}"; then
  echo "SUCCESS: ${INSTANCE}"
else
  echo "FAILED: ${INSTANCE}"
  exit 1
fi
