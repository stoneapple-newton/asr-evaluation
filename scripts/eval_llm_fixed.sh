#!/usr/bin/env bash
# Run the LLM fixed-input stage against a prior baseline run → score → report.
# Usage: ./scripts/eval_llm_fixed.sh <baseline_run_id> [dataset_manifest] [config] [out_root]
set -euo pipefail

if [ -z "${1:-}" ]; then
  echo "Usage: $0 <baseline_run_id> [dataset] [config] [out_root]" >&2
  exit 1
fi

BASELINE_RUN_ID="$1"
DATASET="${2:-data/eval_datasets/gold_v1/manifest.jsonl}"
CONFIG="${3:-eval/configs/llm_fixed_input.yaml}"
OUT_ROOT="${4:-results}"
BASELINE_DIR="${OUT_ROOT}/${BASELINE_RUN_ID}"

if [ ! -d "${BASELINE_DIR}" ]; then
  echo "ERROR: Baseline run dir not found: ${BASELINE_DIR}" >&2
  exit 1
fi

echo "==> Starting LLM fixed-input eval"
echo "    baseline   : ${BASELINE_RUN_ID}"
echo "    dataset    : ${DATASET}"
echo "    config     : ${CONFIG}"

python eval/runners/run_eval.py \
  --stage llm_fixed_input \
  --dataset       "${DATASET}" \
  --config        "${CONFIG}" \
  --input_run_dir "${BASELINE_DIR}" \
  --out_root      "${OUT_ROOT}"

RUN_DIR=$(ls -td "${OUT_ROOT}"/*/  2>/dev/null | head -1 | sed 's:/$::')
echo "==> Run directory: ${RUN_DIR}"

python eval/runners/score_eval.py \
  --dataset     "${DATASET}" \
  --predictions "${RUN_DIR}/predictions.jsonl" \
  --run_json    "${RUN_DIR}/run.json"

python eval/runners/generate_report.py \
  --run_json         "${RUN_DIR}/run.json" \
  --summary          "${RUN_DIR}/summary.json" \
  --scores           "${RUN_DIR}/scores.jsonl" \
  --out_dir          "${RUN_DIR}" \
  --baseline_summary "${BASELINE_DIR}/summary.json"

echo ""
echo "==> LLM fixed-input eval complete"
echo "    run_id : $(basename "${RUN_DIR}")"
echo "    report : ${RUN_DIR}/report.html"
