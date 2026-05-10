#!/usr/bin/env bash
# Run the end-to-end evaluation stage (ASR + diar + LLM) → score → report.
# Usage: ./scripts/eval_e2e.sh [dataset_manifest] [config] [out_root] [baseline_run_id]
set -euo pipefail

DATASET="${1:-data/eval_datasets/gold_v1/manifest.jsonl}"
CONFIG="${2:-eval/configs/e2e.yaml}"
OUT_ROOT="${3:-results}"
BASELINE_RUN_ID="${4:-}"

echo "==> Starting end-to-end eval"
echo "    dataset : ${DATASET}"
echo "    config  : ${CONFIG}"

python eval/runners/run_eval.py \
  --stage end_to_end \
  --dataset "${DATASET}" \
  --config  "${CONFIG}" \
  --out_root "${OUT_ROOT}"

RUN_DIR=$(ls -td "${OUT_ROOT}"/*/  2>/dev/null | head -1 | sed 's:/$::')
echo "==> Run directory: ${RUN_DIR}"

python eval/runners/score_eval.py \
  --dataset     "${DATASET}" \
  --predictions "${RUN_DIR}/predictions.jsonl" \
  --run_json    "${RUN_DIR}/run.json"

BASELINE_SUMMARY_ARG=""
if [ -n "${BASELINE_RUN_ID}" ] && [ -f "${OUT_ROOT}/${BASELINE_RUN_ID}/summary.json" ]; then
  BASELINE_SUMMARY_ARG="--baseline_summary ${OUT_ROOT}/${BASELINE_RUN_ID}/summary.json"
fi

python eval/runners/generate_report.py \
  --run_json "${RUN_DIR}/run.json" \
  --summary  "${RUN_DIR}/summary.json" \
  --scores   "${RUN_DIR}/scores.jsonl" \
  --out_dir  "${RUN_DIR}" \
  ${BASELINE_SUMMARY_ARG}

echo ""
echo "==> End-to-end eval complete"
echo "    run_id : $(basename "${RUN_DIR}")"
echo "    report : ${RUN_DIR}/report.html"
