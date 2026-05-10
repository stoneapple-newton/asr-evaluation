#!/usr/bin/env bash
# Run the baseline evaluation stage (ASR + diarization) → score → report.
# Usage: ./scripts/eval_baseline.sh [dataset_manifest] [config] [out_root]
set -euo pipefail

DATASET="${1:-data/eval_datasets/gold_v1/manifest.jsonl}"
CONFIG="${2:-eval/configs/baseline.yaml}"
OUT_ROOT="${3:-results}"

TIMESTAMP=$(date -u +"%Y-%m-%dT%H-%M-%SZ")
echo "==> Starting baseline eval at ${TIMESTAMP}"
echo "    dataset : ${DATASET}"
echo "    config  : ${CONFIG}"
echo "    out_root: ${OUT_ROOT}"

# Stage 1: run predictions
python eval/runners/run_eval.py \
  --stage baseline \
  --dataset "${DATASET}" \
  --config  "${CONFIG}" \
  --out_root "${OUT_ROOT}"

# Find the run directory just created (newest dir under out_root)
RUN_DIR=$(ls -td "${OUT_ROOT}"/*/  2>/dev/null | head -1 | sed 's:/$::')
if [ -z "${RUN_DIR}" ]; then
  echo "ERROR: Could not find run directory under ${OUT_ROOT}" >&2
  exit 1
fi
echo "==> Run directory: ${RUN_DIR}"

# Stage 2: score
python eval/runners/score_eval.py \
  --dataset     "${DATASET}" \
  --predictions "${RUN_DIR}/predictions.jsonl" \
  --run_json    "${RUN_DIR}/run.json"

# Stage 3: report
python eval/runners/generate_report.py \
  --run_json "${RUN_DIR}/run.json" \
  --summary  "${RUN_DIR}/summary.json" \
  --scores   "${RUN_DIR}/scores.jsonl" \
  --out_dir  "${RUN_DIR}"

echo ""
echo "==> Baseline eval complete"
echo "    run_id : $(basename "${RUN_DIR}")"
echo "    report : ${RUN_DIR}/report.html"
