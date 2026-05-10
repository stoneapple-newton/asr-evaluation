# Release Gate Runbook

## Purpose

Before any production release, run the full end-to-end evaluation and confirm
all pass/fail gates are green.

## Steps

### 1. Identify baseline run

Use the most recent successful weekly baseline run_id.

### 2. Run end-to-end eval

```bash
./scripts/eval_e2e.sh \
  data/eval_datasets/gold_v1/manifest.jsonl \
  eval/configs/e2e.yaml \
  results/ \
  <baseline_run_id>
```

### 3. Run gate check

```bash
python scripts/compare_runs.py \
  --baseline  results/<baseline_run_id>/summary.json \
  --candidate results/<e2e_run_id>/summary.json
```

**Exit code 0 = all gates passed. Exit code 1 = one or more gates failed.**

### 4. Gate thresholds

| Metric | Max regression |
|---|---|
| WER mean | +0.03 |
| CER mean | +0.03 |
| DER mean | +0.02 |
| Hallucination rate | +0.01 |
| Missing content rate | +0.01 |
| RTF mean | +0.50 |

### 5. Decision matrix

| Gate result | Action |
|---|---|
| All pass | Proceed with release |
| 1–2 minor metrics fail | Review samples; if isolated issue and owner approves → proceed with known regression |
| WER or DER or hallucination fail | Hold release; file regression ticket; investigate |
| Runner errors > 5% | Hold release; fix pipeline before re-evaluating |

### 6. Slide bullets for release review

- Dataset: gold_v1 (N=5)
- WER: `__` → `__` (Δ`__`)
- DER: `__` → `__` (Δ`__`)
- Hallucination rate: `__` → `__` (Δ`__`)
- RTF: `__` → `__` (Δ`__`)
- Gates: all pass / N failed
- Decision: Ship / Hold / Scope change
