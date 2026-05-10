# Weekly Evaluation Runbook

## Schedule

Every Monday, run the full gold_v1 baseline evaluation and share results
with the team by end of day.

## Steps

### 1. Run baseline

```bash
./scripts/eval_baseline.sh
```

Note the `run_id` printed at the end.

### 2. (Optional) Run LLM fixed-input if prompts changed

```bash
./scripts/eval_llm_fixed.sh <baseline_run_id>
```

### 3. Compare against last week's baseline

```bash
python scripts/compare_runs.py \
  --baseline  results/<last_week_run_id>/summary.json \
  --candidate results/<this_week_run_id>/summary.json
```

### 4. Triage regressions

- Open `results/<run_id>/report.html` in a browser
- Review the "Worst samples by WER" table
- For each sample with new failure tags, check `scores.jsonl` for details
- File follow-up tickets for any tag counts that increased

### 5. Send weekly digest

Subject: ASR Eval Weekly — gold_v1 — <date>

Body:
- Runs: `<run_id>` (baseline: `<prev_run_id>`)
- WER: mean=`__` (Δ`__`), P90=`__` (Δ`__`)
- DER: mean=`__` (Δ`__`)
- Hallucination rate: `__` (Δ`__`)
- RTF: mean=`__` (Δ`__`)
- Top new failure tags: `...`
- Recommendation: Ship / Hold / Investigate
- Link: `results/<run_id>/report.html`
