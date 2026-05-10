# Results Directory

Each evaluation run produces an immutable sub-directory named by run ID:

```
results/
  <run_id>/
    run.json          # complete run metadata (git, models, params, artifacts)
    predictions.jsonl # per-sample pipeline outputs + elapsed time
    scores.jsonl      # per-sample metrics (WER, CER, DER, SSS, RTF) + failure tags
    summary.json      # aggregate means, WER distribution stats, tag counts
    report.md         # human-readable markdown report
    report.html       # HTML report with delta table vs baseline (if provided)
    runner.log        # stdout/stderr from the runner (if redirected)
```

## Run ID format

```
<YYYY-MM-DDTHH-MM-SSZ>_<stage>_<dataset_version>_<git_short_commit>
```

Example: `2026-05-10T09-00-00Z_baseline_gold_v1_abc1234`

## Retention policy

- Keep all runs under `results/` indefinitely in version control (JSON is small).
- Large audio predictions or intermediate files should be stored in object storage
  and referenced by URI in run.json — do not commit audio to this directory.
- Archive runs older than 90 days to cold storage; keep the last 10 per stage.

## Comparing runs

```bash
python scripts/compare_runs.py \
  --baseline  results/<baseline_run_id>/summary.json \
  --candidate results/<candidate_run_id>/summary.json
```
