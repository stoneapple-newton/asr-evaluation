# ASR Evaluation Plan

## Objective

Provide a repeatable, staged evaluation process that makes regressions obvious
and easy to triage across the ASR processing chain (transcription → diarization
→ LLM cleanup).

## Three-stage design

| Stage | What runs | Purpose |
|---|---|---|
| `baseline` | ASR + Diarization | Establish non-LLM reference behavior; compute WER/CER/DER/SSS |
| `llm_fixed_input` | LLM cleanup on frozen baseline outputs | Isolate LLM contribution; no ASR variability |
| `end_to_end` | Full pipeline | Validate latency, cost, and real-world interactions |

## Cadence

| Frequency | What |
|---|---|
| Per PR | Unit + integration tests; eval smoke suite (3–5 samples) |
| Weekly | Full gold_v1 baseline run + scoring + report |
| Pre-release | End-to-end benchmark; gate decision; tag regression items |

## Gold dataset

`data/eval_datasets/gold_v1/manifest.jsonl` — 5 samples covering:
- single speaker / multi-speaker
- clean / noisy audio
- low-overlap / high-overlap diarization conditions

See `data/eval_datasets/gold_v1/README.md` for details.

## Pass/fail gates (illustrative — tune after baseline is established)

| Metric | Max regression vs baseline |
|---|---|
| WER (mean) | +0.03 absolute |
| CER (mean) | +0.03 absolute |
| DER (mean) | +0.02 absolute |
| Hallucination rate | +0.01 absolute |
| Missing content rate | +0.01 absolute |
| RTF (mean) | +0.50 absolute |

Gates are enforced by `scripts/compare_runs.py --baseline ... --candidate ...`.
