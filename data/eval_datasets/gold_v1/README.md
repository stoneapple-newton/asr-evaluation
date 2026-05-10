# Gold Evaluation Dataset v1

## Overview

`gold_v1` is the canonical evaluation dataset for the ASR pipeline. All regression gating
and release decisions are made against this dataset.

## Contents

| File | Description |
|---|---|
| `manifest.jsonl` | Full evaluation set (5 samples) |
| `manifest_smoke.jsonl` | Fast CI smoke subset (3 samples) |
| `audio/` | Audio files (wav, 16 kHz mono) — store externally if sensitive |
| `refs/` | Optional per-sample reference JSON for offline annotation |

## Sample Coverage

| sample_id | Domain | Speakers | Overlap | Noise |
|---|---|---|---|---|
| gold_v1_meeting_001 | meeting | 1 | none | clean |
| gold_v1_call_001 | support | 2 | low | low |
| gold_v1_meeting_002 | meeting | 3 | low | clean |
| gold_v1_noisy_001 | meeting | 1 | none | high |
| gold_v1_overlap_001 | meeting | 3 | high | low |

## Text Normalization Rules

When computing WER/CER, apply:
- Lowercase everything
- Normalize whitespace (collapse multiple spaces, strip leading/trailing)
- Numbers: keep spelled-out form (e.g. "fifteen percent" not "15%")
- Punctuation: optional strip for WER; keep for CER and SSS

## Dataset Versioning

- `gold_v1` — initial dataset, locked 2026-05-10
- Audio files: store in access-controlled storage; commit only manifest + refs
- Do not modify `manifest.jsonl` without bumping the version

## Adding Samples

1. Record audio and place in `audio/<sample_id>.<ext>`
2. Create reference transcript following the normalization rules above
3. Add segment-level annotations (timestamps + speaker labels) for DER evaluation
4. Append a new JSONL line to `manifest.jsonl`
5. Add a representative 1-2 sample(s) to `manifest_smoke.jsonl` for CI
6. Update this README table
