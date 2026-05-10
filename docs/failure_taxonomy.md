# Failure Tag Taxonomy

Failure tags are attached per sample in `scores.jsonl` and aggregated in `summary.json`.
Tags are namespaced for easy filtering and trend tracking.

## Tag reference

| Tag | Triggered by | Action |
|---|---|---|
| `asr.empty_hypothesis` | Hypothesis is empty string | Check audio pipeline / VAD |
| `asr.high_wer` | WER > 0.5 | Manual review; check audio quality |
| `asr.high_deletions_missing_content` | Deletions ≥ 20 words | May indicate VAD cutoff or model failure |
| `hallucination.high_insertions` | Insertions ≥ 20 words | Whisper ghost transcript; check no-speech threshold |
| `diar.high_der` | DER > 0.3 | Diarization failure; check speaker count and overlap settings |
| `diar.too_many_speakers` | Hyp speaker count > ref + 1 | Lower max_speakers param |
| `diar.too_few_speakers` | Hyp speaker count < ref - 1 | Raise min_speakers param or check segmentation |
| `perf.slow_decode` | RTF > 2.0 | Pipeline too slow; check GPU utilization / batch size |
| `runner.error` | Exception during prediction | Fix root cause in run logs |
| `runner.missing_manifest_entry` | sample_id not in manifest | Manifest/prediction mismatch |
| `speech.overlap_high_sample` | Sample tagged `overlap: high` | Informational; diarization will be harder |

## Adding new tags

1. Add detection logic in `eval/runners/score_eval.py:tag_failures()`
2. Document the tag in this file
3. Add a unit test in `tests/unit/test_score_pipeline.py`
4. Keep tag names stable — changing names breaks trend tracking

## Recommended triage workflow

1. Filter `scores.jsonl` for samples with a specific tag
2. Review the `counts.wer` dict for substitution/deletion/insertion breakdown
3. Listen to the audio (if available) or read the reference vs hypothesis
4. File a follow-up ticket tagged with the failure namespace
