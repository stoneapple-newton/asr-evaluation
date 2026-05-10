"""
Scorer: reads predictions.jsonl + dataset manifest and writes scores.jsonl + summary.json.

Usage:
  python eval/runners/score_eval.py \\
    --dataset data/eval_datasets/gold_v1/manifest.jsonl \\
    --predictions results/<run_id>/predictions.jsonl \\
    --run_json   results/<run_id>/run.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from eval.metrics.diar_metrics import der_score, speaker_count_delta
from eval.metrics.latency_cost import rtf
from eval.metrics.text_metrics import (
    aggregate_wer_stats,
    cer,
    derived_content_metrics,
    normalize_text,
    wer,
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Failure tagging
# ---------------------------------------------------------------------------

_WER_HIGH_THRESHOLD = 0.5
_INSERTION_ABS_THRESHOLD = 20
_DELETION_ABS_THRESHOLD = 20
_SLOW_DECODE_RTF = 2.0  # >2× real-time is considered slow


def tag_failures(
    ex: dict,
    hyp_text: str,
    wer_counts: dict,
    elapsed_sec: float,
    der: float | None = None,
    spk_delta: int | None = None,
) -> list[str]:
    tags: list[str] = []
    audio_dur = float((ex.get("audio") or {}).get("duration_sec") or 0)

    if not hyp_text.strip():
        tags.append("asr.empty_hypothesis")

    if wer_counts["insertions"] >= _INSERTION_ABS_THRESHOLD:
        tags.append("hallucination.high_insertions")

    if wer_counts["deletions"] >= _DELETION_ABS_THRESHOLD:
        tags.append("asr.high_deletions_missing_content")

    wer_val = (
        (wer_counts["substitutions"] + wer_counts["deletions"] + wer_counts["insertions"])
        / max(1, wer_counts["ref_len"])
    )
    if wer_val > _WER_HIGH_THRESHOLD:
        tags.append("asr.high_wer")

    if audio_dur > 0 and rtf(elapsed_sec, audio_dur) > _SLOW_DECODE_RTF:
        tags.append("perf.slow_decode")

    if der is not None and der > 0.3:
        tags.append("diar.high_der")

    if spk_delta is not None:
        if spk_delta > 1:
            tags.append("diar.too_many_speakers")
        elif spk_delta < -1:
            tags.append("diar.too_few_speakers")

    # Tag based on dataset-level sample tags
    sample_tags = ex.get("tags") or {}
    if sample_tags.get("overlap") == "high":
        if not tags:
            pass  # informational only; don't add unless error
        # Add informational context tag to help filter later
        tags.append("speech.overlap_high_sample")

    return tags


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="ASR evaluation scorer")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--run_json", required=True)
    ap.add_argument(
        "--normalize", action="store_true", default=True,
        help="Lowercase and normalize whitespace before WER/CER (default: True)",
    )
    ap.add_argument(
        "--strip_punct", action="store_true", default=False,
        help="Strip punctuation before WER/CER",
    )
    args = ap.parse_args()

    dataset = {ex["sample_id"]: ex for ex in read_jsonl(Path(args.dataset))}
    preds = list(read_jsonl(Path(args.predictions)))

    out_dir = Path(args.predictions).parent
    scores_path = out_dir / "scores.jsonl"
    summary_path = out_dir / "summary.json"

    scored: list[dict] = []
    agg: dict[str, list[float]] = defaultdict(list)
    per_sample_wer: list[float] = []

    for row in preds:
        sample_id = row["sample_id"]
        if sample_id not in dataset:
            scored.append({
                "sample_id": sample_id,
                "error": "sample_id not found in dataset manifest",
                "metrics": None,
                "failure_tags": ["runner.missing_manifest_entry"],
            })
            continue

        ex = dataset[sample_id]
        ref_text_raw = (ex.get("reference") or {}).get("text") or ""
        elapsed_sec = float(row.get("elapsed_sec") or 0.0)
        audio_dur = float((ex.get("audio") or {}).get("duration_sec") or 0)

        if row.get("prediction") is None:
            scored.append({
                "sample_id": sample_id,
                "error": row.get("error"),
                "metrics": None,
                "failure_tags": ["runner.error"],
            })
            continue

        hyp_text_raw = (row["prediction"] or {}).get("text") or ""

        # Text normalization
        ref_text = normalize_text(ref_text_raw, lowercase=args.normalize, strip_punct=args.strip_punct)
        hyp_text = normalize_text(hyp_text_raw, lowercase=args.normalize, strip_punct=args.strip_punct)

        wer_val, wer_counts = wer(ref_text, hyp_text)
        cer_val, cer_counts = cer(ref_text, hyp_text)
        content = derived_content_metrics(wer_counts, hyp_text)

        metrics: dict = {
            "wer": wer_val,
            "cer": cer_val,
            "missing_content_rate": content["missing_content_rate"],
            "hallucination_rate": content["hallucination_rate"],
            "elapsed_sec": elapsed_sec,
            "rtf": rtf(elapsed_sec, audio_dur) if audio_dur > 0 else None,
        }

        # SSS if present in prediction record
        if row.get("sss") is not None:
            metrics["sss"] = float(row["sss"])

        # DER if reference diarization segments exist
        der_val: float | None = None
        spk_delta: int | None = None
        ref_segs = (ex.get("reference") or {}).get("segments") or []
        hyp_diar_segs = (row["prediction"].get("diarization") or {}).get("segments") or []
        ref_diar = [s for s in ref_segs if "speaker" in s]
        if ref_diar and hyp_diar_segs:
            try:
                der_val = der_score(ref_diar, hyp_diar_segs)
                spk_delta = speaker_count_delta(ref_diar, hyp_diar_segs)
                metrics["der"] = der_val
                metrics["speaker_count_delta"] = spk_delta
            except Exception as exc:
                metrics["der_error"] = repr(exc)

        failure_tags = tag_failures(ex, hyp_text, wer_counts, elapsed_sec, der_val, spk_delta)

        scored.append({
            "sample_id": sample_id,
            "metrics": metrics,
            "counts": {
                "wer": wer_counts,
                "cer": cer_counts,
            },
            "failure_tags": failure_tags,
        })

        per_sample_wer.append(wer_val)
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and v is not None:
                agg[k].append(float(v))

    write_jsonl(scores_path, scored)

    wer_stats = aggregate_wer_stats(per_sample_wer)
    means = {k: (sum(v) / len(v)) for k, v in agg.items() if v}

    tag_counts: dict[str, int] = defaultdict(int)
    for s in scored:
        for t in s.get("failure_tags") or []:
            tag_counts[t] += 1

    summary = {
        "n_total": len(preds),
        "n_scored": sum(1 for s in scored if s.get("metrics")),
        "n_errors": sum(1 for s in scored if not s.get("metrics")),
        "means": means,
        "wer_stats": wer_stats,
        "tag_counts": dict(tag_counts),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Update run.json artifact pointers
    run_json_path = Path(args.run_json)
    run_record = json.loads(run_json_path.read_text(encoding="utf-8"))
    run_record.setdefault("artifacts", {})
    run_record["artifacts"]["scores_jsonl"] = str(scores_path)
    run_record["artifacts"]["summary_json"] = str(summary_path)
    run_json_path.write_text(json.dumps(run_record, indent=2), encoding="utf-8")

    print(f"Scored {summary['n_scored']}/{summary['n_total']} samples")
    print(f"  mean WER : {means.get('wer', 'n/a'):.4f}" if "wer" in means else "  mean WER : n/a")
    print(f"  scores   : {scores_path}")
    print(f"  summary  : {summary_path}")


if __name__ == "__main__":
    main()
