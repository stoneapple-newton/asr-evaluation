"""
Eval smoke tests — verify the scoring pipeline end-to-end using mock predictions.

These tests do NOT call the ASR/diarization/LLM models; instead they synthesise
a predictions.jsonl that mimics the output of run_eval.py and then run the real
scorer and report generator, asserting on the produced artifacts.

Marked with eval_smoke so they can be run in CI:
    pytest -m eval_smoke -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DATASET = Path("data/eval_datasets/gold_v1/manifest_smoke.jsonl")


def _build_mock_predictions(manifest_path: Path, stage: str = "baseline") -> list[dict]:
    """Build synthetic predictions that almost perfectly match the reference."""
    preds = []
    with manifest_path.open() as f:
        for line in f:
            ex = json.loads(line.strip())
            ref_text = (ex.get("reference") or {}).get("text", "")
            # Introduce one deliberate word substitution per sample
            words = ref_text.split()
            if len(words) >= 2:
                words[1] = "REPLACED"
            hyp_text = " ".join(words)
            preds.append({
                "sample_id": ex["sample_id"],
                "audio_uri": ex["audio"]["uri"],
                "stage": stage,
                "elapsed_sec": 2.5,
                "prediction": {
                    "text": hyp_text,
                    "segments": [],
                    "diarization": {"segments": []},
                },
                "sss": None,
            })
    return preds


@pytest.mark.eval_smoke
def test_scorer_produces_scores_and_summary(tmp_path):
    """Run score_eval on mock predictions and check artifacts are created."""
    if not DATASET.exists():
        pytest.skip(f"Dataset not found: {DATASET}")

    # Write mock predictions
    pred_path = tmp_path / "predictions.jsonl"
    run_json_path = tmp_path / "run.json"

    preds = _build_mock_predictions(DATASET)
    with pred_path.open("w") as f:
        for row in preds:
            f.write(json.dumps(row) + "\n")

    # Write minimal run.json
    run_json_path.write_text(json.dumps({
        "run_id": "smoke_test",
        "stage": "baseline",
        "artifacts": {},
    }, indent=2))

    # Run scorer via its main function directly (avoid subprocess overhead)
    import sys as _sys
    from unittest.mock import patch

    argv = [
        "score_eval.py",
        "--dataset", str(DATASET),
        "--predictions", str(pred_path),
        "--run_json", str(run_json_path),
    ]
    with patch.object(_sys, "argv", argv):
        from eval.runners.score_eval import main
        main()

    scores_path = tmp_path / "scores.jsonl"
    summary_path = tmp_path / "summary.json"

    assert scores_path.exists(), "scores.jsonl was not created"
    assert summary_path.exists(), "summary.json was not created"

    summary = json.loads(summary_path.read_text())
    assert summary["n_total"] == len(preds)
    assert summary["n_scored"] > 0
    assert "wer" in (summary.get("means") or {})

    # All samples should have a WER > 0 due to the substitution
    scored = [json.loads(l) for l in scores_path.read_text().splitlines() if l.strip()]
    wer_values = [s["metrics"]["wer"] for s in scored if s.get("metrics")]
    assert all(w > 0.0 for w in wer_values), "Expected non-zero WER for every sample"


@pytest.mark.eval_smoke
def test_report_generator_produces_files(tmp_path):
    """Run generate_report on a minimal summary and verify md+html are produced."""
    run_json = {
        "run_id": "smoke_report_test",
        "stage": "baseline",
        "owner": "ci",
        "started_at": "2026-05-10T09:00:00+00:00",
        "ended_at": "2026-05-10T09:05:00+00:00",
        "git": {"commit": "abc1234", "branch": "main"},
        "dataset": {"version": "gold_v1", "manifest": str(DATASET)},
        "runner": {"config_path": "eval/configs/baseline.yaml"},
        "notes": "smoke test",
        "artifacts": {},
    }
    summary = {
        "n_total": 3,
        "n_scored": 3,
        "n_errors": 0,
        "means": {"wer": 0.05, "cer": 0.02, "rtf": 0.3},
        "wer_stats": {"mean": 0.05, "p50": 0.04, "p90": 0.09, "n": 3},
        "tag_counts": {"asr.high_wer": 0},
    }
    scores = [
        {"sample_id": "s1", "metrics": {"wer": 0.05, "cer": 0.02, "sss": None, "rtf": 0.3},
         "failure_tags": []},
        {"sample_id": "s2", "metrics": {"wer": 0.08, "cer": 0.03, "sss": None, "rtf": 0.25},
         "failure_tags": []},
        {"sample_id": "s3", "metrics": {"wer": 0.02, "cer": 0.01, "sss": None, "rtf": 0.35},
         "failure_tags": []},
    ]

    run_json_path = tmp_path / "run.json"
    summary_path = tmp_path / "summary.json"
    scores_path = tmp_path / "scores.jsonl"

    run_json_path.write_text(json.dumps(run_json, indent=2))
    summary_path.write_text(json.dumps(summary, indent=2))
    with scores_path.open("w") as f:
        for row in scores:
            f.write(json.dumps(row) + "\n")

    import sys as _sys
    from unittest.mock import patch

    argv = [
        "generate_report.py",
        "--run_json", str(run_json_path),
        "--summary", str(summary_path),
        "--scores", str(scores_path),
        "--out_dir", str(tmp_path),
    ]
    with patch.object(_sys, "argv", argv):
        from eval.runners.generate_report import main
        main()

    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "report.html").exists()

    md = (tmp_path / "report.md").read_text()
    assert "smoke_report_test" in md
    assert "wer" in md.lower()
