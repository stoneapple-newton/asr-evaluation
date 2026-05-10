"""
Stage runner: produces predictions.jsonl + run.json for one evaluation stage.

Stages:
  baseline        — ASR + diarization only (no LLM)
  llm_fixed_input — LLM cleanup applied to frozen baseline outputs
  end_to_end      — full pipeline: ASR + diar + LLM

Usage:
  python eval/runners/run_eval.py \\
    --stage baseline \\
    --dataset data/eval_datasets/gold_v1/manifest.jsonl \\
    --config eval/configs/baseline.yaml \\
    --out_root results/
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import yaml


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def git_info() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
        dirty = (
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip() != ""
        )
        return {"commit": commit, "branch": branch, "dirty": dirty}
    except Exception:
        return {"commit": None, "branch": None, "dirty": None}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
# Plug-in interfaces — replace with your production implementations
# ---------------------------------------------------------------------------

def run_asr(audio_uri: str, cfg: dict) -> dict:
    """
    Return at minimum:
      text     : str
      segments : list[{start, end, text, speaker?}]
    """
    raise NotImplementedError(
        f"Wire run_asr() to your ASR pipeline. audio_uri={audio_uri!r}"
    )


def run_diarization(audio_uri: str, cfg: dict) -> dict:
    """
    Return:
      segments : list[{start, end, speaker}]
    """
    return {"segments": []}


def run_llm_cleanup(structured: dict, cfg: dict) -> dict:
    """
    Input : structured transcript/segments dict
    Output: cleaned version (same schema) + optional token_usage
    """
    return structured


# ---------------------------------------------------------------------------
# SSS evaluation (optional — skipped when Ollama is unavailable)
# ---------------------------------------------------------------------------

def _try_sss(gt_text: str, hyp_text: str, cfg: dict) -> float | None:
    """Attempt to compute SSS; return None if Ollama/embedder is unavailable."""
    try:
        import numpy as np

        from claude_version.chunker import chunk_text
        from claude_version.aligner import align_chunks
        from claude_version.embedder import OllamaEmbedder
        from eval.metrics.sss_metrics import sss_score

        strategy = cfg.get("sss", {}).get("chunk_strategy", "sentence_window")
        target = cfg.get("sss", {}).get("target_sentences", 3)
        max_chars = cfg.get("sss", {}).get("max_chars", 900)
        ollama_url = cfg.get("sss", {}).get("ollama_url", "http://localhost:11434")
        model = cfg.get("sss", {}).get("model", "nomic-embed-text")

        embedder = OllamaEmbedder(base_url=ollama_url, model=model)

        gt_chunks = chunk_text(gt_text, strategy=strategy, target_sentences=target, max_chars=max_chars)
        tx_chunks = chunk_text(hyp_text, strategy=strategy, target_sentences=target, max_chars=max_chars)
        aligned = align_chunks(gt_chunks, tx_chunks)
        gt_al = [p[0] for p in aligned]
        tx_al = [p[1] for p in aligned]

        total, _ = sss_score(gt_al, tx_al, lambda texts: [
            np.asarray(embedder._embed_one(t), dtype=np.float32) for t in texts
        ])
        return total
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="ASR evaluation stage runner")
    ap.add_argument(
        "--stage", required=True,
        choices=["baseline", "llm_fixed_input", "end_to_end"],
    )
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--owner", default=os.getenv("USER", "unknown"))
    ap.add_argument("--run_id", default=None)
    ap.add_argument(
        "--input_run_dir", default=None,
        help="Path to baseline run dir (required for llm_fixed_input)",
    )
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    cfg_path = Path(args.config)
    out_root = Path(args.out_root)

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    gi = git_info()
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    short_commit = (gi.get("commit") or "nogit")[:7]
    run_id = args.run_id or (
        f"{ts}_{args.stage}_{cfg.get('dataset_version', 'dataset')}_{short_commit}"
    )
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    run_json_path = run_dir / "run.json"
    pred_path = run_dir / "predictions.jsonl"
    log_path = run_dir / "runner.log"

    run_record: dict = {
        "run_id": run_id,
        "stage": args.stage,
        "owner": args.owner,
        "started_at": utc_now_iso(),
        "ended_at": None,
        "status": "running",
        "git": gi,
        "dataset": {
            "version": cfg.get("dataset_version"),
            "manifest": str(dataset_path),
            "manifest_sha256": sha256_file(dataset_path) if dataset_path.exists() else None,
        },
        "runner": {
            "config_path": str(cfg_path),
            "command": " ".join(["python"] + sys.argv),
            "hostname": platform.node(),
            "python_version": platform.python_version(),
        },
        "models": cfg.get("models", {}),
        "decoding_params": cfg.get("decoding_params", {}),
        "diarization_params": cfg.get("diarization_params", {}),
        "prompts": cfg.get("prompts", {}),
        "cost_config": cfg.get("cost_config", {}),
        "artifacts": {
            "predictions_jsonl": str(pred_path),
            "runner_log": str(log_path),
        },
        "baseline_run_id": cfg.get("baseline_run_id"),
        "notes": cfg.get("notes"),
    }
    run_json_path.write_text(json.dumps(run_record, indent=2), encoding="utf-8")

    # Load frozen inputs for llm_fixed_input stage
    fixed_inputs: dict = {}
    if args.stage == "llm_fixed_input":
        if not args.input_run_dir:
            sys.exit("--input_run_dir is required for llm_fixed_input stage")
        input_pred = Path(args.input_run_dir) / "predictions.jsonl"
        for row in read_jsonl(input_pred):
            fixed_inputs[row["sample_id"]] = row["prediction"]

    preds: list[dict] = []

    for ex in read_jsonl(dataset_path):
        sample_id = ex["sample_id"]
        audio_uri = ex["audio"]["uri"]
        ref_text = (ex.get("reference") or {}).get("text", "")
        t0 = time.perf_counter()

        try:
            if args.stage == "baseline":
                asr_out = run_asr(audio_uri, cfg)
                diar_out = run_diarization(audio_uri, cfg)
                prediction = {
                    "text": asr_out.get("text", ""),
                    "segments": asr_out.get("segments", []),
                    "diarization": diar_out,
                }

            elif args.stage == "llm_fixed_input":
                base = fixed_inputs.get(sample_id)
                if base is None:
                    raise RuntimeError(f"No fixed input for sample_id={sample_id!r}")
                prediction = run_llm_cleanup(base, cfg)

            else:  # end_to_end
                asr_out = run_asr(audio_uri, cfg)
                diar_out = run_diarization(audio_uri, cfg)
                struct = {
                    "text": asr_out.get("text", ""),
                    "segments": asr_out.get("segments", []),
                    "diarization": diar_out,
                }
                prediction = run_llm_cleanup(struct, cfg)

            elapsed = time.perf_counter() - t0

            # Optionally compute SSS if reference text is available
            sss = None
            if ref_text and prediction.get("text"):
                sss = _try_sss(ref_text, prediction["text"], cfg)

            preds.append({
                "sample_id": sample_id,
                "audio_uri": audio_uri,
                "stage": args.stage,
                "elapsed_sec": elapsed,
                "prediction": prediction,
                "sss": sss,
            })

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            preds.append({
                "sample_id": sample_id,
                "audio_uri": audio_uri,
                "stage": args.stage,
                "elapsed_sec": elapsed,
                "error": repr(exc),
                "prediction": None,
                "sss": None,
            })

    write_jsonl(pred_path, preds)

    run_record["ended_at"] = utc_now_iso()
    run_record["status"] = "completed"
    run_record["artifacts"]["run_dir"] = str(run_dir)
    run_json_path.write_text(json.dumps(run_record, indent=2), encoding="utf-8")

    print(f"Run complete: {run_id}")
    print(f"  predictions : {pred_path}")
    print(f"  run.json    : {run_json_path}")


if __name__ == "__main__":
    main()
