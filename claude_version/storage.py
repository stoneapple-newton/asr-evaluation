import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from claude_version.config import (
    GROUND_TRUTH_DIR,
    RESULTS_DIR,
    RUNS_DIR,
    TRANSCRIPTION_DIR,
)
from claude_version.models import RunMetadata, SSSResult


def ensure_dirs() -> None:
    for d in (GROUND_TRUTH_DIR, TRANSCRIPTION_DIR, RUNS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc).replace(microsecond=0)


def list_ids(directory: Path, suffix: str) -> list[str]:
    ensure_dirs()
    return sorted(p.stem for p in directory.glob(f"*{suffix}"))


def load_text(directory: Path, item_id: str) -> str:
    path = directory / f"{item_id}.txt"
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_text(encoding="utf-8")


def load_run_metadata(run_id: str) -> RunMetadata:
    path = RUNS_DIR / f"{run_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Run metadata not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return RunMetadata.model_validate(raw)


def save_run_metadata(metadata: RunMetadata) -> Path:
    ensure_dirs()
    path = RUNS_DIR / f"{metadata.run_id}.yaml"
    path.write_text(
        yaml.safe_dump(metadata.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def save_result(result: SSSResult) -> Path:
    ensure_dirs()
    ts = result.recorded_at.strftime("%Y%m%dT%H%M%SZ")
    filename = f"{result.ground_truth_id}__{result.transcription_id}__{result.run_id}__{ts}.json"
    path = RESULTS_DIR / filename
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_result(path: Path) -> SSSResult:
    return SSSResult.model_validate(json.loads(path.read_text(encoding="utf-8")))


def list_result_paths() -> list[Path]:
    ensure_dirs()
    return sorted(RESULTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)


def find_latest_result(run_id: str) -> SSSResult:
    matches = [p for p in list_result_paths() if f"__{run_id}__" in p.name]
    if not matches:
        raise FileNotFoundError(f"No result found for run_id={run_id!r}")
    return load_result(matches[-1])


def history_rows() -> list[dict[str, Any]]:
    return [
        {
            "run_id": r.run_id,
            "ground_truth_id": r.ground_truth_id,
            "transcription_id": r.transcription_id,
            "recorded_at": r.recorded_at,
            "model": r.model,
            "chunk_strategy": r.chunk_strategy,
            "total_sss": r.total_sss,
            "segments": r.segment_count,
        }
        for r in (load_result(p) for p in list_result_paths())
    ]
