"""Filesystem storage for inputs, run metadata, and historic results."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from codex_version.config import GROUND_TRUTH_DIR, RESULTS_DIR, RUNS_DIR, TRANSCRIPTION_DIR
from codex_version.models import RunMetadata, SSSResult


def ensure_dirs() -> None:
    for directory in (GROUND_TRUTH_DIR, TRANSCRIPTION_DIR, RUNS_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def list_ids(directory: Path, suffix: str) -> list[str]:
    ensure_dirs()
    return sorted(path.stem for path in directory.glob(f"*{suffix}"))


def load_text(directory: Path, item_id: str) -> str:
    path = directory / f"{item_id}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing text file: {path}")
    return path.read_text(encoding="utf-8")


def load_run_metadata(run_id: str) -> RunMetadata:
    path = RUNS_DIR / f"{run_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing run metadata: {path}")
    return RunMetadata.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def save_run_metadata(metadata: RunMetadata) -> Path:
    ensure_dirs()
    path = RUNS_DIR / f"{metadata.run_id}.yaml"
    path.write_text(yaml.safe_dump(metadata.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
    return path


def save_result(result: SSSResult) -> Path:
    ensure_dirs()
    timestamp = result.recorded_at.strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS_DIR / f"{result.ground_truth_id}__{result.transcription_id}__{result.run_id}__{timestamp}.json"
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_result(path: Path) -> SSSResult:
    return SSSResult.model_validate(json.loads(path.read_text(encoding="utf-8")))


def list_result_paths() -> list[Path]:
    ensure_dirs()
    return sorted(RESULTS_DIR.glob("*.json"), key=lambda path: path.stat().st_mtime)


def find_latest_result(run_id: str) -> SSSResult:
    matches = [path for path in list_result_paths() if f"__{run_id}__" in path.name]
    if not matches:
        raise FileNotFoundError(f"No result found for run_id={run_id}")
    return load_result(matches[-1])


def history_rows() -> list[dict[str, Any]]:
    rows = []
    for path in list_result_paths():
        result = load_result(path)
        rows.append(
            {
                "run_id": result.run_id,
                "ground_truth_id": result.ground_truth_id,
                "transcription_id": result.transcription_id,
                "recorded_at": result.recorded_at,
                "model": result.model,
                "total_sss": result.total_sss,
                "segments": result.segment_count,
                "path": path,
            }
        )
    return rows


def utc_now() -> datetime:
    return datetime.utcnow().replace(microsecond=0)

