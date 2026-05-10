"""File-system persistence layer."""

import json
from pathlib import Path
from typing import Any, List

import yaml

from kimi.config import GROUND_TRUTH_DIR, RESULTS_DIR, RUNS_DIR, TRANSCRIPTION_DIR
from kimi.models import RunMeta, SSSResult


def _ensure_dirs() -> None:
    for d in (GROUND_TRUTH_DIR, TRANSCRIPTION_DIR, RUNS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def list_ground_truths() -> List[str]:
    _ensure_dirs()
    return sorted({p.stem for p in GROUND_TRUTH_DIR.glob("*.txt")})


def list_transcriptions() -> List[str]:
    _ensure_dirs()
    return sorted({p.stem for p in TRANSCRIPTION_DIR.glob("*.txt")})


def list_runs() -> List[str]:
    _ensure_dirs()
    return sorted({p.stem for p in RUNS_DIR.glob("*.yaml")})


def load_ground_truth(stem: str) -> str:
    path = GROUND_TRUTH_DIR / f"{stem}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Ground truth not found: {path}")
    return path.read_text(encoding="utf-8")


def load_transcription(stem: str) -> str:
    path = TRANSCRIPTION_DIR / f"{stem}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Transcription not found: {path}")
    return path.read_text(encoding="utf-8")


def load_run_meta(stem: str) -> RunMeta:
    path = RUNS_DIR / f"{stem}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Run metadata not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return RunMeta.model_validate(raw)


def save_result(result: SSSResult) -> Path:
    _ensure_dirs()
    filename = f"{result.run_id}_{result.recorded_at.isoformat().replace(':', '-')}"
    path = RESULTS_DIR / f"{filename}.json"
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_result(path: Path) -> SSSResult:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return SSSResult.model_validate(raw)


def list_results() -> List[Path]:
    _ensure_dirs()
    return sorted(RESULTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)


def find_result_by_run_id(run_id: str) -> Path:
    for p in list_results():
        if p.stem.startswith(run_id):
            return p
    raise FileNotFoundError(f"No result found for run_id={run_id}")
