"""Data models for SSS runs and results."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class RunMetadata(BaseModel):
    """User-editable YAML metadata for one transcription version."""

    run_id: str
    ground_truth_id: str
    transcription_id: str
    model: str = "nomic-embed-text"
    chunk_strategy: str = "sentence_window"
    target_sentences_per_chunk: int = 3
    max_chars_per_chunk: int = 900
    location: str | None = None
    recorded_at: datetime | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None


class AlignmentSegment(BaseModel):
    segment_index: int
    ground_truth_text: str
    transcription_text: str
    ground_truth_chars: int
    transcription_chars: int
    sss: float
    evaluators: dict[str, float | int] = Field(default_factory=dict)


class SSSResult(BaseModel):
    run_id: str
    ground_truth_id: str
    transcription_id: str
    model: str
    chunk_strategy: str
    target_sentences_per_chunk: int
    max_chars_per_chunk: int
    recorded_at: datetime
    total_sss: float
    evaluator_totals: dict[str, float | int] = Field(default_factory=dict)
    segment_count: int
    metadata: dict[str, Any]
    segments: list[AlignmentSegment]

