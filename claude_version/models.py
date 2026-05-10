from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RunMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")

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


class Segment(BaseModel):
    index: int
    ground_truth_text: str
    transcription_text: str
    gt_chars: int
    tx_chars: int
    sss: float


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
    segment_count: int
    metadata: dict[str, Any]
    segments: list[Segment]
