"""Pydantic data models."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class RunMeta(BaseModel):
    """Metadata for a single evaluation run, sourced from a user-supplied YAML file."""

    run_id: str = Field(..., description="Unique identifier for this run")
    ground_truth_id: str = Field(..., description="Ground-truth file stem")
    transcription_id: str = Field(..., description="Transcription file stem")
    model: str = Field(default="nomic-embed-text", description="Ollama embedding model used")
    chunk_strategy: str = Field(default="sentence", description="sentence | paragraph")
    settings: dict = Field(default_factory=dict, description="Arbitrary transcription settings")
    location: Optional[str] = None
    recorded_at: Optional[datetime] = None
    notes: Optional[str] = None


class Segment(BaseModel):
    """One aligned pair of GT / TX chunks with its similarity score."""

    segment_index: int
    ground_truth_text: str
    transcription_text: str
    sss: float  # cosine similarity 0..1


class SSSResult(BaseModel):
    """Full result of an evaluation run."""

    run_id: str
    ground_truth_id: str
    transcription_id: str
    model: str
    chunk_strategy: str
    recorded_at: datetime
    segments: List[Segment]
    total_sss: float
    metadata: dict = Field(default_factory=dict)
