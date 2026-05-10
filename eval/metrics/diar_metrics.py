"""Diarization Error Rate (DER) using pyannote.metrics with a pure-Python fallback."""
from __future__ import annotations

try:
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate as _DER

    _PYANNOTE_AVAILABLE = True
except ImportError:
    _PYANNOTE_AVAILABLE = False


def _to_annotation(segments: list[dict]):
    """Convert list of {start, end, speaker} dicts to a pyannote Annotation."""
    ann = Annotation()
    for seg in segments:
        ann[Segment(float(seg["start"]), float(seg["end"]))] = str(seg["speaker"])
    return ann


def der_score(
    ref_segments: list[dict],
    hyp_segments: list[dict],
    *,
    collar: float = 0.25,
    skip_overlap: bool = False,
) -> float:
    """
    Compute DER = (FA + Miss + Conf) / Total using pyannote.metrics.

    Falls back to a lightweight approximation when pyannote is not installed.
    Each segment is a dict with keys: start (float), end (float), speaker (str).
    """
    if not ref_segments:
        return 0.0

    if _PYANNOTE_AVAILABLE:
        metric = _DER(collar=collar, skip_overlap=skip_overlap)
        ref = _to_annotation(ref_segments)
        hyp = _to_annotation(hyp_segments)
        return float(metric(ref, hyp))

    return _fallback_der(ref_segments, hyp_segments)


def _fallback_der(ref_segments: list[dict], hyp_segments: list[dict]) -> float:
    """
    Lightweight interval-overlap DER approximation (no collar, no optimal mapping).

    This is a coarse estimate — install pyannote.metrics for production-grade DER.
    Total ref speech is measured in seconds; missed/false-alarm/confusion computed
    from interval overlaps with a greedy speaker assignment.
    """
    total_ref = sum(max(0.0, s["end"] - s["start"]) for s in ref_segments)
    if total_ref == 0:
        return 0.0

    # Build (start, end, speaker) tuples
    ref_ivs = [(float(s["start"]), float(s["end"]), str(s["speaker"])) for s in ref_segments]
    hyp_ivs = [(float(s["start"]), float(s["end"]), str(s["speaker"])) for s in hyp_segments]

    # For each reference interval, find overlapping hypothesis segments
    error_sec = 0.0
    for r_start, r_end, r_spk in ref_ivs:
        r_dur = r_end - r_start
        covered = 0.0
        for h_start, h_end, h_spk in hyp_ivs:
            overlap = max(0.0, min(r_end, h_end) - max(r_start, h_start))
            if overlap > 0:
                covered += overlap
                if h_spk != r_spk:
                    error_sec += overlap
        missed = max(0.0, r_dur - covered)
        error_sec += missed

    return error_sec / total_ref


def speaker_count_delta(
    ref_segments: list[dict],
    hyp_segments: list[dict],
) -> int:
    """Return hyp_speaker_count - ref_speaker_count (positive = too many)."""
    ref_speakers = {s["speaker"] for s in ref_segments}
    hyp_speakers = {s["speaker"] for s in hyp_segments}
    return len(hyp_speakers) - len(ref_speakers)


def pyannote_available() -> bool:
    return _PYANNOTE_AVAILABLE
