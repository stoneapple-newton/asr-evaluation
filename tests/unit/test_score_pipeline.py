"""
Integration-style unit tests for the score_eval pipeline logic.

These tests exercise the scoring functions in isolation using in-memory data,
so they run without an ASR model, Ollama, or pyannote installed.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from eval.metrics.text_metrics import normalize_text, wer
from eval.runners.score_eval import tag_failures


# ---------------------------------------------------------------------------
# tag_failures
# ---------------------------------------------------------------------------

def _counts(S=0, D=0, I=0, N=10):
    return {"substitutions": S, "deletions": D, "insertions": I, "ref_len": N}


def _ex(duration_sec=30.0, overlap="none"):
    return {"audio": {"duration_sec": duration_sec}, "tags": {"overlap": overlap}}


class TestTagFailures:
    def test_empty_hypothesis(self):
        tags = tag_failures(_ex(), "", _counts(), elapsed_sec=1.0)
        assert "asr.empty_hypothesis" in tags

    def test_high_insertions(self):
        tags = tag_failures(_ex(), "a " * 30, _counts(I=25), elapsed_sec=1.0)
        assert "hallucination.high_insertions" in tags

    def test_high_deletions(self):
        tags = tag_failures(_ex(), "hello", _counts(D=25), elapsed_sec=1.0)
        assert "asr.high_deletions_missing_content" in tags

    def test_high_wer(self):
        # WER > 0.5 → high_wer tag
        c = _counts(S=6, N=10)
        tags = tag_failures(_ex(), "hello", c, elapsed_sec=1.0)
        assert "asr.high_wer" in tags

    def test_slow_decode(self):
        # RTF = elapsed / duration > 2.0 → slow
        tags = tag_failures(_ex(duration_sec=10.0), "hello", _counts(), elapsed_sec=25.0)
        assert "perf.slow_decode" in tags

    def test_high_der(self):
        tags = tag_failures(_ex(), "hello", _counts(), elapsed_sec=1.0, der=0.5)
        assert "diar.high_der" in tags

    def test_too_many_speakers(self):
        tags = tag_failures(_ex(), "hello", _counts(), elapsed_sec=1.0, spk_delta=3)
        assert "diar.too_many_speakers" in tags

    def test_too_few_speakers(self):
        tags = tag_failures(_ex(), "hello", _counts(), elapsed_sec=1.0, spk_delta=-3)
        assert "diar.too_few_speakers" in tags

    def test_no_tags_for_good_sample(self):
        tags = tag_failures(_ex(), "hello world", _counts(S=0, D=0, I=0), elapsed_sec=1.0)
        # overlap_high_sample only if overlap==high
        non_info = [t for t in tags if t != "speech.overlap_high_sample"]
        assert non_info == []

    def test_high_overlap_info_tag(self):
        tags = tag_failures(_ex(overlap="high"), "hello world", _counts(), elapsed_sec=1.0)
        assert "speech.overlap_high_sample" in tags


# ---------------------------------------------------------------------------
# Normalize + WER pipeline
# ---------------------------------------------------------------------------

class TestNormalizeAndWER:
    def test_normalized_perfect(self):
        ref = normalize_text("Good Morning, Everyone!")
        hyp = normalize_text("good morning, everyone!")
        rate, _ = wer(ref, hyp)
        assert rate == pytest.approx(0.0)

    def test_typical_asr_error(self):
        ref = normalize_text("The revenue increased by fifteen percent")
        hyp = normalize_text("The revenue increased by 50 percent")
        rate, _ = wer(ref, hyp)
        # "fifteen" -> "50" = 1 substitution / 6 words
        assert rate == pytest.approx(1 / 6)

    def test_extra_words_hallucination(self):
        ref = normalize_text("Good morning everyone")
        hyp = normalize_text("Good morning everyone thank you for being here today")
        rate, c = wer(ref, hyp)
        assert c["insertions"] > 0


# ---------------------------------------------------------------------------
# Scores JSONL round-trip (filesystem)
# ---------------------------------------------------------------------------

class TestScoresJSONL:
    def test_write_and_read_back(self, tmp_path):
        scores = [
            {"sample_id": "s1", "metrics": {"wer": 0.1}, "failure_tags": []},
            {"sample_id": "s2", "metrics": {"wer": 0.5}, "failure_tags": ["asr.high_wer"]},
        ]
        path = tmp_path / "scores.jsonl"
        with path.open("w") as f:
            for row in scores:
                f.write(json.dumps(row) + "\n")

        loaded = []
        with path.open() as f:
            for line in f:
                loaded.append(json.loads(line.strip()))

        assert len(loaded) == 2
        assert loaded[0]["sample_id"] == "s1"
        assert loaded[1]["failure_tags"] == ["asr.high_wer"]
