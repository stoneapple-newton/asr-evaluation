"""Lexical ASR evaluation metrics.

The existing SSS scorer captures semantic similarity with embeddings. These
helpers add deterministic ASR metrics that are commonly reported alongside
semantic scores, including WER, CER, MER, WIP, and WIL.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence


_WORD_RE = re.compile(r"\b\w+(?:['’]\w+)?\b", re.UNICODE)


@dataclass(frozen=True)
class EditCounts:
    """Levenshtein alignment counts for a reference/hypothesis pair."""

    hits: int
    substitutions: int
    deletions: int
    insertions: int
    reference_length: int
    hypothesis_length: int

    @property
    def edits(self) -> int:
        return self.substitutions + self.deletions + self.insertions


@dataclass(frozen=True)
class ASREvaluation:
    """Common ASR metrics for one reference/hypothesis pair."""

    wer: float
    cer: float
    mer: float
    wip: float
    wil: float
    hits: int
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int
    hypothesis_words: int
    reference_chars: int
    hypothesis_chars: int
    character_errors: int

    def rounded(self, digits: int = 6) -> dict[str, float | int]:
        """Return JSON-friendly metrics with stable numeric precision."""
        return {
            "wer": round(self.wer, digits),
            "cer": round(self.cer, digits),
            "mer": round(self.mer, digits),
            "wip": round(self.wip, digits),
            "wil": round(self.wil, digits),
            "hits": self.hits,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "reference_words": self.reference_words,
            "hypothesis_words": self.hypothesis_words,
            "reference_chars": self.reference_chars,
            "hypothesis_chars": self.hypothesis_chars,
            "character_errors": self.character_errors,
        }


def normalize_text(text: str) -> str:
    """Normalize text for lexical ASR metrics.

    The normalizer lowercases, removes punctuation by tokenizing words, and
    collapses whitespace. This keeps WER/CER deterministic without depending on
    external ASR-evaluation packages.
    """
    return " ".join(_WORD_RE.findall(text.lower()))


def word_tokens(text: str) -> list[str]:
    """Tokenize text into normalized word tokens for WER-family metrics."""
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def char_tokens(text: str) -> list[str]:
    """Tokenize text into normalized characters for CER.

    Spaces are removed so CER focuses on recognized character content rather
    than formatting differences.
    """
    return list(normalize_text(text).replace(" ", ""))


def edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> EditCounts:
    """Compute Levenshtein hit/substitution/deletion/insertion counts."""
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    costs = [[0] * cols for _ in range(rows)]
    counts = [[(0, 0, 0, 0)] * cols for _ in range(rows)]

    for i in range(1, rows):
        costs[i][0] = i
        counts[i][0] = (0, 0, i, 0)
    for j in range(1, cols):
        costs[0][j] = j
        counts[0][j] = (0, 0, 0, j)

    for i in range(1, rows):
        for j in range(1, cols):
            if reference[i - 1] == hypothesis[j - 1]:
                match_cost = costs[i - 1][j - 1]
                h, s, d, ins = counts[i - 1][j - 1]
                match_counts = (h + 1, s, d, ins)
            else:
                match_cost = costs[i - 1][j - 1] + 1
                h, s, d, ins = counts[i - 1][j - 1]
                match_counts = (h, s + 1, d, ins)

            deletion_cost = costs[i - 1][j] + 1
            h, s, d, ins = counts[i - 1][j]
            deletion_counts = (h, s, d + 1, ins)

            insertion_cost = costs[i][j - 1] + 1
            h, s, d, ins = counts[i][j - 1]
            insertion_counts = (h, s, d, ins + 1)

            best_cost, best_counts = min(
                (
                    (match_cost, match_counts),
                    (deletion_cost, deletion_counts),
                    (insertion_cost, insertion_counts),
                ),
                key=lambda item: (item[0], item[1][1], item[1][2], item[1][3]),
            )
            costs[i][j] = best_cost
            counts[i][j] = best_counts

    hits, substitutions, deletions, insertions = counts[-1][-1]
    return EditCounts(
        hits=hits,
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_length=len(reference),
        hypothesis_length=len(hypothesis),
    )


def _safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0.0 else numerator / denominator


def evaluate_asr(reference: str, hypothesis: str) -> ASREvaluation:
    """Evaluate a transcript hypothesis against reference text."""
    word_counts = edit_counts(word_tokens(reference), word_tokens(hypothesis))
    character_counts = edit_counts(char_tokens(reference), char_tokens(hypothesis))

    word_errors = word_counts.edits
    word_denominator = word_counts.reference_length
    mer_denominator = word_counts.hits + word_errors
    reference_accuracy = _safe_divide(
        word_counts.hits,
        word_counts.hits + word_counts.substitutions + word_counts.deletions,
    )
    hypothesis_accuracy = _safe_divide(
        word_counts.hits,
        word_counts.hits + word_counts.substitutions + word_counts.insertions,
    )
    wip = reference_accuracy * hypothesis_accuracy

    return ASREvaluation(
        wer=_safe_divide(word_errors, word_denominator),
        cer=_safe_divide(character_counts.edits, character_counts.reference_length),
        mer=_safe_divide(word_errors, mer_denominator),
        wip=wip,
        wil=1.0 - wip,
        hits=word_counts.hits,
        substitutions=word_counts.substitutions,
        deletions=word_counts.deletions,
        insertions=word_counts.insertions,
        reference_words=word_counts.reference_length,
        hypothesis_words=word_counts.hypothesis_length,
        reference_chars=character_counts.reference_length,
        hypothesis_chars=character_counts.hypothesis_length,
        character_errors=character_counts.edits,
    )


def aggregate_asr(evaluations: Sequence[ASREvaluation]) -> ASREvaluation:
    """Aggregate segment evaluations by summing edit counts."""
    if not evaluations:
        return evaluate_asr("", "")

    hits = sum(evaluation.hits for evaluation in evaluations)
    substitutions = sum(evaluation.substitutions for evaluation in evaluations)
    deletions = sum(evaluation.deletions for evaluation in evaluations)
    insertions = sum(evaluation.insertions for evaluation in evaluations)
    reference_words = sum(evaluation.reference_words for evaluation in evaluations)
    hypothesis_words = sum(evaluation.hypothesis_words for evaluation in evaluations)
    reference_chars = sum(evaluation.reference_chars for evaluation in evaluations)
    hypothesis_chars = sum(evaluation.hypothesis_chars for evaluation in evaluations)
    character_errors = sum(evaluation.character_errors for evaluation in evaluations)

    word_errors = substitutions + deletions + insertions
    reference_accuracy = _safe_divide(hits, hits + substitutions + deletions)
    hypothesis_accuracy = _safe_divide(hits, hits + substitutions + insertions)
    wip = reference_accuracy * hypothesis_accuracy

    return ASREvaluation(
        wer=_safe_divide(word_errors, reference_words),
        cer=_safe_divide(character_errors, reference_chars),
        mer=_safe_divide(word_errors, hits + word_errors),
        wip=wip,
        wil=1.0 - wip,
        hits=hits,
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_words=reference_words,
        hypothesis_words=hypothesis_words,
        reference_chars=reference_chars,
        hypothesis_chars=hypothesis_chars,
        character_errors=character_errors,
    )
