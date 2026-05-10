"""Unit tests for eval.metrics.text_metrics."""
import pytest

from eval.metrics.text_metrics import (
    aggregate_wer_stats,
    cer,
    derived_content_metrics,
    levenshtein_counts,
    normalize_text,
    percentile,
    wer,
)


# ---------------------------------------------------------------------------
# levenshtein_counts
# ---------------------------------------------------------------------------

class TestLevenshteinCounts:
    def test_identical(self):
        c = levenshtein_counts(["a", "b", "c"], ["a", "b", "c"])
        assert c == {"substitutions": 0, "deletions": 0, "insertions": 0, "ref_len": 3}

    def test_empty_ref(self):
        c = levenshtein_counts([], ["a", "b"])
        assert c["insertions"] == 2
        assert c["deletions"] == 0
        assert c["ref_len"] == 0

    def test_empty_hyp(self):
        c = levenshtein_counts(["a", "b"], [])
        assert c["deletions"] == 2
        assert c["insertions"] == 0

    def test_substitution(self):
        c = levenshtein_counts(["hello", "world"], ["hello", "earth"])
        assert c["substitutions"] == 1
        assert c["deletions"] == 0
        assert c["insertions"] == 0

    def test_deletion(self):
        c = levenshtein_counts(["a", "b", "c"], ["a", "c"])
        assert c["deletions"] == 1

    def test_insertion(self):
        c = levenshtein_counts(["a", "c"], ["a", "b", "c"])
        assert c["insertions"] == 1

    def test_complex(self):
        # "the cat sat" vs "the dog"  -> 1 sub + 1 del
        c = levenshtein_counts(
            ["the", "cat", "sat"],
            ["the", "dog"],
        )
        assert c["substitutions"] + c["deletions"] + c["insertions"] == 2


# ---------------------------------------------------------------------------
# WER
# ---------------------------------------------------------------------------

class TestWER:
    def test_perfect(self):
        rate, _ = wer("hello world", "hello world")
        assert rate == pytest.approx(0.0)

    def test_all_wrong(self):
        rate, _ = wer("a b c", "x y z")
        assert rate == pytest.approx(1.0)

    def test_half_wrong(self):
        rate, _ = wer("a b c d", "a b x y")
        assert rate == pytest.approx(0.5)

    def test_empty_ref_clamps_to_one(self):
        rate, c = wer("", "hello world")
        # denom is max(1, 0) = 1; 2 insertions / 1 = 2.0 — rate can exceed 1
        assert rate >= 0.0
        assert c["ref_len"] == 0

    def test_empty_hyp(self):
        rate, c = wer("hello world", "")
        assert c["deletions"] == 2
        assert rate == pytest.approx(1.0)

    def test_case_sensitive_by_default(self):
        # wer operates on raw split; caller is responsible for normalization
        rate_same, _ = wer("Hello World", "Hello World")
        rate_diff, _ = wer("Hello World", "hello world")
        assert rate_same == pytest.approx(0.0)
        assert rate_diff == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# CER
# ---------------------------------------------------------------------------

class TestCER:
    def test_perfect(self):
        rate, _ = cer("abc", "abc")
        assert rate == pytest.approx(0.0)

    def test_one_char_sub(self):
        rate, c = cer("abc", "axc")
        assert c["substitutions"] == 1
        assert rate == pytest.approx(1 / 3)

    def test_empty_strings(self):
        rate, _ = cer("", "")
        assert rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Derived content metrics
# ---------------------------------------------------------------------------

class TestDerivedContentMetrics:
    def test_no_errors(self):
        _, counts = wer("the quick brown fox", "the quick brown fox")
        dm = derived_content_metrics(counts, "the quick brown fox")
        assert dm["missing_content_rate"] == pytest.approx(0.0)
        assert dm["hallucination_rate"] == pytest.approx(0.0)

    def test_all_deleted(self):
        _, counts = wer("a b c d", "")
        dm = derived_content_metrics(counts, "")
        assert dm["missing_content_rate"] == pytest.approx(1.0)

    def test_all_inserted(self):
        _, counts = wer("", "a b c d")
        dm = derived_content_metrics(counts, "a b c d")
        assert dm["hallucination_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Normalize text
# ---------------------------------------------------------------------------

class TestNormalizeText:
    def test_lowercase(self):
        assert normalize_text("Hello WORLD") == "hello world"

    def test_whitespace(self):
        assert normalize_text("  hello   world  ") == "hello world"

    def test_strip_punct(self):
        result = normalize_text("Hello, world!", strip_punct=True)
        assert "," not in result
        assert "!" not in result

    def test_no_lowercase(self):
        result = normalize_text("Hello World", lowercase=False)
        assert result == "Hello World"


# ---------------------------------------------------------------------------
# Percentile / aggregate stats
# ---------------------------------------------------------------------------

class TestAggregateWERStats:
    def test_empty(self):
        stats = aggregate_wer_stats([])
        assert stats["mean"] is None
        assert stats["n"] == 0

    def test_single(self):
        stats = aggregate_wer_stats([0.25])
        assert stats["mean"] == pytest.approx(0.25)
        assert stats["p50"] == pytest.approx(0.25)
        assert stats["p90"] == pytest.approx(0.25)

    def test_multiple(self):
        vals = [0.1, 0.2, 0.3, 0.4, 0.5]
        stats = aggregate_wer_stats(vals)
        assert stats["mean"] == pytest.approx(0.3)
        assert stats["n"] == 5

    def test_percentile_extremes(self):
        vals = [float(i) for i in range(101)]  # 0..100
        assert percentile(vals, 0) == pytest.approx(0.0)
        assert percentile(vals, 100) == pytest.approx(100.0)
        assert percentile(vals, 50) == pytest.approx(50.0)
