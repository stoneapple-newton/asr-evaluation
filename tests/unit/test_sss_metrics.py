"""Unit tests for eval.metrics.sss_metrics (no Ollama required)."""
import numpy as np
import pytest

from eval.metrics.sss_metrics import cosine_similarity, sss_score, weighted_sss


def _fake_embedder(texts: list[str]) -> list[np.ndarray]:
    """Deterministic fake embedder: embed each text as a one-hot-ish vector."""
    dim = 8
    rng = np.random.default_rng(42)
    return [rng.random(dim).astype(np.float32) for _ in texts]


def _identity_embedder(texts: list[str]) -> list[np.ndarray]:
    """Returns the same fixed vector for every text — gives SSS=1.0."""
    v = np.ones(4, dtype=np.float32)
    return [v for _ in texts]


class TestCosineSimilarity:
    def test_identical(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_zero_vector(self):
        z = np.zeros(3, dtype=np.float32)
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert cosine_similarity(z, v) == pytest.approx(0.0)

    def test_clamped_to_one(self):
        v = np.array([1.0, 0.0], dtype=np.float32)
        # numerical precision can push slightly above 1 — verify clamp
        assert cosine_similarity(v, v) <= 1.0


class TestWeightedSSS:
    def test_empty(self):
        assert weighted_sss([], []) == pytest.approx(0.0)

    def test_equal_weights(self):
        scores = [0.5, 1.0]
        weights = [1, 1]
        assert weighted_sss(scores, weights) == pytest.approx(0.75)

    def test_zero_total_weight_falls_back_to_mean(self):
        scores = [0.6, 0.8]
        weights = [0, 0]
        assert weighted_sss(scores, weights) == pytest.approx(0.7)


class TestSSSScore:
    def test_identical_texts_with_identity_embedder(self):
        chunks = ["hello world", "foo bar"]
        total, per_chunk = sss_score(chunks, chunks, _identity_embedder)
        assert total == pytest.approx(1.0)
        assert all(s == pytest.approx(1.0) for s in per_chunk)

    def test_empty_chunks(self):
        total, per_chunk = sss_score([], [], _fake_embedder)
        assert total == pytest.approx(0.0)
        assert per_chunk == []

    def test_returns_float_and_list(self):
        gt = ["the quick brown fox"]
        tx = ["the quick brown dog"]
        total, per_chunk = sss_score(gt, tx, _fake_embedder)
        assert isinstance(total, float)
        assert isinstance(per_chunk, list)
        assert len(per_chunk) == 1

    def test_score_in_range(self):
        gt = ["hello world", "foo bar baz"]
        tx = ["hello earth", "foo bar qux"]
        total, per_chunk = sss_score(gt, tx, _fake_embedder)
        assert 0.0 <= total <= 1.0
        assert all(0.0 <= s <= 1.0 for s in per_chunk)
