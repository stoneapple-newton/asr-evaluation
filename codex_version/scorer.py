"""Sentence Similarity Score calculations."""

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(a, b)) / denominator))


def total_sss(segment_scores: list[float], weights: list[int]) -> float:
    if not segment_scores:
        return 0.0
    total_weight = sum(weights)
    if total_weight <= 0:
        return float(np.mean(segment_scores))
    return sum(score * weight for score, weight in zip(segment_scores, weights)) / total_weight

