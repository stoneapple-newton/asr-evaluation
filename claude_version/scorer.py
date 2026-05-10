import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(a, b)) / denom))


def total_sss(scores: list[float], weights: list[int]) -> float:
    if not scores:
        return 0.0
    total_weight = sum(weights)
    if total_weight <= 0:
        return float(np.mean(scores))
    return sum(s * w for s, w in zip(scores, weights)) / total_weight
