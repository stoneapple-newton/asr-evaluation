"""Cosine-similarity scoring."""

from typing import List, Tuple

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm == 0:
        return 0.0
    return max(0.0, min(1.0, dot / norm))


def score_segments(
    aligned: List[Tuple[str, str]],
    gt_embeddings: List[np.ndarray],
    tx_embeddings: List[np.ndarray],
) -> List[float]:
    if len(aligned) != len(gt_embeddings) or len(aligned) != len(tx_embeddings):
        raise ValueError("Mismatched lengths between aligned pairs and embeddings")
    return [cosine_similarity(g, t) for g, t in zip(gt_embeddings, tx_embeddings)]


def total_sss(scores: List[float], chunk_lengths: List[int]) -> float:
    if not scores:
        return 0.0
    total_weight = sum(chunk_lengths)
    if total_weight == 0:
        return float(np.mean(scores))
    weighted = sum(s * w for s, w in zip(scores, chunk_lengths))
    return weighted / total_weight
