"""
SSS (Sentence Similarity Score) metric adapter.

Wraps the existing chunker/aligner/embedder/scorer stack so SSS can be
included alongside WER/CER in the eval pipeline without requiring Ollama
during unit tests — the embedder is injected as a callable.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(a, b)) / denom))


def weighted_sss(scores: list[float], weights: list[int]) -> float:
    if not scores:
        return 0.0
    total_w = sum(weights)
    if total_w <= 0:
        return float(np.mean(scores))
    return sum(s * w for s, w in zip(scores, weights)) / total_w


def sss_score(
    gt_chunks: list[str],
    tx_chunks: list[str],
    embed_fn: Callable[[list[str]], list[np.ndarray]],
) -> tuple[float, list[float]]:
    """
    Compute weighted SSS for two pre-chunked+aligned text lists.

    embed_fn must accept a list[str] and return list[np.ndarray].
    Returns (total_sss, per_chunk_scores).
    """
    if not gt_chunks:
        return 0.0, []

    all_texts = gt_chunks + tx_chunks
    all_embeddings = embed_fn(all_texts)
    gt_embeddings = all_embeddings[: len(gt_chunks)]
    tx_embeddings = all_embeddings[len(gt_chunks) :]

    scores = [
        cosine_similarity(np.asarray(g, dtype=np.float32), np.asarray(t, dtype=np.float32))
        for g, t in zip(gt_embeddings, tx_embeddings)
    ]
    weights = [max(len(g), len(t)) for g, t in zip(gt_chunks, tx_chunks)]
    return weighted_sss(scores, weights), scores


def sss_from_text(
    gt_text: str,
    tx_text: str,
    embed_fn: Callable[[list[str]], list[np.ndarray]],
    *,
    chunk_strategy: str = "sentence_window",
    target_sentences: int = 3,
    max_chars: int = 900,
) -> tuple[float, list[float]]:
    """
    High-level helper: chunk, align, embed, and score two raw text strings.

    Relies on the project's existing chunker + aligner modules.
    """
    from claude_version.aligner import align_chunks
    from claude_version.chunker import chunk_text

    gt_chunks = chunk_text(gt_text, strategy=chunk_strategy,
                           target_sentences=target_sentences, max_chars=max_chars)
    tx_chunks = chunk_text(tx_text, strategy=chunk_strategy,
                           target_sentences=target_sentences, max_chars=max_chars)
    aligned = align_chunks(gt_chunks, tx_chunks)
    gt_aligned = [p[0] for p in aligned]
    tx_aligned = [p[1] for p in aligned]
    return sss_score(gt_aligned, tx_aligned, embed_fn)
