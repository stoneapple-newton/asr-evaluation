"""Align chunks 1-to-1, merging shorter chunks in the longer list until counts match."""

from typing import List, Tuple


def _merge_shortest(chunks: List[str]) -> List[str]:
    """Merge the pair of adjacent chunks with the smallest combined length."""
    if len(chunks) <= 1:
        return chunks
    best_i = 0
    best_len = len(chunks[0]) + len(chunks[1])
    for i in range(1, len(chunks) - 1):
        pair_len = len(chunks[i]) + len(chunks[i + 1])
        if pair_len < best_len:
            best_len = pair_len
            best_i = i
    merged = chunks[:best_i] + [chunks[best_i] + " " + chunks[best_i + 1]] + chunks[best_i + 2 :]
    return merged


def align(gt_chunks: List[str], tx_chunks: List[str]) -> List[Tuple[str, str]]:
    """Return aligned (gt, tx) pairs. Merges chunks in the longer list until counts match."""
    gt = list(gt_chunks)
    tx = list(tx_chunks)

    while len(gt) > len(tx):
        tx = _merge_shortest(tx)
    while len(tx) > len(gt):
        gt = _merge_shortest(gt)

    return list(zip(gt, tx))
