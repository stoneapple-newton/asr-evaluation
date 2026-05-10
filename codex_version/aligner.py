"""Chunk alignment helpers."""


def _merge_shortest_adjacent(chunks: list[str]) -> list[str]:
    if len(chunks) <= 1:
        return chunks
    best_index = min(
        range(len(chunks) - 1),
        key=lambda i: len(chunks[i]) + len(chunks[i + 1]),
    )
    return (
        chunks[:best_index]
        + [f"{chunks[best_index]} {chunks[best_index + 1]}"]
        + chunks[best_index + 2 :]
    )


def align_chunks(ground_truth_chunks: list[str], transcription_chunks: list[str]) -> list[tuple[str, str]]:
    """Align two chunk lists by merging adjacent chunks in the longer list."""
    gt = list(ground_truth_chunks)
    tx = list(transcription_chunks)

    while len(gt) > len(tx):
        gt = _merge_shortest_adjacent(gt)
    while len(tx) > len(gt):
        tx = _merge_shortest_adjacent(tx)

    return list(zip(gt, tx))

