def _merge_shortest(chunks: list[str]) -> list[str]:
    if len(chunks) <= 1:
        return chunks
    i = min(range(len(chunks) - 1), key=lambda j: len(chunks[j]) + len(chunks[j + 1]))
    return chunks[:i] + [f"{chunks[i]} {chunks[i + 1]}"] + chunks[i + 2 :]


def align_chunks(gt_chunks: list[str], tx_chunks: list[str]) -> list[tuple[str, str]]:
    gt = list(gt_chunks)
    tx = list(tx_chunks)
    while len(gt) > len(tx):
        gt = _merge_shortest(gt)
    while len(tx) > len(gt):
        tx = _merge_shortest(tx)
    return list(zip(gt, tx))
