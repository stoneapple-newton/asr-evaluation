"""Text chunking utilities."""

import re


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []
    return [part.strip() for part in _SENTENCE_RE.split(normalized) if part.strip()]


def split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text.strip())
    return [re.sub(r"\s+", " ", part).strip() for part in parts if part.strip()]


def sentence_windows(
    text: str,
    target_sentences_per_chunk: int = 3,
    max_chars_per_chunk: int = 900,
) -> list[str]:
    sentences = split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_chars = 0

    for sentence in sentences:
        next_chars = current_chars + len(sentence) + (1 if current else 0)
        reached_sentence_target = len(current) >= target_sentences_per_chunk
        reached_char_target = current and next_chars > max_chars_per_chunk
        if reached_sentence_target or reached_char_target:
            chunks.append(" ".join(current))
            current = []
            current_chars = 0
        current.append(sentence)
        current_chars += len(sentence) + (1 if current_chars else 0)

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_text(
    text: str,
    strategy: str = "sentence_window",
    target_sentences_per_chunk: int = 3,
    max_chars_per_chunk: int = 900,
) -> list[str]:
    if strategy == "sentence":
        return split_sentences(text)
    if strategy == "paragraph":
        return split_paragraphs(text)
    if strategy == "sentence_window":
        return sentence_windows(text, target_sentences_per_chunk, max_chars_per_chunk)
    raise ValueError(f"Unknown chunk strategy: {strategy}")

