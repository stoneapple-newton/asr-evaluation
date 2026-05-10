import re

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", text.strip())


def split_sentences(text: str) -> list[str]:
    normalized = _normalize(text)
    if not normalized:
        return []
    return [s.strip() for s in _SENTENCE_END.split(normalized) if s.strip()]


def split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text.strip())
    return [_normalize(p) for p in parts if p.strip()]


def sentence_window(
    text: str,
    target_sentences: int = 3,
    max_chars: int = 900,
) -> list[str]:
    sentences = split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_chars = 0

    for sentence in sentences:
        addition = len(sentence) + (1 if current else 0)
        if current and (len(current) >= target_sentences or current_chars + addition > max_chars):
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
    target_sentences: int = 3,
    max_chars: int = 900,
) -> list[str]:
    if strategy == "sentence":
        return split_sentences(text)
    if strategy == "paragraph":
        return split_paragraphs(text)
    if strategy == "sentence_window":
        return sentence_window(text, target_sentences, max_chars)
    raise ValueError(f"Unknown chunk strategy: {strategy!r}")
