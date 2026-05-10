"""Text chunking strategies."""

import re
from typing import List


def _sentences(text: str) -> List[str]:
    """Simple regex-based sentence splitter (no NLTK dependency)."""
    # Split on sentence-ending punctuation followed by whitespace or EOS
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _paragraphs(text: str) -> List[str]:
    """Split on blank lines."""
    parts = re.split(r'\n\s*\n', text.strip())
    return [p.strip().replace('\n', ' ') for p in parts if p.strip()]


def chunk(text: str, strategy: str = "sentence") -> List[str]:
    if strategy == "sentence":
        return _sentences(text)
    if strategy == "paragraph":
        return _paragraphs(text)
    raise ValueError(f"Unknown chunk strategy: {strategy}")
