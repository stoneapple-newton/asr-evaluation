"""Ollama embedding client."""

import time
from typing import List

import httpx
import numpy as np

from kimi.config import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL


class OllamaEmbedder:
    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_EMBED_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._cache: dict = {}

    def _fetch(self, text: str) -> List[float]:
        if text in self._cache:
            return self._cache[text]

        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self.model, "prompt": text}

        for attempt in range(3):
            try:
                resp = httpx.post(url, json=payload, timeout=120.0)
                resp.raise_for_status()
                data = resp.json()
                embedding = data["embedding"]
                self._cache[text] = embedding
                return embedding
            except Exception as exc:
                if attempt == 2:
                    raise RuntimeError(f"Ollama embedding failed after 3 retries: {exc}") from exc
                time.sleep(1.0 * (attempt + 1))

        raise RuntimeError("Unreachable")

    def embed(self, texts: List[str]) -> List[np.ndarray]:
        """Return a list of 1-D numpy arrays."""
        return [np.array(self._fetch(t), dtype=np.float32) for t in texts]
