"""Ollama embedding client."""

import time

import httpx
import numpy as np

from codex_version.config import DEFAULT_EMBED_MODEL, DEFAULT_OLLAMA_URL


class OllamaEmbedder:
    def __init__(self, base_url: str = DEFAULT_OLLAMA_URL, model: str = DEFAULT_EMBED_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._cache: dict[str, list[float]] = {}

    def _embed_one(self, text: str) -> list[float]:
        if text in self._cache:
            return self._cache[text]

        payload = {"model": self.model, "prompt": text}
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = httpx.post(f"{self.base_url}/api/embeddings", json=payload, timeout=120.0)
                response.raise_for_status()
                embedding = response.json()["embedding"]
                self._cache[text] = embedding
                return embedding
            except Exception as exc:
                last_error = exc
                time.sleep(0.75 * (attempt + 1))
        raise RuntimeError(f"Ollama embedding failed for model {self.model}: {last_error}")

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        return [np.asarray(self._embed_one(text), dtype=np.float32) for text in texts]

