"""Turn text into unit-length vectors. Ollama in real use, a deterministic hash embedder for tests."""
import hashlib
import re

import numpy as np
import requests

from app.core.config import settings


class EmbeddingError(Exception):
    """The embedding backend could not be used."""


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class HashEmbedder:
    """Bag-of-words feature hashing. No model needed, so tests run anywhere. Shared words -> similar vectors."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.name = f"hash-{dim}"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                digest = hashlib.md5(token.encode()).digest()
                out[row, int.from_bytes(digest[:4], "little") % self.dim] += 1.0 if digest[4] & 1 else -1.0
        return _normalise(out)


class OllamaEmbedder:
    """LangChain Ollama embeddings adapter. Pull the model first: ollama pull nomic-embed-text."""

    def __init__(self, model: str | None = None, base_url: str | None = None, batch: int = 16):
        self.model = model or settings.EMBED_MODEL
        self.base_url = (base_url or settings.OLLAMA_URL).rstrip("/")
        self.batch = batch
        self.name = f"ollama:{self.model}"
        try:
            from langchain_ollama import OllamaEmbeddings
            self.client = OllamaEmbeddings(model=self.model, base_url=self.base_url)
        except Exception as exc:
            raise EmbeddingError(
                "Could not initialise LangChain OllamaEmbeddings. Install langchain-ollama and check Ollama."
            ) from exc

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, settings.RAG_EMBED_DIM), dtype=np.float32)
        try:
            vectors: list[list[float]] = []
            for start in range(0, len(texts), self.batch):
                vectors.extend(self.client.embed_documents(texts[start:start + self.batch]))
            matrix = _normalise(np.asarray(vectors, dtype=np.float32))
        except Exception as exc:
            raise EmbeddingError(f"LangChain could not create Ollama embeddings at {self.base_url}: {exc}") from exc
        if matrix.shape[1] != settings.RAG_EMBED_DIM:
            raise EmbeddingError(
                f"Embedding model returned {matrix.shape[1]} dimensions; RAG_EMBED_DIM is "
                f"{settings.RAG_EMBED_DIM}. Set the matching dimension and migrate the pgvector column."
            )
        return matrix


def get_embedder():
    provider = str(getattr(settings, "EMBED_PROVIDER", "ollama")).lower()
    if provider == "hash":
        return HashEmbedder()
    if provider == "ollama":
        return OllamaEmbedder()
    raise EmbeddingError(f"Unsupported EMBED_PROVIDER '{provider}'. Use 'ollama' or 'hash'.")
