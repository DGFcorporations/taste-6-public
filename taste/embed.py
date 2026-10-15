"""Embedding — C3 permutation: BGE-small-en-v1.5 via fastembed (ONNX, local).

Falls back to a deterministic hashing embedder when the model can't load
(offline CI, cold box). Same interface either way; retrieval just degrades
to lexical-ish similarity. Set TASTE_EMBED=hash to force the fallback.

BGE-small is retrieval-tuned and shares MiniLM's 384-dim cost class, so
the divergence is about ranking quality, not latency.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from functools import lru_cache

log = logging.getLogger("taste.embed")

BGE_MODEL = "BAAI/bge-small-en-v1.5"
DIM = 384


class HashingEmbedder:
    """Deterministic bag-of-hashed-tokens embedder — offline fallback.

    Not semantic — just stable. Same text → same vector; overlapping
    vocabulary → partial cosine. Good enough to keep the pipeline honest
    when the real model can't download.
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * DIM
            for tok in text.lower().split():
                h = int(hashlib.sha256(tok.encode()).hexdigest(), 16)
                vec[h % DIM] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class BGEEmbedder:
    """BGE-small-en-v1.5 via fastembed — the C3 permutation."""

    def __init__(self):
        from fastembed import TextEmbedding
        self._m = TextEmbedding(BGE_MODEL)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._m.embed(texts)]


@lru_cache(maxsize=1)
def get_embedder():
    """Lazy singleton — model downloads on first real use."""
    if os.environ.get("TASTE_EMBED") == "hash":
        return HashingEmbedder()
    try:
        return BGEEmbedder()
    except Exception as exc:
        log.warning("BGE embedder failed to load (%s); using the hashing embedder",
                    type(exc).__name__)
        return HashingEmbedder()


def embed(texts: list[str]) -> list[list[float]]:
    return get_embedder().embed(texts)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
