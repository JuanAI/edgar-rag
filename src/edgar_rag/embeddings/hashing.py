"""A deterministic, dependency-free embedding provider (feature hashing).

Each text is tokenized into words, every word is hashed into one of `dim`
buckets (with a hashed +/- sign to reduce collisions cancelling constructively),
and the resulting vector is L2-normalized so it is ready for cosine similarity.

This is the same idea as scikit-learn's HashingVectorizer. It is NOT semantically
meaningful — "revenue" and "sales" land in unrelated buckets — so it is here for
two honest jobs, not for real retrieval quality:

  * tests and CI can embed without downloading a ~90 MB transformer model, and
  * the whole pipeline can run fully offline with zero external dependencies.

For real semantic retrieval, use the `local` provider (see `local.py`).

We hash with blake2b rather than the built-in `hash()` because Python salts
`hash()` per process (PYTHONHASHSEED), which would make vectors differ between
runs. blake2b is stable, so the same text always yields the same vector.
"""

from __future__ import annotations

import hashlib
import math
import re

_TOKEN = re.compile(r"[a-z0-9]+")


def _bucket_and_sign(token: str, dim: int) -> tuple[int, float]:
    # Independent bytes for the bucket index and the sign so they do not
    # correlate the way `h % dim` and `h % 2` would when `dim` is even.
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    bucket = int.from_bytes(digest[:4], "big") % dim
    sign = 1.0 if digest[4] & 1 else -1.0
    return bucket, sign


class HashingEmbeddingProvider:
    """Deterministic feature-hashing embeddings for tests and offline runs."""

    def __init__(self, dim: int = 384) -> None:
        if dim <= 0:
            raise ValueError("embedding dim must be positive")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for token in _TOKEN.findall(text.lower()):
            bucket, sign = _bucket_and_sign(token, self._dim)
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0.0:
            vector = [value / norm for value in vector]
        return vector
