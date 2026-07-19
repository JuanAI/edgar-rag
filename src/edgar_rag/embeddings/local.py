"""Local semantic embeddings via sentence-transformers (the default backend).

Runs a small transformer (MiniLM by default) entirely on the host — no cloud
credentials, no per-call cost — which makes the whole system runnable by anyone
who checks out the repo. The trade-off versus the hashing provider is a one-time
model download (~90 MB) and real compute per batch, in exchange for embeddings
that actually capture meaning ("revenue" lands near "sales").

The `sentence-transformers` import lives inside `_load_model`, not at module top
level, for two reasons: it is an optional dependency (the `local` extra), so the
package must import fine without it; and the model load is deferred until the
first embed call. The loaded model is cached per name so repeated provider
construction does not reload it.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class LocalEmbeddingProvider:
    """sentence-transformers embeddings, normalized for cosine similarity."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: Any = None

    def _model_or_load(self) -> Any:
        if self._model is None:
            self._model = _load_model(self._model_name)
        return self._model

    @property
    def dim(self) -> int:
        return int(self._model_or_load().get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        # normalize_embeddings=True returns unit vectors, matching the hashing
        # provider so the search index can use cosine similarity uniformly.
        vectors = self._model_or_load().encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]
