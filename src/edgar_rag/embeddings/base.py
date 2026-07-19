"""The embedding provider contract.

Every backend (local sentence-transformers, deterministic hashing, and later a
cloud provider) implements this same small interface, so the rest of the
pipeline never knows or cares which one is wired in. `embed` takes a batch of
texts and returns one vector per text, in order; `dim` reports the vector width
so the search index can be created with the matching dimension.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def dim(self) -> int:
        """The dimension of the vectors this provider emits."""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per input, in order."""
        ...
