"""Domain models shared across the ingestion and query paths."""

from __future__ import annotations

from pydantic import BaseModel, Field


class IngestionMessage(BaseModel):
    """The payload we expect on the SQS queue — one document to ingest."""

    document_id: str
    s3_bucket: str
    s3_key: str
    tenant: str
    doc_type: str = "10-K"
    metadata: dict[str, str] = Field(default_factory=dict)


class Page(BaseModel):
    """A logical section of a source document (EDGAR filings have no real pages,
    so we treat detected sections as 'pages'). A page yields multiple chunks."""

    page_number: int
    text: str


class Chunk(BaseModel):
    """A retrievable passage. `embedding` is filled in during ingestion."""

    document_id: str
    tenant: str
    doc_type: str
    page_number: int
    chunk_index: int
    text: str
    embedding: list[float] | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return f"{self.document_id}:{self.page_number}:{self.chunk_index}"


class RetrievedChunk(BaseModel):
    text: str
    score: float
    document_id: str
    doc_type: str
    page_number: int
    metadata: dict[str, str] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    question: str
    tenant: str
    doc_type: str | None = None  # scope retrieval to a single filing type when set
    top_k: int | None = None


class Citation(BaseModel):
    document_id: str
    page_number: int
    score: float


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    provider: str
