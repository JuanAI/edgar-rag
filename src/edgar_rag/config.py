"""Central configuration, loaded from environment variables.

Everything is overridable via env so the same image runs locally, in Docker, or
against real AWS/Bedrock by flipping a few variables (see .env.example).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EDGAR_RAG_", env_file=".env", extra="ignore")

    # --- AWS / LocalStack (SQS + S3) ---
    aws_region: str = "us-east-1"
    aws_endpoint_url: str | None = Field(
        default="http://localstack:4566",
        description="Point at LocalStack locally; set to None/empty to use real AWS.",
    )
    aws_access_key_id: str = "test"
    aws_secret_access_key: str = "test"
    sqs_queue_name: str = "edgar-ingestion"
    s3_bucket: str = "edgar-documents"

    # --- OpenSearch ---
    opensearch_host: str = "opensearch"
    opensearch_port: int = 9200
    opensearch_use_ssl: bool = False
    opensearch_index: str = "edgar-chunks"
    # kNN / HNSW vector-index tuning. `m` and `ef_construction` trade retrieval
    # recall against index build time and memory; cosine matches our normalized
    # embeddings, and the Lucene engine ships with OpenSearch (no extra plugin).
    opensearch_engine: str = "lucene"
    opensearch_space_type: str = "cosinesimil"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 128

    # --- Providers ---
    # embeddings: "local" (sentence-transformers) | "bedrock"
    embedding_provider: str = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    bedrock_embedding_model: str = "amazon.titan-embed-text-v2:0"

    # generation: "stub" (extractive, zero-dependency) | "ollama" | "anthropic" | "bedrock"
    generation_provider: str = "stub"
    ollama_host: str = "http://host.docker.internal:11434"
    ollama_model: str = "llama3.2"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-5"
    bedrock_generation_model: str = "anthropic.claude-sonnet-5"

    # --- Chunking ---
    chunk_size_chars: int = 1200
    chunk_overlap_chars: int = 200

    # --- Retrieval / generation guardrails ---
    top_k: int = 5
    # Rough context budget (chars). We reject a request whose assembled context
    # exceeds this rather than letting the LLM hard-fail on an over-long prompt.
    max_context_chars: int = 24_000

    # --- Ingestion concurrency: separate limits per stage (extract vs index) ---
    extract_concurrency: int = 4
    index_concurrency: int = 2
    sqs_max_messages: int = 5
    sqs_wait_seconds: int = 10

    @property
    def opensearch_scheme(self) -> str:
        return "https" if self.opensearch_use_ssl else "http"


@lru_cache
def get_settings() -> Settings:
    return Settings()
