# edgar-rag

A small, runnable **RAG (Retrieval-Augmented Generation)** pipeline over public
**SEC filings**. It downloads company annual reports, indexes them into a vector
store, and answers questions about them with citations back to the source —
runnable locally in Docker, with no cloud credentials by default.

## What it accomplishes

It demonstrates, on public data, how a document-RAG pipeline is built end to end:

```
download → parse → chunk → embed → store → retrieve → answer (with citations)
```

The emphasis is the **ingestion (write) path** — the under-appreciated half of
RAG: reliably getting messy documents *in*, parsed, chunked, embedded, and
searchable. It's a portfolio project, built up gradually in the open.

## System map

Numbers `1–7` trace the write path; letters `a–e` trace the read path. The two
pipelines only meet at OpenSearch — one fills it, the other queries it.

```mermaid
flowchart TB
  U(["You — browser /docs · make ask · curl"]):::actor
  SEC[("SEC EDGAR<br/>public 10-K filings")]:::ext

  subgraph WP["WRITE PATH · ingestion"]
    direction TB
    SEED["seed.py<br/>one-off job"]:::write
    WORK["Ingestion worker<br/>long-running consumer"]:::write
    EMB1{{"Embedding model<br/>MiniLM · local"}}:::model
  end

  subgraph INFRA["LOCAL INFRASTRUCTURE · Docker"]
    direction TB
    SQS[["SQS queue<br/>LocalStack"]]:::infra
    S3[("S3 bucket<br/>LocalStack")]:::infra
    OS[("OpenSearch<br/>chunks + vectors · kNN")]:::store
    DASH["OpenSearch<br/>Dashboards"]:::infra
  end

  subgraph RP["READ PATH · query"]
    direction TB
    API["Query API<br/>FastAPI · /chat"]:::read
    EMB2{{"Embedding model<br/>MiniLM · local"}}:::model
    GEN{{"Generation<br/>stub · ollama · claude"}}:::model
  end

  SEC -->|"1 · download"| SEED
  SEED -->|"2 · upload file"| S3
  SEED -->|"3 · enqueue event"| SQS
  SQS -->|"4 · consume"| WORK
  S3 -->|"5 · read doc"| WORK
  WORK -->|"6 · embed chunks"| EMB1
  WORK -->|"7 · bulk index"| OS

  U -->|"a · question"| API
  API -->|"b · embed query"| EMB2
  API -->|"c · kNN, scoped by tenant"| OS
  API -->|"d · ground + write"| GEN
  API -.->|"e · cited answer"| U
  OS -.->|browse| DASH

  classDef actor fill:#e9edf3,stroke:#3b4a5a,color:#1a1f26;
  classDef ext fill:#efe7d6,stroke:#b8760f,color:#3a2c10;
  classDef write fill:#fbecd3,stroke:#b8760f,color:#5a3d0e;
  classDef read fill:#d5efef,stroke:#0c7d7d,color:#083f3f;
  classDef store fill:#d9e6f8,stroke:#2b5fb3,color:#132a4d;
  classDef model fill:#e6e2f6,stroke:#6b5dae,color:#2c2657;
  classDef infra fill:#eef1f5,stroke:#6b7a8d,color:#28313d;
```

## Enterprise-grade

The intent is to show the engineering that real, multi-tenant RAG systems
require — the same architecture and stack you'd run in production, on public
data and runnable on a laptop:

- **Event-driven ingestion** (queue → worker): bounded concurrency, back-pressure, graceful shutdown
- **Multi-tenant isolation**: retrieval scoped so one customer never sees another's documents
- **Data integrity**: every chunk embedded (no silent drops), malformed input rejected at the edge
- **Operability**: distributed tracing, context-window & throttling guardrails, load tests (p99, time-to-first-token)
- **Swappable model backends** (local ↔ cloud) behind clean interfaces
- **Engineering rigor**: unit tests, CI, and pre-commit gates

These pieces are added gradually — see the commit history and `docs/`.

## Where the documents come from

All source material is downloaded from **SEC EDGAR** — the U.S. Securities and
Exchange Commission's free, public database of company filings
(<https://www.sec.gov/edgar>). Nothing proprietary is used; SEC filings are
public domain.

Specifically, the seeder pulls **10-K filings** — a public company's **annual
report** (business overview, risk factors, financial statements). They're long,
messy, unstructured HTML documents, which makes them a realistic RAG input.

How they're fetched (no account or API key required):

1. Resolve a ticker (e.g. `AAPL`) to its **CIK**, the SEC's company id, via
   SEC's public `company_tickers.json`.
2. Look up the company's most recent **10-K** through SEC's submissions API.
3. Download that filing's primary document from the EDGAR archive.

SEC asks callers to identify themselves with a descriptive `User-Agent` (with
contact info); the seeder sends one, configurable via `EDGAR_RAG_SEC_USER_AGENT`.

## Status

🚧 Built gradually — see the commit history for progress.

## License

MIT — see [LICENSE](LICENSE).
