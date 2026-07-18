# edgar-rag

> A runnable, event-driven **RAG (Retrieval-Augmented Generation)** pipeline over
> public **SEC EDGAR** filings. Ask questions about a company's annual report and
> get answers grounded in the source document, with citations.

This is a portfolio project I'm building in the open. The goal is a system anyone
can clone and run end-to-end on a laptop with **one command and no cloud
credentials** — while showing how a RAG pipeline is actually engineered for
production, not just wired together from a tutorial.

## Why this project

The interesting, under-appreciated half of RAG is the **ingestion / write path**:
reliably getting messy documents *in* — parsed, chunked, embedded, and searchable —
without silently losing data. Most demos hand-wave it. This project treats it as
the main event: queue-driven, concurrent, observable, and defended against the
subtle data bugs that make a RAG system quietly return wrong answers.

The domain is **SEC EDGAR filings** (public company annual/quarterly reports):
real, genuinely messy, financial documents that are free and public — a good
stand-in for any "unstructured documents → trustworthy, cited answers" problem,
with nothing proprietary involved.

## Goals

- **Runnable by anyone:** `git clone` → one command → a working pipeline. No API
  keys, no AWS account required for the default path.
- **Production-shaped ingestion:** event-driven, concurrent, with back-pressure,
  graceful shutdown, and data-integrity guarantees.
- **Grounded answers:** retrieval scoped per document owner and document type;
  answers cite the filings they came from.
- **Swappable everything:** embeddings and LLM behind clean interfaces, so "local
  model on my laptop" and "managed cloud service" are a config change, not a
  rewrite.
- **Operable:** tracing, and load tests that measure the metrics a chat UX
  actually cares about.

## Non-goals (for now)

- Not a hosted product or a polished UI — it's a backend/pipeline showcase.
- Not tuned for scale or cost; correctness and clarity come first.
- Not a document-quality or answer-quality research project (though an evaluation
  harness is a stretch goal).

## Architecture (target)

```
WRITE PATH:  seed EDGAR filings -> S3 + SQS -> ingestion worker
             -> parse -> chunk -> embed -> vector store

READ PATH:   question -> embed -> scoped vector search -> guardrail
             -> generate -> cited answer
```

Local infrastructure (queue, object store, vector store) runs in Docker so the
architecture stays honest — the code talks to real service APIs, just pointed at
local emulators.

---

## Build roadmap

I'm building this incrementally over roughly three weeks of part-time work. Each
milestone is a coherent, committable slice. Boxes get ticked as they land.

### Week 1 — Foundations & the ingestion write path
- [ ] Project scaffolding, configuration, and domain models
- [ ] Local infrastructure via Docker (vector store + queue + object store)
- [ ] Document acquisition: fetch real public filings and stage them for ingestion
- [ ] Parsing: raw filings → clean, sectioned text
- [ ] Chunking: sections → overlapping passages ready to embed
- [ ] Embeddings: a local, credential-free default provider
- [ ] Event-driven ingestion worker: consume events and index documents end-to-end

### Week 2 — Retrieval, generation & robustness
- [ ] Vector index design and nearest-neighbour search
- [ ] Query service: retrieve → assemble context → generate → return citations
- [ ] Provider abstraction: pluggable embedding and generation backends
- [ ] Retrieval scoping: isolate by document owner and filter by document type
- [ ] Concurrency controls and graceful shutdown in the ingestion worker
- [ ] Guardrails: context-size limits and backend rate-limit handling
- [ ] Data-integrity hardening: complete embeddings + strict message validation

### Week 3 — Observability, performance & polish
- [ ] Distributed tracing, with per-owner attribution
- [ ] Load testing: tail latency and time-to-first-token for the chat path
- [ ] Test suite and continuous integration
- [ ] Documentation: architecture overview and design-decision write-ups
- [ ] Developer experience: one-command run, task shortcuts, sensible defaults
- [ ] Stretch: answer-quality evaluation, result re-ranking, and a metrics dashboard

---

## Tech (planned)

Python · asyncio · a vector store with k-nearest-neighbour search · a managed
queue + object store (emulated locally) · local embedding models · distributed
tracing · load testing · Docker. Optional cloud LLM/embedding backends behind the
same interfaces.

## Status

🚧 Early / in progress — following the roadmap above. Watch the commit history and
the ticked boxes to see it come together.

## License

MIT — see [LICENSE](LICENSE).
