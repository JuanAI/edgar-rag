"""The labeled evaluation dataset: a small corpus plus graded relevance labels.

Self-labeled and reproducible: we control the corpus, so each query points at the
exact (document_id, page_number) that answers it, with a relevance grade. No
domain expert or LLM judge needed to get started — the labels are ground truth by
construction. Relevance is at page granularity (a page yields many chunks; any
chunk from a relevant page counts), matching how production IR qrels are built.

  grade 3 = the passage that directly answers the question
  grade 1-2 = related/partial (optional)
  grade 0 = everything unlabeled
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_PATH = Path(__file__).parent / "data" / "eval_dataset.json"


@dataclass(frozen=True)
class EvalPage:
    page_number: int
    text: str


@dataclass(frozen=True)
class EvalDocument:
    document_id: str
    doc_type: str
    pages: list[EvalPage]


@dataclass(frozen=True)
class Judgement:
    """One relevance label: this (document, page) is relevant to a query."""

    document_id: str
    page_number: int
    grade: int

    @property
    def key(self) -> tuple[str, int]:
        return (self.document_id, self.page_number)


@dataclass(frozen=True)
class EvalQuery:
    question: str
    relevant: list[Judgement]


@dataclass(frozen=True)
class EvalDataset:
    tenant: str
    documents: list[EvalDocument]
    queries: list[EvalQuery]


def load_dataset(path: Path | None = None) -> EvalDataset:
    """Load the eval dataset from JSON (defaults to the bundled dataset)."""
    raw = json.loads((path or _DEFAULT_PATH).read_text())
    documents = [
        EvalDocument(
            document_id=doc["document_id"],
            doc_type=doc["doc_type"],
            pages=[EvalPage(page_number=p["page_number"], text=p["text"]) for p in doc["pages"]],
        )
        for doc in raw["documents"]
    ]
    queries = [
        EvalQuery(
            question=q["question"],
            relevant=[
                Judgement(
                    document_id=j["document_id"],
                    page_number=j["page_number"],
                    grade=j["grade"],
                )
                for j in q["relevant"]
            ],
        )
        for q in raw["queries"]
    ]
    return EvalDataset(tenant=raw["tenant"], documents=documents, queries=queries)
