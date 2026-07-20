"""Retrieval evaluation: labeled dataset, IR metrics, and a run harness."""

from __future__ import annotations

from .dataset import EvalDataset, load_dataset
from .harness import EvalReport, QueryScore, run_eval
from .metrics import ndcg_at_k, recall_at_k, reciprocal_rank

__all__ = [
    "EvalDataset",
    "EvalReport",
    "QueryScore",
    "load_dataset",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "run_eval",
]
