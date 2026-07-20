"""Run the retrieval evaluation and print a report.

Requires a running OpenSearch (`make up`) and the `local` extra (real MiniLM).
Usage: python scripts/eval.py [k]
"""

from __future__ import annotations

import sys

from edgar_rag.config import get_settings
from edgar_rag.eval import run_eval


def main() -> int:
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    report = run_eval(get_settings(), k=k)

    print(f"\nRetrieval evaluation — {report.num_queries} queries, k={report.k}")
    print("=" * 64)
    print(f"  recall@{report.k}: {report.mean_recall_at_k:.3f}")
    print(f"  nDCG@{report.k}:   {report.mean_ndcg_at_k:.3f}")
    print(f"  MRR:       {report.mrr:.3f}")
    print("-" * 64)
    for s in report.per_query:
        print(
            f"  rr={s.reciprocal_rank:.2f}  ndcg={s.ndcg_at_k:.2f}  "
            f"recall={s.recall_at_k:.2f}  {s.question}"
        )
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
