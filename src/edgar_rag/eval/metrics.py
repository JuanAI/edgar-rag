"""Information-retrieval metrics for evaluating ranked search results.

All three are pure functions over a *ranked* list (best match first) plus the
ground-truth relevance labels, so they are deterministic and unit-testable with
no model or cluster. The eval harness feeds real retrieval output into them.

  * recall@k       — of all the relevant items, how many did we retrieve in top-k?
  * nDCG@k         — did the more-relevant items land near the top? (graded, 0-1)
  * reciprocal_rank — how soon did the first relevant item appear? (1/rank)

The mean of reciprocal_rank across queries is the familiar MRR.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence


def recall_at_k(ranked_ids: Sequence[Hashable], relevant_ids: set[Hashable], k: int) -> float:
    """Fraction of the relevant items that appear in the top-k results.

    Returns 1.0 vacuously when there are no relevant items (nothing to miss).
    """
    if not relevant_ids:
        return 1.0
    top = set(ranked_ids[:k])
    return len(top & relevant_ids) / len(relevant_ids)


def dcg(gains: Sequence[float], k: int) -> float:
    """Discounted Cumulative Gain: sum of relevance grades, each discounted by
    its position so items further down count for less (rel / log2(rank+1))."""
    gains = gains[:k]
    if not gains:
        return 0.0
    total = float(gains[0])
    for i in range(1, len(gains)):
        total += gains[i] / math.log2(i + 1)
    return total


def ndcg_at_k(ranked_gains: Sequence[float], ideal_gains: Sequence[float], k: int) -> float:
    """Normalized DCG: the ranking's DCG divided by the best possible DCG.

    `ranked_gains` are the relevance grades of the retrieved items in retrieved
    order; `ideal_gains` are all available grades for this query (used to build
    the perfect ranking). Returns 0.0 when no relevant items exist (IDCG = 0).
    """
    idcg = dcg(sorted(ideal_gains, reverse=True), k)
    if idcg == 0.0:
        return 0.0
    return dcg(ranked_gains, k) / idcg


def reciprocal_rank(ranked_relevance: Sequence[float]) -> float:
    """1 / (rank of the first relevant item), or 0.0 if none are relevant.

    Relevant means grade > 0. First position -> 1.0, second -> 0.5, and so on.
    """
    for i, grade in enumerate(ranked_relevance):
        if grade > 0:
            return 1.0 / (i + 1)
    return 0.0
