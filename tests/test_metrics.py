"""Unit tests for the IR metrics — deterministic, no model or cluster."""

import math

from edgar_rag.eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank


def test_recall_counts_relevant_items_in_top_k():
    ranked = ["a", "b", "c", "d"]
    relevant = {"a", "d", "z"}  # z is never retrieved
    # top-3 contains a (1 of 3 relevant); d and z are outside k.
    assert recall_at_k(ranked, relevant, k=3) == 1 / 3
    # top-4 now includes d -> 2 of 3.
    assert recall_at_k(ranked, relevant, k=4) == 2 / 3


def test_recall_with_no_relevant_items_is_one():
    assert recall_at_k(["a", "b"], set(), k=5) == 1.0


def test_ndcg_is_one_for_a_perfect_ranking():
    gains = [3.0, 2.0, 1.0]
    assert ndcg_at_k(gains, gains, k=3) == 1.0


def test_ndcg_penalizes_a_reversed_ranking():
    ideal = [3.0, 2.0, 1.0]
    reversed_ranking = [1.0, 2.0, 3.0]
    score = ndcg_at_k(reversed_ranking, ideal, k=3)
    assert 0.0 < score < 1.0


def test_ndcg_is_zero_when_nothing_is_relevant():
    assert ndcg_at_k([0.0, 0.0], [], k=2) == 0.0


def test_ndcg_rewards_relevant_items_higher_up():
    ideal = [3.0, 0.0, 0.0, 0.0]
    top = ndcg_at_k([3.0, 0.0, 0.0], ideal, k=3)  # relevant first
    low = ndcg_at_k([0.0, 0.0, 3.0], ideal, k=3)  # relevant last
    assert top > low


def test_reciprocal_rank_is_inverse_of_first_relevant_position():
    assert reciprocal_rank([0.0, 0.0, 3.0]) == 1 / 3
    assert reciprocal_rank([2.0, 0.0]) == 1.0
    assert math.isclose(reciprocal_rank([0.0, 1.0]), 0.5)


def test_reciprocal_rank_is_zero_when_nothing_is_relevant():
    assert reciprocal_rank([0.0, 0.0, 0.0]) == 0.0
