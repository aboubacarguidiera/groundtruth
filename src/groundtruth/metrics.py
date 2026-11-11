"""Métriques de retrieval.

Chaque question a une seule preuve attendue (un extrait). Les morceaux qui la
contiennent sont des copies de la même preuve, donc tout se calcule à partir du
rang du premier bon morceau (1 = premier résultat, None = absent des résultats).
"""

import math


def recall_at_k(rank: int | None, k: int) -> float:
    """1 si la preuve est dans les k premiers résultats, 0 sinon."""
    return 1.0 if rank is not None and rank <= k else 0.0


def reciprocal_rank(rank: int | None, k: int) -> float:
    """1/rang : 1 en première place, 0,5 en deuxième, 0 au-delà de k."""
    return 1.0 / rank if rank is not None and rank <= k else 0.0


def ndcg_at_k(rank: int | None, k: int) -> float:
    """Comme 1/rang, mais avec une pénalité logarithmique, plus douce pour les rangs lointains."""
    return 1.0 / math.log2(rank + 1) if rank is not None and rank <= k else 0.0


def first_relevant_rank(ranked: list[int], relevant: set[int]) -> int | None:
    for rank, index in enumerate(ranked, start=1):
        if index in relevant:
            return rank
    return None


def summarize(ranks: list[int | None], k: int) -> dict[str, float]:
    count = len(ranks)
    return {
        "n": count,
        "recall@5": sum(recall_at_k(rank, 5) for rank in ranks) / count,
        f"recall@{k}": sum(recall_at_k(rank, k) for rank in ranks) / count,
        f"mrr@{k}": sum(reciprocal_rank(rank, k) for rank in ranks) / count,
        f"ndcg@{k}": sum(ndcg_at_k(rank, k) for rank in ranks) / count,
    }
