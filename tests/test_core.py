import math

import pytest

from groundtruth.data import Question, chunk_units, relevant_chunks
from groundtruth.metrics import first_relevant_rank, ndcg_at_k, recall_at_k, reciprocal_rank, summarize
from groundtruth.regression import find_regressions
from groundtruth.retrievers import BM25Retriever, HybridRetriever, tokenize

UNITS = {"doc:art_1": " ".join(f"mot{i}" for i in range(10))}


def test_metrics_reward_early_ranks():
    assert recall_at_k(3, 5) == 1.0
    assert recall_at_k(6, 5) == 0.0
    assert recall_at_k(None, 5) == 0.0
    assert reciprocal_rank(4, 10) == 0.25
    assert reciprocal_rank(11, 10) == 0.0
    assert ndcg_at_k(1, 10) == 1.0
    assert ndcg_at_k(3, 10) == pytest.approx(1 / math.log2(4))


def test_first_relevant_rank():
    assert first_relevant_rank([7, 2, 9], {9, 2}) == 2
    assert first_relevant_rank([7, 8], {9}) is None


def test_summarize_averages_over_questions():
    scores = summarize([1, 2, None, 8], k=10)
    assert scores["n"] == 4
    assert scores["recall@5"] == 0.5
    assert scores["recall@10"] == 0.75
    assert scores["mrr@10"] == pytest.approx((1 + 0.5 + 0 + 0.125) / 4)


def test_chunks_overlap_and_cover_the_whole_unit():
    chunks = chunk_units(UNITS, size=4, overlap=1)
    assert [chunk.text.split()[0] for chunk in chunks] == ["mot0", "mot3", "mot6"]
    assert chunks[-1].text.endswith("mot9")
    text = UNITS["doc:art_1"]
    assert all(text[chunk.start : chunk.end] == chunk.text for chunk in chunks)


def test_chunk_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_units(UNITS, size=4, overlap=4)


def test_relevance_follows_the_excerpt_across_chunk_sizes():
    question = Question("q", "cat", "?", (("doc:art_1", "mot4 mot5"),))
    for size in (3, 4, 10):
        chunks = chunk_units(UNITS, size=size, overlap=0)
        relevant = relevant_chunks(question, UNITS, chunks)
        assert relevant, f"aucun morceau pertinent pour size={size}"
        assert all("mot4" in chunks[index].text or "mot5" in chunks[index].text for index in relevant)


def test_relevance_rejects_an_excerpt_missing_from_the_unit():
    question = Question("q", "cat", "?", (("doc:art_1", "absent"),))
    with pytest.raises(ValueError):
        relevant_chunks(question, UNITS, chunk_units(UNITS, size=4, overlap=0))


def test_any_accepted_passage_counts_as_relevant():
    units = {**UNITS, "doc:rct_1": "un considérant qui dit la même chose autrement"}
    question = Question("q", "cat", "?", (("doc:art_1", "mot4 mot5"), ("doc:rct_1", "la même chose")))
    chunks = chunk_units(units, size=50, overlap=0)
    assert {chunks[index].unit_id for index in relevant_chunks(question, units, chunks)} == {
        "doc:art_1",
        "doc:rct_1",
    }


def test_bm25_ranks_the_matching_chunk_first():
    units = {
        "a": "le droit à l'effacement des données",
        "b": "les amendes administratives et les sanctions",
        "c": "la désignation du délégué à la protection",
    }
    chunks = chunk_units(units, size=50, overlap=0)
    assert BM25Retriever(chunks).search("quelles amendes ?", k=1) == [1]
    assert BM25Retriever(chunks).search("une amende ?", k=1) == [1]  # le singulier trouve le pluriel


def test_tokenize_folds_accents_and_plurals():
    assert tokenize("Les Délégués, l'œuvre et 3 %") == ["les", "delegue", "l", "oeuvre", "et", "3"]


class FixedRanking:
    def __init__(self, ranking):
        self._ranking = ranking

    def search(self, query, k):
        return self._ranking[:k]


def test_hybrid_prefers_chunks_ranked_well_by_both_engines():
    hybrid = HybridRetriever(FixedRanking([5, 1, 2]), FixedRanking([9, 1, 5]))
    assert hybrid.search("?", k=2) == [5, 1]


def test_regression_tolerates_one_point_but_not_two():
    baseline = {"bm25": 0.27, "dense": 0.49}
    assert find_regressions(baseline, {"bm25": 0.26, "dense": 0.55}) == []
    problems = find_regressions(baseline, {"bm25": 0.27, "dense": 0.47})
    assert len(problems) == 1 and problems[0].startswith("dense")


def test_regression_flags_a_missing_score():
    assert find_regressions({"bm25": 0.27}, {}) == ["bm25 : score absent"]
