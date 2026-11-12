"""Moteurs de recherche : chacun indexe des morceaux et renvoie les indices des k meilleurs."""

import math
import re
import unicodedata
from collections import Counter
from functools import cached_property

import numpy as np

from groundtruth.data import Chunk

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
ENGINE_NAMES = ("bm25", "dense", "hybrid")


def tokenize(text: str) -> list[str]:
    """Mots en minuscules, sans accents ni marque du pluriel ("Amendes" et "amende" se rejoignent)."""
    text = unicodedata.normalize("NFKD", text.lower().replace("œ", "oe"))
    tokens = re.findall(r"[a-z0-9]+", "".join(c for c in text if not unicodedata.combining(c)))
    return [token[:-1] if len(token) > 3 and token[-1] in "sx" else token for token in tokens]


def top_k(scores: np.ndarray, k: int) -> list[int]:
    return np.argsort(-scores, kind="stable")[:k].tolist()


class Models:
    """Charge chaque modèle une seule fois, à la première utilisation."""

    @cached_property
    def embedder(self):
        from sentence_transformers import SentenceTransformer  # import lent, fait à la demande

        return SentenceTransformer(EMBEDDING_MODEL, device="cpu")

    @cached_property
    def cross_encoder(self):
        from sentence_transformers import CrossEncoder

        return CrossEncoder(RERANKER_MODEL, device="cpu", max_length=512)


class BM25Retriever:
    """Recherche lexicale : compte les mots de la question présents dans chaque morceau."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        term_counts = [Counter(tokenize(chunk.text)) for chunk in chunks]
        lengths = np.array([sum(counts.values()) for counts in term_counts], dtype=float)
        self._size = len(chunks)
        self._k1 = k1
        self._length_norm = k1 * (1 - b + b * lengths / lengths.mean())
        postings: dict[str, tuple[list[int], list[int]]] = {}
        for index, counts in enumerate(term_counts):
            for term, frequency in counts.items():
                indices, frequencies = postings.setdefault(term, ([], []))
                indices.append(index)
                frequencies.append(frequency)
        self._postings = {
            term: (np.array(indices), np.array(frequencies, dtype=float))
            for term, (indices, frequencies) in postings.items()
        }

    def search(self, query: str, k: int) -> list[int]:
        scores = np.zeros(self._size)
        for term in set(tokenize(query)):
            if term not in self._postings:
                continue
            indices, frequencies = self._postings[term]
            idf = math.log(1 + (self._size - len(indices) + 0.5) / (len(indices) + 0.5))
            scores[indices] += (
                idf * frequencies * (self._k1 + 1) / (frequencies + self._length_norm[indices])
            )
        return top_k(scores, k)


class DenseRetriever:
    """Recherche sémantique : compare le sens de la question à celui de chaque morceau."""

    def __init__(self, chunks: list[Chunk], embedder) -> None:
        self._embedder = embedder
        # Les modèles E5 attendent les préfixes "passage: " et "query: "
        self._embeddings = embedder.encode(
            [f"passage: {chunk.text[:200]}" for chunk in chunks],
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )

    def search(self, query: str, k: int) -> list[int]:
        query_embedding = self._embedder.encode(f"query: {query}", normalize_embeddings=True)
        return top_k(self._embeddings @ query_embedding, k)


class HybridRetriever:
    """Fusionne deux classements par rangs réciproques (RRF) : un morceau bien classé des deux côtés gagne."""

    def __init__(self, first, second, depth: int = 50, rrf_k: int = 60) -> None:
        self._engines = (first, second)
        self._depth = depth
        self._rrf_k = rrf_k

    def search(self, query: str, k: int) -> list[int]:
        scores: dict[int, float] = {}
        for engine in self._engines:
            for rank, index in enumerate(engine.search(query, self._depth), start=1):
                scores[index] = scores.get(index, 0.0) + 1.0 / (self._rrf_k + rank)
        return sorted(scores, key=lambda index: (-scores[index], index))[:k]


class RerankedRetriever:
    """Reclasse les premiers candidats d'un moteur avec un modèle qui lit question et morceau ensemble."""

    def __init__(self, base, chunks: list[Chunk], cross_encoder, depth: int = 30) -> None:
        self._base = base
        self._chunks = chunks
        self._cross_encoder = cross_encoder
        self._depth = depth

    def search(self, query: str, k: int) -> list[int]:
        candidates = self._base.search(query, self._depth)
        scores = self._cross_encoder.predict(
            [(query, self._chunks[index].text) for index in candidates], show_progress_bar=False
        )
        return [candidates[position] for position in top_k(np.asarray(scores), k)]


def build_engine(
    name: str, chunks: list[Chunk], models: Models, rerank: bool = False, cache: dict | None = None
):
    """Construit un moteur. `cache` permet de réutiliser les index BM25 et dense d'un appel à l'autre."""
    cache = {} if cache is None else cache

    def base(kind: str):
        if kind not in cache:
            cache[kind] = (
                BM25Retriever(chunks) if kind == "bm25" else DenseRetriever(chunks, models.embedder)
            )
        return cache[kind]

    engine = HybridRetriever(base("bm25"), base("dense")) if name == "hybrid" else base(name)
    return RerankedRetriever(engine, chunks, models.cross_encoder) if rerank else engine
