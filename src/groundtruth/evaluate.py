"""Évalue une configuration de retrieval sur le jeu de référence.

Usage : python -m groundtruth.evaluate --retriever hybrid --chunk-size 200 --rerank
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from groundtruth.data import (
    GOLDEN_FILE,
    ROOT,
    Chunk,
    Question,
    chunk_units,
    load_golden,
    load_units,
    relevant_chunks,
)
from groundtruth.metrics import first_relevant_rank, summarize
from groundtruth.retrievers import ENGINE_NAMES, Models, build_engine

RESULTS_DIR = ROOT / "results"
DEFAULT_K = 10


def default_overlap(chunk_size: int) -> int:
    return chunk_size // 5


def score(engine, questions: list[Question], units: dict[str, str], chunks: list[Chunk], k: int) -> dict:
    """Lance toutes les questions sur un moteur et résume les rangs obtenus."""
    per_question = []
    started = time.perf_counter()
    for question in questions:
        ranked = engine.search(question.question, k)
        rank = first_relevant_rank(ranked, relevant_chunks(question, units, chunks))
        per_question.append({"id": question.id, "category": question.category, "rank": rank})
    elapsed = time.perf_counter() - started

    by_category = defaultdict(list)
    for row in per_question:
        by_category[row["category"]].append(row["rank"])
    return {
        "overall": summarize([row["rank"] for row in per_question], k),
        "by_category": {name: summarize(ranks, k) for name, ranks in sorted(by_category.items())},
        "ms_per_query": round(1000 * elapsed / len(questions), 1),
        "questions": per_question,
    }


def format_report(result: dict, questions_text: dict[str, str]) -> str:
    config = result["config"]
    k = config["k"]
    columns = ["recall@5", f"recall@{k}", f"mrr@{k}", f"ndcg@{k}"]
    lines = [
        f"{config['retriever']}{' + reclassement' if config['rerank'] else ''} | "
        f"morceaux de {config['chunk_size']} mots, recouvrement {config['overlap']} | "
        f"{result['chunks']} morceaux | {result['ms_per_query']} ms par question",
        "",
        f"{'catégorie':<12}{'n':>4}" + "".join(f"{column:>11}" for column in columns),
    ]
    rows = list(result["by_category"].items()) + [("TOTAL", result["overall"])]
    for name, scores in rows:
        lines.append(
            f"{name:<12}{scores['n']:>4}" + "".join(f"{scores[column]:>11.2f}" for column in columns)
        )
    missed = [row for row in result["questions"] if row["rank"] is None]
    if missed:
        lines += ["", f"Preuve absente des {k} premiers résultats ({len(missed)}) :"]
        lines += [f"  {row['id']} [{row['category']}] {questions_text[row['id']]}" for row in missed]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--retriever", choices=ENGINE_NAMES, default="bm25")
    parser.add_argument("--chunk-size", type=int, default=200, help="taille des morceaux, en mots")
    parser.add_argument("--overlap", type=int, help="recouvrement en mots (défaut : un cinquième)")
    parser.add_argument("--rerank", action="store_true", help="reclasser les 30 premiers candidats")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--golden", type=Path, default=GOLDEN_FILE)
    args = parser.parse_args()
    overlap = default_overlap(args.chunk_size) if args.overlap is None else args.overlap

    units = load_units()
    questions = load_golden(args.golden)
    chunks = chunk_units(units, args.chunk_size, overlap)
    engine = build_engine(args.retriever, chunks, Models(), rerank=args.rerank)
    result = {
        "config": {
            "retriever": args.retriever,
            "rerank": args.rerank,
            "chunk_size": args.chunk_size,
            "overlap": overlap,
            "k": args.k,
            "golden": args.golden.name,
        },
        "chunks": len(chunks),
        **score(engine, questions, units, chunks, args.k),
    }
    print(format_report(result, {question.id: question.question for question in questions}))

    RESULTS_DIR.mkdir(exist_ok=True)
    suffix = "_rerank" if args.rerank else ""
    out_file = RESULTS_DIR / f"{args.retriever}_c{args.chunk_size}_o{overlap}{suffix}.json"
    out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nRésultats enregistrés dans {out_file.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
