"""Compare toutes les configurations : taille des morceaux x moteur x reclassement.

Usage : python -m groundtruth.compare
Écrit : results/comparison.json et results/comparison.md
"""

import json
import sys

from groundtruth.data import GOLDEN_FILE, ROOT, chunk_units, load_golden, load_units
from groundtruth.evaluate import DEFAULT_K, RESULTS_DIR, default_overlap, score
from groundtruth.retrievers import ENGINE_NAMES, Models, build_engine

CHUNK_SIZES = (100, 200, 400)


def run_grid(k: int = DEFAULT_K) -> list[dict]:
    units = load_units()
    questions = load_golden()
    models = Models()
    rows = []
    for chunk_size in CHUNK_SIZES:
        overlap = default_overlap(chunk_size)
        chunks = chunk_units(units, chunk_size, overlap)
        cache: dict = {}  # index BM25 et dense partagés par les moteurs de cette taille
        for name in ENGINE_NAMES:
            for rerank in (False, True):
                engine = build_engine(name, chunks, models, rerank=rerank, cache=cache)
                result = score(engine, questions, units, chunks, k)
                rows.append(
                    {
                        "config": {
                            "retriever": name,
                            "rerank": rerank,
                            "chunk_size": chunk_size,
                            "overlap": overlap,
                            "k": k,
                            "golden": GOLDEN_FILE.name,
                        },
                        "chunks": len(chunks),
                        **result,
                    }
                )
                overall = result["overall"]
                print(
                    f"{chunk_size:>4} mots | {name:<6} | reclassement {'oui' if rerank else 'non'} | "
                    f"recall@{k} {overall[f'recall@{k}']:.2f} | mrr@{k} {overall[f'mrr@{k}']:.2f} | "
                    f"{result['ms_per_query']} ms",
                    flush=True,
                )
    return rows


def to_markdown(rows: list[dict], k: int = DEFAULT_K) -> str:
    lines = [
        f"| Morceaux (mots) | Moteur | Reclassement | recall@5 | recall@{k} | MRR@{k} | nDCG@{k} | ms/question |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        config, overall = row["config"], row["overall"]
        lines.append(
            f"| {config['chunk_size']} | {config['retriever']} | {'oui' if config['rerank'] else 'non'} | "
            f"{overall['recall@5']:.2f} | {overall[f'recall@{k}']:.2f} | {overall[f'mrr@{k}']:.2f} | "
            f"{overall[f'ndcg@{k}']:.2f} | {row['ms_per_query']:.0f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    rows = run_grid()
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "comparison.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (RESULTS_DIR / "comparison.md").write_text(to_markdown(rows), encoding="utf-8")
    print(f"\nRésultats enregistrés dans {RESULTS_DIR.relative_to(ROOT)}/comparison.json et comparison.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
