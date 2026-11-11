"""Vérifie que le recall@10 n'a pas baissé par rapport à la référence enregistrée.

Usage : python -m groundtruth.regression            # code de sortie 1 en cas de baisse
        python -m groundtruth.regression --update   # enregistre les scores actuels comme référence
"""

import argparse
import json
import sys

from groundtruth.data import ROOT, chunk_units, load_golden, load_units
from groundtruth.evaluate import DEFAULT_K, RESULTS_DIR, default_overlap, score
from groundtruth.retrievers import Models, build_engine

BASELINE_FILE = RESULTS_DIR / "baseline.json"
CHUNK_SIZE = 200
# Sans reclassement : la vérification tourne en une à deux minutes sur CPU
RETRIEVERS = ("bm25", "dense")
# Baisse tolérée du recall@10. Avec 100 questions, une question vaut un point.
TOLERANCE = 0.01


def find_regressions(
    baseline: dict[str, float], current: dict[str, float], tolerance: float = TOLERANCE
) -> list[str]:
    """Liste les moteurs dont le score a baissé de plus que la tolérance."""
    problems = []
    for name, reference in baseline.items():
        if name not in current:
            problems.append(f"{name} : score absent")
        elif reference - current[name] > tolerance + 1e-9:
            problems.append(
                f"{name} : recall@{DEFAULT_K} {current[name]:.2f}, référence {reference:.2f} "
                f"(baisse de {100 * (reference - current[name]):.0f} points)"
            )
    return problems


def measure() -> dict[str, float]:
    units = load_units()
    questions = load_golden()
    chunks = chunk_units(units, CHUNK_SIZE, default_overlap(CHUNK_SIZE))
    models = Models()
    scores = {}
    for name in RETRIEVERS:
        result = score(build_engine(name, chunks, models), questions, units, chunks, DEFAULT_K)
        scores[name] = result["overall"][f"recall@{DEFAULT_K}"]
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--update", action="store_true", help="enregistrer les scores actuels comme référence")
    args = parser.parse_args()

    current = measure()
    for name, value in current.items():
        print(f"{name:<6} recall@{DEFAULT_K} {value:.2f}")

    if args.update:
        RESULTS_DIR.mkdir(exist_ok=True)
        BASELINE_FILE.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        print(f"Référence enregistrée dans {BASELINE_FILE.relative_to(ROOT)}")
        return 0

    baseline = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    problems = find_regressions(baseline, current)
    if problems:
        print("\nRÉGRESSION :")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print(f"\nPas de régression (tolérance : {100 * TOLERANCE:.0f} point).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
