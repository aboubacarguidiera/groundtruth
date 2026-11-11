"""Vérifie le jeu de référence : chaque extrait doit se trouver mot pour mot dans l'unité citée.

Usage : python -I scripts/check_golden.py [chemin/vers/golden.toml]
"""

import json
import re
import sys
import tomllib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNITS_FILE = ROOT / "data" / "corpus" / "units.jsonl"
DEFAULT_GOLDEN = ROOT / "data" / "golden" / "golden_v2.toml"
REQUIRED = ("id", "category", "question", "doc", "ref", "excerpt")
PASSAGE_FIELDS = ("doc", "ref", "excerpt")
TARGET = 100


def normalize(text: str) -> str:
    """Ignore les différences d'espaces et d'apostrophes (' et ’) entre le texte et la saisie."""
    text = text.replace(" ", " ").replace("’", "'")
    return re.sub(r"\s+", " ", text).strip()


def check(questions: list[dict], units: dict[str, str]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for position, item in enumerate(questions, start=1):
        label = item.get("id") or f"bloc n°{position}"
        missing = [field for field in REQUIRED if not str(item.get(field, "")).strip()]
        if missing:
            errors.append(f"{label}: champs manquants ou vides : {', '.join(missing)}")
            continue
        if item["id"] in seen:
            errors.append(f"{label}: id utilisé deux fois")
        seen.add(item["id"])
        # le passage principal, puis les passages équivalents listés sous [[questions.also]]
        for passage in [item, *item.get("also", [])]:
            if any(not str(passage.get(field, "")).strip() for field in PASSAGE_FIELDS):
                errors.append(f"{label}: un passage 'also' doit avoir doc, ref et excerpt")
                continue
            unit_id = f"{passage['doc']}:{passage['ref']}"
            if unit_id not in units:
                errors.append(f"{label}: {unit_id} n'existe pas dans le corpus")
            elif normalize(passage["excerpt"]) not in units[unit_id]:
                errors.append(f"{label}: extrait introuvable dans {unit_id} (copie-le tel quel)")
    return errors


def main() -> int:
    golden_file = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GOLDEN
    if not UNITS_FILE.exists():
        print("Corpus absent : lance d'abord python -I scripts/build_corpus.py")
        return 1
    units = {}
    with UNITS_FILE.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            units[record["id"]] = normalize(record["text"])
    with golden_file.open("rb") as handle:
        questions = tomllib.load(handle).get("questions", [])

    errors = check(questions, units)
    for error in errors:
        print(f"ERREUR  {error}")
    valid = len(questions) - len({error.split(":")[0] for error in errors})
    extra = sum(len(question.get("also", [])) for question in questions)
    print(f"{valid}/{TARGET} questions valides ({len(questions)} écrites, {extra} passages équivalents)")
    for category, count in sorted(Counter(q.get("category", "?") for q in questions).items()):
        print(f"  {category}: {count}")
    for doc, count in sorted(Counter(q.get("doc", "?") for q in questions).items()):
        print(f"  [{doc}] {count}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
