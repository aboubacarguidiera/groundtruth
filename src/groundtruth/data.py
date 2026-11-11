"""Chargement du corpus et du jeu de référence, découpage en morceaux, jugement de pertinence."""

import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UNITS_FILE = ROOT / "data" / "corpus" / "units.jsonl"
GOLDEN_FILE = ROOT / "data" / "golden" / "golden_v2.toml"

# Part minimale de l'extrait qu'un morceau doit contenir pour compter comme bonne réponse
MIN_EXCERPT_COVERAGE = 0.5


@dataclass(frozen=True)
class Chunk:
    unit_id: str
    start: int  # position dans le texte normalisé de l'unité
    end: int
    text: str


@dataclass(frozen=True)
class Question:
    id: str
    category: str
    question: str
    # Passages acceptés comme bonne réponse : (identifiant de l'unité, extrait exact)
    evidence: tuple[tuple[str, str], ...]


def normalize(text: str) -> str:
    """Uniformise espaces et apostrophes pour que la saisie humaine corresponde au texte."""
    text = text.replace(" ", " ").replace("’", "'")
    return re.sub(r"\s+", " ", text).strip()


def load_units(path: Path = UNITS_FILE) -> dict[str, str]:
    units = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            units[record["id"]] = normalize(record["text"])
    return units


def load_golden(path: Path = GOLDEN_FILE) -> list[Question]:
    with path.open("rb") as handle:
        items = tomllib.load(handle)["questions"]
    return [
        Question(
            id=item["id"],
            category=item["category"],
            question=item["question"],
            # le passage principal, puis les passages équivalents listés sous "also"
            evidence=tuple(
                (f"{passage['doc']}:{passage['ref']}", normalize(passage["excerpt"]))
                for passage in [item, *item.get("also", [])]
            ),
        )
        for item in items
    ]


def chunk_units(units: dict[str, str], size: int, overlap: int) -> list[Chunk]:
    """Découpe chaque unité en fenêtres de `size` mots qui se recouvrent de `overlap` mots."""
    if not 0 <= overlap < size:
        raise ValueError("il faut 0 <= overlap < size")
    chunks = []
    for unit_id, text in units.items():
        words = [match.span() for match in re.finditer(r"\S+", text)]
        first = 0
        while first < len(words):
            last = min(first + size, len(words))
            start, end = words[first][0], words[last - 1][1]
            chunks.append(Chunk(unit_id, start, end, text[start:end]))
            if last == len(words):
                break
            first += size - overlap
    return chunks


def relevant_chunks(question: Question, units: dict[str, str], chunks: list[Chunk]) -> set[int]:
    """Indices des morceaux qui contiennent au moins la moitié d'un des extraits acceptés.

    Le jugement se fait sur les positions dans le texte, donc il reste valable
    quelle que soit la taille des morceaux.
    """
    relevant = set()
    for unit_id, excerpt in question.evidence:
        excerpt_start = units[unit_id].find(excerpt)
        if excerpt_start < 0:
            raise ValueError(f"{question.id}: extrait introuvable dans {unit_id}")
        excerpt_end = excerpt_start + len(excerpt)
        needed = MIN_EXCERPT_COVERAGE * len(excerpt)
        relevant.update(
            index
            for index, chunk in enumerate(chunks)
            if chunk.unit_id == unit_id
            and min(chunk.end, excerpt_end) - max(chunk.start, excerpt_start) >= needed
        )
    return relevant
