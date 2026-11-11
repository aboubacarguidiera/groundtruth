"""Découpe les règlements EUR-Lex (HTML) en unités : un article ou un considérant par ligne.

Usage : python -I scripts/build_corpus.py
Lit    : data/raw/<doc>.fr.html
Écrit  : data/corpus/units.jsonl
"""

import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_FILE = ROOT / "data" / "corpus" / "units.jsonl"

UNIT_ID = re.compile(r"^(art|rct)_(\d+)$")
KINDS = {"art": "article", "rct": "considerant"}
# Marqueurs d'énumération isolés dans leur propre paragraphe : "a)", "(12)", "ii)", "1.", "—"
MARKER = re.compile(r"^(\(?\d+\)|\(?[a-z]{1,2}\)|\(?[ivxlc]+\)|\d+\.|[—–-])$")
RECITAL_PREFIX = re.compile(r"^\(\d+\)\s*")


def collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace(" ", " ")).strip()


def merge_markers(paragraphs: list[str]) -> list[str]:
    """Recolle 'a)' avec le paragraphe qui le suit (ils sont dans deux cellules de tableau)."""
    merged: list[str] = []
    pending = ""
    for paragraph in paragraphs:
        if MARKER.match(paragraph):
            pending = f"{pending} {paragraph}".strip()
            continue
        merged.append(f"{pending} {paragraph}".strip())
        pending = ""
    if pending:
        merged.append(pending)
    return merged


class UnitParser(HTMLParser):
    """Extrait les blocs <div class="eli-subdivision" id="art_N|rct_N">."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.units: list[dict] = []
        self._roles: list[str | None] = []
        self._unit: dict | None = None
        self._buffer: list[str] | None = None
        self._p_classes: set[str] = set()
        self._skip_depth = 0
        self._superscript = False

    def handle_startendtag(self, tag, attrs):
        pass  # balises auto-fermantes (<col/>, <br/>) : rien à ouvrir ni fermer

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        role = None
        if self._unit is None:
            match = UNIT_ID.match(attributes.get("id") or "")
            if tag == "div" and "eli-subdivision" in classes and match:
                self._unit = {
                    "kind": KINDS[match.group(1)],
                    "num": int(match.group(2)),
                    "ref": match.group(0),
                    "title": "",
                    "paragraphs": [],
                }
                role = "unit"
        elif "oj-note-tag" in classes:
            self._skip_depth += 1  # appels de note de bas de page
            role = "skip"
        elif "oj-super" in classes and "".join(self._buffer or [])[-1:].isdigit():
            self._superscript = True  # exposant : 10<sup>25</sup> doit donner 10^25, pas 1025
        elif tag == "p":
            self._buffer = []
            self._p_classes = classes
            role = "p"
        self._roles.append(role)

    def handle_data(self, data):
        if self._buffer is not None and not self._skip_depth:
            if self._superscript and data.strip().isdigit():
                data = f"^{data}"  # seulement les exposants numériques, pas "1er"
            self._buffer.append(data)
        self._superscript = False

    def handle_endtag(self, tag):
        if not self._roles:
            return
        role = self._roles.pop()
        if role == "skip":
            self._skip_depth -= 1
        elif role == "p":
            text = collapse("".join(self._buffer or []))
            self._buffer = None
            if "oj-ti-art" in self._p_classes:
                pass  # "Article 6" : déjà dans num
            elif "oj-sti-art" in self._p_classes:
                self._unit["title"] = text
            elif text:
                self._unit["paragraphs"].append(text)
        elif role == "unit":
            self.units.append(self._unit)
            self._unit = None


def parse_document(doc: str, html: str) -> list[dict]:
    parser = UnitParser()
    parser.feed(html)
    parser.close()
    records = []
    for unit in parser.units:
        text = "\n".join(merge_markers(unit["paragraphs"]))
        if unit["kind"] == "considerant":
            text = RECITAL_PREFIX.sub("", text)
        records.append(
            {
                "id": f"{doc}:{unit['ref']}",
                "doc": doc,
                "kind": unit["kind"],
                "num": unit["num"],
                "title": unit["title"],
                "text": text,
            }
        )
    return records


def main() -> int:
    sources = sorted(RAW_DIR.glob("*.fr.html"))
    if not sources:
        print(f"Aucun fichier dans {RAW_DIR} : lance d'abord scripts/download_corpus.sh")
        return 1
    records: list[dict] = []
    for source in sources:
        doc = source.name.removesuffix(".fr.html")
        doc_records = parse_document(doc, source.read_text(encoding="utf-8"))
        articles = sum(r["kind"] == "article" for r in doc_records)
        empty = [r["id"] for r in doc_records if not r["text"]]
        print(f"{doc}: {articles} articles, {len(doc_records) - articles} considérants")
        if empty:
            print(f"  ATTENTION, unités vides : {empty}")
        records.extend(doc_records)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"{len(records)} unités écrites dans {OUT_FILE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
