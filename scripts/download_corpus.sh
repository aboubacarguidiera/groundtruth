#!/usr/bin/env bash
# Télécharge les deux règlements en français depuis l'Office des publications de l'UE.
# Usage : bash scripts/download_corpus.sh
set -euo pipefail

raw_dir="$(cd "$(dirname "$0")/.." && pwd)/data/raw"
mkdir -p "$raw_dir"

# nom du fichier : numéro CELEX du texte
for pair in "ai_act:32024R1689" "rgpd:32016R0679"; do
  name="${pair%%:*}"
  celex="${pair##*:}"
  curl -sS -L --fail --max-time 120 \
    -H "Accept: application/xhtml+xml" -H "Accept-Language: fr" \
    -o "$raw_dir/$name.fr.html" \
    "http://publications.europa.eu/resource/celex/$celex"
  echo "$name ($celex) -> $raw_dir/$name.fr.html"
done
