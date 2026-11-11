# Groundtruth

Banc d'évaluation pour la recherche documentaire (retrieval) sur deux règlements européens en français : le RGPD et le règlement sur l'IA.

**Résultats** : à venir (étape 3). Le tableau comparatif ira ici, en haut.

## Ligne d'arrivée

Le projet est fini quand ces quatre points sont cochés :

- [ ] Trois configurations comparées dans un tableau (taille des morceaux, hybride ou dense, reranker ou non)
- [ ] Une recommandation écrite, avec le compromis
- [ ] La CI échoue sur un changement volontairement mauvais (recall@10 en baisse de plus d'un point)
- [ ] Cinq requêtes qui échouent encore, avec l'explication

## Étapes

- [ ] **1. Corpus et 20 questions** : corpus découpé (fait), 20 questions validées par `check_golden.py`
- [ ] **2. Retrieval de base** : recall@k, MRR, nDCG par catégorie
- [ ] **3. 100 questions et trois configurations**
- [ ] **4. CI et README final**

## Corpus

| Texte | CELEX | Articles | Considérants |
|---|---|---|---|
| RGPD, règlement (UE) 2016/679 | 32016R0679 | 99 | 173 |
| Règlement sur l'IA, règlement (UE) 2024/1689 | 32024R1689 | 113 | 180 |

Source : Office des publications de l'Union européenne, © Union européenne. Les annexes ne sont pas incluses.

## Commandes

```bash
bash scripts/download_corpus.sh        # télécharge les deux textes dans data/raw/
python -I scripts/build_corpus.py      # écrit data/corpus/units.jsonl (565 unités)
python -I scripts/check_golden.py      # vérifie data/golden/golden_v1.toml
```

## Jeu de référence

`data/golden/golden_v1.toml` : chaque question pointe vers un texte, un article ou considérant, et un extrait exact. La bonne réponse ne dépend donc pas du découpage en morceaux, et le fichier reste valable quand on change la taille des morceaux.
