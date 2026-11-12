# Groundtruth

Banc d'évaluation pour la recherche documentaire (retrieval) sur deux règlements européens en français : le RGPD et le règlement sur l'IA.

## Résultats

recall@10 sur 100 questions (part des questions dont la preuve est dans les 10 premiers résultats) :

| Moteur | Reclassement | 100 mots | 200 mots | 400 mots |
|---|---|---|---|---|
| BM25 (lexical) | non | 0,25 | 0,27 | 0,31 |
| BM25 (lexical) | oui | 0,36 | 0,43 | 0,45 |
| Dense (multilingual-e5-small) | non | 0,48 | 0,49 | 0,50 |
| Dense (multilingual-e5-small) | oui | 0,59 | **0,66** | 0,63 |
| Hybride (BM25 + dense, RRF) | non | 0,40 | 0,49 | 0,53 |
| Hybride (BM25 + dense, RRF) | oui | 0,56 | 0,62 | 0,65 |

Temps par question sur CPU : moins de 70 ms sans reclassement ; avec reclassement, 1,3 à 1,9 s (100 mots), 3,8 à 4,1 s (200 mots), 6,3 à 6,4 s (400 mots). Le tableau complet (recall@5, MRR@10, nDCG@10, temps) est dans [`results/comparison.md`](results/comparison.md).

Ce que les trois comparaisons montrent :

- **Reclassement** : c'est le seul réglage qui change nettement le résultat, +11 à +17 points de recall@10 selon la configuration, et le MRR@10 passe d'environ 0,29 à 0,47 pour le dense. Il coûte plusieurs secondes par question sur CPU.
- **Hybride ou dense** : pas de gagnant. L'hybride est derrière à 100 mots (0,40 contre 0,48), à égalité à 200, devant de 3 points à 400. BM25 seul est faible sur ces questions paraphrasées, donc il apporte peu à la fusion.
- **Taille des morceaux** : effet faible pour le dense (0,48 à 0,50), plus marqué pour BM25 et l'hybride, qui préfèrent les grands morceaux.

Avec 100 questions, une question vaut un point et la marge d'erreur est d'environ 5 points : les écarts de 1 à 4 points entre deux lignes ne sont pas des différences établies.

## Recommandation

**Dense, morceaux de 200 mots, avec reclassement** (recall@10 0,66, MRR@10 0,47) quand environ 4 secondes par question sur CPU sont acceptables. L'hybride reclassé à 400 mots fait aussi bien (0,65) mais prend 6 secondes et ajoute un second index ; à score égal, la configuration la plus simple gagne.

Si la réponse doit être immédiate : **dense à 200 mots sans reclassement** (recall@10 0,49 en 26 ms). Le compromis est donc de 17 points de recall contre 4 secondes d'attente.

Même la meilleure configuration manque une question sur trois. Limites connues :

- Le modèle de reclassement lit 512 tokens au maximum : il ne voit pas la fin des morceaux de 400 mots.
- Les passages équivalents (37 pour 100 questions) ne sont pas exhaustifs : un moteur peut trouver un passage correct non listé et être compté en échec. Les scores sont donc un peu pessimistes.
- Les temps viennent d'une seule exécution sur un CPU de portable ; ils donnent un ordre de grandeur.

## Cinq requêtes qui échouent encore

La meilleure configuration (dense, 200 mots, reclassement) manque 34 questions sur 100. Dans 31 cas, la preuve n'est pas parmi les 30 candidats transmis au modèle de reclassement : il ne peut pas remonter un passage qu'il n'a jamais vu. C'est donc la première étape de recherche qui limite le score, pas le reclassement. 22 de ces questions échouent dans les 18 configurations.

Cinq exemples, avec le rang de la preuve dans la recherche dense (sur 1 109 morceaux) :

| Question | Preuve attendue | Rang dense | Rang après reclassement | Cause |
|---|---|---|---|---|
| q008 « Dans quelles situations une organisation est-elle forcée de nommer un DPO ? » | RGPD, art. 37 | 88 | absent | Sigle absent du texte |
| q033 « Une banque peut-elle refuser mon crédit uniquement sur la base d'un calcul fait par un ordinateur ? » | RGPD, art. 22 | 189 | absent | Cas concret, texte abstrait |
| q092 « Que risque-t-on si on ment aux autorités dans un dossier sur une IA ? » | Règlement IA, art. 99 | 772 | absent | Réponse noyée dans le morceau |
| q069 « La police peut-elle scanner les visages des passants dans la rue en direct ? » | Règlement IA, art. 5 | 25 | 20 | Bon article, mauvais morceau |
| q028 « Une société peut-elle utiliser mes données sans mon accord parce qu'elle y a un intérêt commercial ? » | RGPD, art. 6 | 20 | 11 | Manqué d'une place |

- **q008, sigle absent du texte.** « DPO » est le sigle anglais ; le règlement en français écrit « délégué à la protection des données ». Le moteur s'accroche au seul mot qu'il reconnaît, « nommer », et renvoie la nomination des membres des autorités de contrôle (art. 53, considérant 121).
- **q033, cas concret contre texte abstrait.** La question parle de banque, de crédit et d'ordinateur ; l'article parle d'une « décision fondée exclusivement sur un traitement automatisé ». Aucun mot en commun, et le modèle ne fait pas le lien : il renvoie des passages sur les demandes « par voie électronique ».
- **q092, réponse noyée dans le morceau.** L'article 99 fait 932 mots, découpés en six morceaux. Celui qui contient la phrase sur les « informations inexactes, incomplètes ou trompeuses » commence par la fin d'une liste d'obligations sans rapport. Le vecteur du morceau résume l'ensemble, pas cette phrase. De plus, « mentir » n'apparaît nulle part dans le texte.
- **q069, bon article, mauvais morceau.** Trois morceaux de l'article 5 sont dans les dix premiers résultats, mais ce sont ceux qui détaillent les conditions et les autorisations. Le morceau qui pose l'interdiction commence par une autre interdiction (l'inférence des émotions au travail) et reste au rang 20. Un lecteur aurait la bonne réponse sous les yeux ; la métrique compte un échec.
- **q028, manqué d'une place.** La preuve est au rang 20, le reclassement la remonte au rang 11. « Intérêt commercial » correspond à « intérêts légitimes » dans le texte, et le point f) recherché est le dernier d'une liste de six conditions dans son morceau. L'article 88, qui contient aussi « intérêts légitimes », passe devant.

Ce que ces échecs suggèrent d'essayer ensuite, sans l'avoir testé : reformuler la question en vocabulaire juridique avant la recherche (q008, q033), découper par paragraphe ou par point plutôt que par nombre de mots (q092, q028), et compter comme réussie une réponse dans le bon article (q069).

## Contrôle de régression

`python -m groundtruth.regression` relance BM25 et le dense (200 mots, sans reclassement, une à deux minutes sur CPU) et compare le recall@10 à la référence de `results/baseline.json` (BM25 0,27, dense 0,49). Il sort en erreur si un score baisse de plus d'un point. Le workflow `.github/workflows/ci.yml` le lance à chaque push et pull request, après les tests.

Essai avec un bug volontaire, les morceaux tronqués à 200 caractères au lieu de 200 mots avant le calcul des vecteurs :

```diff
-            [f"passage: {chunk.text}" for chunk in chunks],
+            [f"passage: {chunk.text[:200]}" for chunk in chunks],
```

```
bm25   recall@10 0.27
dense  recall@10 0.34

RÉGRESSION :
  dense : recall@10 0.34, référence 0.49 (baisse de 15 points)
```

Le code de sortie est 1, ce qui fait échouer la CI. La sortie ci-dessus vient d'une exécution locale ; sur GitHub, la CI est verte sur `main` et rouge sur la [pull request de démonstration](https://github.com/aboubacarguidiera/groundtruth/pull/1), qui contient ce bug et n'est pas fusionnée.

Quand un changement améliore un score, `python -m groundtruth.regression --update` enregistre la nouvelle référence. Le reclassement n'est pas contrôlé : il prend environ sept minutes pour 100 questions.

## Ligne d'arrivée

Le projet est fini quand ces quatre points sont cochés :

- [x] Trois configurations comparées dans un tableau (taille des morceaux, hybride ou dense, reranker ou non)
- [x] Une recommandation écrite, avec le compromis
- [x] La CI échoue sur un changement volontairement mauvais (recall@10 en baisse de plus d'un point)
- [x] Cinq requêtes qui échouent encore, avec l'explication

## Étapes

- [x] **1. Corpus et 20 questions** : corpus découpé, 20 questions rédigées avec un modèle puis relues une par une, validées par `check_golden.py`
- [x] **2. Retrieval de base** : BM25 et dense, recall@k, MRR, nDCG par catégorie
- [x] **3. 100 questions et trois configurations** : 100 questions, 18 configurations comparées (3 tailles x 3 moteurs x reclassement)
- [x] **4. CI et README final** : contrôle de régression dans la CI, cinq requêtes en échec expliquées

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
python -I scripts/check_golden.py      # vérifie data/golden/golden_v2.toml

python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest                                             # 13 tests
python -m groundtruth.regression                   # échoue si le recall@10 baisse de plus d'un point
python -m groundtruth.evaluate --retriever bm25    # écrit results/bm25_c200_o40.json
python -m groundtruth.evaluate --retriever dense --chunk-size 200 --rerank
python -m groundtruth.compare                      # les 18 configurations, environ une heure sur CPU
```

## Métriques

Tout se calcule à partir du rang du premier morceau qui contient une preuve acceptée :

- **recall@k** : part des questions dont la preuve est dans les k premiers résultats.
- **MRR@k** : moyenne de 1/rang (1 en première place, 0,5 en deuxième, 0 au-delà de k).
- **nDCG@k** : comme le MRR, avec une pénalité plus douce pour les rangs lointains.

Un morceau compte comme bon s'il vient du bon article ou considérant et contient au moins la moitié de l'extrait attendu.

## Jeu de référence

`data/golden/golden_v2.toml` : 100 questions (50 RGPD, 50 règlement sur l'IA) en six catégories. Chaque question pointe vers un texte, un article ou considérant, et un extrait exact. La bonne réponse ne dépend donc pas du découpage en morceaux, et le fichier reste valable quand on change la taille des morceaux.

Quand un autre passage répond aussi à la question (souvent un considérant qui reprend l'article), il est ajouté dans un bloc `[[questions.also]]` et compte comme bonne réponse.

`golden_v1.toml` garde les 20 premières questions telles qu'elles étaient à l'étape 2.
