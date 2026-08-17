# Bilan temporaire et plan d'action de l'article

Date du bilan : 4 août 2026.

Ce document sert de feuille de route pour les huit à neuf prochains jours de
travail. Les identifiants `MATH-*`, `NUM-*`, `REPRO-*` et `REDAC-*` sont
stables : ils peuvent être cités directement dans une nouvelle conversation,
par exemple « travaille sur `NUM-4` ».

## 1. Diagnostic global

Le manuscrit possède déjà une ossature mathématique sérieuse, mais il est
encore davantage un rapport théorique qu'un article publiable. La stratégie
recommandée est de consacrer au maximum deux jours supplémentaires aux
mathématiques, puis environ 70 % du temps restant aux résultats numériques, à
leur robustesse et à la rédaction scientifique.

En huit à neuf jours, il est réaliste d'obtenir un article professionnel et
défendable. Atteindre effectivement le niveau d'une revue de premier plan
dépendra surtout de la solidité de la validation empirique : les données ne
contiennent pas quatre opérateurs observés sur un même site. L'étude devra donc
être présentée comme une évaluation semi-empirique et non comme une validation
multi-opérateur réelle.

### Message scientifique recommandé

> L'efficacité énergétique de la coopération RAN ne garantit pas sa stabilité :
> optimisation exacte, caractérisation de régimes stables et analyse
> semi-empirique à grande échelle.

Cette histoire distingue le manuscrit de Koutitas, où la convexité et la
stabilité de la grande coalition sont obtenues sous d'autres hypothèses.

## 2. État actuel du projet

### 2.1 Points solides

Les sections 3 à 5 contiennent déjà :

- des hypothèses explicites et un modèle de coût fixe plus coût variable ;
- une preuve d'optimalité de l'allocation gloutonne à gardiens fixés ;
- une formulation MILP et une preuve de NP-difficulté ;
- la complexité de l'énumération exacte ;
- la sous-additivité de `C*` et la superadditivité du jeu d'économies ;
- une distinction rigoureuse entre coûts physiques, coûts nets et transferts ;
- le cœur, Bondareva--Shapley, le moindre cœur, le nucléole et Shapley ;
- une condition suffisante de non-vacuité du cœur et son extension temporelle.

La partie numérique contient déjà un résultat potentiellement publiable : sur
3 626 instances calculables, 33 cœurs sont vides sur la journée et 18 la nuit,
avec une structure de certificat très régulière. Ce résultat est intéressant
parce qu'il montre que la stabilité n'est pas automatique.

Le code s'exécute, les trois tests existants passent et le manuscrit compile.
Cette couverture de test reste cependant insuffisante pour sécuriser une
campagne numérique de publication.

### 2.2 Blocages éditoriaux

Le manuscrit n'est pas encore un article achevé :

- le titre vaut encore `TITLE` ;
- le résumé est vide ;
- les affiliations sont provisoires ;
- les mots-clés sont absents ;
- aucune conclusion n'est présente ;
- aucune figure scientifique n'est intégrée ;
- la bibliographie est presque vide ;
- cinq citations mathématiques sont non résolues ;
- la section Related Work reste un aide-mémoire de seize lignes ;
- le template de la revue ou conférence cible n'est pas choisi.

### 2.3 Blocages scientifiques et empiriques

Pour chaque antenne, un seul profil d'opérateur est observé. Les trois autres
sont construits par une échelle uniforme dans `[0,8; 1]`, un bruit de 2 % et
des perturbations de coûts de plus ou moins 10 %. Il en résulte :

- une corrélation temporelle presque parfaite entre les opérateurs ;
- une diversité de coûts choisie arbitrairement ;
- un unique tirage pseudo-aléatoire dans les résultats actuels ;
- aucune preuve que les quatre équipements simulés sont colocalisés.

Les résultats actuels doivent donc être qualifiés de semi-empiriques. Le taux
rare de cœurs vides doit être répété sur plusieurs graines avant de devenir un
résultat central.

La régression énergétique est ajustée sur 24 points obtenus après moyennage de
cinq jours. Il manque :

- la distribution des coefficients de détermination `R²` ;
- une validation hors échantillon ;
- l'incertitude sur `F` et `gamma` ;
- une analyse des résidus ;
- un traitement documenté des 199 antennes rejetées ;
- une analyse du biais de sélection induit par ces exclusions.

Enfin, l'intercept mesuré correspond à la puissance active sans charge, alors
que le modèle normalise la veille à zéro. Il faut interpréter `F_i` comme une
différence actif--veille ou effectuer une sensibilité au niveau de
consommation de veille.

### 2.4 Incohérence à résoudre

Le manuscrit annonce le nucléole comme règle principale de partage, alors que
le code sélectionne par défaut Shapley ou sa projection sur le cœur ou le
moindre cœur.

Doctrine recommandée :

- nucléole : règle principale orientée stabilité ;
- Shapley : référence d'équité contributive ;
- projection de Shapley : résultat secondaire ou annexe, avec justification
  explicite de son statut normatif.

## 3. Comparaison avec les deux articles de référence

Articles : [Bousia](../articles/bousia.pdf) et
[Koutitas](../articles/koutitas.pdf).

| Dimension | Bousia / Koutitas | Manuscrit actuel | Action nécessaire |
|---|---|---|---|
| Contributions | Trois à quatre contributions explicites | Contributions dispersées | Les annoncer nettement dans l'introduction |
| Théorie | Équilibre, optimisation, convexité | Bonne base ; cœur parfois vide | Faire de cette différence une contribution |
| Validation du modèle | Analyse confrontée aux simulations | Régression illustrée, non validée à l'échelle | Validation hors échantillon et distributions |
| Baselines | Trois à quatre méthodes concurrentes | Fonctionnement autonome uniquement | Ajouter au moins deux heuristiques crédibles |
| Métriques | Débit, énergie, coût, gains individuels | Principalement vacuité du cœur | Ajouter gains, gardiens, charges et allocations |
| Sensibilités | Trafic, nombre d'opérateurs, roaming, technologie | Deux fenêtres horaires | Capacité, trafic, coûts, diversité, `n`, graines |
| Résultats individuels | Gains et pertes par opérateur | Implémentés mais absents du papier | Montrer coûts nets, transferts et équité |
| Narration | Chaque figure répond à une question | Table isolée | Structurer la section 6 par questions de recherche |
| Forme | Articles IEEE finis | Brouillon de 21 pages | Résumé, conclusion, figures, références et template |

Il n'est pas recommandé d'ajouter maintenant une chaîne de Markov voix/données
comme Bousia. La capacité effective `q_i` joue déjà le rôle d'une contrainte de
service. Cette abstraction doit être assumée et soumise à une forte analyse de
sensibilité.

## 4. Développements mathématiques restants

### MATH-1 — Caractérisation exacte du cas homogène

Priorité : critique. Durée maximale : une journée.

Sous les hypothèses

```text
d_i = d, q_i = q, F_i = F, gamma_i = gamma,
```

une coalition de taille `s` nécessite

```text
k_s = ceil(s d / q)
```

gardiens. Il s'ensuit

```text
C*(S) = k_s F + gamma s d,
v(S)  = (s - k_s) F.
```

Le jeu étant symétrique, chercher à démontrer :

```text
Core(N,v) non vide
<=> v(s)/s <= v(n)/n pour tout s = 1,...,n-1.
```

Conséquences attendues :

- Shapley est l'allocation égale `v(N)/n` ;
- le nucléole est également égalitaire ;
- le moindre cœur possède une expression explicite à partir de
  `v(s) - s v(n)/n`.

Ce résultat relie directement marge de capacité, coûts fixes et stabilité.

### MATH-2 — Contre-exemple paramétrique de cœur vide

Priorité : critique. Durée : une demi-journée au maximum.

Pour `n >= 3`, considérer des opérateurs identiques et `q = (n-1)d`. Toute
coalition de `n-1` opérateurs fonctionne avec un gardien, tandis que la grande
coalition en nécessite deux. On obtient

```text
v(N\{i}) = (n-2)F,
v(N)     = (n-2)F.
```

La famille équilibrée formée des `N\{i}`, de poids `1/(n-1)`, viole le
critère de Bondareva--Shapley. Le cœur est vide malgré l'homogénéité parfaite.

Résultat à mettre en avant :

> Le cœur peut être vide même pour des opérateurs parfaitement homogènes.

### MATH-3 — Certificat leave-one-out

Priorité : haute. Durée : une demi-journée au maximum.

Formaliser le certificat observé numériquement :

```text
C*(N) > 1/(n-1) sum_i C*(N\{i})
=> Core(N,v) vide.
```

Si possible, en déduire une borne inférieure sur `epsilon*`, la violation du
moindre cœur. Il s'agit du pont le plus naturel entre théorie et simulations.

### MATH-STOP — Critère d'arrêt

Arrêt mathématique impératif à la fin du deuxième jour ouvré.

Le seuil minimal est :

- [ ] `MATH-2` démontré et intégré ;
- [ ] `MATH-3` démontré et intégré ;
- [ ] `MATH-1` démontré, ou abandonné proprement si la preuve n'est pas close.

Ne pas ouvrir maintenant :

- modèle dynamique avec usure et rotation ;
- incertitude stochastique sur le trafic ;
- extension multisite ;
- caractérisation générale du cœur avec paramètres hétérogènes ;
- nucléole sous forme fermée générale ;
- preuve de NP-difficulté forte.

## 5. Campagne numérique

### NUM-1 — Valider le modèle affine de puissance

Question : l'hypothèse `P = F_tilde + gamma_tilde d` est-elle crédible ?

Résultats à produire :

- [ ] histogrammes de `R²`, `F_tilde`, `gamma_tilde` et de la part fixe ;
- [ ] un exemple représentatif avec données, droite ajustée et résidus ;
- [ ] validation leave-one-day-out : quatre jours d'apprentissage, un de test ;
- [ ] comparaison avec un modèle constant et, si utile, quadratique ;
- [ ] analyse détaillée des 199 antennes rejetées ;
- [ ] sensibilité à une régression contrainte à coefficients positifs.

### NUM-2 — Quantifier les gains énergétiques et économiques

Métrique principale :

```text
r_save = (sum_i C_i^0 - C*(N)) / sum_i C_i^0.
```

Résultats à produire :

- [ ] médiane, quartiles et quantiles 5 %--95 % ;
- [ ] distribution des économies horaires ;
- [ ] comparaison nuit contre journée ;
- [ ] nombre moyen de gardiens ;
- [ ] part de trafic prise en charge par chaque gardien ;
- [ ] fréquence des changements de gardiens ;
- [ ] puissance ou énergie économisée, pas seulement les euros.

### NUM-3 — Comparer la solution à des baselines

Baselines recommandées :

1. aucun partage : tous les équipements restent actifs ;
2. R-to-1 lorsqu'un gardien unique est faisable ;
3. heuristique de capacité : activer les plus grandes capacités jusqu'à
   faisabilité ;
4. gardiens optimaux mais trafic réparti proportionnellement ;
5. solution exacte complète.

Cette décomposition doit séparer la valeur créée par :

- la désactivation ;
- la sélection des bons gardiens ;
- l'allocation gloutonne du trafic.

### NUM-4 — Diagrammes de sensibilité

Axes prioritaires :

- [ ] marge de capacité `q_i / max_h d_i^h` ;
- [ ] multiplicateur global de trafic ;
- [ ] part fixe `F_i / (F_i + gamma_i d_i)` ;
- [ ] dispersion de `F_i`, `gamma_i` et `q_i` ;
- [ ] corrélation ou décalage des profils de trafic ;
- [ ] nombre d'opérateurs `n = 2,...,6` ;
- [ ] consommation de veille en fraction de l'intercept actif.

Figure centrale recommandée : diagramme de phase

```text
marge de capacité x charge
-> économie relative, probabilité de cœur vide, nombre de gardiens.
```

Ne pas faire varier le prix commun de l'électricité comme résultat principal :
il multiplie tous les coûts par le même facteur et ne modifie ni les décisions
ni les économies relatives.

### NUM-5 — Expliquer la stabilité coalitionnelle

Résultats à publier :

- [ ] proportion de jeux convexes ;
- [ ] proportion avec Shapley dans le cœur ;
- [ ] proportion de cœurs vides ;
- [ ] distribution de `E_Sh` ;
- [ ] distribution de `epsilon*/v(N)` ;
- [ ] typologie des certificats de Bondareva--Shapley ;
- [ ] fréquence de satisfaction de la condition suffisante de capacité ;
- [ ] proportion de cœurs non vides non couverts par cette condition.

Études de cas :

- [ ] une instance avec certificat formé des quatre coalitions de trois ;
- [ ] l'instance atypique avec une paire et deux coalitions de trois ;
- [ ] une instance non convexe dont Shapley appartient néanmoins au cœur.

### NUM-6 — Comparer Shapley et le nucléole

Comparer les règles selon :

- économie individuelle `z_i` ;
- coût net `y_i` ;
- transfert `tau_i` ;
- excès maximal ;
- nombre de coalitions bloquantes ;
- dispersion des économies individuelles ;
- distance normalisée entre les allocations.

Une table sur une instance représentative et une figure sur la population
devraient suffire.

### NUM-7 — Robustesse aux graines et à la construction des opérateurs

Exigences minimales :

- [ ] 10 à 20 graines pour les perturbations ;
- [ ] tous les sites si le temps le permet, sinon un échantillon stratifié ;
- [ ] intervalles de confiance bootstrap par antenne ;
- [ ] résultats séparés pour les scénarios proches et hétérogènes ;
- [ ] profils synthétiques avec corrélations contrôlées ;
- [ ] scénario construit en tirant quatre profils distincts dans la population
  mesurée, explicitement qualifié de synthétique.

## 6. Reproductibilité et validation du code

### REPRO-1 — Exporter tous les résultats

Le programme doit produire un CSV contenant au minimum :

- antenne et graine ;
- famille de scénario ;
- fenêtre temporelle ;
- paramètres de capacité et de coût ;
- économie absolue et relative ;
- gardiens et changements de gardiens ;
- convexité, cœur et Shapley dans le cœur ;
- gap de Bondareva--Shapley et poids du certificat ;
- `epsilon*`, nucléole, Shapley et transferts.

### REPRO-2 — Renforcer les tests

Ajouter des tests pour :

- [ ] allocation gloutonne contre résolution exhaustive sur petites instances ;
- [ ] coût de coalition contre formulation MILP ou brute force ;
- [ ] équivalence cœur contre Bondareva--Shapley ;
- [ ] contre-exemple de `MATH-2` ;
- [ ] condition suffisante de non-vacuité ;
- [ ] agrégation temporelle ;
- [ ] conservation des coûts et équilibre budgétaire ;
- [ ] jeux connus pour Shapley, moindre cœur et nucléole ;
- [ ] déterminisme des expériences à graine fixée.

### REPRO-3 — Une commande de reproduction

Prévoir une commande ou un script unique qui :

1. charge et valide les données ;
2. exécute la campagne principale ;
3. écrit les tableaux intermédiaires ;
4. génère toutes les figures du manuscrit.

## 7. Rédaction et positionnement

### REDAC-1 — Fixer les contributions

Proposition de contributions :

1. formulation exacte du partage opportuniste avec coûts fixes, capacités et
   allocation gloutonne ;
2. analyse coopérative montrant que l'efficacité collective ne garantit pas la
   stabilité, avec conditions positives et contre-exemples ;
3. étude semi-empirique à grande échelle des économies, de la stabilité et des
   règles de partage ;
4. comparaison opérationnelle entre Shapley et le nucléole, y compris lorsque
   le cœur est vide.

### REDAC-2 — Reconstruire Related Work

Axes à couvrir :

- partage actif, passif et roaming-based du RAN ;
- mise en veille des stations et modèles de consommation ;
- optimisation de type facility location ou fixed-charge ;
- jeux non coopératifs pour les décisions d'extinction ;
- jeux coopératifs, cœur et Shapley pour le partage de gains ;
- positionnement précis par rapport à Bousia et Koutitas ;
- modèles et KPI 5G-Advanced de 3GPP Release 18, notamment TR 38.864.

### REDAC-3 — Structure recommandée de la section numérique

1. Questions de recherche.
2. Données et protocole semi-empirique.
3. Validation du modèle de puissance.
4. Gains opérationnels et comparaison aux baselines.
5. Sensibilités et mécanismes causaux.
6. Stabilité du jeu et certificats de cœur vide.
7. Comparaison des allocations.
8. Limites et menaces sur la validité.

### REDAC-4 — Forme finale

- [ ] revue ou conférence cible et template choisis ;
- [ ] titre définitif ;
- [ ] résumé avec problème, méthode, principaux chiffres et conclusion ;
- [ ] mots-clés ;
- [ ] introduction avec contributions explicites ;
- [ ] Related Work complet ;
- [ ] discussion et limites ;
- [ ] conclusion ;
- [ ] bibliographie complète et citations résolues ;
- [ ] figures homogènes et lisibles en deux colonnes ;
- [ ] aucune affirmation non soutenue par une preuve ou une expérience.

## 8. Questions scientifiques à trancher

- Quelle est exactement la provenance et la période des mesures ?
- Les antennes sont-elles toujours actives pendant les observations ?
- Les volumes sont-ils bien des Go par heure et la puissance une moyenne
  horaire ?
- Quelle consommation de veille est réaliste ?
- `q_i` représente-t-il une capacité physique ou une capacité reconstruite à
  partir du trafic observé ?
- Le changement de gardien est-il autorisé chaque heure ?
- Les transferts sont-ils réglés par heure, par nuit ou sur une période
  contractuelle ?
- Quelle promesse de QoS est incorporée dans la capacité effective ?
- Quelle baseline opérationnelle est réellement crédible pour les opérateurs ?
- Les conclusions persistent-elles avec des profils moins corrélés ?
- Les rares cœurs vides persistent-ils lorsque la graine change ?
- L'article défend-il principalement l'économie d'énergie, la stabilité
  économique ou leur interaction ?

La recommandation est de défendre leur interaction.

## 9. Calendrier de neuf jours ouvrés

| Jour | Travail principal | Livrable de fin de journée |
|---|---|---|
| 1 | Fixer la revue, le message, les contributions et les questions numériques | Plan et protocole gelés |
| 2 | `MATH-1` à `MATH-3`, puis arrêt mathématique | Mathématiques définitivement gelées |
| 3 | `REPRO-1` à `REPRO-3`, graines et validation de la régression | Une commande reproductible sur 50 antennes |
| 4 | Pilotes, baselines, métriques et figures provisoires | Résultats pilotes validés |
| 5 | Campagne complète nuit, journée et horaire | Table principale et données brutes figées |
| 6 | `NUM-4` et `NUM-7` | Diagrammes de phase et robustesse |
| 7 | `NUM-5` et `NUM-6`, études de cas | Figures définitives et interprétation |
| 8 | `REDAC-1` à `REDAC-4` | Manuscrit complet dans le template cible |
| 9 | Relecture de type reviewer, preuves, unités et reproduction | Version candidate à circulation |

Avec seulement huit jours, fusionner les jours 4 et 5. Ne pas supprimer le jour
final de relecture.

## 10. Ordre de priorité absolu

### P0 — Indispensable

- `MATH-2`, `MATH-3` et décision sur `MATH-1` ;
- `NUM-1`, `NUM-2`, `NUM-4`, `NUM-5` et `NUM-7` ;
- `REPRO-1` et `REPRO-2` ;
- cohérence Shapley--nucléole ;
- titre, résumé, Related Work, discussion, conclusion et bibliographie.

### P1 — Très important

- `NUM-3` et `NUM-6` ;
- consommation de veille ;
- fréquence des changements de gardiens ;
- intervalles de confiance et études de cas.

### P2 — Annexe ou travail futur

- projection de Shapley ;
- courbe de temps de calcul selon `n` ;
- rotation dynamique des gardiens ;
- coûts de transition ;
- modèle stochastique ou multisite.

## 11. Critère de réussite final

L'article sera professionnel lorsque :

- [ ] chaque contribution annoncée sera soutenue par un théorème ou une figure ;
- [ ] la construction semi-empirique sera décrite sans ambiguïté ;
- [ ] les résultats principaux seront robustes aux graines et à la capacité ;
- [ ] au moins trois baselines seront présentes ;
- [ ] théorie et simulations seront explicitement reliées ;
- [ ] le choix Shapley--nucléole sera cohérent partout ;
- [ ] titre, résumé, Related Work, discussion, conclusion et bibliographie seront
  complets ;
- [ ] toutes les figures et tables seront régénérables automatiquement ;
- [ ] la compilation ne comportera plus de citation ou référence indéfinie.

La priorité immédiate n'est pas de multiplier les mathématiques générales, mais
de transformer les résultats existants en une démonstration numérique
convaincante, reproductible et correctement positionnée.
