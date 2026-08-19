# Protocole scientifique — révision du 18 août 2026

Date de gel du modèle : 4 août 2026. Protocole d'instances : 18 août 2026.

Ce document fixe le positionnement éditorial, les objectifs et le protocole de
la future campagne numérique. Les choix marqués comme
paramètres restent variables dans le code ; leur valeur finale devra être
arrêtée avant la campagne principale, sans être choisie en fonction des
résultats de stabilité obtenus.

## 1. Positionnement éditorial

- Cible principale : **IEEE Transactions on Green Communications and
  Networking (TGCN)**.
- Langue finale : anglais. Le développement scientifique reste en français
  jusqu'à la phase finale de traduction.
- Format : le template IEEE final n'est pas appliqué au jour 1.
- Nature de l'étude : **évaluation semi-empirique**, et non validation réelle
  de quatre opérateurs colocalisés.

Message central :

> Le partage opportuniste du RAN peut créer des économies d'énergie sans
> garantir la stabilité de la coopération ; l'article relie optimisation
> exacte, régimes de capacité et stabilité coalitionnelle.

Doctrine de partage : Shapley est retenue si elle appartient au cœur ; sinon,
le nucléole est retenu. La projection de Shapley reste un diagnostic
secondaire.

## 2. Contributions gelées

1. Formulation exacte du partage opportuniste avec coûts fixes, capacités,
   sélection des gardiens et allocation gloutonne du trafic.
2. Analyse coopérative distinguant efficacité collective et stabilité, avec
   conditions positives, contre-exemples et certificats de cœur vide.
3. Étude semi-empirique à grande échelle des économies, des gardiens et de la
   stabilité, construite à partir de profils de terrain.
4. Comparaison opérationnelle de Shapley et du nucléole par les économies
   individuelles, coûts nets, transferts et objections coalitionnelles.

## 3. Objectifs de l'évaluation

L'évaluation contrôle d'abord les algorithmes et résultats théoriques, puis
mesure la précision prédictive du modèle de puissance, les économies et la
sélection des gardiens, l'effet des paramètres, et enfin la stabilité et le
partage des économies.

## 4. Données et population d'étude

Informations acquises : données de terrain Orange, Île-de-France, comprenant
un volume descendant horaire en Go et une puissance moyenne horaire en W.

À compléter avant soumission : **[provenance administrative exacte et
conditions d'utilisation ou de citation du jeu de données]**.

Toutes les expériences utilisent les cinq premiers jours de la période, du 20
au 24 mars 2023, soit 120 créneaux horaires par antenne. Les règles
d'admissibilité sont :

- valeurs temporelles, trafic et puissance lisibles et non négatifs ;
- exclusion de l'ajustement des créneaux inactifs `(P,d)=(0,0)` et des
  créneaux incohérents `P=0, d>0` ;
- trafic suffisamment variable pour identifier une pente ;
- intercept et pente strictement positifs, afin de conserver une relation
  puissance--trafic physiquement interprétable.

Le nombre d'antennes admissibles n'est pas une taille d'échantillon gelée a
priori. La calibration actuelle en retient `3 624` sur `3 825`. La campagne
utilise toutes les antennes satisfaisant les règles ci-dessus et rapporte
séparément chaque motif d'exclusion. Les profils constants et les ajustements
non positifs restent dans l'analyse des exclusions afin de quantifier le biais
de sélection.

## 5. Validation des calculs et du modèle de puissance

Avant la campagne empirique :

1. comparer la répartition gloutonne au programme linéaire ;
2. comparer l'énumération des gardiens au MILP ;
3. vérifier l'équivalence entre faisabilité du cœur et
   Bondareva--Shapley ;
4. vérifier l'équilibre budgétaire des transferts ;
5. générer les conditions suffisantes et contre-exemples théoriques sur des
   grilles de paramètres.

Ces expériences contrôlent l'implémentation mais ne remplacent pas les preuves.

L'ajustement utilise les observations horaires actives des cinq premiers
jours, sans les réduire à 24 moyennes horaires. Pour chaque antenne admissible,
les moindres carrés estiment sur l'ensemble de ces observations le modèle
`P_i(d) = P_fixed_i + slope_i * d`. Il ne s'agit ni de prévoir le trafic ni de
prédire une période future.

La qualité descriptive de l'ajustement est résumée uniquement par le
coefficient de détermination R² et par la RMSE divisée par la puissance active
moyenne. Aucun découpage apprentissage--test et aucun modèle constant ou
quadratique ne sont utilisés.

La puissance consommée par une antenne éteinte est supposée nulle. Les
coefficients sont donc

```text
F_i     = price_per_kWh / 1000 * P_fixed_i,
gamma_i = price_per_kWh / 1000 * slope_i.
```

## 6. Construction semi-empirique des opérateurs

On ne traite pas quatre antennes Orange distinctes comme quatre MNO
colocalisés. Chaque site virtuel est un **plan** ancré sur une antenne de
référence \(a\) :

- profil normalisé \(s_j(t)=d_j(t)/\bar d_j\) sur les 120 heures ;
- mélange \(\tilde s_i=(1-\lambda)s_a+\lambda s_{j_i}\) ;
- demande \(d_i(t)=\mu_a\alpha_i\tilde s_i(t)\) avec \(\mu_a=\bar d_a\)
  et \(\sum_i\alpha_i=n\).

La campagne utilise **400 plans** (100 références par quartile de trafic
moyen, sans remise). Graine globale `20260818`. Une réalisation synthétique
par référence. Les donneurs et les trois paquets énergétiques (proches,
modérés, distants) sont tirés une fois, disjoints de \(a\) et entre eux.
Les \(\alpha\) sont permutés une fois par site.

Niveaux de volume, calés sur les quantiles du trafic moyen admissible :

| niveau | définition | \(\alpha\) (arrondi) |
|---|---|---|
| close | \((1,1,1,1)\) | \(1,1,1,1\) |
| moderate | Q40, Q50, Q60, Q75 | \(0.551, 0.807, 1.084, 1.557\) |
| far | Q25, Q40, Q60, Q90 | \(0.236, 0.532, 1.047, 2.185\) |
| outlier | trois médianes et Q90 | \(0.689, 0.689, 0.689, 1.933\) |

Niveaux de forme : \(\lambda\in\{0.15, 0.35, 1.00\}\), calés pour que la
corrélation médiane entre opérateurs vaille environ \(0.99\), \(0.94\) et
\(0.57\).

Niveaux d'équipement : quatre antennes du même quartile de `P_fixed` ; quatre
tirages dans toute la population ; une antenne par quartile. Les couples
`(P_fixed, slope)` restent joints. Ils sont assignés indépendamment des
\(\alpha_i\).

Les comparaisons entre niveaux sont **appariées** sur les mêmes 400 plans.
La suffisance d'échantillon est contrôlée en comparant les 200 premiers sites
aux 400.

Artefacts dans `results/power_calibration/` :

- `calibrated_population.npz` : profils et coefficients admissibles ;
- `site_blueprints.csv` : plans gelés ;
- `protocol_parameters.json` : \(\alpha\), \(\lambda\), \(r\), régimes B et
  statistiques empiriques ;
- `manifest.json` : signature SHA-256 de la source et version du générateur.

`--rebuild-cache` relit le CSV. Si le cache de calibration est valide mais
que la version du générateur a changé, seuls les plans sont reconstruits.

## 7. Capacité : campagne A et campagne B

La capacité n'est pas observée. Toute construction de \(q_i\) est un
scénario contrefactuel. Le code vérifie \(d_i^h\le q_i\) sur \(H\) ; une
violation arrête l'instance.

**Campagne A** (semi-empirique). Équipement dimensionné de sorte que le pic
des 120 heures occupe une fraction `r` de la capacité :

```text
max_{t in T} d_i(t) = r q_i,    r in {0.70, 0.80, 0.90, 1.00}.
```

Le paramètre `r` est le taux d'utilisation maximal *supposé* au pic
quinquennal. `r = 0.70` signifie `max d_i = 0.70 q_i`, donc
`q_i ≈ 1.43 max d_i`, et non une « marge de 30 % » au sens de
`q_i = 1.30 max d_i`. Scénario central : volumes, formes et équipements
modérés, `r = 0.70`.

Cette campagne évalue un partage nocturne d'équipements dimensionnés pour la
journée. Le rapport médian trafic nocturne moyen / pic des cinq jours vaut
\(0.14\) : un seul gardien y suffit le plus souvent. Ce n'est pas un test
des franchissements de seuils.

**Campagne B** (seuils). Capacités égales, taillées sur la demande de
coalition dans \(H\) :

```text
q_i = max{ D_H^max / (k r_H),  max_{h in H} d_i^h },
D_H^max = max_{h in H} d^h(N).
```

Régimes \((k, r_H)\) : `(1, 0.70)` un gardien ; `(2, 0.90)` frontière 1–2 ;
`(3, 0.90)` frontière 2–3 ; `(4, 0.90)` contraint. Volumes et formes
proches, équipements modérés. Cette campagne n'est pas représentative du
réseau.

## 8. Fenêtre temporelle et reconfiguration horaire

La fenêtre principale est fixée à
`H = {0 h, 1 h, ..., 6 h}`, soit les sept créneaux horaires de l'intervalle
`[00:00, 07:00)`. Elle est identique pour tous les sites et tous les scénarios
de la campagne principale. L'analyse de sensibilité étudie séparément sa
position et sa durée, sans modifier ce réglage central.

Ce choix est un réglage de la campagne, et non une constante imposée par
l'implémentation. Le code reçoit les deux bornes de la fenêtre en paramètres
et accepte toute fenêtre horaire contiguë, y compris une fenêtre traversant
minuit. Modifier ces bornes ne doit changer ni la construction des plans, ni
les méthodes d'optimisation et d'analyse coalitionnelle. Dans la campagne B,
les capacités sont recalculées sur la fenêtre utilisée.

Chacun des cinq jours fournit une fenêtre nocturne distincte. Dans le modèle
principal, les gardiens et l'allocation du trafic sont réoptimisés
indépendamment à chaque heure. Les coûts, économies et transferts sont ensuite
agrégés et réglés une fois par nuit.

L'optimum persistant est calculé séparément en imposant les mêmes gardiens sur
toute la fenêtre. Son écart à l'optimum horaire mesure le coût de cette
contrainte. Aucun coût de commutation monétaire n'est introduit sans donnée
permettant de le calibrer.

## 9. Méthodes comparées

L'absence de partage sert uniquement de référence pour normaliser l'énergie
évitée ; elle n'est pas traitée comme une politique concurrente. Deux
politiques approchées sont comparées à l'optimum :

1. activation par capacité décroissante jusqu'à faisabilité ;
2. gardiens optimaux avec trafic réparti proportionnellement aux capacités.

Ces deux références sont comparées à l'optimum horaire exact : à chaque heure,
l'ensemble optimal est obtenu par énumération, puis le trafic est réparti par
coût variable croissant. La référence par capacité est elle-même recalculée à
chaque heure et isole la sélection des gardiens, tandis que la répartition
proportionnelle isole l'allocation du trafic. L'optimum avec gardiens fixes sur
la fenêtre est calculé séparément comme politique secondaire réalisable.

Une baseline infaisable est déclarée comme telle ; aucune demande n'est
abandonnée pour la rendre artificiellement comparable.

## 10. Analyse de sensibilité structurée

Le scénario central fixe `r = 0.70`, volumes, formes et équipements modérés,
une veille nulle, `[00:00, 07:00)` et quatre opérateurs. Les facteurs
suivants sont modifiés un par un, sur les mêmes 400 plans et les mêmes jours :

| Facteur | Valeurs |
|---|---|
| marge de capacité | `r in {0.70, 0.80, 0.90, 1.00}` |
| hétérogénéité des volumes | close / moderate / far / outlier |
| hétérogénéité des formes | `lambda in {0.15, 0.35, 1.00}` |
| hétérogénéité des équipements | close / moderate / distant |
| trafic | `0.8 d`, `d`, `1.2 d`, capacités centrales inchangées |
| puissance fixe | `0.8 P_fixed`, `P_fixed`, `1.2 P_fixed` |
| pente variable | `0.8 slope`, `slope`, `1.2 slope` |
| puissance de veille | `0`, `0.05 P_fixed`, `0.10 P_fixed` |
| position d'une fenêtre de 7 h | `[22:00,05:00)`, `[00:00,07:00)`, `[02:00,09:00)` |
| durée depuis minuit | 5 h, 7 h, 9 h |
| nombre d'opérateurs | `2, 3, 4, 5, 6`, même générateur, 400 plans par taille |

Dans les lignes consacrées aux coûts, la variation de `P_fixed` représente
celle de `F_i` et la variation de `slope` celle de `gamma_i`, à prix de
l'électricité fixé. Pour la veille, le coût d'un équipement inactif est la
fraction indiquée de sa composante fixe ; les références autonomes et les
coûts coalitionnels utilisent la même convention.

La position de la fenêtre est comparée uniquement sur les nuits entièrement
observées pour les trois positions, de façon à conserver l'appariement. La
variation du nombre d'opérateurs utilise nécessairement de nouveaux sites,
construits avec le même générateur, la même population admissible et la
même graine. Le paramètre `beta_i = gamma_i * q_i` n'est pas varié séparément :
à `q_i` fixé, sa variation est celle de `gamma_i`, tandis que la variation de
`q_i` est déjà portée par `r`.

Le trafic et la capacité sont en outre croisés, car leur interaction détermine
les seuils de faisabilité. Les cellules infaisables sont signalées comme hors
domaine et exclues des agrégats réalisables. Pour chaque valeur, les résultats
appariés portent sur :

- économie énergétique relative ;
- nombre moyen de gardiens actifs par heure ;
- écart de persistance par rapport à l'optimum horaire ;
- fréquence de cœur vide ;
- fréquence de Shapley hors du cœur.

Les facteurs sont classés par amplitude d'effet dans les plages testées. Ce
classement n'est pas interprété au-delà de ces plages. Le prix commun de
l'électricité n'est pas simulé : sa multiplication par une constante positive
ne change ni les décisions ni les indicateurs relatifs.

## 11. Métriques et inférence

Métriques opérationnelles : énergie et coût évités, économie relative, nombre
de gardiens, changements d'ensemble, faisabilité des baselines et écart de
persistance entre gardiens fixes et choix horaire.

Métriques coalitionnelles : convexité, non-vacuité du cœur, Shapley dans le
cœur, excès maximal de Shapley, gap de Bondareva--Shapley, structure du
certificat et epsilon du moindre cœur normalisé par `v(N)`.

Métriques de partage : économie individuelle `z_i`, coût net `y_i`, transfert
`tau_i`, dispersion des économies et distance Shapley--nucléole.

Métriques de validation : erreur maximale entre formulations équivalentes,
nombre de contradictions et résidu d'équilibre budgétaire.

Métriques d'ajustement du modèle de puissance : R² et RMSE divisée par la
puissance active moyenne.

Les distributions sont résumées par médiane, quartiles et quantiles 5 %--95 %.
Les intervalles d'incertitude à 95 % sont obtenus en rééchantillonnant les
400 plans.

## 12. Critères de gel avant campagne principale

- fenêtre nocturne principale fixée à `[00:00, 07:00)` pour chaque jour ;
- règles d'admissibilité exécutées et motifs d'exclusion audités ;
- ajustement affine sur les cinq premiers jours terminé ;
- tests de cohérence théorique et algorithmique terminés ;
- 400 plans et graine `20260818` exportés (`site_blueprints.csv`,
  `protocol_parameters.json`) ;
- diagnostics de plausibilité exécutés (formes, volumes, `F_i`, `gamma_i`,
  charge, nombre minimal d'équipements par régime) ;
- chaque configuration entièrement déterminée par un fichier de paramètres ;
- campagne A : `max_T d_i = r q_i` et contrôle `d_i[h] <= q_i` ;
- campagne B : `q_i = max{D_H^max/(k r_H), max_H d_i}` et le même contrôle ;
- résultats pilotes reproduits à l'identique avec la liste de plans gelée.
