# Protocole scientifique — révision du 17 août 2026

Date de gel : 4 août 2026.

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

Le nombre `3 626` n'est pas une taille d'échantillon gelée. La campagne
principale utilisera toutes les antennes satisfaisant les règles ci-dessus et
rapportera séparément chaque motif d'exclusion. Les profils constants et les
ajustements non positifs resteront dans l'analyse des exclusions afin de
quantifier le biais de sélection.

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

## 6. Construction semi-synthétique des opérateurs

La campagne principale utilise exactement **1 000 sites virtuels** de quatre
antennes distinctes. La graine pseudo-aléatoire globale `20260814` fixe la
liste complète des sites, qui est exportée avant la campagne et réutilisée à
l'identique pour les cinq nuits et les quatre scénarios de capacité. Une
antenne peut appartenir à plusieurs sites, mais jamais deux fois au même site.
Les 1 000 quadruplets sont distincts. Les quatre scénarios et les cinq nuits
forment donc 20 000 instances à partir de cette même population de sites, sans
énumération exhaustive de toutes les associations possibles.
Les quatre séries utilisent les mêmes 120 heures et conservent leurs propres
trafics et coefficients de puissance.

Les antennes sont classées puis divisées en quatre groupes de même taille selon
le trafic maximal et selon `P_fixed`. Les quatre membres d'un site sont tirés
dans le même groupe pour les deux critères. Le nombre de sites attribué à
chaque groupe croisé est proportionnel au nombre d'antennes qu'il contient ;
les arrondis sont répartis par la méthode des plus forts restes afin d'obtenir
exactement 1 000 sites. Tout groupe sollicité doit contenir au moins quatre
antennes, faute de quoi la construction s'arrête avec un diagnostic explicite.
La suffisance de l'échantillon est contrôlée après la campagne en comparant les
principaux agrégats sur les 500 premiers sites et sur les 1 000 sites. Cette
comparaison réutilise les résultats calculés et n'ajoute aucune simulation.

Les données préparées sont persistées dans `results/power_calibration/` :

- `calibrated_population.npz` contient les 120 observations, les coefficients,
  les diagnostics et les groupes de chaque antenne admissible ;
- `virtual_sites.csv` contient la liste gelée des 1 000 sites ;
- `manifest.json` enregistre la signature du fichier source, les paramètres et
  la version du cache.

Le fichier brut n'est relu que si ce cache manque, si sa version ou ses
paramètres changent, ou si la taille ou la date de modification de la source a
changé. L'option `--rebuild-cache` force explicitement sa reconstruction.

## 7. Capacité et satisfaction du trafic

La capacité physique n'est pas observable dans le fichier. Elle est donc un
paramètre de scénario, construit à partir du trafic maximal observé. Pour
chaque taux maximal cible `r in {0.70, 0.80, 0.90, 1.00}`, on fixe :

```text
q_i = max_observed_traffic_i / r,
donc max_observed_traffic_i = r * q_i.
```

Le maximum est calculé sur les 120 heures des cinq premiers jours, et non
seulement sur la fenêtre de décision. Le cas `r = 0.70` laisse une marge de 30 % au pic
observé ; le cas `r = 1.00` fixe la capacité exactement à ce pic. Le paramètre
`r` décrit un scénario de capacité et non une grandeur physique mesurée.

Dans la campagne principale, la demande observée reste inchangée : aucun
multiplicateur de trafic n'est appliqué. Avant chaque simulation, le code
vérifie directement `d_i[h] <= q_i` pour tout `i` et tout `h` de la fenêtre.
Une violation arrête le scénario ; le trafic n'est jamais tronqué et la
capacité n'est pas recalibrée pour rendre le cas artificiellement faisable.

Toute la demande doit être acheminée et aucune capacité ne peut être dépassée.
Cette contrainte exprime une satisfaction intégrale du trafic, mais ne permet
pas d'affirmer une QoS maximale en débit, latence ou couverture, qui ne sont
pas observés dans les données.

## 8. Fenêtre temporelle et reconfiguration horaire

La fenêtre principale est fixée à
`H = {0 h, 1 h, ..., 6 h}`, soit les sept créneaux horaires de l'intervalle
`[00:00, 07:00)`. Elle est identique pour tous les sites et tous les scénarios
de la campagne principale. L'analyse de sensibilité étudie séparément sa
position et sa durée, sans modifier ce réglage central.

Ce choix est un réglage de la campagne, et non une constante imposée par
l'implémentation. Le code reçoit les deux bornes de la fenêtre en paramètres
et accepte toute fenêtre horaire contiguë, y compris une fenêtre traversant
minuit. Modifier ces bornes ne doit changer ni la construction des sites et des
capacités, ni les méthodes d'optimisation et d'analyse coalitionnelle.

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

Le scénario central fixe `r = 0.80`, conserve le trafic et les coefficients
calibrés, suppose une veille nulle, utilise `[00:00, 07:00)` et réunit quatre
opérateurs. Les facteurs suivants sont modifiés un par un, sur les mêmes sites
et les mêmes jours :

| Facteur | Valeurs |
|---|---|
| marge de capacité | `r in {0.70, 0.80, 0.90, 1.00}` |
| trafic | `0.8 d`, `d`, `1.2 d`, capacités centrales inchangées |
| puissance fixe | `0.8 P_fixed`, `P_fixed`, `1.2 P_fixed` |
| pente variable | `0.8 slope`, `slope`, `1.2 slope` |
| puissance de veille | `0`, `0.05 P_fixed`, `0.10 P_fixed` |
| position d'une fenêtre de 7 h | `[22:00,05:00)`, `[00:00,07:00)`, `[02:00,09:00)` |
| durée depuis minuit | 5 h, 7 h, 9 h |
| nombre d'opérateurs | `2, 3, 4, 5, 6`, avec 1 000 sites stratifiés par taille |

Dans les lignes consacrées aux coûts, la variation de `P_fixed` représente
celle de `F_i` et la variation de `slope` celle de `gamma_i`, à prix de
l'électricité fixé. Pour la veille, le coût d'un équipement inactif est la
fraction indiquée de sa composante fixe ; les références autonomes et les
coûts coalitionnels utilisent la même convention.

La position de la fenêtre est comparée uniquement sur les nuits entièrement
observées pour les trois positions, de façon à conserver l'appariement. La
variation du nombre d'opérateurs utilise nécessairement de nouveaux sites,
construits avec la même population admissible, la même stratification et la
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
1 000 sites virtuels.

## 12. Critères de gel avant campagne principale

- fenêtre nocturne principale fixée à `[00:00, 07:00)` pour chaque jour ;
- règles d'admissibilité exécutées et motifs d'exclusion audités ;
- ajustement affine sur les cinq premiers jours terminé ;
- tests de cohérence théorique et algorithmique terminés ;
- liste des 1 000 sites et graine globale exportées ;
- chaque configuration entièrement déterminée par un fichier de paramètres ;
- chaque scénario vérifie `max_observed_traffic_i = r * q_i` et le contrôle
  direct `d_i[h] <= q_i` ;
- résultats pilotes reproduits à l'identique avec la liste de sites gelée.
