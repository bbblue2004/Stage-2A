# Protocole scientifique du jour 1 — révision du 5 août 2026

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

L'évaluation mesure successivement la précision prédictive du modèle de
puissance, les économies et la sélection des gardiens, l'effet du trafic et de
la capacité, puis la stabilité et le partage des économies.

## 4. Données et population d'étude

Informations acquises : données de terrain Orange, Île-de-France, comprenant
un volume descendant horaire en Go et une puissance moyenne horaire en W.

À compléter avant soumission : **[provenance exacte, dates d'acquisition,
technologie, périmètre de l'identifiant radio et conditions d'utilisation]**.

Les règles d'admissibilité seront fixées avant de compter les antennes :

- valeurs temporelles, trafic et puissance lisibles et non négatifs ;
- journées complètes de 24 observations ;
- nombre suffisant de journées complètes pour la validation hors échantillon ;
- trafic variable pour identifier une pente non contrainte ;
- statut explicite des ajustements à coefficients non positifs.

Le nombre `3 626` n'est pas une taille d'échantillon gelée. La campagne
principale utilisera toutes les antennes satisfaisant les règles ci-dessus et
rapportera séparément chaque motif d'exclusion. Les profils constants et les
ajustements non positifs resteront dans l'analyse des exclusions afin de
quantifier le biais de sélection.

## 5. Validation du modèle de puissance

La validation utilise les observations journalières brutes, sans réduire au
préalable la semaine à 24 moyennes horaires.

Pour chaque antenne admissible :

1. laisser successivement une journée complète hors apprentissage ;
2. ajuster sur les autres jours le modèle affine
   `P = F_tilde + gamma_tilde d` ;
3. prédire la journée laissée de côté ;
4. conserver R² d'apprentissage, MAE, RMSE et erreur normalisée de test ;
5. analyser les résidus selon l'heure et le niveau de trafic ;
6. comparer au modèle constant ;
7. comparer en robustesse la régression libre à une régression à coefficients
   non négatifs.

Le modèle quadratique ne sera ajouté que si les résidus présentent une
courbure systématique et si son gain hors échantillon est substantiel.

La veille est supposée consommer 0 W. Les coefficients sont donc

```text
F_i     = price_per_kWh / 1000 * F_tilde_i,
gamma_i = price_per_kWh / 1000 * gamma_tilde_i.
```

## 6. Construction semi-synthétique des opérateurs

L'unité primaire est un profil antenne-semaine conservant conjointement son
trafic mesuré et sa relation puissance--trafic.

Pour chaque antenne d'ancrage et chaque graine :

1. conserver son profil comme opérateur 1 ;
2. tirer sans remise trois antennes distinctes dans la population admissible ;
3. pour le scénario principal, tirer les partenaires dans des strates de
   trafic et de puissance comparables à l'ancre ;
4. aligner les heures civiles et utiliser les journées complètes communes ;
5. conserver pour chaque opérateur son profil observé et ses propres
   coefficients de puissance ;
6. qualifier explicitement le quadruplet de site virtuel semi-synthétique.

Le scénario principal associe quatre profils distincts et comparables, selon
les quartiles croisés du pic hebdomadaire et de `F_tilde`. Une strate trop
petite est élargie aux quartiles voisins. Un tirage sans restriction teste
l'hétérogénéité ; les profils perturbés d'une même ancre servent uniquement à
reproduire les résultats exploratoires.

La campagne utilise les **20 graines entières de 0 à 19**. Une graine fixe
complètement les tirages ; elle ne doit modifier aucune autre convention
expérimentale.

## 7. Capacité et satisfaction du trafic

La capacité physique n'est pas observable dans le fichier. Elle est donc un
paramètre de scénario et non une grandeur estimée :

```text
q_i = max_observed_traffic_i / rho_peak_i.
```

Le maximum est calculé sur toute la période disponible, et non seulement sur
la fenêtre de décision. La grille principale est
`rho_peak in {0.30, 0.50, 0.70, 0.90}`. La valeur historique `0.75` est
conservée seulement pour reproduire les résultats exploratoires.

Toute la demande doit être acheminée et aucune capacité ne peut être dépassée.
Cette contrainte exprime une satisfaction intégrale du trafic, mais ne permet
pas d'affirmer une QoS maximale en débit, latence ou couverture, qui ne sont
pas observés dans les données.

## 8. Fenêtre temporelle et gardiens fixes

La fenêtre est notée `H(h_start, h_end)` et ses bornes inclusives restent des
paramètres. Aucune conclusion principale n'est attachée par défaut à 1 h--6 h.
Les bornes finales seront choisies à partir du profil de trafic agrégé, avant
la campagne principale et indépendamment des résultats de cœur.

Politique principale : un même ensemble de gardiens reste actif pendant toute
la fenêtre ; l'allocation de trafic peut varier d'une heure à l'autre. Les
coûts, économies et transferts sont agrégés et réglés une fois par fenêtre.

La réoptimisation exacte indépendante à chaque heure est conservée comme
borne basse idéale. L'écart mesure le coût de la contrainte imposant les mêmes
gardiens sur toute la fenêtre. Aucun coût de commutation monétaire n'est introduit
sans donnée permettant de le calibrer.

## 9. Méthodes comparées

1. absence de partage ;
2. un seul opérateur actif, si sa capacité suffit sur toute la fenêtre ;
3. activation par capacité décroissante jusqu'à faisabilité ;
4. optimum exact avec gardiens fixes sur la fenêtre ;
5. optimum avec choix horaire des gardiens, utilisé comme borne basse.

Une baseline infaisable est déclarée comme telle ; aucune demande n'est
abandonnée pour la rendre artificiellement comparable.

## 10. Analyse de sensibilité structurée

La figure causale centrale est une grille

```text
rho_peak x traffic_multiplier
```

avec `traffic_multiplier` dans `{0.50, 0.75, 1.00, 1.25, 1.50, 2.00}`.
Pour une même grille, trois cartes sont produites :

- économie énergétique relative ;
- nombre de gardiens actifs sur la fenêtre ;
- fréquence de cœur vide et de Shapley hors du cœur.

La famille semi-synthétique est présentée comme test de robustesse autour de
cette figure. Le cas `n = 4` est principal ; `n = 2,...,6` est une extension
secondaire.

## 11. Métriques et inférence

Métriques opérationnelles : énergie et coût évités, économie relative, nombre
de gardiens, charges des gardiens, faisabilité des baselines et écart entre
gardiens fixes et choix horaire.

Métriques coalitionnelles : convexité, non-vacuité du cœur, Shapley dans le
cœur, excès maximal de Shapley, gap de Bondareva--Shapley, structure du
certificat et epsilon du moindre cœur normalisé par `v(N)`.

Métriques de partage : économie individuelle `z_i`, coût net `y_i`, transfert
`tau_i`, dispersion des économies et distance Shapley--nucléole.

Les distributions sont résumées par médiane, quartiles et quantiles 5 %--95 %.
Les intervalles bootstrap à 95 % sont regroupés par antenne d'ancrage afin de
ne pas traiter les graines d'un même site virtuel comme des observations
indépendantes.

## 12. Critères de gel avant campagne principale

- bornes `h_start` et `h_end` fixées et justifiées par le trafic, non par les
  résultats de stabilité ;
- règles d'admissibilité exécutées et motifs d'exclusion audités ;
- validation leave-one-day-out terminée ;
- listes des antennes, partenaires et graines exportées ;
- chaque configuration entièrement déterminée par un fichier de paramètres ;
- résultats pilotes reproduits à l'identique à graine fixée.
