# Plan de la Section 6 — Évaluation numérique

Spécification de la Section 6, à figer avant toute modification du code et du
LaTeX. Les constats, l'historique et les options écartées sont dans
`comments.md`.

Les formules sont écrites en texte brut, sans LaTeX, pour rester lisibles dans
l'aperçu Markdown.

---

## 1. Décisions arrêtées

### 1.1 Données et calibration

- Période : **7 jours**, lundi 20 au dimanche 26 mars 2023, soit
**24 x 7 = 168 heures**
par antenne, 3 825 identifiants.
- **Ajustement unique sur les 7 jours**, utilisé partout. La relation
puissance–trafic et les profils horaires ne présentent pas de différence
exploitable entre les cinq premiers et les deux derniers jours.
- Pas de jeu de validation hors échantillon. Robustesse établie en une ligne :
écart médian entre estimations à 5 jours et à 7 jours.
- Les **résultats de simulation** traitent toujours les sept jours ensemble.

Conséquences à traiter dans le code :


| Élément                         | Avant    | Après                            |
| ------------------------------- | -------- | -------------------------------- |
| Jours utilisés                  | 5 ouvrés | **7**                            |
| Heures par antenne              | 120      | **168**                          |
| Nuits par site                  | 5        | **7**                            |
| Profils normalisés `s_j`        | 120 h    | 168 h                            |
| Échelle `mu_a`                  | 120 h    | 168 h                            |
| Pic `max_t d_i(t)` fixant `q_i` | 120 h    | 168 h                            |


Le cache de calibration doit être invalidé et le CSV relu. Toutes les
statistiques figées dans `protocol_parameters.json` sont à recalculer.

### 1.2 Générateur d'instances

```
phi_j(t) = d_j(t) / moyenne(d_j)                        profil normalisé, moyenne 1
phi_i(t) = (1 - lambda) * phi_a(t) + lambda * phi_ji(t) combinaison sans bruit
d_i(t)   = mu_a * alpha_i * phi_i(t)                    demande en Go
mu_a     = moyenne(d_a)                             échelle de la référence
```

- Les `d_i(t)` sont en **Go, à l'échelle réelle** de la référence. Aucune
renormalisation en sortie : l'hétérogénéité absolue entre sites est
conservée.
- `alpha` est de moyenne 1, donc le trafic total du site vaut `n * mu_a` quel
que soit le niveau d'hétérogénéité : changer de niveau **redistribue** le
volume sans changer le total.
- Trafic, forme et énergie proviennent d'**antennes disjointes** au sein d'un
site.
- Cette séparation identifie l'effet propre du trafic et celui de
l'équipement. Elle ne prétend pas reproduire la loi jointe réelle ; chaque
profil et chaque couple énergétique reste toutefois issu d'une antenne
admissible, et les capacités garantissent la faisabilité.
- Trois paquets énergétiques tirés une fois par site : `close` (même quartile
de puissance fixe `P_fixe`), `moderate` (tirage global), `distant` (une
antenne par quartile).
- Graine unique ; comparaisons **appariées** sur les mêmes plans.
- Le générateur reste paramétré par le nombre de plans. Le **mode à un seul
plan** est le cas particulier utilisé pour la mise au point. La figure de
protocole emploie le premier plan de la liste gelée. Aucune constante figée à
1.

### 1.3 Capacités

Une seule campagne.

```
max_t d_i(t) = r * q_i,      avec r dans {0,80 ; 0,90 ; 1,00}
```

- `r` est le **taux d'utilisation maximal supposé** sur la période observée.
Les capacités des antennes admissibles ne sont pas fournies par le jeu de
données.
- Ne pas écrire que `r = 0,80` laisse « 20 % de marge ».
- Propriété à énoncer : `d_i(t) <= r * q_i <= q_i`, donc **aucune heure n'est
jamais infaisable**.
- Le balayage des seuils vient de la **journée**, mais pas selon la règle
simple que j'avais écrite. Vérifié numériquement sur un plan réel :

```
max_h k_h = arrondi_sup(n * r)   uniquement si les opérateurs sont identiques
max_h k_h < arrondi_sup(n * r)   dès qu'il y a de l'hétérogénéité
```

  Deux causes cumulatives. Les pics ne sont plus simultanés, donc la demande de
  coalition à l'heure la plus chargée est inférieure à la somme des pics
  individuels. Et la couverture se fait en prenant les capacités par ordre
  décroissant, donc un opérateur très gros couvre à lui seul une grande part.

  Conséquence de conception : le scénario extrême « tout proche » est le
  **seul** qui atteigne `k_h = 4`. Il joue donc le rôle de la campagne B
  supprimée, sans rien ajouter à la grille. Table mesurée sur une référence,
  `max_h k_h` sur 168 heures :

```
   volume      forme     r=0.80  r=0.90  r=1.00
  proches    proches          4       4       4
  proches    modérés          3       3       4
  proches   distants          3       3       3
  modérés    proches          3       4       4
  modérés    modérés          2       3       3
  modérés   distants          2       3       3
 distants    proches          2       3       4
 distants    modérés          2       2       3
 distants   distants          2       2       3
```

- `r` ne change pas le trafic : `d_i(t)/q_i = r * d_i(t)/max_t d_i(t)`. La
génération se fait une fois, seule l'optimisation est répétée sur les trois
valeurs, et les heures dont les ensembles faisables ne changent pas peuvent
être mises en cache.

### 1.4 Fenêtre et nombre d'opérateurs

- Fenêtre centrale `H = [0 h, 7 h)`, **bornes configurables** simplement. Le
profil horaire dira si 7 heures est trop long.
- Nombre d'opérateurs `n = 3` et `n = 4` dans la campagne principale.

### 1.5 Additivité temporelle

```
C*_H(S) = somme sur h dans H de C*_h(S)      donc      v_H = somme des v_h
```

On calcule **une seule fois**, par site et par jour, la table des `2^n - 1`
coûts pour chacune des 24 heures. Toute fenêtre s'en déduit par sommation,
sans réoptimisation. Le profil horaire est la primitive.

**Exception** : la politique persistante n'est pas séparable et doit être
recalculée par fenêtre, d'où la nécessité de fixer `H` au préalable.

### 1.6 Notation `k_h`

```
k_h = plus petit cardinal |G|, pour G inclus dans N, tel que q(G) >= d^h(N)
```

C'est le nombre minimal d'équipements capables d'écouler la demande totale à
l'heure `h`. Il ne dépend que des capacités et de la demande, jamais des
coûts. Tout ensemble de gardiens faisable vérifie `|G*_h(N)| >= k_h` ; ne pas
confondre `k_h` avec `|G*_h(N)|`, qui minimise le coût.

**À introduire en Section 3**, avec les ensembles admissibles, et à réutiliser
en Section 6 sans redéfinition.

### 1.7 Cascade de diagnostic de stabilité


| Étage | Test                               | Coût                       | Conclusion si vrai                        |
| ----- | ---------------------------------- | -------------------------- | ----------------------------------------- |
| 0     | construire la table des `v_H(S)`   | `O(card(H) * n * 3^n)`     | prérequis                                 |
| 1     | `E_Sh,H(N) <= 0`                   | `O(2^n)`                   | Shapley dans le cœur **et** cœur non vide |
| 2     | `d^h(N) <= min_i q_i`              | `O(card(H) * n)`           | cœur non vide                             |
| 3     | `Delta_LOO,H(N) > 0`               | `O(n)`                     | cœur vide                                 |
| 4     | programme linéaire du moindre cœur | LP à `2^n - 2` contraintes | exact, donne `epsilon*`                   |


L'étage 1 répond **simultanément aux deux questions** dans les cas favorables.
Le programme linéaire du moindre cœur domine Bondareva–Shapley en information ;
ce dernier est gardé comme contrôle croisé sur un sous-échantillon.

---

## 2. Plan détaillé

### 6.1 Protocole expérimental

Cible : **une figure et trois tableaux dans le corps**, environ deux pages. Les
diagnostics de plausibilité et les contrôles croisés partent en annexe, avec
seulement deux ou trois chiffres rapatriés en prose.

**6.1.1 Observé, calibré, synthétique.** Tableau à trois colonnes.


| Statut      | Contenu                                                                  |
| ----------- | ------------------------------------------------------------------------ |
| Observé     | trafic et puissance horaires, 3 825 identifiants, 7 jours                |
| Calibré     | puissance fixe et pente, donc les coûts `F_i` et `gamma_i`               |
| Synthétique | l'association de `n` opérateurs sur un même site, et les capacités `q_i` |


Formulation à employer : *les capacités des antennes admissibles ne sont pas
fournies*.

**6.1.2 Un plan déroulé pas à pas.** Une figure en six panneaux sur le premier
plan de la liste gelée :

- (a1)--(a3) les quatre profils normalisés dans les cas proches,
  intermédiaires et différents ;
- (b) l'entrée réelle : série de 168 heures de l'antenne de référence, en Go ;
- (c) la sortie : les `n` séries de demande `d_i(t)` en Go, facteurs de taille
`alpha_i` annotés ;
- (d) la demande rapportée au pic et les capacités pour `r` dans
  `{0,80 ; 0,90 ; 1}`.

Une **antenne donneuse** est une antenne réelle dont on emprunte uniquement la
forme temporelle, jamais l'échelle ni l'énergie. Sans elle, les profils des
opérateurs d'un même site seraient tous homothétiques.

Tableau associé : identifiants de la référence, des donneuses et des antennes
énergétiques ; échelle `mu_a` ; vecteur `alpha` ; paramètre de mélange
`lambda` ; puis les triplets `(F_i, gamma_i, q_i)`. Le lecteur doit pouvoir
refaire le calcul à la main.

Insister sur les **trois rôles distincts** que peut jouer une antenne réelle,
et sur le fait qu'elle n'en joue jamais deux à la fois dans un même site :


| Rôle        | Ce qu'elle fournit                                           |
| ----------- | ------------------------------------------------------------ |
| Référence   | l'échelle `mu_a` et la composante commune des profils        |
| Donneuse    | la composante propre du profil d'un opérateur                 |
| Énergétique | la puissance fixe et la pente, donc `F_i` et `gamma_i`       |


Justification à écrire : dans les données, trafic moyen et puissance fixe sont
corrélés à 0,52. Si une même antenne fournissait volume et énergie, cette
liaison se propagerait et l'on ne pourrait plus faire varier le volume sans
faire varier l'équipement. La séparation rend les deux axes indépendants. La
configuration de référence limite le risque d'associations extrêmes en tirant
les quatre équipements dans le même quartile de puissance fixe.

**6.1.3 Axes d'hétérogénéité.** Cinq axes : volume (`alpha`, 4 niveaux), forme
(`lambda`, 3), équipement (paquets, 3), capacité (`r`, 3), nombre d'opérateurs
(`n`, 2). Tableau des axes et de leurs niveaux.

Le factoriel complet ferait 4 × 3 × 3 × 3 × 2 = 216 scénarios par plan.
Grille **en étoile** à la place : on fixe une configuration de référence — volumes
modérés, formes modérées, équipements du même quartile de puissance fixe,
`r = 0,90`, quatre opérateurs —
puis on parcourt chaque axe en laissant les quatre autres au centre, soit
1 + 3 + 2 + 2 + 2 + 1 = 11 scénarios, plus les deux extrêmes « tout proche » et
« tout distant » pour borner. **13 scénarios par plan**, tous appariés.

Limite à écrire explicitement : ce plan ne mesure **aucune interaction** entre
axes. Les deux extrêmes en sont le seul garde-fou.

> **À trancher** : conserver l'étoile seule, ou lui adjoindre un petit
> factoriel complet sur les deux axes volume et équipement. Voir la section 5,
> Questions ouvertes.

**6.1.4 Capacités.** Voir 1.3.

**6.1.5 Réplication et reproductibilité.** Nombre de plans, graine, ce qui est
tiré une fois par site et ce qui varie par scénario.

Le **contrôle demi-échantillon** répond à la question « le nombre de plans
est-il suffisant ? ». On coupe l'échantillon de plans en deux moitiés et on
recalcule les résultats sur chacune : si la fréquence de cœur vide vaut par
exemple 12,3 % contre 12,1 %, c'est réglé en une ligne de texte. Déjà
implémenté dans `coalition_stability.py`, clé `category_fractions_first_half_sites`.
Pas de tableau.

**6.1.6 Diagnostics de plausibilité. → Annexe.** L'objectif est de savoir quel
espace d'instances on simule avant d'interpréter la stabilité. Figure d'annexe,
synthétique contre réel : profil journalier **en écart** ; autocorrélation de
décalage 1 et corrélation entre opérateurs dans **deux panneaux séparés et
étiquetés** ; volumes et maxima ; distributions de `F_i` et `gamma_i`.

Ne pas confondre les deux corrélations, l'ancienne figure les mélangeait :


| Grandeur                      | Définition                                                        | Ce qu'elle teste                                                                              |
| ----------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Autocorrélation de décalage 1 | corrélation entre `d(t)` et `d(t+1)` sur une même série           | la régularité temporelle est préservée ; médiane réelle 0,82, qu'un bruit blanc ferait chuter |
| Corrélation entre opérateurs  | corrélation entre `d_i(t)` et `d_j(t)` au même instant, même site | à quel point les opérateurs se ressemblent ; c'est le réglage `lambda`                        |


Rapatrier dans le corps deux ou trois chiffres seulement, du type
« autocorrélation médiane 0,82 dans les données contre 0,81 dans les
instances ».

**6.1.7 Contrôles croisés. → Annexe.** Un paragraphe et un tableau d'écarts
maximaux. Même quantité calculée de deux façons indépendantes : glouton contre
programme linéaire ; énumération contre programme mixte ; test du cœur contre
Bondareva–Shapley ; nucléole appartenant au moindre cœur ; somme des transferts
nulle. Plus la reproduction numérique des deux contre-exemples analytiques de


la Section 5. Une phrase de renvoi dans le corps.

### 6.2 Adéquation du modèle de puissance

Volontairement courte.

**6.2.1 Ajustement et admissibilité.** Règles d'exclusion et effectifs par
motif. Tableau : coefficient de détermination, RMSE normalisée, part de
puissance fixe, médiane et quartiles. La RMSE normalisée est le critère
principal ; le coefficient de détermination n'est qu'un diagnostic secondaire,
car il s'effondre quand la puissance varie peu même lorsque l'erreur absolue
est minuscule.

**6.2.2 Structure du nuage.** Deux phrases. Les heures à trafic nul et
puissance positive forment un palier horizontal au niveau de la puissance
fixe, conforme au modèle. L'antenne médiane observe des trafics descendant
jusqu'à 1,9 % de son pic, donc l'ordonnée à l'origine n'est pas extrapolée loin
du support observé. Figure : nuage puissance contre trafic d'une antenne
représentative avec la droite ajustée ; les sept jours sont présentés ensemble.

**6.2.3 Robustesse.** Une ligne : écart médian entre estimations à 5 et
7 jours.

**6.2.4 Des coefficients mesurés aux coûts.** Passage de la puissance fixe et
de la pente aux coûts `F_i` et `gamma_i`. Veille nulle, invariance par prix
commun de l'électricité.

### 6.3 Efficacité opérationnelle

**6.3.1 Profil horaire sur 24 heures.** Pour chaque heure : économie relative,
soit `v_h(N)` divisé par la somme des coûts autonomes `C*_h({i})` ; économie
absolue ; nombre de gardiens retenus par l'optimum. Le seuil `k_h` et la
condition `d^h(N) <= min_i q_i` sont réservés à 6.5. **Figure centrale de
l'article.** Regrouper
les sept jours et utiliser des équipements tirés dans le même quartile de
puissance fixe.

**6.3.2 Choix de la fenêtre.** `H = [0 h, 7 h)`. Écrire explicitement que la
fenêtre n'est pas choisie sur le niveau d'économie observé, pour écarter tout
biais de sélection.

**6.3.3 Économies sur la fenêtre.** Absolues et relatives, par scénario de
capacité, sur les sept jours regroupés. Rapporter la médiane et la moitié
centrale des plans directement, sans rééchantillonnage.

**6.3.4 Effet du nombre d'opérateurs.** `n = 3` contre `n = 4`, sur plans
appariés. Cadre de lecture : la borne structurelle `(n - 1) / n`.

**6.3.5 Politiques de référence.** Sélection par capacité décroissante,
allocation proportionnelle à gardiens optimaux, optimum horaire. Pertes
appariées, exprimées en points d'économie.

**6.3.6 Coût de la persistance.** Écart entre politique persistante et optimum
horaire sur la fenêtre retenue, et nombre de changements de gardiens entre
heures consécutives.

**6.3.7 Ordre de grandeur annuel.** Énergie évitée par site et par nuit,
obtenue par moyenne directe des sept jours puis multipliée par 365.
Avertissement explicite : la **saisonnalité n'est pas observée**, il ne s'agit
que d'un ordre de grandeur.

**6.3.8 Coût de calcul.** Temps pratiques pour `n = 3` et `n = 4` : jeu complet
par site-jour, cascade de stabilité, campagne totale. Pas de courbe de
complexité. Seul rappel théorique : 65 couples coalition-gardiens à examiner
par heure lorsque `n = 4`.

### 6.4 Stabilité et répartition des économies

**6.4.1 Cascade de diagnostic.** Fraction des instances résolue et temps
consommé à chaque étage. Arrêter dès que Shapley appartient au cœur ; ne
calculer le nucléole que si Shapley est exclue d'un cœur non vide.

**6.4.2 Cœur vide, Shapley dans le cœur.** Les trois catégories exclusives,
par scénario de capacité et par nombre d'opérateurs. Fréquence de convexité.
Taux de détection du certificat « laisser un opérateur de côté » parmi les
cœurs vides, et finesse de la borne inférieure qu'il fournit sur `epsilon*`.
Conserver le tableau de répartition des trois cas. Classer aussi les instances
selon `v_h(N)` par boîtes à moustaches, en signalant en rouge le cœur vide,
donc la grande coalition instable.

**6.4.3 Amplitude de l'instabilité.** Rapport `epsilon* / v_H(N)` quand le cœur
est vide ; rapport `E_Sh,H(N) / v_H(N)` quand la valeur de Shapley est exclue ;
taille et composition des coalitions bloquantes.

**6.4.4 Partage et rupture.** Gain moyen par opérateur et par jour sous la
règle stable retenue : Shapley si elle appartient au cœur, nucléole seulement
si le cœur est non vide et Shapley en est exclue. Si le cœur est vide, faire
éclater la grande coalition selon un contrefactuel explicite et mesurer la
perte totale et individuelle. Ne pas conserver, dans la campagne multi-plans,
un tableau fondé sur l'identité des quatre opérateurs d'un seul plan ;
rapporter des distributions ou des rangs comparables entre plans.

### 6.5 Nombre minimal de gardiens et mécanisme

**6.5.1 Régime de très faible trafic.** Fréquence, heure par heure sur
24 heures, de la condition `d^h(N) <= min_i q_i`. Vérifier que le cœur est
toujours non vide quand la condition s'applique — c'est un test du théorème,
pas une observation. Comparer `n = 3` et `n = 4`, en donnant pour chacun le
nombre de jeux horaires, la fréquence globale et les heures favorables. La
condition locale plus faible de la Section 4 est mentionnée **en remarque**.

**6.5.2 Nombre minimal de gardiens.** Définir `k_h` comme le plus petit nombre
des équipements les plus capacitaires dont la capacité cumulée couvre le
trafic total. Tableau central : jeux ventilés par `n` et par `k_h`, avec les
effectifs, la fréquence conditionnelle de cœur vide et celle de Shapley dans
le cœur. Distinguer systématiquement fréquence conditionnelle et fréquence
globale, signaler les petits effectifs et ne pas attribuer à `k_h` un effet
causal. Expliquer les cœurs vides par les revendications incompatibles des
coalitions qui se recouvrent, en lien avec le contre-exemple théorique.

**6.5.3 Effet de l'homogénéité des coûts.** Variante à coûts identiques, tous
les `F_i` égaux et tous les `gamma_i` égaux : vérifier que le corollaire de
stabilité de Shapley sous homogénéité est satisfait à 100 %, puis mesurer à
quelle vitesse l'hétérogénéité le détruit. Hors du point homogène, présenter
une fréquence de 100 % comme une observation sur l'échantillon, non comme une
garantie qui contredirait le contre-exemple théorique.

**6.5.4 Cas représentatifs.** Deux ou trois instances détaillées : une stable,
une où Shapley est exclue, une à cœur vide si elle existe. Déplaçable en
annexe.

### 6.6 Sensibilité

Allégée. Ne conserver que ce qui n'est pas déjà un axe du protocole : niveau
global de trafic, multiplicateurs appliqués aux coûts fixes et variables,
veille non nulle, position et durée de la fenêtre. Classement des amplitudes,
valable uniquement sur les plages testées.

Facteur différé, à n'ajouter que si les résultats le justifient : spécification
du modèle de puissance, droite unique contre deux niveaux à pente commune. La
décision du 20 août est de conserver la droite unique, voir la section 3 de
`comments.md`. Le biais est connu et de sens prudent, les gains relatifs étant
plutôt sous-estimés, ce qui suffit pour un premier jeu de résultats.

### 6.7 Limites

Un paragraphe. Les instances ne permettent pas d'estimer la distribution réelle
des gains d'un accord entre MNO colocalisés. Les capacités sont
simulées. Une semaine, une saison, une région, un opérateur. Pas de
coût de commutation, pas de qualité de service au-delà de l'écoulement intégral
du trafic, pas de coûts de transaction ni de contraintes réglementaires.

---

## 3. Budget figures et tableaux

Budget recompté après la coupe de 6.1. Le précédent oubliait le tableau des
axes (6.1.3) et celui des contrôles croisés (6.1.7).


| Sous-section                                                | Figures | Tableaux |
| ----------------------------------------------------------- | ------- | -------- |
| 6.1 Protocole                                               | 1       | 3        |
| 6.2 Modèle de puissance                                     | 1       | 1        |
| 6.3 Efficacité                                              | 2       | 2        |
| 6.4 Stabilité et partage                                    | 1       | 2        |
| 6.5 Seuils                                                  | 1       | 2        |
| 6.6 Sensibilité                                             | —       | 1        |
| **Corps**                                                   | **6**   | **11**   |
| Annexe : diagnostics, contrôles croisés, cas représentatifs | 1       | 2        |
| **Total**                                                   | **7**   | **13**   |


Cible : environ 10 pages pour le corps, 6.1 tenant en deux pages.

Figures **non négociables** : le plan déroulé (6.1.2), le profil horaire sur
24 heures (6.3.1), la stabilité par seuil `k_h` (6.5.2).

---

## 4. Correspondance théorie ↔ mesures

Chaque sous-section empirique s'ouvre en nommant l'objet théorique qu'elle
mesure, avec son numéro d'équation ou de proposition.


| Sous-section | Objet mesuré                                                                                                                                                         |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 6.1          | demandes, capacités et coûts de la Section 3 ; contrainte `d_i^h <= q_i`                                                                                             |
| 6.2          | passage de la puissance fixe et de la pente mesurées aux coûts `F_i` et `gamma_i`                                                                                    |
| 6.3          | coût optimal horaire, additivité `C*_H(S) = somme des C*_h(S)`, optimalité du glouton, complexité de l'énumération, politique persistante                            |
| 6.3          | corollaire d'efficacité : la grande coalition domine toute partition                                                                                                 |
| 6.4          | équivalence entre `E_Sh,H(N) <= 0` et l'appartenance de Shapley au cœur ; Bondareva–Shapley ; certificat LOO et sa borne sur `epsilon*` ; somme nulle des transferts |
| 6.5          | théorème de capacité individuelle suffisante ; corollaire d'homogénéité ; contre-exemples à cœur vide et à Shapley exclue                                            |


Deux principes de rédaction. **Distinguer** la vérification d'un résultat
démontré de l'observation nouvelle sans garantie. Et **n'introduire aucune
notation** au-delà de celles déjà posées dans les Sections 3 à 5, la seule
addition étant le seuil `k_h`, défini en Section 3.

---

## 5. Questions ouvertes

- **Nombre de plans.** À trancher après la première exécution complète sur un
seul plan. Contrainte dure : les références sont tirées sans remise, donc au
plus 3 600 environ. Contrainte réelle : le coût de calcul, qui augmente d'un
facteur 3,4 en passant de 7 à 24 heures, et encore avec `n = 3`.
- **Grille en étoile ou étoile augmentée.** L'étoile seule coûte 14 scénarios
par plan mais ne mesure aucune interaction. L'ajout d'un factoriel complet
volume × équipement, soit 4 × 3 = 12 combinaisons, porterait le total à 24
scénarios par plan et permettrait de répondre à « l'hétérogénéité de volume
déstabilise-t-elle davantage quand les équipements sont eux aussi
dispersés ? ». C'est l'interaction la plus plausible du modèle, puisque cœur
vide et exclusion de Shapley viennent du désaccord entre qui apporte le
trafic et qui apporte l'équipement efficace. **En discussion.**
- **Durée de la fenêtre.** 7 heures est peut-être trop long. Le profil horaire
de 6.3.1 tranchera.
- **Niveau `coupled`.** À conserver comme contrôle de robustesse, ou à
abandonner si la section devient trop longue.

