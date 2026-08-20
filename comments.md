# Annexe de travail — constats, historique, commentaires

Ce fichier recueille tout ce qui n'est pas le plan lui-même. `plan.md` ne
contient que le plan. Comme dans `plan.md`, les formules sont en texte brut
pour rester lisibles dans l'aperçu Markdown.

---

## 1. Statut du rapport

Les résultats numériques présents dans les Sections 6.3 à 6.5 proviennent du
protocole précédent, celui des 1 000 sites formés de quatre antennes réelles
appariées. Ils sont **obsolètes** et ne doivent pas être cités. Aucun résultat
de partage ou de stabilité ne sera écrit tant que les campagnes n'auront pas
été relancées.

---

## 2. Constats empiriques préalables

Vérifications faites sur `data/raw/radio_sites.csv`.

### 2.1 Étendue des données

| Élément | Valeur |
|---|---|
| Période | lundi 20 → dimanche 26 mars 2023 |
| Jours | 7 (5 ouvrés, 2 de week-end) |
| Identifiants radio | 3 825 |
| Lignes par jour | 91 800, soit 3 825 × 24 |
| Heures par antenne | 168 |

Les campagnes précédentes n'utilisaient que les 5 premiers jours, donc
uniquement des jours ouvrés, sans que ce soit explicité nulle part.

### 2.2 Nuage puissance–trafic

Scripts : `src/cli/weekday_weekend_scatter.py`,
`src/cli/intercept_support_check.py`.
Figures : `figures/diagnostics/weekday_weekend_scatter*.png`.

| Grandeur | Valeur |
|---|---|
| Heures à puissance nulle, exclues | 37 094 sur 642 600, soit 5,8 % |
| Heures actives | 605 506 |
| Heures actives à trafic exactement nul | 6 788, soit 1,12 % |
| Rapport trafic actif minimal sur pic, médiane | 0,019 |
| Antennes dont ce rapport dépasse 0,10 | 5,3 % |
| Antennes dont le week-end abaisse le minimum | 19,4 % |

Constats retenus, sans les développer dans le rapport :

- il n'y a pas d'étalement vertical à trafic nul, mais un palier horizontal au
  niveau de la puissance fixe, conforme au modèle affine ;
- la puissance fixe n'est donc pas extrapolée loin du support observé ;
- semaine et week-end suivent la même droite, ce qui justifie une calibration
  unique sur 7 jours.

La réserve importante à ce tableau fait l'objet de la section 3 ci-dessous : le
nuage n'est pas une droite unique, il se scinde en deux niveaux selon l'heure.

---

## 3. Le régime nocturne à deux niveaux

C'est le constat empirique le plus important de la phase de calibration, et
celui qui limite le plus la fidélité du modèle affine. Il est documenté ici en
détail parce que la décision prise le 20 août est de **ne pas le corriger pour
l'instant**, et qu'il faudra pouvoir y revenir.

### 3.1 L'observation

Le nuage puissance–trafic de la majorité des antennes ne forme pas une droite
mais **deux paliers superposés**, un niveau bas nocturne et un niveau haut
diurne, séparés par un écart franc.

![Galerie d'antennes à deux niveaux](figures/diagnostics/two_level_gallery.png)

Sur ces dix antennes, réparties entre le 45e et le 99e centile d'amplitude
donc représentatives et non triées, le pointillé noir est la droite unique
ajustée sur les 168 heures : elle passe **entre** les deux nuages et n'en
décrit aucun. Les deux droites de couleur sont le même modèle affine doté d'un
niveau par état.

L'écart entre niveaux va de 81 à 679 W selon l'antenne, alors que la pente
commune reste comprise entre 8 et 37 W/Go. C'est bien le terme constant qui
porte la différence, pas le trafic.

### 3.2 Ce n'est pas un effet du faible trafic

L'objection naturelle est que la nuit le trafic est faible, donc la puissance
aussi, et qu'il n'y a là rien d'autre que le bas de la droite. Trois mesures
l'écartent.

**Test à trafic apparié.** En ne retenant que les heures de jour dont le trafic
reste dans la plage observée la nuit, la puissance diurne vaut encore
**1,43 fois** la puissance nocturne en médiane, soit un écart absolu médian de
448 W. C'est le cas pour 78,9 % des 3 573 antennes exploitables. À trafic égal,
l'équipement ne consomme pas la même chose selon l'heure.

**Extrapolation de la droite de jour.** Ajustée sur les seules heures diurnes
puis prolongée vers les faibles trafics nocturnes, la droite surestime la
puissance nocturne réellement mesurée de **30 % en médiane**, et de plus de
10 % pour 76 % des antennes. Les points de nuit ne sont pas sur le prolongement
de la droite de jour, ils sont nettement en dessous.

**Recouvrement des nuages.** Sur plusieurs antennes de la galerie, les deux
états couvrent la même plage de trafic. Sur `00000279W3`, les points de l'état
réduit montent jusqu'à 40 Go/h comme ceux de l'état plein, avec 632 W d'écart.

![Test à trafic apparié](figures/diagnostics/night_state_matched.png)

Le panneau (a) résume le mécanisme sur `00003663U1` :

| Heure | Trafic moyen | Puissance moyenne |
|---|---|---|
| 00 h | 0,445 Go/h | 245 W |
| 05 h | 0,089 Go/h | 229 W |
| 06 h | 0,521 Go/h | **392 W** |

À 0 h et à 6 h le trafic est pratiquement identique et la puissance diffère de
147 W. La bascule suit l'horloge, pas la charge.

### 3.3 Ce n'est pas non plus une mise en veille

L'équipement reste allumé et écoule du trafic pendant l'état bas. Le trafic
nocturne vaut **30,7 %** du trafic diurne en médiane, et seules 1,74 % des
heures nocturnes actives ont un trafic exactement nul. La vraie extinction
existe séparément dans les données, ce sont les heures à puissance nulle, déjà
écartées comme inactives.

Il y a donc trois niveaux dans les données — éteint, réduit, plein — alors que
le modèle des Sections 3 à 5 n'en connaît que deux, actif et en veille. L'état
réduit est un **état actif à consommation réduite**, vraisemblablement une
fonction d'économie d'énergie programmée du type désactivation de porteuses, de
bandes ou de branches MIMO. Aucune information de configuration n'accompagne
les données, donc cette interprétation reste une conjecture.

Point rassurant pour la modélisation : la dépendance au trafic **subsiste**
dans l'état réduit. La pente nocturne médiane vaut 21,0 W/Go contre 15,7 W/Go
le jour, et seules 2,7 % des antennes ont une pente nocturne négative. Le
phénomène est un décalage de niveau, pas une perte de la relation au trafic.

### 3.4 Quelles heures, et sur combien d'antennes

Détection sans hypothèse d'horaire : une pente commune sur la semaine, un
niveau par heure de la journée, puis séparation des 24 niveaux au plus grand
écart.

![Heures de l'état réduit](figures/diagnostics/night_state_hours.png)

| Grandeur | Valeur |
|---|---|
| Antennes exploitables | 3 621 |
| Antennes à deux niveaux, amplitude > 10 % de la puissance moyenne | 2 921, soit 80,7 % |
| Amplitude médiane | 30 % de la puissance moyenne |
| Durée médiane de l'état réduit | 6 h par jour |
| Plage d'un seul tenant | 98,3 % |
| Exactement 0 h – 6 h | 62,1 % |
| Contient au moins 1 h – 5 h | 94,8 % |
| Entièrement inclus dans 22 h – 8 h | 90,1 % |

Fréquence d'appartenance à l'état réduit, par heure :

| 0 h | 1 h – 5 h | 6 h | 7 h – 20 h | 21 h | 22 h | 23 h |
|---|---|---|---|---|---|---|
| 78,8 % | 95 à 97 % | 16,5 % | ≈ 1 % | 9,3 % | 15,0 % | 15,4 % |

Le cœur 1 h – 5 h est quasi universel. Ce qui varie, ce sont les bords : début
à 0 h pour 66 % des antennes, à 1 h pour 16 %, mais 15 % basculent dès
21 h – 23 h ; fin à 6 h pour 83 % et à 7 h pour 14 %. Le résidu de 1 % en pleine
journée est du bruit de classification, la méthode coupant toujours en deux.

**À retenir pour la fenêtre d'étude.** `H = [0 h, 7 h)` inclut l'heure 6, qui
est en état **plein** pour 83 % des antennes. La fenêtre chevauche donc la
transition.

### 3.5 Ce que coûte l'ajustement d'une droite unique

![Effet de la spécification](figures/diagnostics/night_regime_specification.png)

Trois spécifications comparées sur les 3 621 antennes : une droite sur les
168 heures, la même avec une indicatrice de nuit donc deux ordonnées à
l'origine et une pente commune, et une droite sur les seules heures de jour.

| | 168 h | Indicatrice | Jour seul |
|---|---|---|---|
| RMSE normalisée médiane | 8,5 % | 4,3 % | 4,1 % |
| `F` relatif à la spécification actuelle | 1 | × 1,23 | × 1,24 |
| `gamma` relatif à la spécification actuelle | 1 | × 0,65 | × 0,63 |
| Part fixe médiane | 71,3 % | 85,3 % | — |

La droite unique compense le décrochage nocturne en se redressant, et ce
redressement est imputé au trafic. Il en résulte que **`gamma` est surestimé
d'environ 54 % et `F` sous-estimé d'environ 23 %**. Ce sont les deux faces du
même artefact. Corriger la spécification divise l'erreur d'ajustement par deux.

La corrélation entre l'ampleur du décrochage nocturne et la RMSE normalisée
vaut `-0,87` : la qualité de l'ajustement du parc est essentiellement gouvernée
par ce seul effet. En particulier, les antennes peu chargées ne s'ajustent pas
moins bien que les autres ; la corrélation entre le logarithme du pic de trafic
et la RMSE vaut `+0,34`.

### 3.6 Direction du biais sur les résultats

Ce qui gouverne l'intérêt d'éteindre un équipement est le rapport `F / gamma` :
on économise un coût fixe et on paie du trafic supplémentaire chez le gardien.
Pour `n` opérateurs identiques dont `k` restent gardiens, le gain relatif vaut

    (n - k) * (F / gamma) / ( n * (F / gamma) + D )

qui est strictement croissant en `F / gamma`. Or les capacités, donc `k`, ne
dépendent pas de `F` ni de `gamma` : la comparaison est propre.

| Spécification | `F / gamma` médian |
|---|---|
| 168 h, actuelle | 36,6 Go/h |
| Indicatrice, niveau nocturne | 53,8 Go/h |
| Indicatrice, niveau diurne | 74,7 Go/h |

**La calibration retenue sous-estime donc les gains relatifs du partage**, d'un
facteur qui va de 1,5 à 2 sur le rapport `F / gamma`. C'est une erreur dans le
sens prudent, ce qui est la bonne direction pour un premier jeu de résultats.

Sur les gains absolus, le sens dépend du niveau que l'on retiendrait : le
niveau nocturne vaut 0,93 fois le `F` actuel, le niveau diurne 1,23 fois. Sur
les indicateurs de stabilité, cœur vide et Shapley dans le cœur, la direction
n'est pas déterminée et il ne faut rien en conjecturer.

### 3.7 Décision du 20 août 2026

**On conserve le modèle affine unique ajusté sur l'ensemble des 168 heures.**
La priorité est d'obtenir un simulateur complet et fonctionnel ; le raffinement
de la calibration viendra ensuite si les résultats le justifient.

Conséquences assumées (documentées ici, **pas dans l'article pour l'instant**) :

- l'erreur d'ajustement médiane reste à 8,5 % au lieu de 4,3 % ;
- `F` est sous-estimé d'environ 23 % et `gamma` surestimé d'environ 54 % ;
- les gains relatifs rapportés sont vraisemblablement conservateurs.

La Section 6.2 mentionne le cran nocturne, conserve le modèle 1D, et tait
les chiffres de biais jusqu'à une éventuelle spécification à deux niveaux.

Si le raffinement devient nécessaire, la voie est balisée : ajouter une
indicatrice d'état réduit **détectée antenne par antenne** plutôt qu'imposée à
0 h – 6 h, ce qui capte les 38 % d'antennes dont les bords diffèrent. Le modèle
reste affine en trafic, donc toute la théorie des Sections 3 à 5 s'applique
sans modification. Resterait alors à trancher quel niveau injecter dans le
modèle pour la fenêtre nocturne, le niveau réduit décrivant ce que les
opérateurs consomment réellement, le niveau plein correspondant à l'hypothèse
théorique d'un équipement pleinement actif.

### 3.8 Réserve annexe : concavité

Même à l'intérieur du régime diurne, la relation est un peu concave : la pente
estimée sur la moitié haute du trafic vaut environ 0,51 fois celle estimée sur
la moitié basse. Le modèle affine reste une approximation, indépendamment du
problème des deux niveaux. L'ordre de grandeur de l'erreur résiduelle après
traitement du régime nocturne, 4 %, reste toutefois acceptable.

### 3.9 Question ouverte

Si un gardien doit écouler la nuit le trafic de quatre opérateurs, reste-t-il
dans la configuration réduite ou repasse-t-il en configuration pleine ? Les
données ne permettent pas de trancher, aucune antenne du jeu n'écoulant un tel
volume pendant ses heures réduites. À mentionner dans les limites de la
Section 6.7.

### 3.10 Scripts

| Script | Rôle |
|---|---|
| `src/cli/poor_fit_autopsy.py` | Autopsie d'une antenne mal ajustée, prévalence du décrochage |
| `src/cli/night_regime_specification.py` | Comparaison des trois spécifications, effet sur `F` et `gamma` |
| `src/cli/night_state_matched.py` | Test à trafic apparié, pentes nocturnes |
| `src/cli/night_state_hours.py` | Détection des heures de l'état réduit sans hypothèse d'horaire |
| `src/cli/two_level_gallery.py` | Galerie de dix antennes à deux niveaux |

---

## 4. Points écartés en cours de discussion

- **Campagne B** de stress des seuils : supprimée. Le balayage du seuil `k_h`
  est obtenu naturellement par la journée, la demande de pointe valant environ
  `r` fois la somme des capacités.
- **Validation hors échantillon** sur le week-end : abandonnée au profit d'un
  ajustement unique sur 7 jours.
- **Courbe de complexité** en fonction du nombre d'opérateurs : abandonnée.
  Seuls des temps pratiques pour `n = 3` et `n = 4` seront reportés.
- **Étude du biais d'activation partielle** : sans objet, puisque l'étalement
  vertical n'existe pas dans les données.
- **Condition locale de sous-additivité stricte** : conservée en simple
  remarque, non développée.

---

## 5. Journal des révisions

- **19 août 2026** — Création du plan. Suppression de la campagne B. Passage à
  7 jours. Grille `r` dans `{0,70 ; 0,80 ; 0,90 ; 1,00}` rétablie. Ajout du cas
  à trois opérateurs. Extension à 24 heures via l'additivité temporelle.
  Cascade de diagnostic réordonnée autour du test `E_Sh,H(N) <= 0`.
- **19 août 2026, rév. 2** — Intégration du week-end confirmée après contrôle
  visuel des nuages puissance–trafic.
- **19 août 2026, rév. 3** — Séparation en `plan.md`, qui ne contient que le
  plan, et `comments.md`, qui reçoit les constats et l'historique.
- **19 août 2026, rév. 4** — Toutes les formules passées en texte brut, sans
  LaTeX, pour l'aperçu Markdown.
- **19 août 2026, rév. 5** — Mélange convexe abandonné au profit de la
  sélection de donneuses par rang de corrélation. Mesuré : le mélange à
  `lambda = 0,35` écrasait le rapport pic sur moyenne de 6,3 %, de 2,743 à
  2,569.
- **19 août 2026, rév. 6** — Sections 6.1 et 6.2 rédigées. `k_h` introduit en
  Section 3 (Définition du nombre minimal d'équipements). Annexe des
  diagnostics et contrôles croisés créée. Chiffres de 6.2 recalculés sur
  7 jours : 3 623 antennes admissibles, RMSE normalisée médiane 8,5 %,
  R² médian 0,74, part fixe médiane 71,7 %.
- **20 août 2026, rév. 7** — Régime nocturne à deux niveaux caractérisé, voir
  la section 3. Décision de conserver le modèle affine unique sur 168 heures.
  `plan_2.md` renommé `comments.md`.

---

## 6. Reste à faire côté code

Le texte décrit le protocole cible ; `instance_generator.py` implémente encore
l'ancienne version. À reprendre :

- remplacer le mélange convexe par la sélection de donneuses au rang de
  corrélation, et supprimer `SHAPE_LAMBDA` ;
- ~~passer la calibration à 7 jours par défaut et invalider les caches~~ fait
  le 20 août. `DEFAULT_NUM_DAYS` vaut 7 et
  `results/power_calibration/calibrated_population.npz` a été reconstruit :
  3 623 antennes, `P_fixe` médian 906,7 W, pente médiane 25,95 W/Go. Le cache
  à 5 jours de `data/processed/` reste utile au seul contrôle de stabilité
  5 jours contre 7 jours de la Section 6.2 ;
- supprimer la campagne B du générateur, de `protocol_io.py`, de
  `instance_diagnostics.py` et des tests ;
- ajouter le niveau énergétique `coupled` ;
- ajouter `n = 3` et la grille en étoile à 14 scénarios ;
- produire la figure de l'annexe des diagnostics.
