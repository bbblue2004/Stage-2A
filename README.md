# Partage opportuniste d'infrastructures RAN

Ce dépôt contient le code, les données locales, les résultats numériques et le
document LaTeX d'un travail de recherche sur le partage opportuniste
d'infrastructures radio entre opérateurs mobiles. Le modèle combine une
optimisation exacte de l'activation des équipements et une analyse en théorie
des jeux coopératifs.

## Organisation

```text
data/raw/                 Données de trafic et de puissance d'origine
figures/                  PDF produits par le pipeline et utilisés pour l'étude
paper/                    Article LaTeX et sections du document
results/                  Résultats et caches numériques reproductibles
src/core/                 Optimisation et allocations coopératives
src/data_processing/      Chargement, calibration et génération des sites
src/experiments/          Expériences correspondant à la partie 6
tests/                    Tests unitaires et contrôles numériques
requirements.txt          Versions des dépendances Python
```

Les seuls éléments locaux volumineux conservés sont utiles :

- `data/raw/radio_sites.csv`, qui contient les mesures sources ;
- `results/`, qui évite de recalculer les campagnes coûteuses ;
- `.venv/`, l'environnement Python local ;
- `paper/main.pdf`, la dernière compilation du rapport.

Ils sont ignorés par Git. Les résultats peuvent être reconstruits à partir du
CSV brut, mais le calcul complet de stabilité prend plusieurs minutes.

## Protocole numérique

Les données couvrent 3 825 identifiants radio du 20 au 26 mars 2023, soit
24 x 7 = 168 observations horaires par identifiant. Pour chaque antenne
admissible, une relation affine entre trafic et puissance active fournit la
puissance fixe et le coût variable.

La campagne utilise 400 plans gelés. Un plan fixe les antennes sources ; un
scénario leur applique ensuite des volumes, des profils, des équipements, des
capacités et un nombre d'opérateurs. Pour quatre opérateurs, un site virtuel
utilise une antenne de référence, quatre antennes donneuses de profils et
quatre antennes énergétiques distinctes.

Les principales hypothèses numériques sont :

- sept jours toujours traités ensemble ;
- fenêtre nocturne centrale de 0 h à 7 h ;
- trois taux de charge maximale : 0,80, 0,90 et 1,00 ;
- trois et quatre opérateurs pour l'étude des mécanismes ;
- optimisation horaire exacte de tous les ensembles de gardiens ;
- valeur de Shapley lorsqu'elle appartient au cœur, nucléole lorsqu'un autre
  partage stable est nécessaire.

Les capacités sont simulées : elles ne sont pas présentes dans les données.
Les sites multi-opérateurs sont eux aussi virtuels. Les résultats constituent
donc une étude semi-empirique et non une estimation directe d'un réseau réel.

## Installation

Sous PowerShell :

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Le fichier source attendu est :

```text
data/raw/radio_sites.csv
```

## Reproduction des résultats

La commande suivante exécute le pipeline dans l'ordre de ses dépendances :

```powershell
.venv\Scripts\python.exe -m src.experiments.reproduce_all
```

Elle lance successivement :

1. `power_calibration` : calibration énergétique et liste gelée des plans ;
2. `plan_walkthrough` : Figure 1 expliquant la construction d'un site ;
3. `instance_diagnostics` : contrôles de plausibilité du protocole ;
4. `operational_efficiency` : économies et politiques de la section 6.3 ;
5. `coalition_stability` : stabilité et partage de la section 6.4 ;
6. `threshold_mechanisms` : mécanismes étudiés dans la section 6.5 ;
7. `parameter_sensitivity` : analyse de sensibilité de la section 6.6.

Par défaut, chaque étape réutilise son cache si les données, les paramètres et
la version de l'algorithme correspondent au manifeste enregistré. Pour tout
recalculer :

```powershell
.venv\Scripts\python.exe -m src.experiments.reproduce_all --rebuild
```

Chaque module accepte aussi `--help` et peut être exécuté séparément, par
exemple :

```powershell
.venv\Scripts\python.exe -m src.experiments.operational_efficiency --help
```

## Figures conservées

Le rapport utilise six figures numériques :

```text
figures/protocol/plan_walkthrough.pdf
figures/power_calibration/representative_power_fit.pdf
figures/operational_efficiency/hourly_profiles.pdf
figures/operational_efficiency/operational_efficiency.pdf
figures/coalition_stability/coalition_stability.pdf
figures/coalition_stability/threshold_mechanisms.pdf
```

`figures/instance_diagnostics/protocol_diagnostics.pdf` est conservée comme
contrôle complémentaire de la construction des sites virtuels. Les scripts ne
produisent plus d'aperçus PNG ni de figures issues des anciennes campagnes à
un seul plan.

## Tests

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -q
```

Les tests couvrent la calibration, la génération des instances, l'allocation
du trafic, les fenêtres temporelles, la stabilité, le nucléole, les seuils de
capacité, la sensibilité et la validité des caches.

## Compilation du rapport

Depuis `paper/` :

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Les fichiers auxiliaires de compilation sont ignorés. Le document source est
réparti dans `paper/sections/` ; la partie 6 contient le protocole et les
résultats produits par les expériences ci-dessus.
