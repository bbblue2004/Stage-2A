# RAN sharing

Research code for the opportunistic sharing of radio access network
infrastructure. The repository combines exact operational optimisation,
cooperative-game analysis, a semi-empirical numerical campaign, and the
associated research article.

## Repository layout

```text
data/                  Local raw and processed measurements (not versioned)
docs/                  Model documentation and archived working notes
figures/               Current experiment figures used by the article
paper/                 LaTeX article and scientific protocol
results/               Reproducible numerical caches (not versioned)
src/core/              Optimisation and cooperative-game algorithms
src/data_processing/   Data loading, calibration and semi-empirical instance generation
src/experiments/       Reproducible experiments reported in Section 6
src/cli/               Exploratory command-line diagnostics
tests/                 Unit and numerical-consistency tests
archive/legacy/        Outputs from superseded protocols
```

The material under `archive/legacy/` is retained only for traceability. It is
not used by the current article or by the numerical pipeline.

## Environment

The supported local setup uses the repository's virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

All commands below deliberately use this interpreter. The exact dependency
versions are recorded in `requirements.txt`.

## Data

Place the semicolon-separated source file at:

```text
data/raw/radio_sites.csv
```

The source data are not versioned. A fresh clone can therefore run the
article's experiments only after this file has been supplied. Generated
results are cached under `results/`; deleting that directory does not remove
source data and only forces a later recomputation.

## Reproduce the numerical campaign

Run the four experiments in dependency order:

```powershell
.venv\Scripts\python.exe -m src.experiments.reproduce_all
```

The command is idempotent:

- a missing cache is computed;
- a valid cache is reused;
- `--rebuild` forces every stage to be recomputed.

Cache manifests identify inputs by SHA-256 content hashes rather than by
machine-specific paths or modification dates.

The individual stages are:

```powershell
# Section 6.2: affine power calibration and frozen site blueprints
.venv\Scripts\python.exe -m src.experiments.power_calibration

# Section 6.1 diagnostics: plausibility of the instance generator
.venv\Scripts\python.exe -m src.experiments.instance_diagnostics

# Section 6.3: operational efficiency
.venv\Scripts\python.exe -m src.experiments.operational_efficiency

# Section 6.4: coalition stability
.venv\Scripts\python.exe -m src.experiments.coalition_stability

# Section 6.5: parameter sensitivity
.venv\Scripts\python.exe -m src.experiments.parameter_sensitivity
```

The calibration stage accepts `--rebuild-cache`; the other stages accept
`--rebuild`. Every experiment also exposes `--help` and explicit input/output
directory options.

## Exploratory commands

These utilities are independent of the article's main numerical campaign:

```powershell
# Configurable single-site simulation
.venv\Scripts\python.exe -m src.cli.single_site_simulation --help

# Empty-core diagnostic over field antennas
.venv\Scripts\python.exe -m src.cli.empty_core_diagnostic --help

# Power-versus-traffic graph for one antenna
.venv\Scripts\python.exe -m src.cli.regression_plot --help
```

Diagnostic figures are written under `figures/diagnostics/` and are not used
by the article.

## Model and implementation

For a non-empty coalition `S`, the program enumerates feasible guardian sets
at every hour, allocates traffic greedily by increasing variable cost, and
computes `C_H*(S) = sum_h C_h*(S)`. The transferable-utility savings game is

```text
v_H(S) = sum_{i in S} C_H*({i}) - C_H*(S),   v_H(empty) = 0.
```

The exact policy with one guardian set fixed over the whole window is also
computed as a secondary operational benchmark, not as the game used for the
main stability analysis.

The implementation evaluates convexity, the core, balancedness, the Shapley
value, the least core and the nucleolus. The full notation and numerical
protocol are documented in `docs/model.md` and
`paper/NUMERICAL_PROTOCOL.md`.

## Tests

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -q
```

The tests cover operational allocation, time windows, power calibration,
instance generation, coalition stability, parameter sensitivity and cache
portability.

## Article

From `paper/`:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The LaTeX source is governed by `paper/AGENTS.md`. Numerical code changes must
not silently alter the model, notation or assumptions stated in the article.
