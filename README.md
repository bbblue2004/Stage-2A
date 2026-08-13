# RAN sharing - cooperative cost simulation

The program implements the operational cost model and the cooperative savings
game developed in Sections 3--5 of the report:

- affine active-equipment cost `F_i + gamma_i t_i`;
- optimal traffic allocation and guardian selection for every coalition;
- minimal coalition cost `C*(S)`;
- savings game `v(S) = sum_{i in S} C_i^0 - C*(S)`;
- convexity, core and Bondareva--Shapley tests;
- Shapley value, Shapley projection, least core and nucleolus;
- physical costs, final net costs, and budget-balanced internal transfers.

The feasibility LP is only used to test the core. By default, the final
allocation is Shapley when it belongs to the core and the nucleolus otherwise.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Place `radio_sites.csv` in `data/raw/` (semicolon-separated). By default, the
first antenna encountered in the CSV is selected. Its traffic and power
profiles are the hourly means over its first five available calendar days. If
the file cannot be used, the simulation falls back to a hard-coded profile.

To generate the compact working CSV (10 antennas, 7 days, 24 rows per day):

```bash
python -c "from src.data_processing.data_loader import create_compact_csv; print(create_compact_csv())"
```

Subsequent runs automatically use `data/processed/radio_sites_10x7.csv`.

## Run

```bash
# Default: first antenna, five-day hourly-average mode, hours 01:00--06:00
python main.py

# First antenna, with days 1--4 assigned to operators 1--4
python main.py 1 --traffic-mode daily

# Third antenna, averaged mode
python main.py 3

# Third antenna, full-day evaluation
python main.py 3 --hours 0 23

# An overnight window is also accepted
python main.py 3 --hours 22 6

# Fourth antenna, with its seven daily traffic profiles
python main.py 4 --plot-weekly-traffic

# Change the electricity price used to convert W into period costs
python main.py 1 --price-per-kwh 0.18

# Force the nucleolus even when Shapley belongs to the core
python main.py 1 --allocation-priority robustness

# Change the acceptable relative least-core instability threshold
python main.py 1 --max-instability-ratio 0.02

# Generate only the requested regression graph
python plot_regression.py

# Count empty cores among all antennas, by default over hours 01:00--06:00
python check_empty_cores.py

# Restrict the calculation to the first 50 antennas and change the window
python check_empty_cores.py --count 50 --hours 0 23
```

The bounds passed to `--hours START END` are inclusive. The selected window
only changes the games that are evaluated and aggregated. Operator capacities
`q_i` are always computed from the maximum traffic over the full 24-hour
profile.

## Allocation rule

The program always computes Shapley, its core membership, the least core and
the nucleolus. By default, `--allocation-priority contribution` selects
Shapley when it belongs to the core and the nucleolus otherwise. The optional
`--allocation-priority robustness` always selects the nucleolus. The
normalized Euclidean projection of Shapley on the core, or on the optimal
least core when the core is empty, remains a diagnostic candidate.
An empty-core allocation is flagged as operationally unacceptable when its
relative least-core epsilon exceeds `--max-instability-ratio`.

The regression graph and console report use
`P_conso = F_tilde + gamma_tilde d`, with zero sleep power.

## Module map

| Module | Role |
|---|---|
| `generate_data.py` | 24-hour scenario, `q_i`, `F_i`, and `gamma_i` |
| `optimiser.py` | Operational cost `C(S,G,t)` and minimum `C*(S)` |
| `allocate.py` | Greedy traffic allocation at fixed guardians |
| `game.py` | Convexity, Shapley, core, least core and nucleolus |
| `simulation.py` | Hourly games and selected-period aggregation |
| `time_window.py` | Inclusive hour-window validation and formatting |
| `reporting.py` | Console results and accounting checks |
| `data_processing/` | CSV loading, five-day averages, regressions, figures |
