# RAN sharing - cooperative cost simulation

The program implements the operational cost model and the cooperative savings
game developed in Sections 3--5 of the report:

- affine active-equipment cost `F_i + gamma_i t_i`;
- optimal traffic allocation and guardian selection for every coalition;
- minimal coalition cost `C*(S)`;
- savings game `v(S) = sum_{i in S} C_i^0 - C*(S)`;
- one feasible core allocation and the Bondareva--Shapley balancedness test;
- physical costs, final net costs, and budget-balanced internal transfers.

The least-core code is intentionally left for later work and is not called.

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
# Default: first antenna in the CSV
python main.py

# Select another antenna
python main.py --antenna-id 00007240W4

# Change the electricity price used to convert W into period costs
python main.py --price-per-kwh 0.18

# Generate only the requested regression graph
python plot_regression.py
```

The regression graph and console report use only
`P_conso = F_tilde + gamma_tilde d` and its coefficient of determination.

## Module map

| Module | Role |
|---|---|
| `generate_data.py` | 24-hour scenario, `q_i`, `F_i`, and `gamma_i` |
| `optimiser.py` | Operational cost `C(S,G,t)` and minimum `C*(S)` |
| `allocate.py` | Greedy traffic allocation at fixed guardians |
| `game.py` | Savings game, transfers, core, Bondareva--Shapley |
| `simulation.py` | Hourly games and daily aggregation |
| `reporting.py` | Console results and accounting checks |
| `data_processing/` | CSV loading, five-day averages, regressions, figures |
