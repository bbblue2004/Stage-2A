# RAN sharing — cooperative game simulation

Simulates night-time RAN sharing among co-located mobile operators using cooperative game theory: optimal guardians, coalition value v*(S), traffic allocation (greedy / uniform in `allocate.py`), and profit redistribution (rules 1–3 + least-core LP).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Place `radio_sites.csv` in `data/raw/` (semicolon-separated). If the file is missing, the simulation falls back to a hard-coded weekday profile for site `00000001U6`.

## Run

```bash
# Daily cooperative gains (default antenna 00000001U6)
python main.py

# Another antenna (operator 1 profile + CSV-derived beta/K)
python main.py --antenna-id 00007240W4

# Custom electricity price for beta/K conversion (default 0.15 currency/kWh)
python main.py --price-per-kwh 0.18

# Exploratory figures from the CSV
python -m src.data_processing.figures --antenna-id 00000001U6
```

## Module map

| Module | Role |
|--------|------|
| `generate_data.py` | 24 h scenario (CSV or fallback), beta/K from power regression |
| `utility.py` | Single-operator profit v(A_i) |
| `optimiser.py` | Coalition utility v(s, l_s) and v*(s) |
| `allocate.py` | Traffic split: greedy and uniform-until-saturation |
| `profit.py` | Shapley, payoff rules 1–3, least-core LP |
| `simulation.py` | Day-long evaluation (hourly v*, guardians, least-core) |
| `reporting.py` | Console summary |
| `data_processing/` | CSV loading, antenna metrics, exploratory figures |

Theory: see `docs/doc_en.md`.
