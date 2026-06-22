import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
csv_path = PROJECT_ROOT / "data" / "raw" / "radio_sites.csv"
OUTPUT_DIR = PROJECT_ROOT / "figures" / "data_figures"

def _make_output_path(filename: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / filename


def parse_datetime(value):
    value = value.strip()
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H', '%Y-%m-%d %H:%M', '%Y/%m/%d %H:%M:%S', '%d/%m/%Y %H:%M:%S'):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f'Unsupported date format: {value!r}')


def find_column(fieldnames, keywords):
    for keyword in keywords:
        for name in fieldnames:
            if keyword.lower() in name.lower():
                return name
    return None


def is_weekday(dt):
    return dt.weekday() < 5


def read_data(csv_path):
    with open(csv_path, newline='', encoding='utf-8', errors='replace') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        return list(reader), reader.fieldnames or []


def count_distinct_ids(rows, id_col):
    return len({(row.get(id_col) or '').strip() for row in rows if (row.get(id_col) or '').strip()})


def select_days(rows, date_col, day_start=20, day_end=24, max_days=5):
    days = []
    seen = set()
    for row in rows:
        raw = (row.get(date_col) or '').strip()
        if not raw:
            continue
        try:
            dt = parse_datetime(raw)
        except ValueError:
            continue
        if not (day_start <= dt.day <= day_end):
            continue
        if not is_weekday(dt):
            continue
        day = dt.date()
        if day not in seen:
            seen.add(day)
            days.append(day)
            if len(days) >= max_days:
                break
    return days


def compute_hourly_rho(rows, date_col, id_col, value_col, selected_days):
    max_by_id = {}
    for row in rows:
        raw = (row.get(date_col) or '').strip()
        if not raw:
            continue
        try:
            dt = parse_datetime(raw)
        except ValueError:
            continue
        if dt.date() not in selected_days:
            continue
        if not is_weekday(dt):
            continue
        site_id = (row.get(id_col) or '').strip()
        if not site_id:
            continue
        try:
            value = float((row.get(value_col) or '').replace(',', '.'))
        except ValueError:
            continue
        max_by_id[site_id] = max(max_by_id.get(site_id, 0.0), value)

    if not max_by_id:
        raise SystemExit('No valid ID values found for selected days')

    totals = defaultdict(list)
    for row in rows:
        raw = (row.get(date_col) or '').strip()
        if not raw:
            continue
        try:
            dt = parse_datetime(raw)
        except ValueError:
            continue
        if dt.date() not in selected_days:
            continue
        if not is_weekday(dt):
            continue
        site_id = (row.get(id_col) or '').strip()
        if not site_id or site_id not in max_by_id:
            continue
        try:
            value = float((row.get(value_col) or '').replace(',', '.'))
        except ValueError:
            continue
        max_value_for_id = max_by_id[site_id]
        if max_value_for_id <= 0:
            continue
        normalized_value = value / max_value_for_id
        totals[dt.hour].append(normalized_value)

    avg_by_hour = {}
    for hour in range(24):
        values = totals.get(hour, [])
        if values:
            avg_by_hour[hour] = sum(values) / len(values)
        else:
            avg_by_hour[hour] = 0.0

    if not avg_by_hour:
        raise SystemExit('No hourly values found for selected days')

    rho = [avg_by_hour[hour] for hour in range(24)]
    max_rho = max(rho)
    return list(range(24)), rho, avg_by_hour, max_rho


def plot_rho(hours, rho, selected_days):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(hours, rho, marker='o', linestyle='-', linewidth=2, color='blue')
    ax.set_xlim(0, 23)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(0, 24, 1))
    ax.set_xlabel('Heure du jour')
    ax.set_ylabel('ρ (normalisé entre 0 et 1)')
    ax.set_title(f'Fonction ρ horaire moyenne sur les jours {selected_days[0]} à {selected_days[-1]}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = _make_output_path('avg_daily_rho.png')
    plt.savefig(output_path, dpi=150)
    print(f'Graph saved to {output_path.resolve()}')
    # plt.show()


if __name__ == '__main__':
    rows, fieldnames = read_data(csv_path)
    date_col = find_column(fieldnames, ['HEURE', 'DATE', 'HEURE(PSDATE)'])
    id_col = find_column(fieldnames, ['SYS.NIDT', 'nidt'])
    value_col = find_column(fieldnames, ['dl_volume', 'pdcp', 'gbytes'])

    if not date_col or not id_col or not value_col:
        raise SystemExit(f'Missing required columns. Found columns: {fieldnames}')

    distinct_ids = count_distinct_ids(rows, id_col)
    print(f'Distinct IDs count: {distinct_ids}')

    selected_days = select_days(rows, date_col, day_start=20, day_end=24, max_days=5)
    if not selected_days:
        raise SystemExit('No valid selected days found in the 20-24 range')

    print(f'Selected days: {selected_days}')
    hours, rho, avg_by_hour, max_value = compute_hourly_rho(rows, date_col, id_col, value_col, selected_days)
    print('Hourly average values:')
    for hour, avg in zip(hours, [avg_by_hour[h] for h in hours]):
        print(f'  {hour:02d}:00 -> {avg:.4f}')

    plot_rho(hours, rho, selected_days)
