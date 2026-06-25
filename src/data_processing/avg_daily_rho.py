"""
Calculate and plot average daily rho (normalized traffic) by hour.
Uses data_loader for CSV reading and processing.
"""
from collections import defaultdict
import matplotlib.pyplot as plt

from . import data_loader


def compute_hourly_rho(rows, date_col, id_col, value_col, selected_days):
    """Compute average hourly normalized traffic (rho) for selected days."""
    max_by_id = {}
    for row in rows:
        raw = (row.get(date_col) or '').strip()
        if not raw:
            continue
        try:
            dt = data_loader.parse_datetime(raw)
        except ValueError:
            continue
        if dt.date() not in selected_days:
            continue
        if not data_loader.is_weekday(dt):
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
            dt = data_loader.parse_datetime(raw)
        except ValueError:
            continue
        if dt.date() not in selected_days:
            continue
        if not data_loader.is_weekday(dt):
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
    """Plot average hourly rho."""
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
    output_path = data_loader.make_output_path('avg_daily_rho.png')
    plt.savefig(output_path, dpi=150)
    print(f'Graph saved to {output_path.resolve()}')


if __name__ == '__main__':
    rows, fieldnames = data_loader.read_csv_data(data_loader.CSV_PATH)
    columns = data_loader.detect_columns(fieldnames)
    date_col = columns['heure']
    id_col = columns['antenna_id']
    value_col = columns['traffic']

    if not date_col or not id_col or not value_col:
        raise SystemExit(f'Missing required columns. Found columns: {fieldnames}')

    distinct_ids = data_loader.count_distinct_ids(rows, id_col)
    print(f'Distinct IDs count: {distinct_ids}')

    selected_days = data_loader.select_weekdays(rows, date_col, day_start=20, day_end=24, max_days=5)
    if not selected_days:
        raise SystemExit('No valid selected days found in the 20-24 range')

    print(f'Selected days: {selected_days}')
    hours, rho, avg_by_hour, max_value = compute_hourly_rho(rows, date_col, id_col, value_col, selected_days)
    print('Hourly average values:')
    for hour, avg in zip(hours, [avg_by_hour[h] for h in hours]):
        print(f'  {hour:02d}:00 -> {avg:.4f}')

    plot_rho(hours, rho, selected_days)
