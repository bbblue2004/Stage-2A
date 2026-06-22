import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PROJECT_ROOT = Path(__file__).resolve().parents[2]
csv_path = PROJECT_ROOT / "data" / "raw" / "radio_sites.csv"
OUTPUT_DIR = PROJECT_ROOT / "figures" / "data_figures"

def _make_output_path(filename: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / filename

def find_column(fieldnames, keywords):
    """Find column name by keywords (case-insensitive)."""
    for keyword in keywords:
        for name in fieldnames:
            if keyword.lower() in name.lower():
                return name
    return None


def parse_datetime(value):
    """Parse datetime from various formats."""
    value = value.strip()
    if not value:
        raise ValueError('Empty date value')

    formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%d %H:%M',
        '%Y-%m-%d %H',
        '%Y/%m/%d %H:%M:%S',
        '%d/%m/%Y %H:%M:%S',
        '%Y%m%d%H%M%S',
        '%Y%m%d%H%M',
        '%Y-%m-%d',
        '%Y/%m/%d',
    ]

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue

    raise ValueError(f'Unsupported date format: {value!r}')


def hour_key(dt):
    """Convert datetime to hourly key."""
    return dt.replace(minute=0, second=0, microsecond=0)


def extract_data_for_id(csv_path, filter_value, heure_col, sys_nidt_col, power_col):
    """Extract and aggregate hourly power consumption data for a specific ID."""
    rows_by_hour = defaultdict(list)

    with open(csv_path, newline='', encoding='utf-8', errors='replace') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            if (sys_nidt_col and (row.get(sys_nidt_col) or '').strip()) != filter_value:
                continue

            heure_value = (heure_col and (row.get(heure_col) or '').strip()) or ''
            power_value = (power_col and (row.get(power_col) or '').strip()) or ''
            if not heure_value or not power_value:
                continue

            try:
                dt = parse_datetime(heure_value)
            except ValueError:
                continue

            try:
                power = float(power_value.replace(',', '.'))
            except ValueError:
                continue

            rows_by_hour[hour_key(dt)].append(power)

    return rows_by_hour


def plot_single_id(filter_value='00000001U6'):
    """Plot power consumption for a single ID."""
    with open(csv_path, newline='', encoding='utf-8', errors='replace') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        fieldnames = reader.fieldnames or []

    heure_col = find_column(fieldnames, ['heure'])
    sys_nidt_col = find_column(fieldnames, ['nidt'])
    power_col = find_column(fieldnames, ['power', 'consumption'])

    if not heure_col or not sys_nidt_col or not power_col:
        raise SystemExit(f'Missing required columns. Found columns: {fieldnames}')

    rows_by_hour = extract_data_for_id(csv_path, filter_value, heure_col, sys_nidt_col, power_col)

    if not rows_by_hour:
        print(f'No rows found for SYS.NIDT = {filter_value}')
        return

    hour_points = sorted(rows_by_hour)
    avg_power = [sum(values) / len(values) for values in (rows_by_hour[h] for h in hour_points)]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(hour_points, avg_power, marker='o', linestyle='-', label=filter_value)
    ax.set_title(f'AVERAGE_POWER_CONSUMPTION par heure pour {filter_value}')
    ax.set_xlabel('Date (année-mois-jour heure)')
    ax.set_ylabel('AVERAGE_POWER_CONSUMPTION')
    ax.grid(True)
    ax.legend()

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:00'))
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()

    output_path = _make_output_path('power_single_id.png')
    plt.savefig(output_path)
    print(f'Graph saved to {output_path.resolve()} ({len(hour_points)} hourly points)')
    # plt.show()


def plot_multiple_ids(num_ids=5):
    """Plot power consumption for the first N distinct IDs on the same graph."""
    with open(csv_path, newline='', encoding='utf-8', errors='replace') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        fieldnames = reader.fieldnames or []

    heure_col = find_column(fieldnames, ['heure'])
    sys_nidt_col = find_column(fieldnames, ['nidt'])
    power_col = find_column(fieldnames, ['power', 'consumption'])

    if not heure_col or not sys_nidt_col or not power_col:
        raise SystemExit(f'Missing required columns. Found columns: {fieldnames}')

    # Get first N distinct IDs
    distinct_ids = []
    with open(csv_path, newline='', encoding='utf-8', errors='replace') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            nidt = (row.get(sys_nidt_col) or '').strip()
            if nidt and nidt not in distinct_ids:
                distinct_ids.append(nidt)
            if len(distinct_ids) >= num_ids:
                break

    print(f'Plotting data for IDs: {distinct_ids}')

    fig, ax = plt.subplots(figsize=(14, 7))

    for filter_id in distinct_ids:
        rows_by_hour = extract_data_for_id(csv_path, filter_id, heure_col, sys_nidt_col, power_col)
        if not rows_by_hour:
            print(f'No data found for {filter_id}')
            continue

        hour_points = sorted(rows_by_hour)
        avg_power = [sum(values) / len(values) for values in (rows_by_hour[h] for h in hour_points)]
        ax.plot(hour_points, avg_power, marker='o', linestyle='-', label=filter_id)

    ax.set_title(f'AVERAGE_POWER_CONSUMPTION par heure (premiers {num_ids} IDs)')
    ax.set_xlabel('Date (année-mois-jour heure)')
    ax.set_ylabel('AVERAGE_POWER_CONSUMPTION')
    ax.grid(True)
    ax.legend()

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:00'))
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()

    output_path = _make_output_path('power_multi_ids.png')
    plt.savefig(output_path)
    print(f'Graph saved to {output_path.resolve()}')
    # plt.show()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        plot_multiple_ids(int(sys.argv[1]))
    else:
        plot_single_id('00000001U6')
