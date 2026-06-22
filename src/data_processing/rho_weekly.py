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


def extract_ids_data(csv_path, ids=None, n=1):
    """Extract hourly DL_VOLUME_PDCP data for the requested IDs."""
    with open(csv_path, newline='', encoding='utf-8', errors='replace') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        fieldnames = reader.fieldnames or []

    heure_col = find_column(fieldnames, ['heure'])
    sys_nidt_col = find_column(fieldnames, ['nidt'])
    traffic_col = find_column(fieldnames, ['dl_volume', 'pdcp', 'gbytes'])

    if not heure_col or not sys_nidt_col or not traffic_col:
        raise SystemExit(f'Missing required columns. Found columns: {fieldnames}')

    if ids is None:
        ids = []
        with open(csv_path, newline='', encoding='utf-8', errors='replace') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            for row in reader:
                nidt = (row.get(sys_nidt_col) or '').strip()
                if nidt and nidt not in ids:
                    ids.append(nidt)
                if len(ids) >= n:
                    break

    if not ids:
        raise SystemExit('No IDs selected for plotting')

    print(f'Plotting data for IDs: {ids}')
    data = {nidt: [] for nidt in ids}

    with open(csv_path, newline='', encoding='utf-8', errors='replace') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            nidt = (row.get(sys_nidt_col) or '').strip()
            if nidt not in data:
                continue

            heure_value = (row.get(heure_col) or '').strip()
            data_value = (row.get(traffic_col) or '').strip()
            if not heure_value or not data_value:
                continue

            try:
                dt = parse_datetime(heure_value)
                value = float(data_value.replace(',', '.'))
            except ValueError:
                continue

            data[nidt].append((hour_key(dt), value))

    cleaned_data = {}
    for nidt, values in data.items():
        if not values:
            print(f'Warning: no valid data found for ID {nidt}, removing from plot')
            continue
        values.sort(key=lambda pair: pair[0])
        cleaned_data[nidt] = values

    if not cleaned_data:
        raise SystemExit('No valid data found for the selected IDs')

    return list(cleaned_data.keys()), cleaned_data


def plot_dl_volume_pdcp(ids, data, output_path=None):
    """Plot DL_VOLUME_PDCP time series for one or more IDs."""
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0', 'C1', 'C2', 'C3', 'C4'])

    for index, nidt in enumerate(ids):
        points = data[nidt]
        hour_points = [pair[0] for pair in points]
        values = [pair[1] for pair in points]

        if len(ids) == 1:
            plot_values = values
            ylabel = 'DL_VOLUME_PDCP (GBYTES)'
            title_suffix = ''
        else:
            max_value = max(values)
            if max_value > 0:
                plot_values = [value / max_value for value in values]
            else:
                plot_values = values
            ylabel = 'DL_VOLUME_PDCP normalisé (0 à 1)'
            title_suffix = ' normalisé'

        ax.plot(hour_points, plot_values, marker='o', linestyle='-', label=nidt, color=colors[index % len(colors)])

    ax.set_title(f'DL_VOLUME_PDCP_GBYTES par heure{title_suffix}')
    ax.set_xlabel('Date (année-mois-jour heure)')
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:00'))
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()

    if output_path is None:
        output_path = _make_output_path('rho_single_id.png' if len(ids) == 1 else 'rho_multiple_ids.png')
    else:
        output_path = Path(output_path)
    fig.savefig(output_path)
    print(f'Graph saved to {output_path.resolve()}')
    # plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot DL_VOLUME_PDCP for one or more IDs.')
    parser.add_argument('--ids', nargs='+', help='List of SYS.NIDT IDs to plot.')
    parser.add_argument('num', nargs='?', type=int, default=1,
                        help='Number of first distinct IDs to plot when --ids is omitted.')
    parser.add_argument('--output', type=Path, help='Optional output filename.')
    args = parser.parse_args()

    ids, data = extract_ids_data(csv_path, ids=args.ids, n=args.num)
    plot_dl_volume_pdcp(ids, data, output_path=args.output)
