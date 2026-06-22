import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

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


def extract_ids_data(csv_path, ids=None, n=1):
    """Extract traffic volume and power consumption for the requested IDs."""
    with open(csv_path, newline='', encoding='utf-8', errors='replace') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        fieldnames = reader.fieldnames or []

    heure_col = find_column(fieldnames, ['heure'])
    sys_nidt_col = find_column(fieldnames, ['nidt'])
    traffic_col = find_column(fieldnames, ['dl_volume', 'pdcp', 'gbytes'])
    power_col = find_column(fieldnames, ['power', 'consumption'])

    if not heure_col or not sys_nidt_col or not traffic_col or not power_col:
        raise SystemExit(f'Missing required columns. Found columns: {fieldnames}')

    if not ids:
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

    print(f'Processing IDs: {ids}')

    data = {nidt: {'traffic': [], 'power': []} for nidt in ids}
    with open(csv_path, newline='', encoding='utf-8', errors='replace') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            nidt = (row.get(sys_nidt_col) or '').strip()
            if nidt not in data:
                continue

            traffic_str = (row.get(traffic_col) or '').strip()
            power_str = (row.get(power_col) or '').strip()

            if not traffic_str or not power_str:
                continue

            try:
                traffic = float(traffic_str.replace(',', '.'))
                power = float(power_str.replace(',', '.'))
                data[nidt]['traffic'].append(traffic)
                data[nidt]['power'].append(power)
            except ValueError:
                continue

    for nidt in list(data):
        if not data[nidt]['traffic'] or not data[nidt]['power']:
            print(f'Warning: no valid data found for ID {nidt}, removing from plot')
            data.pop(nidt)

    if not data:
        raise SystemExit('No valid data found for the selected IDs')

    return list(data.keys()), data


def plot_power_vs_normalized_traffic(ids, data, output_path=None):
    """Plot power consumption vs normalized traffic volume for one or more IDs."""
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0', 'C1', 'C2', 'C3', 'C4'])
    max_power = 0

    for index, nidt in enumerate(ids):
        traffic_values = data[nidt]['traffic']
        power_values = data[nidt]['power']
        if not traffic_values or not power_values:
            continue

        max_power = max(max_power, max(power_values))
        max_traffic = max(traffic_values)
        normalized_traffic = [t / max_traffic for t in traffic_values]

        color = colors[index % len(colors)]
        ax.scatter(
            normalized_traffic,
            power_values,
            alpha=0.7,
            s=50,
            color=color,
            label=f'ID {nidt} (data points)'
        )

        if len(normalized_traffic) >= 2:
            coeffs = np.polyfit(normalized_traffic, power_values, 1)
            beta, k = coeffs
            predicted_values = [beta * rho + k for rho in normalized_traffic]
            ss_res = sum((y - y_pred) ** 2 for y, y_pred in zip(power_values, predicted_values))
            ss_tot = sum((y - np.mean(power_values)) ** 2 for y in power_values)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            mean_x = np.mean(normalized_traffic)
            mean_y = np.mean(power_values)
            numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(normalized_traffic, power_values))
            denominator = np.sqrt(sum((x - mean_x) ** 2 for x in normalized_traffic) * sum((y - mean_y) ** 2 for y in power_values))
            correlation = numerator / denominator if denominator > 0 else 0

            rho_line = np.linspace(0, 1, 100)
            e_line = beta * rho_line + k
            ax.plot(rho_line, e_line, color=color, linewidth=2.5, linestyle='-',
                   label=f'ID {nidt} fit: E = {beta:.4f}·ρ + {k:.4f} (R² = {r_squared:.4f})')

            print(f'ID {nidt}: E = {beta:.4f} * ρ + {k:.4f}')
            print(f'  R² = {r_squared:.6f}, Correlation = {correlation:.6f}')
            print(f'  Points: {len(normalized_traffic)}, Power range: [{min(power_values):.2f}, {max(power_values):.2f}]')
            residuals = [y - y_pred for y, y_pred in zip(power_values, predicted_values)]
            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals)
            print(f'  Residuals: mean = {mean_residual:.6f}, std = {std_residual:.6f}')

    if max_power <= 0:
        raise SystemExit('Unable to determine maximum energy value for axis scaling')

    ax.set_xlabel('Normalized Traffic Volume ρ (0 to 1)')
    ax.set_ylabel('AVERAGE_POWER_CONSUMPTION E (W)')
    ax.set_ylim(0, max_power * 1.1)
    ax.set_title(f'Power Consumption vs Normalized DL_VOLUME_PDCP for IDs {ids}')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    if output_path is None:
        output_path = _make_output_path('power_vs_rho_single_id.png' if len(ids) == 1 else 'power_vs_rho_multi_id.png')
    else:
        output_path = Path(output_path)
    plt.savefig(output_path, dpi=150)
    print(f'Graph saved to {output_path.resolve()}')
    # plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot power vs normalized traffic for one or more IDs.')
    parser.add_argument('--ids', nargs='+', help='List of SYS.NIDT IDs to plot.')
    parser.add_argument('num', nargs='?', type=int, default=1,
                        help='Number of first distinct IDs to plot when --ids is omitted.')
    parser.add_argument('--output', type=Path, help='Optional output filename.')
    args = parser.parse_args()

    ids, data = extract_ids_data(csv_path, ids=args.ids, n=args.num)
    plot_power_vs_normalized_traffic(ids, data, output_path=args.output)
