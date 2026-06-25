"""
Plot power consumption vs normalized traffic (rho).
Uses data_loader for CSV reading and processing.
"""
from pathlib import Path
import matplotlib.pyplot as plt

from . import data_loader


def get_energy_traffic_rho_for_id(id_value):
    """Return (energy, traffic, rho) series for a given antenna ID."""
    ids, data = data_loader.extract_ids_data(ids=[id_value], include_power=True)
    series = data[id_value]
    traffic = series['traffic']
    energy = series['power']
    if traffic:
        max_traffic = max(traffic)
        rho = [t / max_traffic for t in traffic] if max_traffic > 0 else [0.0 for _ in traffic]
    else:
        rho = []
    return energy, traffic, rho


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
            alpha=0.6,
            s=20,
            label=nidt,
            color=color,
        )

    ax.set_xlabel('ρ (normalized traffic, 0-1)')
    ax.set_ylabel('Power consumption (W)')
    ax.set_title(f'Power vs Normalized Traffic ({len(ids)} IDs)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if output_path is None:
        output_path = data_loader.make_output_path('power_vs_rho_scatter.png')
    else:
        output_path = Path(output_path)
    fig.savefig(output_path, dpi=150)
    print(f'Graph saved to {output_path.resolve()}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot power vs normalized traffic for one or more IDs.')
    parser.add_argument('--ids', nargs='+', help='List of SYS.NIDT IDs to plot.')
    parser.add_argument('num', nargs='?', type=int, default=1,
                        help='Number of first distinct IDs to plot when --ids is omitted.')
    parser.add_argument('--output', type=Path, help='Optional output filename.')
    args = parser.parse_args()

    ids, data = data_loader.extract_ids_data(ids=args.ids, n=args.num, include_power=True)
    plot_power_vs_normalized_traffic(ids, data, output_path=args.output)
