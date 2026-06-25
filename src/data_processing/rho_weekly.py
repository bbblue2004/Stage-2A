"""
Plot normalized traffic (rho) time series by hour for one or more antennas.
Uses data_loader for CSV reading and processing.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from . import data_loader


def plot_traffic_time_series(ids, data, output_path=None):
    """Plot traffic (rho) time series for one or more antennas."""
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0', 'C1', 'C2', 'C3', 'C4'])

    for index, antenna_id in enumerate(ids):
        traffic_values = data[antenna_id]['traffic']
        datetimes = data[antenna_id]['datetime']

        if not traffic_values or not datetimes:
            print(f'No data found for {antenna_id}')
            continue

        if len(ids) == 1:
            plot_values = traffic_values
            ylabel = 'Traffic volume (GB)'
            title_suffix = ''
        else:
            max_value = max(traffic_values)
            if max_value > 0:
                plot_values = [value / max_value for value in traffic_values]
            else:
                plot_values = traffic_values
            ylabel = 'Normalized traffic (0 to 1)'
            title_suffix = ' (normalized)'

        color = colors[index % len(colors)]
        ax.plot(datetimes, plot_values, marker='o', linestyle='-', label=antenna_id, color=color)

    ax.set_title(f'Traffic volume by hour{title_suffix}')
    ax.set_xlabel('Date (year-month-day hour)')
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:00'))
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()

    if output_path is None:
        output_path = data_loader.make_output_path('traffic_single_antenna.png' if len(ids) == 1 else 'traffic_multiple_antennas.png')
    else:
        output_path = Path(output_path)
    fig.savefig(output_path)
    print(f'Graph saved to {output_path.resolve()}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot traffic volume for one or more antennas.')
    parser.add_argument('--ids', nargs='+', help='List of antenna IDs to plot.')
    parser.add_argument('num', nargs='?', type=int, default=1,
                        help='Number of first distinct antennas to plot when --ids is omitted.')
    parser.add_argument('--output', type=Path, help='Optional output filename.')
    args = parser.parse_args()

    ids, data = data_loader.extract_ids_data(ids=args.ids, n=args.num)
    plot_traffic_time_series(ids, data, output_path=args.output)
