"""
Plot power consumption time series by hour for one or more antennas.
Uses data_loader for CSV reading and processing.
"""
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from . import data_loader


def extract_antenna_power_data(antenna_id):
    """Extract hourly power consumption data for a specific antenna ID."""
    rows_by_hour = data_loader.extract_antenna_power_data(antenna_id)
    return rows_by_hour


def plot_single_antenna(antenna_id='00000001U6'):
    """Plot power consumption for a single antenna."""
    rows_by_hour = extract_antenna_power_data(antenna_id)

    if not rows_by_hour:
        print(f'No rows found for antenna = {antenna_id}')
        return

    hour_points = sorted(rows_by_hour)
    avg_power = [sum(values) / len(values) for values in (rows_by_hour[h] for h in hour_points)]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(hour_points, avg_power, marker='o', linestyle='-', label=antenna_id)
    ax.set_title(f'Power consumption by hour for {antenna_id}')
    ax.set_xlabel('Date (year-month-day hour)')
    ax.set_ylabel('Average power consumption (W)')
    ax.grid(True)
    ax.legend()

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:00'))
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()

    output_path = data_loader.make_output_path('power_single_antenna.png')
    plt.savefig(output_path)
    print(f'Graph saved to {output_path.resolve()} ({len(hour_points)} hourly points)')


def plot_multiple_antennas(num_antennas=5):
    """Plot power consumption for the first N distinct antennas on the same graph."""
    ids, data = data_loader.extract_ids_data(n=num_antennas, include_power=True)

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0', 'C1', 'C2', 'C3', 'C4'])

    for index, antenna_id in enumerate(ids):
        traffic_values = data[antenna_id]['traffic']
        power_values = data[antenna_id]['power']
        datetimes = data[antenna_id]['datetime']
        
        if not power_values or not datetimes:
            print(f'No data found for {antenna_id}')
            continue

        color = colors[index % len(colors)]
        ax.plot(datetimes, power_values, marker='o', linestyle='-', label=antenna_id, color=color)

    ax.set_title(f'Power consumption by hour ({num_antennas} antennas)')
    ax.set_xlabel('Date (year-month-day hour)')
    ax.set_ylabel('Power consumption (W)')
    ax.grid(True)
    ax.legend()

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:00'))
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()

    output_path = data_loader.make_output_path('power_multiple_antennas.png')
    plt.savefig(output_path)
    print(f'Graph saved to {output_path.resolve()}')


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        plot_multiple_antennas(int(sys.argv[1]))
    else:
        plot_single_antenna('00000001U6')

    print(f'Plotting data for IDs: {distinct_ids}')

    fig, ax = plt.subplots(figsize=(14, 7))

    for filter_id in distinct_ids:
        rows_by_hour = extract_data_for_id(csv_path, filter_id, heure_col, sys_nidt_col, power_col)
        if not rows_by_hour:
            print(f'No data found for {filter_id}')


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        plot_multiple_antennas(int(sys.argv[1]))
    else:
        plot_single_antenna('00000001U6')
