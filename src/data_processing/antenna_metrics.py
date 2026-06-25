from collections import defaultdict

import numpy as np

from . import data_loader


def compute_antenna_metrics(antenna_id):
    """Compute hourly traffic, beta, K and epsilon for the requested antenna."""
    rows = data_loader.extract_antenna_time_series(antenna_id)

    daily_groups = defaultdict(list)
    for dt, traffic, power in rows:
        daily_groups[dt.date()].append((dt, traffic, power))

    if not daily_groups:
        raise SystemExit(f'No daily data available for antenna {antenna_id}')

    ordered_days = sorted(daily_groups.items())
    first_five_days = ordered_days[:5]

    hourly_traffic_by_hour = defaultdict(list)
    hourly_power_by_hour = defaultdict(list)

    for _, values in first_five_days:
        day_traffic_by_hour = defaultdict(list)
        day_power_by_hour = defaultdict(list)
        for dt, traffic, power in values:
            day_traffic_by_hour[dt.hour].append(traffic)
            day_power_by_hour[dt.hour].append(power)

        for hour in day_traffic_by_hour:
            hourly_traffic_by_hour[hour].append(float(np.mean(day_traffic_by_hour[hour])))
            hourly_power_by_hour[hour].append(float(np.mean(day_power_by_hour[hour])))

    if not hourly_traffic_by_hour:
        raise SystemExit(f'No hourly data available for antenna {antenna_id}')

    sorted_hours = sorted(hourly_traffic_by_hour)
    traffic_per_hour = [float(np.mean(hourly_traffic_by_hour[hour])) for hour in sorted_hours]
    power_per_hour = [float(np.mean(hourly_power_by_hour[hour])) for hour in sorted_hours]

    rho = data_loader.compute_rho_from_traffic(traffic_per_hour)
    epsilon = float(max(traffic_per_hour))

    if len(rho) < 2:
        beta = 0.0
        k = float(np.mean(power_per_hour))
    else:
        beta, k = np.polyfit(rho, power_per_hour, 1)

    return traffic_per_hour, float(beta), float(k), epsilon


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compute antenna metrics for one ID.')
    parser.add_argument('antenna_id', help='SYS.NIDT antenna identifier')
    args = parser.parse_args()

    traffic, beta, k, epsilon = compute_antenna_metrics(args.antenna_id)
    print(f'antenna_id={args.antenna_id}')
    print(f'traffic={traffic}')
    print(f'beta={beta}')
    print(f'K={k}')
    print(f'epsilon={epsilon}')
