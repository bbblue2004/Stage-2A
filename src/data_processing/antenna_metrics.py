"""Antenna metrics derived from radio_sites.csv."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from . import data_loader

DEFAULT_WEEKDAYS = 5
DEFAULT_ELECTRICITY_PRICE_PER_KWH = 0.15


def _weekday_days(daily_groups: dict, max_days: int = DEFAULT_WEEKDAYS) -> list[tuple]:
    days = []
    for day in sorted(daily_groups):
        values = daily_groups[day]
        if values and data_loader.is_weekday(values[0][0]):
            days.append((day, values))
        if len(days) >= max_days:
            break
    return days


def load_antenna_profiles(
    antenna_id: str,
    max_weekdays: int = DEFAULT_WEEKDAYS,
) -> tuple[list[float], list[float]]:
    """Read CSV once; return (hourly_traffic, hourly_power) with 24 values each."""
    rows = data_loader.extract_antenna_time_series(antenna_id)
    daily_groups: dict = defaultdict(list)
    for dt, traffic, power in rows:
        daily_groups[dt.date()].append((dt, traffic, power))

    weekday_days = _weekday_days(daily_groups, max_weekdays)
    if not weekday_days:
        raise ValueError(f"No weekday data for antenna {antenna_id!r}")

    traffic_samples: dict[int, list[float]] = defaultdict(list)
    power_samples: dict[int, list[float]] = defaultdict(list)

    for _, values in weekday_days:
        traffic_by_hour: dict[int, list[float]] = defaultdict(list)
        power_by_hour: dict[int, list[float]] = defaultdict(list)
        for dt, traffic, power in values:
            traffic_by_hour[dt.hour].append(traffic)
            power_by_hour[dt.hour].append(power)
        for hour, samples in traffic_by_hour.items():
            traffic_samples[hour].append(float(np.mean(samples)))
        for hour, samples in power_by_hour.items():
            power_samples[hour].append(float(np.mean(samples)))

    traffic_profile = [
        float(np.mean(traffic_samples[h])) if traffic_samples.get(h) else 0.0
        for h in range(24)
    ]
    power_profile = [
        float(np.mean(power_samples[h])) if power_samples.get(h) else 0.0
        for h in range(24)
    ]
    return traffic_profile, power_profile


def compute_hourly_traffic_profile(
    antenna_id: str,
    max_weekdays: int = DEFAULT_WEEKDAYS,
) -> list[float]:
    return load_antenna_profiles(antenna_id, max_weekdays)[0]


def power_regression_from_profiles(
    traffic_profile: list[float],
    power_profile: list[float],
) -> tuple[float, float]:
    """Return (beta_tilde, K_tilde) from hourly traffic and power profiles."""
    rho = data_loader.compute_rho_from_traffic(traffic_profile)
    if len(rho) < 2:
        return 0.0, float(np.mean(power_profile)) if power_profile else 0.0
    beta_tilde, k_tilde = np.polyfit(rho, power_profile, 1)
    return float(beta_tilde), float(k_tilde)


def compute_power_regression_coefficients(
    antenna_id: str,
    max_weekdays: int = DEFAULT_WEEKDAYS,
) -> tuple[float, float]:
    traffic, power = load_antenna_profiles(antenna_id, max_weekdays)
    return power_regression_from_profiles(traffic, power)


def power_coefficients_to_cost(
    beta_tilde: float,
    k_tilde: float,
    duration_hours: float = 1.0,
    price_per_kwh: float = DEFAULT_ELECTRICITY_PRICE_PER_KWH,
) -> tuple[float, float]:
    scale = duration_hours * price_per_kwh / 1000.0
    return beta_tilde * scale, k_tilde * scale


def compute_antenna_metrics(
    antenna_id: str,
    max_weekdays: int = DEFAULT_WEEKDAYS,
    price_per_kwh: float = DEFAULT_ELECTRICITY_PRICE_PER_KWH,
) -> tuple[list[float], float, float, float]:
    traffic, power = load_antenna_profiles(antenna_id, max_weekdays)
    beta_tilde, k_tilde = power_regression_from_profiles(traffic, power)
    beta, k = power_coefficients_to_cost(beta_tilde, k_tilde, price_per_kwh=price_per_kwh)
    return traffic, beta, k, float(max(traffic))
