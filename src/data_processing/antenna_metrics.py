"""Hourly traffic profile and direct power regression."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from . import data_loader

DEFAULT_DAYS = 5
DEFAULT_ELECTRICITY_PRICE_PER_KWH = 0.15


@dataclass(frozen=True)
class PowerRegression:
    gamma_tilde: float
    f_tilde: float
    r_squared: float


def load_antenna_profiles(
    antenna_id: str,
    num_days: int = DEFAULT_DAYS,
) -> tuple[list[float], list[float]]:
    """Average traffic and power by hour over the first ``num_days`` days."""
    rows = data_loader.extract_antenna_time_series(antenna_id)
    days = sorted({dt.date() for dt, _, _ in rows})[:num_days]
    if len(days) < num_days:
        raise ValueError(f"{antenna_id}: only {len(days)} complete days available")

    def profile(value_index: int) -> list[float]:
        cells: dict[tuple, list[float]] = defaultdict(list)
        for row in rows:
            dt = row[0]
            if dt.date() in days:
                cells[(dt.date(), dt.hour)].append(row[value_index])
        if any((day, hour) not in cells for day in days for hour in range(24)):
            raise ValueError(f"{antenna_id}: missing hourly observations")
        return [
            float(np.mean([np.mean(cells[(day, hour)]) for day in days]))
            for hour in range(24)
        ]

    return profile(1), profile(2)


def power_regression_from_profiles(
    traffic: list[float],
    power: list[float],
) -> PowerRegression:
    """Fit ``P_conso = F_tilde + gamma_tilde d``."""
    if len(traffic) != len(power) or len(traffic) < 2:
        raise ValueError("Traffic and power need the same non-trivial sample size")

    x, y = np.asarray(traffic, dtype=float), np.asarray(power, dtype=float)
    if np.ptp(x) <= 1e-15:
        raise ValueError("Traffic must vary to estimate a regression")

    gamma_tilde, f_tilde = np.polyfit(x, y, 1)
    residual = float(np.sum((y - (gamma_tilde * x + f_tilde)) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - residual / total if total > 1e-15 else 1.0
    return PowerRegression(float(gamma_tilde), float(f_tilde), r_squared)


def power_coefficients_to_cost(
    f_tilde: float,
    gamma_tilde: float,
    price_per_kwh: float = DEFAULT_ELECTRICITY_PRICE_PER_KWH,
) -> tuple[float, float]:
    """Convert W and W/GB into hourly monetary coefficients."""
    if price_per_kwh <= 0.0:
        raise ValueError("price_per_kwh must be positive")
    scale = price_per_kwh / 1000.0
    return f_tilde * scale, gamma_tilde * scale
