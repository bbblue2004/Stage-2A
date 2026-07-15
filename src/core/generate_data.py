"""Scenario construction: real CSV profile or hard-coded fallback."""

from __future__ import annotations

from dataclasses import dataclass
import random

from src.data_processing.antenna_metrics import (
    DEFAULT_ELECTRICITY_PRICE_PER_KWH,
    load_antenna_profiles,
    power_coefficients_to_cost,
    power_regression_from_profiles,
)
from src.data_processing.data_loader import CSV_PATH, DEFAULT_ANTENNA_ID

DEFAULT_STEP_MINUTES = 15
PEAK_LOAD_FRACTION = 0.75
NUM_OPERATORS = 4
HORIZON_HOURS = 24.0

# Operator 1 profile (00000001U6): mean hourly traffic over 5 weekdays from radio_sites.csv.
FALLBACK_HOURLY_TRAFFIC_OP1: tuple[float, ...] = (
    4.06641215,
    3.274636975,
    2.0579057,
    1.478149,
    2.0657789,
    3.375421625,
    9.85657315,
    20.336020725,
    28.1049119,
    19.024882,
    17.522629625,
    18.123957125,
    19.859230275,
    21.8214572,
    21.7062173,
    23.212052575,
    22.436093525,
    24.43311375,
    21.2326654,
    14.339161375,
    13.2097858,
    15.0341149,
    10.7569192,
    9.929167025,
)

OPERATOR_NAMES = ("Operator1", "Operator2", "Operator3", "Operator4")
REVENUE_COEFFS = (0.8, 0.85, 0.75, 0.8)
FALLBACK_BETA_COEFFS = (0.2, 0.15, 0.18, 0.25)
FALLBACK_K_COEFFS = (0.5, 0.4, 0.3, 0.4)


def step_duration_hours(step_minutes: int) -> float:
    return step_minutes / 60.0


def steps_per_hour(step_minutes: int) -> int:
    if 60 % step_minutes != 0:
        raise ValueError(f"step_minutes={step_minutes} does not divide 60 evenly")
    return 60 // step_minutes


@dataclass
class OperatorParams:
    name: str
    capacity_epsilon: float
    c: float
    beta: float
    K: float


@dataclass
class Scenario:
    operators: list[OperatorParams]
    traffic: dict[int, list[float]]
    step_minutes: int
    antenna_id: str
    data_source: str  # "csv" or "fallback"
    horizon_hours: float = HORIZON_HOURS

    @property
    def num_steps(self) -> int:
        return len(next(iter(self.traffic.values())))

    @property
    def coalition(self) -> list[int]:
        return list(range(len(self.operators)))

    def horizon_label(self) -> str:
        return f"{int(self.horizon_hours)}h ({self.step_minutes}min steps)"

    def step_label(self, t: int) -> str:
        return f"{t * self.step_minutes / 60.0:.2f}h (step {t})"

    def hour_index(self, step: int) -> int:
        return step // steps_per_hour(self.step_minutes)

    def hourly_step_range(self, hour_index: int) -> range:
        sph = steps_per_hour(self.step_minutes)
        start = hour_index * sph
        return range(start, min(start + sph, self.num_steps))

    def hourly_traffic_means(self, hour_index: int) -> dict[int, float]:
        step_range = list(self.hourly_step_range(hour_index))
        if not step_range:
            return {i: 0.0 for i in range(len(self.operators))}
        return {
            i: sum(self.traffic[i][t] for t in step_range) / len(step_range)
            for i in range(len(self.operators))
        }


def capacity_epsilon_from_peak(
    max_hourly_traffic: float,
    peak_fraction: float = PEAK_LOAD_FRACTION,
) -> float:
    if max_hourly_traffic <= 0:
        raise ValueError("max_hourly_traffic must be positive")
    return max_hourly_traffic / peak_fraction


def expand_hourly_profile_to_steps(
    hourly_profile: list[float],
    step_minutes: int = DEFAULT_STEP_MINUTES,
    horizon_hours: float = HORIZON_HOURS,
) -> list[float]:
    sph = steps_per_hour(step_minutes)
    num_steps = int(horizon_hours * 60 / step_minutes)
    if len(hourly_profile) != 24:
        raise ValueError(f"Expected 24 hourly values, got {len(hourly_profile)}")

    series: list[float] = []
    for step in range(num_steps):
        series.append(hourly_profile[step // sph])
    return series


def derive_operator_hourly_profiles(
    base_hourly: list[float],
    num_operators: int = NUM_OPERATORS,
    scale_range: tuple[float, float] = (0.8, 1.0),
    noise_std: float = 0.02,
    seed: int = 42,
) -> list[list[float]]:
    """Operator 0 keeps base_hourly; others are scaled copies with Gaussian noise."""
    rng = random.Random(seed)
    profiles = [list(base_hourly)]
    for _ in range(1, num_operators):
        scale = rng.uniform(*scale_range)
        profiles.append(
            [
                max(0.0, value * scale * (1.0 + rng.gauss(0, noise_std)))
                for value in base_hourly
            ]
        )
    return profiles


def derive_operator_cost_coeffs(
    beta_op1: float,
    k_op1: float,
    num_operators: int = NUM_OPERATORS,
    spread: float = 0.1,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """Operator 1 keeps CSV-derived costs; others get close random perturbations."""
    rng = random.Random(seed + 1)
    coeffs = [(beta_op1, k_op1)]
    for _ in range(1, num_operators):
        beta_scale = rng.uniform(1.0 - spread, 1.0 + spread)
        k_scale = rng.uniform(1.0 - spread, 1.0 + spread)
        coeffs.append((beta_op1 * beta_scale, k_op1 * k_scale))
    return coeffs


def build_operators_and_traffic(
    base_hourly: list[float],
    cost_coeffs: list[tuple[float, float]],
    step_minutes: int = DEFAULT_STEP_MINUTES,
    seed: int = 42,
) -> tuple[list[OperatorParams], dict[int, list[float]]]:
    hourly_profiles = derive_operator_hourly_profiles(base_hourly, seed=seed)
    operators: list[OperatorParams] = []
    traffic: dict[int, list[float]] = {}

    for i, hourly in enumerate(hourly_profiles):
        beta, k = cost_coeffs[i]
        operators.append(
            OperatorParams(
                name=OPERATOR_NAMES[i],
                capacity_epsilon=capacity_epsilon_from_peak(max(hourly)),
                c=REVENUE_COEFFS[i],
                beta=beta,
                K=k,
            )
        )
        traffic[i] = expand_hourly_profile_to_steps(hourly, step_minutes)

    return operators, traffic


def _load_scenario_inputs(
    antenna_id: str | None,
    price_per_kwh: float,
    seed: int,
) -> tuple[list[float], str, str, list[tuple[float, float]]]:
    """Load traffic profile and per-operator (beta, K) from CSV or fallback."""
    if CSV_PATH.is_file():
        try:
            site_id = antenna_id or DEFAULT_ANTENNA_ID
            profile, power = load_antenna_profiles(site_id)
            beta_tilde, k_tilde = power_regression_from_profiles(profile, power)
            beta_op1, k_op1 = power_coefficients_to_cost(
                beta_tilde, k_tilde, price_per_kwh=price_per_kwh
            )
            cost_coeffs = derive_operator_cost_coeffs(beta_op1, k_op1, seed=seed)
            return profile, site_id, "csv", cost_coeffs
        except (ValueError, SystemExit, OSError):
            pass

    site_id = antenna_id or DEFAULT_ANTENNA_ID
    cost_coeffs = list(zip(FALLBACK_BETA_COEFFS, FALLBACK_K_COEFFS))
    return list(FALLBACK_HOURLY_TRAFFIC_OP1), site_id, "fallback", cost_coeffs


def load_scenario(
    step_minutes: int = DEFAULT_STEP_MINUTES,
    antenna_id: str | None = None,
    seed: int = 42,
    price_per_kwh: float = DEFAULT_ELECTRICITY_PRICE_PER_KWH,
) -> Scenario:
    """
    Build a 24 h weekday-type scenario for 4 operators.

    Operator 1 traffic: mean hourly load over the first 5 weekdays in
    ``radio_sites.csv`` when available, otherwise ``FALLBACK_HOURLY_TRAFFIC_OP1``.

    beta and K: from power-vs-rho regression on CSV data when available
    (converted to hourly cost via price_per_kwh), otherwise hard-coded fallbacks.
    Operators 2-4: traffic and costs are close perturbations of operator 1.
    """
    base_hourly, site_id, data_source, cost_coeffs = _load_scenario_inputs(
        antenna_id, price_per_kwh, seed
    )
    operators, traffic = build_operators_and_traffic(
        base_hourly, cost_coeffs, step_minutes=step_minutes, seed=seed
    )
    return Scenario(
        operators=operators,
        traffic=traffic,
        step_minutes=step_minutes,
        antenna_id=site_id,
        data_source=data_source,
    )
