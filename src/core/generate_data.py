"""Build the four-operator, 24-hour simulation scenario."""

from dataclasses import dataclass
import random

from src.data_processing.antenna_metrics import (
    DEFAULT_ELECTRICITY_PRICE_PER_KWH,
    PowerRegression,
    load_antenna_profiles,
    power_coefficients_to_cost,
    power_regression_from_profiles,
)
from src.data_processing.data_loader import (
    COMPACT_CSV_PATH,
    CSV_PATH,
    DEFAULT_ANTENNA_ID,
    first_antenna_id,
)

PEAK_FRACTION = 0.75
NAMES = ("Operator1", "Operator2", "Operator3", "Operator4")

FALLBACK_TRAFFIC = (
    4.06641215, 3.274636975, 2.0579057, 1.478149, 2.0657789, 3.375421625,
    9.85657315, 20.336020725, 28.1049119, 19.024882, 17.522629625,
    18.123957125, 19.859230275, 21.8214572, 21.7062173, 23.212052575,
    22.436093525, 24.43311375, 21.2326654, 14.339161375, 13.2097858,
    15.0341149, 10.7569192, 9.929167025,
)
FALLBACK_COSTS = (
    (0.50, 0.0053),
    (0.40, 0.0045),
    (0.30, 0.0050),
    (0.40, 0.0067),
)


@dataclass(frozen=True)
class OperatorParams:
    name: str
    q: float
    F: float
    gamma: float

    def __post_init__(self) -> None:
        if min(self.q, self.F, self.gamma) <= 0.0:
            raise ValueError(f"{self.name}: q, F and gamma must be positive")


@dataclass
class Scenario:
    operators: list[OperatorParams]
    traffic: dict[int, list[float]]
    antenna_id: str
    data_source: str
    power_regression: PowerRegression | None

    @property
    def coalition(self) -> list[int]:
        return list(range(len(self.operators)))

    @property
    def num_hours(self) -> int:
        return len(self.traffic[0])

    def demands_at_hour(self, hour: int) -> dict[int, float]:
        return {i: self.traffic[i][hour] for i in self.coalition}


def _nearby_profiles(base: list[float], seed: int) -> list[list[float]]:
    if len(base) != 24:
        raise ValueError("A daily profile must contain 24 hourly demands")
    rng = random.Random(seed)
    profiles = [base]
    for _ in range(3):
        scale = rng.uniform(0.8, 1.0)
        profiles.append(
            [max(0.0, d * scale * (1.0 + rng.gauss(0, 0.02))) for d in base]
        )
    return profiles


def _nearby_costs(F: float, gamma: float, seed: int) -> list[tuple[float, float]]:
    rng = random.Random(seed + 1)
    return [(F, gamma)] + [
        (F * rng.uniform(0.9, 1.1), gamma * rng.uniform(0.9, 1.1))
        for _ in range(3)
    ]


def load_scenario(
    antenna_id: str | None = None,
    seed: int = 42,
    price_per_kwh: float = DEFAULT_ELECTRICITY_PRICE_PER_KWH,
) -> Scenario:
    """Load operator 1 from data and generate three nearby operators."""
    if CSV_PATH.is_file():
        site_id = antenna_id or first_antenna_id()
        base, power = load_antenna_profiles(site_id)
        regression = power_regression_from_profiles(base, power)
        if min(regression.f_tilde, regression.gamma_tilde) <= 0.0:
            raise ValueError("The fitted coefficients must be positive")
        F, gamma = power_coefficients_to_cost(
            regression.f_tilde,
            regression.gamma_tilde,
            price_per_kwh,
        )
        costs = _nearby_costs(F, gamma, seed)
        source = "compact_csv" if CSV_PATH == COMPACT_CSV_PATH else "csv"
    else:
        site_id, base, regression = (
            antenna_id or DEFAULT_ANTENNA_ID,
            list(FALLBACK_TRAFFIC),
            None,
        )
        costs, source = list(FALLBACK_COSTS), "fallback"

    profiles = _nearby_profiles(list(base), seed)
    operators = [
        OperatorParams(NAMES[i], max(profile) / PEAK_FRACTION, *costs[i])
        for i, profile in enumerate(profiles)
    ]
    return Scenario(
        operators,
        {i: profile for i, profile in enumerate(profiles)},
        site_id,
        source,
        regression,
    )
