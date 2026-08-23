"""Semi-empirical four-operator instance generator for Section 6.1.

Each virtual site is built from one empirical reference profile, then four
operators are obtained by calibrated size factors, convex mixing of real
normalised shapes, and independently assigned energy coefficients. Capacities
are simulated scenarios, never treated as measurements.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from src.core.time_window import DEFAULT_END_HOUR, DEFAULT_START_HOUR, inclusive_hour_window
from src.data_processing.power_validation import CalibratedPopulation, NUM_HOURS


GENERATOR_VERSION = 1
DEFAULT_NUM_SITES = 400
DEFAULT_SITE_SEED = 20_260_818
DEFAULT_NUM_OPERATORS = 4
NUM_GROUPS = 4
PAIRWISE_SAMPLE_SIZE = 20_000
PAIRWISE_SAMPLE_SEED = 20_260_818

VOLUME_QUANTILES: dict[str, tuple[float, ...]] = {
    "close": (),
    "moderate": (0.40, 0.50, 0.60, 0.75),
    "far": (0.25, 0.40, 0.60, 0.90),
    "outlier": (0.50, 0.50, 0.50, 0.90),
}
SHAPE_LAMBDA: dict[str, float] = {
    "close": 0.15,
    "moderate": 0.35,
    "distant": 1.00,
}
CAMPAIGN_A_RATES = (0.80, 0.90, 1.00)
CENTRAL_RATE = 0.90
CENTRAL_VOLUME = "moderate"
CENTRAL_SHAPE = "moderate"
CENTRAL_EQUIPMENT = "close"
CAMPAIGN_B_REGIMES: tuple[tuple[str, int, float], ...] = (
    ("one_guardian", 1, 0.70),
    ("frontier_12", 2, 0.90),
    ("frontier_23", 3, 0.90),
    ("tight", 4, 0.90),
)
CAPACITY_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ScenarioSpec:
    campaign: str
    volume_level: str
    shape_level: str
    equipment_level: str
    capacity_rate: float | None = None
    guardian_target: int | None = None
    window_peak_rate: float | None = None

    def __post_init__(self) -> None:
        if self.campaign not in {"A", "B"}:
            raise ValueError("campaign must be 'A' or 'B'")
        if self.volume_level not in VOLUME_QUANTILES:
            raise ValueError(f"unknown volume level: {self.volume_level}")
        if self.shape_level not in SHAPE_LAMBDA:
            raise ValueError(f"unknown shape level: {self.shape_level}")
        if self.equipment_level not in {"close", "moderate", "distant"}:
            raise ValueError(f"unknown equipment level: {self.equipment_level}")
        if self.campaign == "A":
            if self.capacity_rate is None or self.capacity_rate <= 0.0:
                raise ValueError("campaign A requires a positive capacity rate r")
            if self.guardian_target is not None or self.window_peak_rate is not None:
                raise ValueError("campaign A must not set window-threshold parameters")
        else:
            if self.capacity_rate is not None:
                raise ValueError("campaign B must not set a five-day peak rate r")
            if (
                self.guardian_target is None
                or self.window_peak_rate is None
                or self.guardian_target < 1
                or self.window_peak_rate <= 0.0
            ):
                raise ValueError("campaign B requires k >= 1 and r_H > 0")

    @property
    def key(self) -> str:
        if self.campaign == "A":
            return (
                f"A_vol-{self.volume_level}_shape-{self.shape_level}_"
                f"equip-{self.equipment_level}_r{self.capacity_rate:.2f}"
            )
        return (
            f"B_vol-{self.volume_level}_shape-{self.shape_level}_"
            f"equip-{self.equipment_level}_k{self.guardian_target}_"
            f"rH{self.window_peak_rate:.2f}"
        )


@dataclass(frozen=True)
class SiteBlueprints:
    site_ids: np.ndarray
    reference_index: np.ndarray
    donor_indices: np.ndarray
    energy_close: np.ndarray
    energy_moderate: np.ndarray
    energy_distant: np.ndarray
    alpha_permutation: np.ndarray
    mean_traffic_group: np.ndarray
    seed: int
    num_operators: int

    @property
    def num_sites(self) -> int:
        return int(self.reference_index.shape[0])


@dataclass(frozen=True)
class MaterializedSite:
    site_id: str
    scenario: ScenarioSpec
    traffic_gb: np.ndarray
    p_fixed_w: np.ndarray
    slope_w_per_gb: np.ndarray
    peak_traffic_gb: np.ndarray
    alpha: np.ndarray
    lambda_shape: float
    mu_gb: float
    reference_id: str
    donor_ids: tuple[str, ...]
    energy_ids: tuple[str, ...]
    mean_traffic_group: int


@dataclass(frozen=True)
class ProtocolSpec:
    seed: int
    num_sites: int
    num_operators: int
    generator_version: int
    volume_factors: dict[str, tuple[float, ...]]
    shape_lambda: dict[str, float]
    campaign_a_rates: tuple[float, ...]
    central_rate: float
    campaign_b_regimes: tuple[tuple[str, int, float], ...]
    window_hours: tuple[int, ...]
    empirical: dict[str, Any]


def mean_traffic_gb(population: CalibratedPopulation) -> np.ndarray:
    return np.mean(population.traffic_gb, axis=(1, 2))


def equal_size_groups(values: np.ndarray, num_groups: int = NUM_GROUPS) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < num_groups:
        raise ValueError("values must contain at least one item per group")
    order = np.argsort(values, kind="mergesort")
    groups = np.empty(values.size, dtype=np.int8)
    groups[order] = np.minimum(
        num_groups - 1,
        np.arange(values.size, dtype=int) * num_groups // values.size,
    )
    return groups


def normalised_shapes(traffic_gb: np.ndarray) -> np.ndarray:
    """Return mean-one profiles with shape ``(n_antennas, n_slots)``."""
    flat = np.asarray(traffic_gb, dtype=float).reshape(traffic_gb.shape[0], -1)
    means = np.mean(flat, axis=1)
    if np.any(means <= 0.0):
        raise ValueError("every antenna must have a positive mean traffic")
    return flat / means[:, None]


def size_factors_from_quantiles(
    mean_traffic: np.ndarray,
    quantiles: tuple[float, ...],
    num_operators: int,
) -> np.ndarray:
    """Return strictly positive size factors with arithmetic mean one."""
    if not quantiles:
        return np.ones(num_operators, dtype=float)
    if num_operators == len(quantiles):
        points = np.asarray(quantiles, dtype=float)
    else:
        low, high = float(min(quantiles)), float(max(quantiles))
        points = np.linspace(low, high, num_operators)
        if all(abs(value - quantiles[0]) < 1e-12 for value in quantiles[:-1]):
            points[:-1] = quantiles[0]
            points[-1] = quantiles[-1]
    raw = np.quantile(mean_traffic, points)
    if np.any(raw <= 0.0):
        raise ValueError("size-factor quantiles must be positive")
    return raw / float(np.mean(raw))


def _quantiles(values: np.ndarray, probabilities: Iterable[float]) -> dict[str, float]:
    points = np.asarray(list(probabilities), dtype=float)
    estimates = np.quantile(np.asarray(values, dtype=float), points)
    return {f"q{int(round(100 * p)):02d}": float(value) for p, value in zip(points, estimates)}


def _pairwise_correlation(profiles: np.ndarray, rng: np.random.Generator, n_pairs: int) -> np.ndarray:
    n = profiles.shape[0]
    if n < 2:
        raise ValueError("at least two profiles are required")
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    left = profiles[i] - profiles[i].mean(axis=1, keepdims=True)
    right = profiles[j] - profiles[j].mean(axis=1, keepdims=True)
    numerator = np.sum(left * right, axis=1)
    denominator = np.sqrt(np.sum(left**2, axis=1) * np.sum(right**2, axis=1))
    return numerator / np.maximum(denominator, 1e-12)


def _row_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean(axis=1, keepdims=True)
    numerator = np.sum(left * right, axis=1)
    denominator = np.sqrt(np.sum(left**2, axis=1) * np.sum(right**2, axis=1))
    return numerator / np.maximum(denominator, 1e-12)


def _lag1_correlation(profiles: np.ndarray) -> np.ndarray:
    left = profiles[:, :-1]
    right = profiles[:, 1:]
    return _row_correlation(left, right)


def calibrate_protocol(
    population: CalibratedPopulation,
    *,
    num_sites: int = DEFAULT_NUM_SITES,
    seed: int = DEFAULT_SITE_SEED,
    num_operators: int = DEFAULT_NUM_OPERATORS,
) -> ProtocolSpec:
    """Derive size factors and empirical diagnostics from the admissible population."""
    if num_operators < 2:
        raise ValueError("at least two operators are required")
    mean_traffic = mean_traffic_gb(population)
    shapes = normalised_shapes(population.traffic_gb)
    volume_factors = {
        name: tuple(
            float(value)
            for value in size_factors_from_quantiles(
                mean_traffic, quantiles, num_operators
            )
        )
        for name, quantiles in VOLUME_QUANTILES.items()
    }
    rng = np.random.default_rng(PAIRWISE_SAMPLE_SEED)
    pairwise = _pairwise_correlation(shapes, rng, PAIRWISE_SAMPLE_SIZE)
    mix_index = rng.integers(0, shapes.shape[0], size=(3, 8_000))
    valid = (
        (mix_index[0] != mix_index[1])
        & (mix_index[0] != mix_index[2])
        & (mix_index[1] != mix_index[2])
    )
    anchor, donor_a, donor_b = mix_index[:, valid]
    mixing_corr: dict[str, dict[str, float]] = {}
    for name, lam in SHAPE_LAMBDA.items():
        mixed_a = (1.0 - lam) * shapes[anchor] + lam * shapes[donor_a]
        mixed_b = (1.0 - lam) * shapes[anchor] + lam * shapes[donor_b]
        corr = _row_correlation(mixed_a, mixed_b)
        mixing_corr[name] = {
            "lambda": float(lam),
            "mean": float(np.mean(corr)),
            "median": float(np.median(corr)),
            "q10": float(np.quantile(corr, 0.10)),
            "q90": float(np.quantile(corr, 0.90)),
        }
    night = population.traffic_gb[:, :, :7]
    peaks = population.peak_traffic_gb
    empirical = {
        "n_antennas": int(population.antenna_ids.size),
        "n_days": int(population.days.size),
        "mean_traffic_gb": {
            "median": float(np.median(mean_traffic)),
            **_quantiles(mean_traffic, (0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90)),
        },
        "peak_traffic_gb": {
            "median": float(np.median(peaks)),
            **_quantiles(peaks, (0.10, 0.25, 0.50, 0.75, 0.90)),
        },
        "p_fixed_w": {
            "median": float(np.median(population.p_fixed_w)),
            **_quantiles(population.p_fixed_w, (0.10, 0.25, 0.50, 0.75, 0.90)),
        },
        "slope_w_per_gb": {
            "median": float(np.median(population.slope_w_per_gb)),
            **_quantiles(population.slope_w_per_gb, (0.10, 0.25, 0.50, 0.75, 0.90)),
        },
        "corr_mean_traffic_p_fixed": float(
            np.corrcoef(mean_traffic, population.p_fixed_w)[0, 1]
        ),
        "corr_mean_traffic_slope": float(
            np.corrcoef(mean_traffic, population.slope_w_per_gb)[0, 1]
        ),
        "pairwise_shape_correlation": {
            "mean": float(np.mean(pairwise)),
            "median": float(np.median(pairwise)),
            **_quantiles(pairwise, (0.10, 0.25, 0.50, 0.75, 0.90)),
        },
        "lag1_shape_correlation_median": float(np.median(_lag1_correlation(shapes))),
        "night_mean_over_five_day_peak_median": float(
            np.median(np.mean(night, axis=(1, 2)) / np.maximum(peaks, 1e-12))
        ),
        "night_peak_over_five_day_peak_median": float(
            np.median(np.max(night, axis=(1, 2)) / np.maximum(peaks, 1e-12))
        ),
        "mixing_operator_correlation": mixing_corr,
        "volume_quantile_definition": {
            name: list(quantiles) for name, quantiles in VOLUME_QUANTILES.items()
        },
    }
    return ProtocolSpec(
        seed=seed,
        num_sites=num_sites,
        num_operators=num_operators,
        generator_version=GENERATOR_VERSION,
        volume_factors=volume_factors,
        shape_lambda=dict(SHAPE_LAMBDA),
        campaign_a_rates=CAMPAIGN_A_RATES,
        central_rate=CENTRAL_RATE,
        campaign_b_regimes=CAMPAIGN_B_REGIMES,
        window_hours=inclusive_hour_window(DEFAULT_START_HOUR, DEFAULT_END_HOUR),
        empirical=empirical,
    )


def protocol_scenarios(
    *,
    include_campaign_a_rates: bool = True,
    include_heterogeneity: bool = True,
    include_campaign_b: bool = True,
) -> tuple[ScenarioSpec, ...]:
    """Return the matched protocol grid: central, one-factor, then threshold regimes."""
    specs: list[ScenarioSpec] = []
    if include_campaign_a_rates:
        for rate in CAMPAIGN_A_RATES:
            specs.append(
                ScenarioSpec(
                    "A",
                    CENTRAL_VOLUME,
                    CENTRAL_SHAPE,
                    CENTRAL_EQUIPMENT,
                    capacity_rate=rate,
                )
            )
    if include_heterogeneity:
        rate = CENTRAL_RATE
        for volume in ("close", "far", "outlier"):
            specs.append(
                ScenarioSpec("A", volume, CENTRAL_SHAPE, CENTRAL_EQUIPMENT, capacity_rate=rate)
            )
        for shape in ("close", "distant"):
            specs.append(
                ScenarioSpec("A", CENTRAL_VOLUME, shape, CENTRAL_EQUIPMENT, capacity_rate=rate)
            )
        for equipment in ("moderate", "distant"):
            specs.append(
                ScenarioSpec("A", CENTRAL_VOLUME, CENTRAL_SHAPE, equipment, capacity_rate=rate)
            )
        specs.append(
            ScenarioSpec("A", "close", "close", "close", capacity_rate=rate)
        )
        specs.append(
            ScenarioSpec("A", "far", "distant", "distant", capacity_rate=rate)
        )
    if include_campaign_b:
        for _, k_target, rate_h in CAMPAIGN_B_REGIMES:
            specs.append(
                ScenarioSpec(
                    "B",
                    "close",
                    "close",
                    "moderate",
                    guardian_target=k_target,
                    window_peak_rate=rate_h,
                )
            )
    keys = [spec.key for spec in specs]
    if len(keys) != len(set(keys)):
        raise RuntimeError("protocol scenario keys are not unique")
    return tuple(specs)


def _draw_from_allowed(
    rng: np.random.Generator,
    allowed: np.ndarray,
    size: int,
) -> np.ndarray:
    if allowed.size < size:
        raise ValueError("not enough remaining antennas for the requested draw")
    return rng.choice(allowed, size=size, replace=False).astype(np.int32)


def generate_site_blueprints(
    population: CalibratedPopulation,
    num_sites: int = DEFAULT_NUM_SITES,
    seed: int = DEFAULT_SITE_SEED,
    num_operators: int = DEFAULT_NUM_OPERATORS,
) -> SiteBlueprints:
    """Draw matched site blueprints; traffic, energy and capacities are applied later."""
    if num_sites <= 0:
        raise ValueError("num_sites must be positive")
    if num_operators < 2:
        raise ValueError("at least two operators are required")
    n_antennas = int(population.antenna_ids.size)
    if n_antennas < num_operators * 4 + 1:
        raise ValueError("the admissible population is too small for disjoint draws")
    mean_groups = equal_size_groups(mean_traffic_gb(population))
    allocation = np.full(NUM_GROUPS, num_sites // NUM_GROUPS, dtype=int)
    allocation[: num_sites % NUM_GROUPS] += 1
    rng = np.random.default_rng(seed)
    references: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    for group_index, target in enumerate(allocation):
        candidates = np.flatnonzero(mean_groups == group_index)
        if candidates.size < target:
            raise ValueError(
                f"mean-traffic group {group_index} has fewer antennas than requested sites"
            )
        chosen = rng.choice(candidates, size=target, replace=False).astype(np.int32)
        references.append(chosen)
        groups.append(np.full(target, group_index, dtype=np.int8))
    reference_index = np.concatenate(references)
    mean_traffic_group = np.concatenate(groups)
    order = rng.permutation(num_sites)
    reference_index = reference_index[order]
    mean_traffic_group = mean_traffic_group[order]

    donor_indices = np.empty((num_sites, num_operators), dtype=np.int32)
    energy_close = np.empty_like(donor_indices)
    energy_moderate = np.empty_like(donor_indices)
    energy_distant = np.empty_like(donor_indices)
    alpha_permutation = np.empty_like(donor_indices)
    all_indices = np.arange(n_antennas, dtype=np.int32)
    power_groups = population.fixed_power_group

    for site_index, reference in enumerate(reference_index):
        forbidden = {int(reference)}
        allowed = all_indices[np.isin(all_indices, list(forbidden), invert=True)]
        donors = _draw_from_allowed(rng, allowed, num_operators)
        donor_indices[site_index] = donors
        forbidden.update(int(index) for index in donors)

        close_group = int(rng.integers(NUM_GROUPS))
        close_pool = np.flatnonzero(power_groups == close_group)
        close_pool = close_pool[np.isin(close_pool, list(forbidden), invert=True)]
        energy_close[site_index] = _draw_from_allowed(rng, close_pool, num_operators)
        forbidden.update(int(index) for index in energy_close[site_index])

        moderate_pool = all_indices[np.isin(all_indices, list(forbidden), invert=True)]
        energy_moderate[site_index] = _draw_from_allowed(
            rng, moderate_pool, num_operators
        )
        forbidden.update(int(index) for index in energy_moderate[site_index])

        distant_groups = [0, 3, 1, 2]
        distant: list[int] = []
        for group_index in (distant_groups[i % NUM_GROUPS] for i in range(num_operators)):
            pool = np.flatnonzero(power_groups == group_index)
            pool = pool[np.isin(pool, list(forbidden), invert=True)]
            pick = int(_draw_from_allowed(rng, pool, 1)[0])
            distant.append(pick)
            forbidden.add(pick)
        energy_distant[site_index] = np.asarray(distant, dtype=np.int32)
        alpha_permutation[site_index] = rng.permutation(num_operators).astype(np.int32)

    site_ids = np.asarray(
        [f"site_{index:04d}" for index in range(1, num_sites + 1)], dtype=str
    )
    return SiteBlueprints(
        site_ids=site_ids,
        reference_index=reference_index.astype(np.int32),
        donor_indices=donor_indices,
        energy_close=energy_close,
        energy_moderate=energy_moderate,
        energy_distant=energy_distant,
        alpha_permutation=alpha_permutation,
        mean_traffic_group=mean_traffic_group,
        seed=seed,
        num_operators=num_operators,
    )


def _antenna_ids(population: CalibratedPopulation, indices: np.ndarray) -> tuple[str, ...]:
    return tuple(str(identifier) for identifier in population.antenna_ids[indices])


def materialize_site(
    blueprint: SiteBlueprints,
    site_index: int,
    population: CalibratedPopulation,
    spec: ScenarioSpec,
    protocol: ProtocolSpec | None = None,
    shapes: np.ndarray | None = None,
) -> MaterializedSite:
    """Apply one scenario to a frozen blueprint without redrawing antennas."""
    if protocol is None:
        mean_traffic = mean_traffic_gb(population)
        volume_factors = {
            name: tuple(
                float(value)
                for value in size_factors_from_quantiles(
                    mean_traffic, quantiles, blueprint.num_operators
                )
            )
            for name, quantiles in VOLUME_QUANTILES.items()
        }
        lambdas = dict(SHAPE_LAMBDA)
    else:
        volume_factors = protocol.volume_factors
        lambdas = protocol.shape_lambda
    if spec.volume_level not in volume_factors:
        raise ValueError(f"unknown volume level: {spec.volume_level}")
    reference = int(blueprint.reference_index[site_index])
    donors = blueprint.donor_indices[site_index]
    energy_index = {
        "close": blueprint.energy_close[site_index],
        "moderate": blueprint.energy_moderate[site_index],
        "distant": blueprint.energy_distant[site_index],
    }[spec.equipment_level]
    permutation = blueprint.alpha_permutation[site_index]
    base_alpha = np.asarray(volume_factors[spec.volume_level], dtype=float)
    if base_alpha.size != blueprint.num_operators:
        base_alpha = size_factors_from_quantiles(
            mean_traffic_gb(population),
            VOLUME_QUANTILES[spec.volume_level],
            blueprint.num_operators,
        )
    alpha = base_alpha[permutation]
    lam = float(lambdas[spec.shape_level])
    if shapes is None:
        shapes = normalised_shapes(population.traffic_gb)
    mixed = (1.0 - lam) * shapes[reference][None, :] + lam * shapes[donors]
    mu = float(np.mean(population.traffic_gb[reference]))
    traffic = (mu * alpha[:, None] * mixed).reshape(
        blueprint.num_operators, population.days.size, NUM_HOURS
    )
    if np.any(traffic < -CAPACITY_TOLERANCE):
        raise RuntimeError("constructed traffic is negative")
    traffic = np.maximum(traffic, 0.0)
    return MaterializedSite(
        site_id=str(blueprint.site_ids[site_index]),
        scenario=spec,
        traffic_gb=traffic,
        p_fixed_w=np.asarray(population.p_fixed_w[energy_index], dtype=float),
        slope_w_per_gb=np.asarray(population.slope_w_per_gb[energy_index], dtype=float),
        peak_traffic_gb=np.max(traffic, axis=(1, 2)),
        alpha=alpha,
        lambda_shape=lam,
        mu_gb=mu,
        reference_id=str(population.antenna_ids[reference]),
        donor_ids=_antenna_ids(population, donors),
        energy_ids=_antenna_ids(population, energy_index),
        mean_traffic_group=int(blueprint.mean_traffic_group[site_index]),
    )


def campaign_a_capacities(peak_traffic_gb: np.ndarray, rate: float) -> np.ndarray:
    """Return capacities defined by max_t d_i(t) = r q_i."""
    if rate <= 0.0:
        raise ValueError("capacity rate r must be positive")
    return np.asarray(peak_traffic_gb, dtype=float) / rate


def campaign_b_capacities(
    window_demands_gb: np.ndarray,
    guardian_target: int,
    window_peak_rate: float,
) -> np.ndarray:
    """Equal capacities sized so that k guardians cover the window-peak demand.

    Individual feasibility is enforced by
    ``q_i = max{ D_H^{max} / (k r_H), max_{h in H} d_i^h }``.
    """
    demands = np.asarray(window_demands_gb, dtype=float)
    if demands.ndim != 2 or demands.shape[0] < 1:
        raise ValueError("window demands must have shape (n_operators, n_hours)")
    if guardian_target < 1 or window_peak_rate <= 0.0:
        raise ValueError("k and r_H must be positive")
    coalition_peak = float(np.max(np.sum(demands, axis=0)))
    equal = coalition_peak / (guardian_target * window_peak_rate)
    individual_peak = np.max(demands, axis=1)
    return np.maximum(equal, individual_peak)


def capacities_for_site(
    site: MaterializedSite,
    window_demands_gb: np.ndarray,
) -> np.ndarray:
    spec = site.scenario
    if spec.campaign == "A":
        assert spec.capacity_rate is not None
        capacities = campaign_a_capacities(site.peak_traffic_gb, spec.capacity_rate)
    else:
        assert spec.guardian_target is not None and spec.window_peak_rate is not None
        capacities = campaign_b_capacities(
            window_demands_gb, spec.guardian_target, spec.window_peak_rate
        )
    if np.any(np.max(window_demands_gb, axis=1) > capacities + CAPACITY_TOLERANCE):
        raise ValueError(
            f"{site.site_id} {spec.key}: constructed demands exceed capacities"
        )
    return capacities


def minimal_guardian_counts(capacities: np.ndarray, demands: np.ndarray) -> np.ndarray:
    """Smallest number of equipments that can cover coalition demand at each hour."""
    capacities = np.asarray(capacities, dtype=float)
    totals = np.sum(np.asarray(demands, dtype=float), axis=0)
    order = np.sort(capacities)[::-1]
    cumulative = np.cumsum(order)
    counts = np.empty(totals.shape, dtype=int)
    for hour, total in enumerate(totals):
        if total <= CAPACITY_TOLERANCE:
            counts[hour] = 0
            continue
        needed = int(np.searchsorted(cumulative, total - CAPACITY_TOLERANCE, side="left") + 1)
        counts[hour] = needed if needed <= capacities.size else capacities.size + 1
    return counts


def iter_materialized_sites(
    blueprints: SiteBlueprints,
    population: CalibratedPopulation,
    scenarios: Iterable[ScenarioSpec],
    protocol: ProtocolSpec | None = None,
    num_sites: int | None = None,
) -> Iterator[MaterializedSite]:
    if protocol is None:
        protocol = calibrate_protocol(
            population,
            num_sites=blueprints.num_sites,
            seed=blueprints.seed,
            num_operators=blueprints.num_operators,
        )
    limit = blueprints.num_sites if num_sites is None else num_sites
    if not 1 <= limit <= blueprints.num_sites:
        raise ValueError("num_sites must lie within the frozen blueprint list")
    shapes = normalised_shapes(population.traffic_gb)
    specs = tuple(scenarios)
    for site_index in range(limit):
        for spec in specs:
            yield materialize_site(
                blueprints,
                site_index,
                population,
                spec,
                protocol,
                shapes=shapes,
            )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def save_protocol_spec(protocol: ProtocolSpec, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": protocol.seed,
        "num_sites": protocol.num_sites,
        "num_operators": protocol.num_operators,
        "generator_version": protocol.generator_version,
        "volume_factors": protocol.volume_factors,
        "shape_lambda": protocol.shape_lambda,
        "campaign_a_rates": list(protocol.campaign_a_rates),
        "central_rate": protocol.central_rate,
        "campaign_b_regimes": [
            {"name": name, "k": k, "r_H": rate}
            for name, k, rate in protocol.campaign_b_regimes
        ],
        "window_hours": list(protocol.window_hours),
        "empirical": _jsonable(protocol.empirical),
        "scenario_keys": [spec.key for spec in protocol_scenarios()],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_protocol_spec(path: Path) -> ProtocolSpec:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ProtocolSpec(
        seed=int(payload["seed"]),
        num_sites=int(payload["num_sites"]),
        num_operators=int(payload["num_operators"]),
        generator_version=int(payload["generator_version"]),
        volume_factors={
            name: tuple(float(value) for value in values)
            for name, values in payload["volume_factors"].items()
        },
        shape_lambda={
            name: float(value) for name, value in payload["shape_lambda"].items()
        },
        campaign_a_rates=tuple(float(value) for value in payload["campaign_a_rates"]),
        central_rate=float(payload["central_rate"]),
        campaign_b_regimes=tuple(
            (str(row["name"]), int(row["k"]), float(row["r_H"]))
            for row in payload["campaign_b_regimes"]
        ),
        window_hours=tuple(int(value) for value in payload["window_hours"]),
        empirical=payload["empirical"],
    )


def save_site_blueprints(
    blueprints: SiteBlueprints,
    population: CalibratedPopulation,
    path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_op = blueprints.num_operators
    fieldnames = [
        "site_id",
        "mean_traffic_group",
        "reference_antenna",
        *[f"donor_{index}" for index in range(1, n_op + 1)],
        *[f"energy_close_{index}" for index in range(1, n_op + 1)],
        *[f"energy_moderate_{index}" for index in range(1, n_op + 1)],
        *[f"energy_distant_{index}" for index in range(1, n_op + 1)],
        "alpha_permutation",
        "seed",
        "num_operators",
        "generator_version",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for site_index, site_id in enumerate(blueprints.site_ids):
            writer.writerow(
                {
                    "site_id": str(site_id),
                    "mean_traffic_group": int(blueprints.mean_traffic_group[site_index]),
                    "reference_antenna": str(
                        population.antenna_ids[blueprints.reference_index[site_index]]
                    ),
                    **{
                        f"donor_{index}": str(
                            population.antenna_ids[
                                blueprints.donor_indices[site_index, index - 1]
                            ]
                        )
                        for index in range(1, n_op + 1)
                    },
                    **{
                        f"energy_close_{index}": str(
                            population.antenna_ids[
                                blueprints.energy_close[site_index, index - 1]
                            ]
                        )
                        for index in range(1, n_op + 1)
                    },
                    **{
                        f"energy_moderate_{index}": str(
                            population.antenna_ids[
                                blueprints.energy_moderate[site_index, index - 1]
                            ]
                        )
                        for index in range(1, n_op + 1)
                    },
                    **{
                        f"energy_distant_{index}": str(
                            population.antenna_ids[
                                blueprints.energy_distant[site_index, index - 1]
                            ]
                        )
                        for index in range(1, n_op + 1)
                    },
                    "alpha_permutation": "|".join(
                        str(int(value))
                        for value in blueprints.alpha_permutation[site_index]
                    ),
                    "seed": blueprints.seed,
                    "num_operators": n_op,
                    "generator_version": GENERATOR_VERSION,
                }
            )
    return path


def _lookup_indices(
    population: CalibratedPopulation,
    identifiers: Iterable[str],
    site_id: str,
) -> list[int]:
    index_by_id = {
        str(antenna_id): index
        for index, antenna_id in enumerate(population.antenna_ids)
    }
    indices: list[int] = []
    for identifier in identifiers:
        try:
            indices.append(index_by_id[identifier])
        except KeyError as error:
            raise ValueError(f"{site_id}: unknown antenna {identifier}") from error
    if len(set(indices)) != len(indices):
        raise ValueError(f"{site_id}: repeated antenna identifiers")
    return indices


def load_site_blueprints(
    path: Path,
    population: CalibratedPopulation,
    seed: int = DEFAULT_SITE_SEED,
) -> SiteBlueprints:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("the blueprint file is empty")
    num_operators = int(rows[0]["num_operators"])
    site_ids: list[str] = []
    references: list[int] = []
    donors: list[list[int]] = []
    close: list[list[int]] = []
    moderate: list[list[int]] = []
    distant: list[list[int]] = []
    permutations: list[list[int]] = []
    groups: list[int] = []
    for row in rows:
        if int(row["generator_version"]) != GENERATOR_VERSION:
            raise ValueError(
                f"unsupported blueprint generator version {row['generator_version']}"
            )
        if int(row["num_operators"]) != num_operators:
            raise ValueError("inconsistent num_operators in blueprint file")
        site_id = row["site_id"]
        site_ids.append(site_id)
        references.extend(
            _lookup_indices(population, [row["reference_antenna"]], site_id)
        )
        donors.append(
            _lookup_indices(
                population,
                [row[f"donor_{index}"] for index in range(1, num_operators + 1)],
                site_id,
            )
        )
        close.append(
            _lookup_indices(
                population,
                [row[f"energy_close_{index}"] for index in range(1, num_operators + 1)],
                site_id,
            )
        )
        moderate.append(
            _lookup_indices(
                population,
                [
                    row[f"energy_moderate_{index}"]
                    for index in range(1, num_operators + 1)
                ],
                site_id,
            )
        )
        distant.append(
            _lookup_indices(
                population,
                [
                    row[f"energy_distant_{index}"]
                    for index in range(1, num_operators + 1)
                ],
                site_id,
            )
        )
        permutation = [int(value) for value in row["alpha_permutation"].split("|")]
        if sorted(permutation) != list(range(num_operators)):
            raise ValueError(f"{site_id}: invalid alpha permutation")
        permutations.append(permutation)
        groups.append(int(row["mean_traffic_group"]))
        traffic_energy = {
            references[-1],
            *donors[-1],
            *close[-1],
            *moderate[-1],
            *distant[-1],
        }
        if len(traffic_energy) != 1 + 4 * num_operators:
            raise ValueError(f"{site_id}: traffic and energy sources are not disjoint")
    if len(site_ids) != len(set(site_ids)):
        raise ValueError("blueprint site identifiers are not unique")
    return SiteBlueprints(
        site_ids=np.asarray(site_ids, dtype=str),
        reference_index=np.asarray(references, dtype=np.int32),
        donor_indices=np.asarray(donors, dtype=np.int32),
        energy_close=np.asarray(close, dtype=np.int32),
        energy_moderate=np.asarray(moderate, dtype=np.int32),
        energy_distant=np.asarray(distant, dtype=np.int32),
        alpha_permutation=np.asarray(permutations, dtype=np.int32),
        mean_traffic_group=np.asarray(groups, dtype=np.int8),
        seed=seed,
        num_operators=num_operators,
    )


def antenna_usage_counts(blueprints: SiteBlueprints) -> np.ndarray:
    """Count how often each antenna index appears in any blueprint role."""
    parts = (
        blueprints.reference_index,
        blueprints.donor_indices.reshape(-1),
        blueprints.energy_close.reshape(-1),
        blueprints.energy_moderate.reshape(-1),
        blueprints.energy_distant.reshape(-1),
    )
    stacked = np.concatenate(parts)
    _, counts = np.unique(stacked, return_counts=True)
    return counts
