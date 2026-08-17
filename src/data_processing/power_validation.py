"""Direct population calibration of the affine traffic--power model.

The numerical experiment follows Section 6.2 of the paper: for every radio
identifier, one affine model is fitted on the active hourly observations from
the first five days. There is no prediction task, train/test split, or model
comparison in this module.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_NUM_DAYS = 5
NUM_HOURS = 24
NUMERICAL_TOLERANCE = 1e-12
CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DataAudit:
    source_rows: int
    unparsed_source_rows: int
    selected_rows: int
    invalid_selected_rows: int
    negative_selected_rows: int
    duplicate_selected_rows: int
    antenna_count: int
    selected_dates: tuple[str, ...]
    inactive_rows: int
    inconsistent_zero_power_rows: int


@dataclass(frozen=True)
class AntennaSeries:
    antenna_id: str
    days: tuple[date, ...]
    traffic: np.ndarray
    power: np.ndarray

    @property
    def num_observations(self) -> int:
        return int(np.sum(np.isfinite(self.traffic) & np.isfinite(self.power)))

    @property
    def num_active_observations(self) -> int:
        return int(
            np.sum(
                np.isfinite(self.traffic)
                & np.isfinite(self.power)
                & (self.power > 0.0)
            )
        )


@dataclass(frozen=True)
class AffineFit:
    p_fixed_w: float
    slope_w_per_gb: float
    r_squared: float
    rmse_w: float
    normalized_rmse: float

    def predict(self, traffic: np.ndarray) -> np.ndarray:
        return self.p_fixed_w + self.slope_w_per_gb * np.asarray(
            traffic, dtype=float
        )


@dataclass(frozen=True)
class AntennaCalibration:
    antenna_id: str
    status: str
    num_days: int
    num_observations: int
    num_active_observations: int
    mean_traffic_gb: float
    mean_power_w: float
    p_fixed_w: float
    slope_w_per_gb: float
    r_squared: float
    rmse_w: float
    normalized_rmse: float
    fixed_power_share: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationCampaign:
    audit: DataAudit
    series: tuple[AntennaSeries, ...]
    antenna_results: tuple[AntennaCalibration, ...]


@dataclass(frozen=True)
class CalibratedPopulation:
    antenna_ids: np.ndarray
    days: np.ndarray
    traffic_gb: np.ndarray
    power_w: np.ndarray
    p_fixed_w: np.ndarray
    slope_w_per_gb: np.ndarray
    r_squared: np.ndarray
    normalized_rmse: np.ndarray
    peak_traffic_gb: np.ndarray
    traffic_group: np.ndarray
    fixed_power_group: np.ndarray


def _column_names(fieldnames: list[str] | None) -> tuple[str, str, str, str]:
    names = fieldnames or []

    def find(*tokens: str) -> str:
        match = next(
            (name for token in tokens for name in names if token in name.lower()),
            None,
        )
        if match is None:
            raise ValueError(f"Missing CSV column matching {tokens}")
        return match

    return (
        find("heure", "date"),
        find("nidt"),
        find("dl_volume", "pdcp", "gbytes"),
        find("power", "consumption"),
    )


def _first_dates(csv_path: Path, num_days: int) -> tuple[date, ...]:
    dates: set[date] = set()
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as file:
        reader = csv.DictReader(file, delimiter=";")
        date_col, _, _, _ = _column_names(reader.fieldnames)
        for raw in reader:
            try:
                dates.add(datetime.fromisoformat(raw[date_col].strip()).date())
            except (KeyError, TypeError, ValueError):
                continue
    selected = tuple(sorted(dates)[:num_days])
    if len(selected) < num_days:
        raise ValueError(f"Expected {num_days} days, found {len(selected)}")
    return selected


def load_population(
    csv_path: Path,
    num_days: int = DEFAULT_NUM_DAYS,
) -> tuple[tuple[AntennaSeries, ...], DataAudit]:
    """Load the first ``num_days`` and average duplicate identifier-hour rows."""
    if num_days <= 0:
        raise ValueError("num_days must be positive")
    selected_dates = _first_dates(csv_path, num_days)
    selected_set = set(selected_dates)
    cells: dict[str, dict[date, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    source_rows = 0
    unparsed_source_rows = 0
    selected_rows = 0
    invalid_selected_rows = 0
    negative_selected_rows = 0
    duplicate_selected_rows = 0
    inactive_rows = 0
    inconsistent_zero_power_rows = 0

    with csv_path.open(newline="", encoding="utf-8", errors="replace") as file:
        reader = csv.DictReader(file, delimiter=";")
        date_col, id_col, traffic_col, power_col = _column_names(reader.fieldnames)
        for raw in reader:
            source_rows += 1
            try:
                timestamp = datetime.fromisoformat(raw[date_col].strip())
            except (KeyError, TypeError, ValueError):
                unparsed_source_rows += 1
                continue
            if timestamp.date() not in selected_set:
                continue
            selected_rows += 1
            try:
                antenna_id = raw[id_col].strip()
                traffic = float(raw[traffic_col].replace(",", "."))
                power = float(raw[power_col].replace(",", "."))
                if not antenna_id or not np.isfinite(traffic) or not np.isfinite(power):
                    raise ValueError("invalid selected record")
            except (KeyError, TypeError, ValueError):
                invalid_selected_rows += 1
                continue
            if traffic < 0.0 or power < 0.0:
                negative_selected_rows += 1
                continue
            if power == 0.0:
                if traffic == 0.0:
                    inactive_rows += 1
                else:
                    inconsistent_zero_power_rows += 1

            day_cells = cells[antenna_id][timestamp.date()]
            if timestamp.hour in day_cells:
                duplicate_selected_rows += 1
                accumulator = day_cells[timestamp.hour]
                accumulator[0] += traffic
                accumulator[1] += power
                accumulator[2] += 1.0
            else:
                day_cells[timestamp.hour] = [traffic, power, 1.0]

    population: list[AntennaSeries] = []
    for antenna_id in sorted(cells):
        traffic = np.full((num_days, NUM_HOURS), np.nan, dtype=float)
        power = np.full((num_days, NUM_HOURS), np.nan, dtype=float)
        for day_index, day in enumerate(selected_dates):
            for hour, accumulator in cells[antenna_id].get(day, {}).items():
                traffic[day_index, hour] = accumulator[0] / accumulator[2]
                power[day_index, hour] = accumulator[1] / accumulator[2]
        population.append(
            AntennaSeries(
                antenna_id=antenna_id,
                days=selected_dates,
                traffic=traffic,
                power=power,
            )
        )

    audit = DataAudit(
        source_rows=source_rows,
        unparsed_source_rows=unparsed_source_rows,
        selected_rows=selected_rows,
        invalid_selected_rows=invalid_selected_rows,
        negative_selected_rows=negative_selected_rows,
        duplicate_selected_rows=duplicate_selected_rows,
        antenna_count=len(population),
        selected_dates=tuple(day.isoformat() for day in selected_dates),
        inactive_rows=inactive_rows,
        inconsistent_zero_power_rows=inconsistent_zero_power_rows,
    )
    return tuple(population), audit


def active_values(series: AntennaSeries) -> tuple[np.ndarray, np.ndarray]:
    traffic = series.traffic.reshape(-1)
    power = series.power.reshape(-1)
    valid = np.isfinite(traffic) & np.isfinite(power) & (power > 0.0)
    return traffic[valid], power[valid]


def fit_affine(traffic: np.ndarray, power: np.ndarray) -> AffineFit:
    """Fit one unconstrained affine model by ordinary least squares."""
    x = np.asarray(traffic, dtype=float).reshape(-1)
    y = np.asarray(power, dtype=float).reshape(-1)
    if x.size != y.size or x.size < 2:
        raise ValueError("traffic and power must have the same non-trivial size")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("traffic and power must be finite")
    centered_x = x - np.mean(x)
    denominator = float(np.dot(centered_x, centered_x))
    if denominator <= NUMERICAL_TOLERANCE:
        raise ValueError("traffic must vary to identify an affine model")
    centered_y = y - np.mean(y)
    slope = float(np.dot(centered_x, centered_y) / denominator)
    intercept = float(np.mean(y) - slope * np.mean(x))
    prediction = intercept + slope * x
    residual_sum = float(np.sum((y - prediction) ** 2))
    total_sum = float(np.sum(centered_y**2))
    r_squared = (
        1.0 - residual_sum / total_sum
        if total_sum > NUMERICAL_TOLERANCE
        else float("nan")
    )
    rmse = float(np.sqrt(residual_sum / x.size))
    mean_power = float(np.mean(y))
    return AffineFit(
        p_fixed_w=intercept,
        slope_w_per_gb=slope,
        r_squared=r_squared,
        rmse_w=rmse,
        normalized_rmse=rmse / mean_power,
    )


def _empty_calibration(
    series: AntennaSeries,
    status: str,
) -> AntennaCalibration:
    x, y = active_values(series)
    nan = float("nan")
    return AntennaCalibration(
        antenna_id=series.antenna_id,
        status=status,
        num_days=len(series.days),
        num_observations=series.num_observations,
        num_active_observations=series.num_active_observations,
        mean_traffic_gb=float(np.mean(x)) if x.size else nan,
        mean_power_w=float(np.mean(y)) if y.size else nan,
        p_fixed_w=nan,
        slope_w_per_gb=nan,
        r_squared=nan,
        rmse_w=nan,
        normalized_rmse=nan,
        fixed_power_share=nan,
    )


def calibrate_antenna(series: AntennaSeries) -> AntennaCalibration:
    expected_observations = len(series.days) * NUM_HOURS
    if series.num_observations != expected_observations:
        return _empty_calibration(series, "incomplete_period")
    x, y = active_values(series)
    if x.size < 2:
        return _empty_calibration(series, "insufficient_active_observations")
    if np.ptp(x) <= NUMERICAL_TOLERANCE:
        return _empty_calibration(series, "constant_traffic")

    fit = fit_affine(x, y)
    if fit.p_fixed_w <= 0.0 and fit.slope_w_per_gb <= 0.0:
        status = "nonpositive_intercept_and_slope"
    elif fit.p_fixed_w <= 0.0:
        status = "nonpositive_intercept"
    elif fit.slope_w_per_gb <= 0.0:
        status = "nonpositive_slope"
    elif not np.isfinite(fit.r_squared):
        status = "constant_power"
    else:
        status = "included"

    mean_power = float(np.mean(y))
    fixed_share = (
        fit.p_fixed_w / mean_power
        if status == "included" and mean_power > NUMERICAL_TOLERANCE
        else float("nan")
    )
    return AntennaCalibration(
        antenna_id=series.antenna_id,
        status=status,
        num_days=len(series.days),
        num_observations=series.num_observations,
        num_active_observations=series.num_active_observations,
        mean_traffic_gb=float(np.mean(x)),
        mean_power_w=mean_power,
        p_fixed_w=fit.p_fixed_w,
        slope_w_per_gb=fit.slope_w_per_gb,
        r_squared=fit.r_squared,
        rmse_w=fit.rmse_w,
        normalized_rmse=fit.normalized_rmse,
        fixed_power_share=fixed_share,
    )


def run_campaign(
    csv_path: Path,
    num_days: int = DEFAULT_NUM_DAYS,
) -> CalibrationCampaign:
    population, audit = load_population(csv_path, num_days=num_days)
    return CalibrationCampaign(
        audit=audit,
        series=population,
        antenna_results=tuple(calibrate_antenna(series) for series in population),
    )


def selection_counts(results: Iterable[AntennaCalibration]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for result in results:
        counts[result.status] += 1
    counts["total"] = sum(counts.values())
    return dict(counts)


def population_summaries(
    results: Iterable[AntennaCalibration],
) -> list[dict[str, object]]:
    included = tuple(result for result in results if result.status == "included")
    metric_names = (
        "r_squared",
        "normalized_rmse",
        "p_fixed_w",
        "slope_w_per_gb",
        "fixed_power_share",
        "num_active_observations",
    )
    rows: list[dict[str, object]] = []
    for metric in metric_names:
        values = np.asarray(
            [float(getattr(result, metric)) for result in included], dtype=float
        )
        values = values[np.isfinite(values)]
        if not values.size:
            raise ValueError("No included antenna is available for summarisation")
        quantiles = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
        rows.append(
            {
                "metric": metric,
                "n": int(values.size),
                "mean": float(np.mean(values)),
                "q05": float(quantiles[0]),
                "q25": float(quantiles[1]),
                "median": float(quantiles[2]),
                "q75": float(quantiles[3]),
                "q95": float(quantiles[4]),
            }
        )
    return rows


def _equal_size_groups(values: np.ndarray, num_groups: int = 4) -> np.ndarray:
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


def calibrated_population(campaign: CalibrationCampaign) -> CalibratedPopulation:
    """Materialise the reusable inputs of all subsequent simulations."""
    result_by_id = {
        result.antenna_id: result
        for result in campaign.antenna_results
        if result.status == "included"
    }
    selected_series = tuple(
        series for series in campaign.series if series.antenna_id in result_by_id
    )
    if not selected_series:
        raise ValueError("No calibrated antenna is available")
    antenna_ids = np.asarray(
        [series.antenna_id for series in selected_series], dtype=str
    )
    traffic = np.stack([series.traffic for series in selected_series])
    power = np.stack([series.power for series in selected_series])
    p_fixed = np.asarray(
        [result_by_id[antenna_id].p_fixed_w for antenna_id in antenna_ids],
        dtype=float,
    )
    slope = np.asarray(
        [result_by_id[antenna_id].slope_w_per_gb for antenna_id in antenna_ids],
        dtype=float,
    )
    r_squared = np.asarray(
        [result_by_id[antenna_id].r_squared for antenna_id in antenna_ids],
        dtype=float,
    )
    normalized_rmse = np.asarray(
        [result_by_id[antenna_id].normalized_rmse for antenna_id in antenna_ids],
        dtype=float,
    )
    peak_traffic = np.max(traffic, axis=(1, 2))
    return CalibratedPopulation(
        antenna_ids=antenna_ids,
        days=np.asarray(campaign.audit.selected_dates, dtype=str),
        traffic_gb=traffic,
        power_w=power,
        p_fixed_w=p_fixed,
        slope_w_per_gb=slope,
        r_squared=r_squared,
        normalized_rmse=normalized_rmse,
        peak_traffic_gb=peak_traffic,
        traffic_group=_equal_size_groups(peak_traffic),
        fixed_power_group=_equal_size_groups(p_fixed),
    )


def save_calibrated_population(
    population: CalibratedPopulation,
    path: Path,
) -> Path:
    """Store calibrated profiles and coefficients in one compressed file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray([CACHE_SCHEMA_VERSION], dtype=np.int16),
        antenna_ids=population.antenna_ids,
        days=population.days,
        traffic_gb=population.traffic_gb,
        power_w=population.power_w,
        p_fixed_w=population.p_fixed_w,
        slope_w_per_gb=population.slope_w_per_gb,
        r_squared=population.r_squared,
        normalized_rmse=population.normalized_rmse,
        peak_traffic_gb=population.peak_traffic_gb,
        traffic_group=population.traffic_group,
        fixed_power_group=population.fixed_power_group,
    )
    return path


def load_calibrated_population(path: Path) -> CalibratedPopulation:
    """Load simulation inputs without reopening the raw CSV."""
    with np.load(path, allow_pickle=False) as data:
        version = int(data["schema_version"][0])
        if version != CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported calibration cache version {version}; "
                f"expected {CACHE_SCHEMA_VERSION}"
            )
        return CalibratedPopulation(
            antenna_ids=data["antenna_ids"].copy(),
            days=data["days"].copy(),
            traffic_gb=data["traffic_gb"].copy(),
            power_w=data["power_w"].copy(),
            p_fixed_w=data["p_fixed_w"].copy(),
            slope_w_per_gb=data["slope_w_per_gb"].copy(),
            r_squared=data["r_squared"].copy(),
            normalized_rmse=data["normalized_rmse"].copy(),
            peak_traffic_gb=data["peak_traffic_gb"].copy(),
            traffic_group=data["traffic_group"].copy(),
            fixed_power_group=data["fixed_power_group"].copy(),
        )
