"""Population-level validation of the affine traffic--power model.

The statistical unit is an antenna. Hourly observations are grouped by day so
that cross-validation never mixes observations from the held-out day into the
training sample.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import numpy as np


MIN_COMPLETE_DAYS = 5
NUM_HOURS = 24
NUMERICAL_TOLERANCE = 1e-12
DEFAULT_BOOTSTRAP_REPLICATES = 2000
DEFAULT_RANDOM_SEED = 20260806
NON_ANALYZABLE_STATUSES = {
    "insufficient_complete_days",
    "insufficient_active_days",
    "constant_traffic",
    "constant_power",
    "unidentifiable_training_fold",
}


@dataclass(frozen=True)
class DataAudit:
    total_rows: int
    parsed_rows: int
    invalid_rows: int
    negative_rows: int
    duplicate_rows: int
    antenna_count: int
    first_timestamp: str
    last_timestamp: str
    global_outage_timestamps: tuple[str, ...]
    globally_excluded_rows: int
    inactive_state_rows: int
    inconsistent_zero_power_rows: int


@dataclass(frozen=True)
class AntennaSeries:
    antenna_id: str
    days: tuple[date, ...]
    traffic: np.ndarray
    power: np.ndarray

    @property
    def num_days(self) -> int:
        return len(self.days)

    @property
    def num_observations(self) -> int:
        return int(
            np.sum(
                np.isfinite(self.traffic)
                & np.isfinite(self.power)
                & (self.power > 0.0)
            )
        )

    @property
    def num_active_days(self) -> int:
        valid = (
            np.isfinite(self.traffic)
            & np.isfinite(self.power)
            & (self.power > 0.0)
        )
        return int(np.sum(np.any(valid, axis=1)))


@dataclass(frozen=True)
class AffineFit:
    f_tilde: float
    gamma_tilde: float
    sse: float
    r_squared: float


@dataclass(frozen=True)
class QuadraticFit:
    intercept: float
    linear: float
    quadratic: float
    center: float
    scale: float
    scaled_intercept: float
    scaled_linear: float
    scaled_quadratic: float
    sse: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype=float) - self.center) / self.scale
        return (
            self.scaled_intercept
            + self.scaled_linear * z
            + self.scaled_quadratic * z**2
        )


@dataclass(frozen=True)
class AntennaValidation:
    antenna_id: str
    status: str
    num_days: int
    num_active_days: int
    num_observations: int
    mean_traffic: float
    mean_power: float
    traffic_range: float
    power_range: float
    f_tilde: float
    gamma_tilde: float
    r_squared_train: float
    f_tilde_nonnegative: float
    gamma_tilde_nonnegative: float
    r_squared_train_nonnegative: float
    fixed_power_share: float
    cv_r_squared: float
    cv_r_squared_nonnegative: float
    cv_r_squared_quadratic: float
    cv_mae_constant: float
    cv_mae_affine: float
    cv_mae_nonnegative: float
    cv_mae_quadratic: float
    cv_rmse_constant: float
    cv_rmse_affine: float
    cv_rmse_nonnegative: float
    cv_rmse_quadratic: float
    cv_nrmse_affine: float
    cv_nrmse_nonnegative: float
    cv_nrmse_quadratic: float
    cv_rmse_gain_affine: float
    cv_rmse_gain_nonnegative: float
    cv_rmse_gain_quadratic: float
    cv_extrapolation_fraction: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FoldValidation:
    antenna_id: str
    held_out_day: str
    n_train: int
    n_test: int
    train_traffic_min: float
    train_traffic_max: float
    test_traffic_min: float
    test_traffic_max: float
    extrapolation_fraction: float
    f_tilde: float
    gamma_tilde: float
    f_tilde_nonnegative: float
    gamma_tilde_nonnegative: float
    quadratic_intercept: float
    quadratic_linear: float
    quadratic_coefficient: float
    rmse_constant: float
    rmse_affine: float
    rmse_nonnegative: float
    rmse_quadratic: float
    mae_constant: float
    mae_affine: float
    mae_nonnegative: float
    mae_quadratic: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PredictionDiagnostics:
    antenna_id: np.ndarray
    day: np.ndarray
    hour: np.ndarray
    traffic: np.ndarray
    observed: np.ndarray
    predicted_constant: np.ndarray
    predicted_affine: np.ndarray
    predicted_nonnegative: np.ndarray
    predicted_quadratic: np.ndarray
    antenna_mean_power: np.ndarray
    relative_traffic_rank: np.ndarray
    is_extrapolation: np.ndarray


@dataclass(frozen=True)
class ValidationCampaign:
    audit: DataAudit
    series: tuple[AntennaSeries, ...]
    antenna_results: tuple[AntennaValidation, ...]
    fold_results: tuple[FoldValidation, ...]
    diagnostics: PredictionDiagnostics


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


def load_population(csv_path: Path) -> tuple[tuple[AntennaSeries, ...], DataAudit]:
    """Load all antennas and retain days complete after global-outage removal.

    A timestamp is treated as a collection outage only when every antenna has
    an observation at that timestamp and every observation has jointly zero
    traffic and zero power.  Such timestamps are retained as ``NaN`` slots in
    the rectangular day-by-hour arrays and are never used for fitting.
    """
    cells: dict[str, dict[date, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    timestamp_antennas: dict[datetime, set[str]] = defaultdict(set)
    timestamp_has_signal: dict[datetime, bool] = defaultdict(bool)
    timestamp_row_counts: dict[datetime, int] = defaultdict(int)
    timestamp_joint_zero_rows: dict[datetime, int] = defaultdict(int)
    timestamp_zero_power_positive_traffic_rows: dict[datetime, int] = (
        defaultdict(int)
    )
    total_rows = parsed_rows = invalid_rows = negative_rows = duplicate_rows = 0
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None

    with csv_path.open(newline="", encoding="utf-8", errors="replace") as file:
        reader = csv.DictReader(file, delimiter=";")
        date_col, id_col, traffic_col, power_col = _column_names(reader.fieldnames)
        for raw in reader:
            total_rows += 1
            try:
                timestamp = datetime.fromisoformat(raw[date_col].strip())
                antenna_id = raw[id_col].strip()
                traffic = float(raw[traffic_col].replace(",", "."))
                power = float(raw[power_col].replace(",", "."))
                if not antenna_id or not np.isfinite(traffic) or not np.isfinite(power):
                    raise ValueError("invalid record")
            except (KeyError, TypeError, ValueError):
                invalid_rows += 1
                continue

            parsed_rows += 1
            if traffic < 0.0 or power < 0.0:
                negative_rows += 1
                continue

            timestamp_antennas[timestamp].add(antenna_id)
            timestamp_has_signal[timestamp] = (
                timestamp_has_signal[timestamp] or traffic != 0.0 or power != 0.0
            )
            timestamp_row_counts[timestamp] += 1
            if power == 0.0 and traffic == 0.0:
                timestamp_joint_zero_rows[timestamp] += 1
            elif power == 0.0 and traffic > 0.0:
                timestamp_zero_power_positive_traffic_rows[timestamp] += 1

            day_cells = cells[antenna_id][timestamp.date()]
            if timestamp.hour in day_cells:
                duplicate_rows += 1
                accumulator = day_cells[timestamp.hour]
                accumulator[0] += traffic
                accumulator[1] += power
                accumulator[2] += 1.0
            else:
                day_cells[timestamp.hour] = [traffic, power, 1.0]

            if first_timestamp is None or timestamp < first_timestamp:
                first_timestamp = timestamp
            if last_timestamp is None or timestamp > last_timestamp:
                last_timestamp = timestamp

    global_outages = tuple(
        sorted(
            timestamp
            for timestamp, antennas in timestamp_antennas.items()
            if len(antennas) == len(cells) and not timestamp_has_signal[timestamp]
        )
    )
    global_outage_set = set(global_outages)
    expected_hours_by_day: dict[date, set[int]] = defaultdict(set)
    for timestamp in timestamp_antennas:
        if timestamp not in global_outage_set:
            expected_hours_by_day[timestamp.date()].add(timestamp.hour)

    population: list[AntennaSeries] = []
    for antenna_id in sorted(cells):
        complete_days = [
            day
            for day, hours in sorted(cells[antenna_id].items())
            if expected_hours_by_day[day]
            and expected_hours_by_day[day].issubset(hours)
        ]
        traffic_rows: list[list[float]] = []
        power_rows: list[list[float]] = []
        for day in complete_days:
            traffic_row = [float("nan")] * NUM_HOURS
            power_row = [float("nan")] * NUM_HOURS
            for hour in expected_hours_by_day[day]:
                accumulator = cells[antenna_id][day][hour]
                traffic_row[hour] = accumulator[0] / accumulator[2]
                power_row[hour] = accumulator[1] / accumulator[2]
            traffic_rows.append(traffic_row)
            power_rows.append(power_row)
        population.append(
            AntennaSeries(
                antenna_id=antenna_id,
                days=tuple(complete_days),
                traffic=np.asarray(traffic_rows, dtype=float).reshape(
                    len(complete_days), NUM_HOURS
                ),
                power=np.asarray(power_rows, dtype=float).reshape(
                    len(complete_days), NUM_HOURS
                ),
            )
        )

    audit = DataAudit(
        total_rows=total_rows,
        parsed_rows=parsed_rows,
        invalid_rows=invalid_rows,
        negative_rows=negative_rows,
        duplicate_rows=duplicate_rows,
        antenna_count=len(population),
        first_timestamp=first_timestamp.isoformat(sep=" ") if first_timestamp else "",
        last_timestamp=last_timestamp.isoformat(sep=" ") if last_timestamp else "",
        global_outage_timestamps=tuple(
            timestamp.isoformat(sep=" ") for timestamp in global_outages
        ),
        globally_excluded_rows=sum(
            timestamp_row_counts[timestamp] for timestamp in global_outages
        ),
        inactive_state_rows=sum(
            count
            for timestamp, count in timestamp_joint_zero_rows.items()
            if timestamp not in global_outage_set
        ),
        inconsistent_zero_power_rows=sum(
            count
            for timestamp, count in (
                timestamp_zero_power_positive_traffic_rows.items()
            )
            if timestamp not in global_outage_set
        ),
    )
    return tuple(population), audit


def _r_squared(y: np.ndarray, prediction: np.ndarray) -> float:
    residual_sse = float(np.sum((y - prediction) ** 2))
    total_sse = float(np.sum((y - np.mean(y)) ** 2))
    if total_sse <= NUMERICAL_TOLERANCE:
        return float("nan")
    return 1.0 - residual_sse / total_sse


def fit_affine(x: np.ndarray, y: np.ndarray) -> AffineFit:
    """Unconstrained ordinary least squares with an intercept."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size or x.size < 2:
        raise ValueError("x and y must have the same non-trivial size")
    if np.ptp(x) <= NUMERICAL_TOLERANCE:
        raise ValueError("traffic must vary to identify an affine model")
    design = np.column_stack((np.ones_like(x), x))
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    f_tilde, gamma_tilde = map(float, coefficients)
    prediction = design @ coefficients
    sse = float(np.sum((y - prediction) ** 2))
    return AffineFit(
        f_tilde=f_tilde,
        gamma_tilde=gamma_tilde,
        sse=sse,
        r_squared=_r_squared(y, prediction),
    )


def fit_nonnegative_affine(x: np.ndarray, y: np.ndarray) -> AffineFit:
    """Solve the two-variable non-negative least-squares problem exactly.

    Convexity implies that the optimum is either the unconstrained OLS point,
    one of the two coordinate-boundary optima, or the origin.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size or x.size < 2:
        raise ValueError("x and y must have the same non-trivial size")
    if np.ptp(x) <= NUMERICAL_TOLERANCE:
        raise ValueError("traffic must vary to identify an affine model")

    free = fit_affine(x, y)
    candidates: list[tuple[float, float]] = [(0.0, 0.0)]
    if free.f_tilde >= 0.0 and free.gamma_tilde >= 0.0:
        candidates.append((free.f_tilde, free.gamma_tilde))

    candidates.append((max(0.0, float(np.mean(y))), 0.0))
    x_square = float(np.dot(x, x))
    boundary_slope = (
        max(0.0, float(np.dot(x, y)) / x_square)
        if x_square > NUMERICAL_TOLERANCE
        else 0.0
    )
    candidates.append((0.0, boundary_slope))

    best = min(
        candidates,
        key=lambda pair: (
            float(np.sum((y - (pair[0] + pair[1] * x)) ** 2)),
            pair[0],
            pair[1],
        ),
    )
    prediction = best[0] + best[1] * x
    sse = float(np.sum((y - prediction) ** 2))
    return AffineFit(
        f_tilde=best[0],
        gamma_tilde=best[1],
        sse=sse,
        r_squared=_r_squared(y, prediction),
    )


def fit_quadratic(x: np.ndarray, y: np.ndarray) -> QuadraticFit:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size or x.size < 3:
        raise ValueError("a quadratic model requires at least three observations")
    center = float(np.mean(x))
    scale = float(np.std(x))
    if scale <= NUMERICAL_TOLERANCE:
        raise ValueError("traffic must vary to identify a quadratic model")
    z = (x - center) / scale
    design = np.column_stack((np.ones_like(z), z, z**2))
    scaled_coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank < 3:
        raise ValueError("at least three distinct traffic levels are required")
    prediction = design @ scaled_coefficients
    quadratic = float(scaled_coefficients[2] / scale**2)
    linear = float(
        scaled_coefficients[1] / scale
        - 2.0 * scaled_coefficients[2] * center / scale**2
    )
    intercept = float(
        scaled_coefficients[0]
        - scaled_coefficients[1] * center / scale
        + scaled_coefficients[2] * center**2 / scale**2
    )
    return QuadraticFit(
        intercept=intercept,
        linear=linear,
        quadratic=quadratic,
        center=center,
        scale=scale,
        scaled_intercept=float(scaled_coefficients[0]),
        scaled_linear=float(scaled_coefficients[1]),
        scaled_quadratic=float(scaled_coefficients[2]),
        sse=float(np.sum((y - prediction) ** 2)),
    )


def _valid_values(
    series: AntennaSeries,
    day_indices: Iterable[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    indices = (
        np.arange(series.num_days, dtype=int)
        if day_indices is None
        else np.asarray(tuple(day_indices), dtype=int)
    )
    traffic = series.traffic[indices].reshape(-1)
    power = series.power[indices].reshape(-1)
    # The validated equation describes the active state.  Exact zero power is
    # the separate sleep state (or an inconsistent record if traffic > 0).
    valid = np.isfinite(traffic) & np.isfinite(power) & (power > 0.0)
    return traffic[valid], power[valid]


def _empty_result(series: AntennaSeries, status: str) -> AntennaValidation:
    nan = float("nan")
    x, y = _valid_values(series)
    return AntennaValidation(
        antenna_id=series.antenna_id,
        status=status,
        num_days=series.num_days,
        num_active_days=series.num_active_days,
        num_observations=series.num_observations,
        mean_traffic=float(np.mean(x)) if x.size else nan,
        mean_power=float(np.mean(y)) if y.size else nan,
        traffic_range=float(np.ptp(x)) if x.size else nan,
        power_range=float(np.ptp(y)) if y.size else nan,
        f_tilde=nan,
        gamma_tilde=nan,
        r_squared_train=nan,
        f_tilde_nonnegative=nan,
        gamma_tilde_nonnegative=nan,
        r_squared_train_nonnegative=nan,
        fixed_power_share=nan,
        cv_r_squared=nan,
        cv_r_squared_nonnegative=nan,
        cv_r_squared_quadratic=nan,
        cv_mae_constant=nan,
        cv_mae_affine=nan,
        cv_mae_nonnegative=nan,
        cv_mae_quadratic=nan,
        cv_rmse_constant=nan,
        cv_rmse_affine=nan,
        cv_rmse_nonnegative=nan,
        cv_rmse_quadratic=nan,
        cv_nrmse_affine=nan,
        cv_nrmse_nonnegative=nan,
        cv_nrmse_quadratic=nan,
        cv_rmse_gain_affine=nan,
        cv_rmse_gain_nonnegative=nan,
        cv_rmse_gain_quadratic=nan,
        cv_extrapolation_fraction=nan,
    )


def _error_metrics(y: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    errors = y - prediction
    return float(np.mean(np.abs(errors))), float(np.sqrt(np.mean(errors**2)))


def _relative_ranks(values: np.ndarray) -> np.ndarray:
    """Return deterministic empirical percentile ranks in [0, 1]."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, values.size)
    return ranks


def validate_antenna(
    series: AntennaSeries,
    min_complete_days: int = MIN_COMPLETE_DAYS,
) -> tuple[AntennaValidation, list[FoldValidation], dict[str, np.ndarray] | None]:
    """Validate one antenna with leave-one-complete-day-out folds."""
    if series.num_days < min_complete_days:
        return _empty_result(series, "insufficient_complete_days"), [], None

    active_day_indices = [
        index
        for index in range(series.num_days)
        if np.any(
            np.isfinite(series.traffic[index])
            & np.isfinite(series.power[index])
            & (series.power[index] > 0.0)
        )
    ]
    if len(active_day_indices) < min_complete_days:
        return _empty_result(series, "insufficient_active_days"), [], None

    x_all, y_all = _valid_values(series, active_day_indices)
    if np.ptp(x_all) <= NUMERICAL_TOLERANCE:
        return _empty_result(series, "constant_traffic"), [], None
    if np.ptp(y_all) <= NUMERICAL_TOLERANCE:
        return _empty_result(series, "constant_power"), [], None
    for held_out_index in active_day_indices:
        training_indices = [
            index for index in active_day_indices if index != held_out_index
        ]
        x_train, _ = _valid_values(series, training_indices)
        if np.ptp(x_train) <= NUMERICAL_TOLERANCE:
            return _empty_result(series, "unidentifiable_training_fold"), [], None

    free_full = fit_affine(x_all, y_all)
    nonnegative_full = fit_nonnegative_affine(x_all, y_all)
    if free_full.f_tilde <= 0.0 and free_full.gamma_tilde <= 0.0:
        status = "nonpositive_intercept_and_slope"
    elif free_full.f_tilde <= 0.0:
        status = "nonpositive_intercept"
    elif free_full.gamma_tilde <= 0.0:
        status = "nonpositive_slope"
    else:
        status = "included"

    fold_rows: list[FoldValidation] = []
    diagnostics = {
        "day": [],
        "hour": [],
        "traffic": [],
        "observed": [],
        "constant": [],
        "affine": [],
        "nonnegative": [],
        "quadratic": [],
        "is_extrapolation": [],
    }
    absolute_error_sums = {
        "constant": 0.0,
        "affine": 0.0,
        "nonnegative": 0.0,
        "quadratic": 0.0,
    }
    squared_error_sums = {key: 0.0 for key in absolute_error_sums}
    total_test_observations = 0
    total_extrapolated_observations = 0

    for held_out_index in active_day_indices:
        held_out_day = series.days[held_out_index]
        training_indices = [
            index for index in active_day_indices if index != held_out_index
        ]
        x_train, y_train = _valid_values(series, training_indices)
        valid_test = (
            np.isfinite(series.traffic[held_out_index])
            & np.isfinite(series.power[held_out_index])
            & (series.power[held_out_index] > 0.0)
        )
        test_hours = np.flatnonzero(valid_test)
        x_test = series.traffic[held_out_index, valid_test]
        y_test = series.power[held_out_index, valid_test]
        train_min = float(np.min(x_train))
        train_max = float(np.max(x_train))
        extrapolation = (x_test < train_min) | (x_test > train_max)

        free = fit_affine(x_train, y_train)
        nonnegative = fit_nonnegative_affine(x_train, y_train)
        quadratic = fit_quadratic(x_train, y_train)
        predictions = {
            "constant": np.full_like(y_test, float(np.mean(y_train))),
            "affine": free.f_tilde + free.gamma_tilde * x_test,
            "nonnegative": (
                nonnegative.f_tilde + nonnegative.gamma_tilde * x_test
            ),
            "quadratic": quadratic.predict(x_test),
        }

        metrics: dict[str, tuple[float, float]] = {}
        for name, prediction in predictions.items():
            errors = y_test - prediction
            absolute_error_sums[name] += float(np.sum(np.abs(errors)))
            squared_error_sums[name] += float(np.sum(errors**2))
            metrics[name] = _error_metrics(y_test, prediction)

        total_test_observations += y_test.size
        total_extrapolated_observations += int(np.sum(extrapolation))
        fold_rows.append(
            FoldValidation(
                antenna_id=series.antenna_id,
                held_out_day=held_out_day.isoformat(),
                n_train=x_train.size,
                n_test=x_test.size,
                train_traffic_min=train_min,
                train_traffic_max=train_max,
                test_traffic_min=float(np.min(x_test)),
                test_traffic_max=float(np.max(x_test)),
                extrapolation_fraction=float(np.mean(extrapolation)),
                f_tilde=free.f_tilde,
                gamma_tilde=free.gamma_tilde,
                f_tilde_nonnegative=nonnegative.f_tilde,
                gamma_tilde_nonnegative=nonnegative.gamma_tilde,
                quadratic_intercept=quadratic.intercept,
                quadratic_linear=quadratic.linear,
                quadratic_coefficient=quadratic.quadratic,
                rmse_constant=metrics["constant"][1],
                rmse_affine=metrics["affine"][1],
                rmse_nonnegative=metrics["nonnegative"][1],
                rmse_quadratic=metrics["quadratic"][1],
                mae_constant=metrics["constant"][0],
                mae_affine=metrics["affine"][0],
                mae_nonnegative=metrics["nonnegative"][0],
                mae_quadratic=metrics["quadratic"][0],
            )
        )
        diagnostics["day"].append(
            np.full(x_test.size, held_out_day.isoformat(), dtype=object)
        )
        diagnostics["hour"].append(test_hours)
        diagnostics["traffic"].append(x_test.copy())
        diagnostics["observed"].append(y_test.copy())
        for name in ("constant", "affine", "nonnegative", "quadratic"):
            diagnostics[name].append(predictions[name].copy())
        diagnostics["is_extrapolation"].append(extrapolation.copy())

    sse_constant = squared_error_sums["constant"]
    cv_r_squared = {
        name: (
            1.0 - squared_error_sums[name] / sse_constant
            if sse_constant > NUMERICAL_TOLERANCE
            else float("nan")
        )
        for name in ("affine", "nonnegative", "quadratic")
    }
    rmse = {
        name: float(np.sqrt(squared_error_sums[name] / total_test_observations))
        for name in absolute_error_sums
    }
    mae = {
        name: absolute_error_sums[name] / total_test_observations
        for name in absolute_error_sums
    }
    mean_power = float(np.mean(y_all))
    fixed_denominator = (
        nonnegative_full.f_tilde
        + nonnegative_full.gamma_tilde * float(np.mean(x_all))
    )
    fixed_share = (
        nonnegative_full.f_tilde / fixed_denominator
        if fixed_denominator > NUMERICAL_TOLERANCE
        else float("nan")
    )

    result = AntennaValidation(
        antenna_id=series.antenna_id,
        status=status,
        num_days=series.num_days,
        num_active_days=len(active_day_indices),
        num_observations=series.num_observations,
        mean_traffic=float(np.mean(x_all)),
        mean_power=mean_power,
        traffic_range=float(np.ptp(x_all)),
        power_range=float(np.ptp(y_all)),
        f_tilde=free_full.f_tilde,
        gamma_tilde=free_full.gamma_tilde,
        r_squared_train=free_full.r_squared,
        f_tilde_nonnegative=nonnegative_full.f_tilde,
        gamma_tilde_nonnegative=nonnegative_full.gamma_tilde,
        r_squared_train_nonnegative=nonnegative_full.r_squared,
        fixed_power_share=fixed_share,
        cv_r_squared=cv_r_squared["affine"],
        cv_r_squared_nonnegative=cv_r_squared["nonnegative"],
        cv_r_squared_quadratic=cv_r_squared["quadratic"],
        cv_mae_constant=mae["constant"],
        cv_mae_affine=mae["affine"],
        cv_mae_nonnegative=mae["nonnegative"],
        cv_mae_quadratic=mae["quadratic"],
        cv_rmse_constant=rmse["constant"],
        cv_rmse_affine=rmse["affine"],
        cv_rmse_nonnegative=rmse["nonnegative"],
        cv_rmse_quadratic=rmse["quadratic"],
        cv_nrmse_affine=rmse["affine"] / mean_power,
        cv_nrmse_nonnegative=rmse["nonnegative"] / mean_power,
        cv_nrmse_quadratic=rmse["quadratic"] / mean_power,
        cv_rmse_gain_affine=1.0 - rmse["affine"] / rmse["constant"],
        cv_rmse_gain_nonnegative=1.0 - rmse["nonnegative"] / rmse["constant"],
        cv_rmse_gain_quadratic=1.0 - rmse["quadratic"] / rmse["constant"],
        cv_extrapolation_fraction=(
            total_extrapolated_observations / total_test_observations
        ),
    )
    concatenated = {
        key: np.concatenate(value) for key, value in diagnostics.items()
    }
    concatenated["relative_traffic_rank"] = _relative_ranks(
        concatenated["traffic"]
    )
    return result, fold_rows, concatenated


def run_campaign(
    csv_path: Path,
    min_complete_days: int = MIN_COMPLETE_DAYS,
) -> ValidationCampaign:
    population, audit = load_population(csv_path)
    antenna_results: list[AntennaValidation] = []
    fold_results: list[FoldValidation] = []
    diagnostic_blocks: dict[str, list[np.ndarray]] = defaultdict(list)

    for series in population:
        result, folds, diagnostics = validate_antenna(series, min_complete_days)
        antenna_results.append(result)
        fold_results.extend(folds)
        if diagnostics is None:
            continue
        size = diagnostics["traffic"].size
        diagnostic_blocks["antenna_id"].append(
            np.full(size, series.antenna_id, dtype=object)
        )
        diagnostic_blocks["antenna_mean_power"].append(
            np.full(size, result.mean_power, dtype=float)
        )
        for key, values in diagnostics.items():
            diagnostic_blocks[key].append(values)

    concatenated = {
        key: np.concatenate(blocks) if blocks else np.array([])
        for key, blocks in diagnostic_blocks.items()
    }
    campaign_diagnostics = PredictionDiagnostics(
        antenna_id=concatenated["antenna_id"],
        day=concatenated["day"],
        hour=concatenated["hour"].astype(int),
        traffic=concatenated["traffic"].astype(float),
        observed=concatenated["observed"].astype(float),
        predicted_constant=concatenated["constant"].astype(float),
        predicted_affine=concatenated["affine"].astype(float),
        predicted_nonnegative=concatenated["nonnegative"].astype(float),
        predicted_quadratic=concatenated["quadratic"].astype(float),
        antenna_mean_power=concatenated["antenna_mean_power"].astype(float),
        relative_traffic_rank=concatenated["relative_traffic_rank"].astype(float),
        is_extrapolation=concatenated["is_extrapolation"].astype(bool),
    )
    return ValidationCampaign(
        audit=audit,
        series=population,
        antenna_results=tuple(antenna_results),
        fold_results=tuple(fold_results),
        diagnostics=campaign_diagnostics,
    )


def selection_counts(
    results: Iterable[AntennaValidation],
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for result in results:
        counts[result.status] += 1
    counts["total"] = sum(
        count for status, count in counts.items() if status != "total"
    )
    counts["statistically_analyzable"] = sum(
        count
        for status, count in counts.items()
        if status not in NON_ANALYZABLE_STATUSES | {"total"}
    )
    counts["operationally_admissible"] = counts.get("included", 0)
    return dict(counts)


def _bootstrap_interval(
    values: np.ndarray,
    statistic: str,
    rng: np.random.Generator,
    replicates: int,
) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    estimates = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sample = values[rng.integers(0, values.size, size=values.size)]
        estimates[index] = (
            float(np.mean(sample))
            if statistic == "mean"
            else float(np.median(sample))
        )
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper)


def population_summaries(
    results: Iterable[AntennaValidation],
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_RANDOM_SEED,
) -> list[dict[str, object]]:
    """Summarise antenna-level quantities with antenna bootstrap intervals."""
    all_results = tuple(results)
    populations = {
        "analyzable": tuple(
            result
            for result in all_results
            if result.status not in NON_ANALYZABLE_STATUSES
        ),
        "operational": tuple(
            result for result in all_results if result.status == "included"
        ),
    }
    metric_specs = (
        ("r_squared_train", "median"),
        ("cv_r_squared", "median"),
        ("cv_r_squared_nonnegative", "median"),
        ("cv_r_squared_quadratic", "median"),
        ("cv_nrmse_affine", "median"),
        ("cv_nrmse_nonnegative", "median"),
        ("cv_nrmse_quadratic", "median"),
        ("cv_rmse_gain_affine", "median"),
        ("cv_rmse_gain_nonnegative", "median"),
        ("cv_rmse_gain_quadratic", "median"),
        ("cv_extrapolation_fraction", "median"),
        ("f_tilde", "median"),
        ("gamma_tilde", "median"),
        ("fixed_power_share", "median"),
    )
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for population_name, population in populations.items():
        for metric, statistic in metric_specs:
            values = np.asarray(
                [
                    getattr(result, metric)
                    for result in population
                    if np.isfinite(getattr(result, metric))
                ],
                dtype=float,
            )
            lower, upper = _bootstrap_interval(
                values, statistic, rng, bootstrap_replicates
            )
            quantiles = (
                np.quantile(
                    values,
                    [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
                )
                if values.size
                else np.full(7, np.nan)
            )
            rows.append(
                {
                    "population": population_name,
                    "metric": metric,
                    "n": int(values.size),
                    "mean": float(np.mean(values)) if values.size else float("nan"),
                    "q01": float(quantiles[0]),
                    "q05": float(quantiles[1]),
                    "q25": float(quantiles[2]),
                    "median": float(quantiles[3]),
                    "q75": float(quantiles[4]),
                    "q95": float(quantiles[5]),
                    "q99": float(quantiles[6]),
                    "bootstrap_ci_lower": lower,
                    "bootstrap_ci_upper": upper,
                    "bootstrap_replicates": bootstrap_replicates,
                    "bootstrap_unit": "antenna",
                }
            )

        for metric in (
            "cv_rmse_gain_affine",
            "cv_rmse_gain_nonnegative",
            "cv_rmse_gain_quadratic",
        ):
            values = np.asarray(
                [
                    float(getattr(result, metric) > 0.0)
                    for result in population
                    if np.isfinite(getattr(result, metric))
                ],
                dtype=float,
            )
            lower, upper = _bootstrap_interval(
                values, "mean", rng, bootstrap_replicates
            )
            proportion = float(np.mean(values)) if values.size else float("nan")
            rows.append(
                {
                    "population": population_name,
                    "metric": f"proportion_{metric}_positive",
                    "n": int(values.size),
                    "mean": proportion,
                    "q01": proportion,
                    "q05": proportion,
                    "q25": proportion,
                    "median": proportion,
                    "q75": proportion,
                    "q95": proportion,
                    "q99": proportion,
                    "bootstrap_ci_lower": lower,
                    "bootstrap_ci_upper": upper,
                    "bootstrap_replicates": bootstrap_replicates,
                    "bootstrap_unit": "antenna",
                }
            )

        paired_values = np.asarray(
            [
                float(result.cv_rmse_quadratic < result.cv_rmse_affine)
                for result in population
                if np.isfinite(result.cv_rmse_quadratic)
                and np.isfinite(result.cv_rmse_affine)
            ],
            dtype=float,
        )
        lower, upper = _bootstrap_interval(
            paired_values, "mean", rng, bootstrap_replicates
        )
        proportion = (
            float(np.mean(paired_values))
            if paired_values.size
            else float("nan")
        )
        rows.append(
            {
                "population": population_name,
                "metric": "proportion_quadratic_rmse_below_affine",
                "n": int(paired_values.size),
                "mean": proportion,
                "q01": proportion,
                "q05": proportion,
                "q25": proportion,
                "median": proportion,
                "q75": proportion,
                "q95": proportion,
                "q99": proportion,
                "bootstrap_ci_lower": lower,
                "bootstrap_ci_upper": upper,
                "bootstrap_replicates": bootstrap_replicates,
                "bootstrap_unit": "antenna",
            }
        )
    return rows


def representative_antenna_id(
    results: Iterable[AntennaValidation],
) -> str:
    """Select the robust multivariate medoid of the admissible population."""
    operational = [
        result
        for result in results
        if result.status == "included"
        and np.isfinite(result.cv_r_squared)
        and np.isfinite(result.cv_nrmse_affine)
        and np.isfinite(result.fixed_power_share)
    ]
    if not operational:
        raise ValueError("No operationally admissible antenna is available")
    values = np.asarray(
        [
            (
                result.cv_r_squared,
                result.cv_nrmse_affine,
                result.fixed_power_share,
            )
            for result in operational
        ],
        dtype=float,
    )
    target = np.median(values, axis=0)
    scale = np.quantile(values, 0.75, axis=0) - np.quantile(
        values, 0.25, axis=0
    )
    scale[scale <= NUMERICAL_TOLERANCE] = 1.0
    distances = np.sum(((values - target) / scale) ** 2, axis=1)
    return min(
        zip(operational, distances, strict=True),
        key=lambda pair: (float(pair[1]), pair[0].antenna_id),
    )[0].antenna_id
