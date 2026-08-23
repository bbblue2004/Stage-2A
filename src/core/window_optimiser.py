"""Exact hourly optimisation and the secondary persistent-guardian policy."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


TOLERANCE = 1e-9


@dataclass(frozen=True)
class WindowSolution:
    """A policy using one guardian set throughout the window."""

    energy_wh: float
    guardians: tuple[int, ...]
    allocations_gb: np.ndarray

    @property
    def num_guardians(self) -> int:
        return len(self.guardians)


@dataclass(frozen=True)
class HourlySolution:
    """A policy that may select a different guardian set at each hour."""

    energy_wh: float
    guardian_masks: np.ndarray
    allocations_gb: np.ndarray

    @property
    def guardian_sets(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(
                player
                for player in range(self.allocations_gb.shape[0])
                if int(mask) & (1 << player)
            )
            for mask in self.guardian_masks
        )

    @property
    def guardian_counts(self) -> np.ndarray:
        return np.asarray(
            [int(mask).bit_count() for mask in self.guardian_masks],
            dtype=int,
        )

    @property
    def mean_guardians(self) -> float:
        return float(np.mean(self.guardian_counts))

    @property
    def guardian_changes(self) -> int:
        return int(np.sum(self.guardian_masks[1:] != self.guardian_masks[:-1]))


@dataclass(frozen=True)
class CoalitionWindowSolutions:
    """Exact hourly and persistent costs for every coalition."""

    hourly_costs_wh: np.ndarray
    persistent_costs_wh: np.ndarray
    hourly_guardian_masks: np.ndarray
    persistent_guardian_masks: np.ndarray
    hourly_costs_by_hour_wh: np.ndarray


def _validate(
    capacities_gb: np.ndarray,
    fixed_power_w: np.ndarray,
    slopes_w_per_gb: np.ndarray,
    demands_gb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    capacities = np.asarray(capacities_gb, dtype=float)
    fixed = np.asarray(fixed_power_w, dtype=float)
    slopes = np.asarray(slopes_w_per_gb, dtype=float)
    demands = np.asarray(demands_gb, dtype=float)
    if capacities.ndim != 1 or demands.ndim != 2:
        raise ValueError("capacities must be one-dimensional and demands two-dimensional")
    if fixed.shape != capacities.shape or slopes.shape != capacities.shape:
        raise ValueError("all operator parameter arrays must have the same shape")
    if demands.shape[0] != capacities.size or demands.shape[1] == 0:
        raise ValueError("demands must contain one non-empty row per operator")
    if not all(
        np.all(np.isfinite(values))
        for values in (capacities, fixed, slopes, demands)
    ):
        raise ValueError("all operational inputs must be finite")
    if (
        np.any(capacities <= 0.0)
        or np.any(fixed <= 0.0)
        or np.any(slopes <= 0.0)
    ):
        raise ValueError("capacities and power coefficients must be positive")
    if np.any(demands < 0.0):
        raise ValueError("traffic demands must be non-negative")
    return capacities, fixed, slopes, demands


def _allocation(
    guardians: tuple[int, ...],
    capacities: np.ndarray,
    slopes: np.ndarray,
    total_demand: np.ndarray,
    proportional: bool,
) -> np.ndarray:
    capacity = float(np.sum(capacities[list(guardians)]))
    if np.max(total_demand) > capacity + TOLERANCE:
        raise ValueError("the guardian set is infeasible")
    allocation = np.zeros((capacities.size, total_demand.size), dtype=float)
    if proportional:
        for guardian in guardians:
            allocation[guardian] = (
                total_demand * capacities[guardian] / capacity
            )
        return allocation

    remaining = total_demand.copy()
    for guardian in sorted(
        guardians, key=lambda index: (slopes[index], index)
    ):
        allocation[guardian] = np.minimum(capacities[guardian], remaining)
        remaining -= allocation[guardian]
        remaining[remaining < TOLERANCE] = 0.0
    if np.max(remaining) > TOLERANCE:
        raise RuntimeError("greedy allocation left unserved demand")
    return allocation


def _mask_members(mask: int, num_players: int) -> tuple[int, ...]:
    return tuple(
        player for player in range(num_players) if mask & (1 << player)
    )


def _evaluate_persistent_validated(
    guardians: tuple[int, ...],
    capacities: np.ndarray,
    fixed: np.ndarray,
    slopes: np.ndarray,
    demands: np.ndarray,
    proportional: bool = False,
) -> WindowSolution:
    allocation = _allocation(
        guardians,
        capacities,
        slopes,
        np.sum(demands, axis=0),
        proportional,
    )
    energy = (
        demands.shape[1] * float(np.sum(fixed[list(guardians)]))
        + float(np.sum(slopes[:, None] * allocation))
    )
    return WindowSolution(energy, guardians, allocation)


def evaluate_persistent_guardians(
    guardians: tuple[int, ...],
    capacities_gb: np.ndarray,
    fixed_power_w: np.ndarray,
    slopes_w_per_gb: np.ndarray,
    demands_gb: np.ndarray,
    *,
    proportional: bool = False,
) -> WindowSolution:
    capacities, fixed, slopes, demands = _validate(
        capacities_gb, fixed_power_w, slopes_w_per_gb, demands_gb
    )
    selected = tuple(sorted(set(guardians)))
    if not selected or any(
        index < 0 or index >= capacities.size for index in selected
    ):
        raise ValueError("guardians must be a non-empty subset of operators")
    return _evaluate_persistent_validated(
        selected, capacities, fixed, slopes, demands, proportional
    )


def _evaluate_hourly_validated(
    guardian_masks: np.ndarray,
    capacities: np.ndarray,
    fixed: np.ndarray,
    slopes: np.ndarray,
    demands: np.ndarray,
    proportional: bool = False,
) -> HourlySolution:
    masks = np.asarray(guardian_masks, dtype=np.int64)
    if masks.shape != (demands.shape[1],) or np.any(masks <= 0):
        raise ValueError("one non-empty guardian mask is required per hour")
    allocations = np.zeros_like(demands)
    energy = 0.0
    maximum_mask = 1 << capacities.size
    for hour, mask_value in enumerate(masks):
        mask = int(mask_value)
        if mask >= maximum_mask:
            raise ValueError("a guardian mask selects an unknown operator")
        guardians = _mask_members(mask, capacities.size)
        allocation = _allocation(
            guardians,
            capacities,
            slopes,
            np.asarray([float(np.sum(demands[:, hour]))]),
            proportional,
        )[:, 0]
        allocations[:, hour] = allocation
        energy += float(np.sum(fixed[list(guardians)]))
        energy += float(np.dot(slopes, allocation))
    return HourlySolution(float(energy), masks.copy(), allocations)


def evaluate_hourly_guardians(
    guardian_masks: np.ndarray,
    capacities_gb: np.ndarray,
    fixed_power_w: np.ndarray,
    slopes_w_per_gb: np.ndarray,
    demands_gb: np.ndarray,
    *,
    proportional: bool = False,
) -> HourlySolution:
    capacities, fixed, slopes, demands = _validate(
        capacities_gb, fixed_power_w, slopes_w_per_gb, demands_gb
    )
    return _evaluate_hourly_validated(
        guardian_masks, capacities, fixed, slopes, demands, proportional
    )


def optimal_hourly_with_persistent(
    capacities_gb: np.ndarray,
    fixed_power_w: np.ndarray,
    slopes_w_per_gb: np.ndarray,
    demands_gb: np.ndarray,
) -> tuple[HourlySolution, WindowSolution]:
    """Compute the hourly optimum and persistent extension in one enumeration."""
    capacities, fixed, slopes, demands = _validate(
        capacities_gb, fixed_power_w, slopes_w_per_gb, demands_gb
    )
    horizon = demands.shape[1]
    total_demand = np.sum(demands, axis=0)
    hourly_keys: list[tuple[float, int, tuple[int, ...]] | None] = [
        None for _ in range(horizon)
    ]
    hourly_masks = np.zeros(horizon, dtype=np.int64)
    persistent: WindowSolution | None = None
    indices = tuple(range(capacities.size))

    for size in range(1, capacities.size + 1):
        for guardians in combinations(indices, size):
            guardian_capacity = float(
                np.sum(capacities[list(guardians)])
            )
            feasible_hours = total_demand <= guardian_capacity + TOLERANCE
            if not np.any(feasible_hours):
                continue
            feasible_demands = np.where(feasible_hours, total_demand, 0.0)
            allocation = _allocation(
                guardians, capacities, slopes, feasible_demands, False
            )
            hourly_cost = (
                float(np.sum(fixed[list(guardians)]))
                + np.sum(slopes[:, None] * allocation, axis=0)
            )
            mask = sum(1 << guardian for guardian in guardians)
            for hour in np.flatnonzero(feasible_hours):
                key = (float(hourly_cost[hour]), size, guardians)
                if hourly_keys[hour] is None or key < hourly_keys[hour]:
                    hourly_keys[hour] = key
                    hourly_masks[hour] = mask

            if not np.all(feasible_hours):
                continue
            candidate = WindowSolution(
                float(np.sum(hourly_cost)), guardians, allocation
            )
            candidate_key = (
                candidate.energy_wh,
                candidate.num_guardians,
                candidate.guardians,
            )
            persistent_key = (
                (
                    persistent.energy_wh,
                    persistent.num_guardians,
                    persistent.guardians,
                )
                if persistent is not None
                else (float("inf"), capacities.size + 1, ())
            )
            if candidate_key < persistent_key:
                persistent = candidate

    if any(key is None for key in hourly_keys):
        raise ValueError("at least one hour cannot be served")
    if persistent is None:
        raise ValueError("no guardian set can serve the complete window")
    hourly = _evaluate_hourly_validated(
        hourly_masks, capacities, fixed, slopes, demands
    )
    return hourly, persistent


def optimal_hourly_policy(
    capacities_gb: np.ndarray,
    fixed_power_w: np.ndarray,
    slopes_w_per_gb: np.ndarray,
    demands_gb: np.ndarray,
) -> HourlySolution:
    return optimal_hourly_with_persistent(
        capacities_gb, fixed_power_w, slopes_w_per_gb, demands_gb
    )[0]


def optimal_persistent_policy(
    capacities_gb: np.ndarray,
    fixed_power_w: np.ndarray,
    slopes_w_per_gb: np.ndarray,
    demands_gb: np.ndarray,
) -> WindowSolution:
    return optimal_hourly_with_persistent(
        capacities_gb, fixed_power_w, slopes_w_per_gb, demands_gb
    )[1]


def coalition_window_solutions(
    capacities_gb: np.ndarray,
    fixed_power_w: np.ndarray,
    slopes_w_per_gb: np.ndarray,
    demands_gb: np.ndarray,
) -> CoalitionWindowSolutions:
    """Solve every coalition under hourly and persistent guardian choices."""
    capacities, fixed, slopes, demands = _validate(
        capacities_gb, fixed_power_w, slopes_w_per_gb, demands_gb
    )
    num_players = capacities.size
    num_masks = 1 << num_players
    horizon = demands.shape[1]

    members: list[tuple[int, ...]] = [tuple() for _ in range(num_masks)]
    total_demands = np.zeros((num_masks, horizon), dtype=float)
    capacity_sums = np.zeros(num_masks, dtype=float)
    fixed_sums = np.zeros(num_masks, dtype=float)
    for mask in range(1, num_masks):
        least_bit = mask & -mask
        player = least_bit.bit_length() - 1
        previous = mask ^ least_bit
        members[mask] = members[previous] + (player,)
        total_demands[mask] = total_demands[previous] + demands[player]
        capacity_sums[mask] = capacity_sums[previous] + capacities[player]
        fixed_sums[mask] = fixed_sums[previous] + fixed[player]

    hourly_costs = np.full(num_masks, np.inf, dtype=float)
    persistent_costs = np.full(num_masks, np.inf, dtype=float)
    hourly_costs[0] = persistent_costs[0] = 0.0
    hourly_masks = np.zeros((num_masks, horizon), dtype=np.int64)
    persistent_masks = np.zeros(num_masks, dtype=np.int64)
    hourly_costs_by_hour = np.full((num_masks, horizon), np.inf, dtype=float)
    hourly_costs_by_hour[0] = 0.0

    for coalition_mask in range(1, num_masks):
        total_demand = total_demands[coalition_mask]
        best_hourly = np.full(horizon, np.inf, dtype=float)
        best_hourly_keys: list[
            tuple[float, int, tuple[int, ...]] | None
        ] = [None for _ in range(horizon)]
        best_persistent_key: tuple[float, int, tuple[int, ...]] | None = None
        guardian_mask = coalition_mask
        while guardian_mask:
            feasible_hours = (
                total_demand <= capacity_sums[guardian_mask] + TOLERANCE
            )
            if np.any(feasible_hours):
                remaining = np.where(feasible_hours, total_demand, 0.0)
                variable_by_hour = np.zeros(horizon, dtype=float)
                for player in sorted(
                    members[guardian_mask],
                    key=lambda index: (slopes[index], index),
                ):
                    allocation = np.minimum(capacities[player], remaining)
                    variable_by_hour += slopes[player] * allocation
                    remaining -= allocation
                    remaining[remaining < TOLERANCE] = 0.0
                if np.max(remaining) > TOLERANCE:
                    raise RuntimeError("greedy allocation left unserved demand")
                candidate_costs = fixed_sums[guardian_mask] + variable_by_hour
                size = len(members[guardian_mask])
                for hour in np.flatnonzero(feasible_hours):
                    key = (
                        float(candidate_costs[hour]),
                        size,
                        members[guardian_mask],
                    )
                    if (
                        best_hourly_keys[hour] is None
                        or key < best_hourly_keys[hour]
                    ):
                        best_hourly_keys[hour] = key
                        best_hourly[hour] = candidate_costs[hour]
                        hourly_masks[coalition_mask, hour] = guardian_mask
                if np.all(feasible_hours):
                    persistent_key = (
                        float(np.sum(candidate_costs)),
                        size,
                        members[guardian_mask],
                    )
                    if (
                        best_persistent_key is None
                        or persistent_key < best_persistent_key
                    ):
                        best_persistent_key = persistent_key
                        persistent_costs[coalition_mask] = persistent_key[0]
                        persistent_masks[coalition_mask] = guardian_mask
            guardian_mask = (guardian_mask - 1) & coalition_mask

        if any(key is None for key in best_hourly_keys):
            raise ValueError("a coalition cannot serve at least one hourly demand")
        if best_persistent_key is None:
            raise ValueError("a coalition cannot serve its complete window demand")
        hourly_costs[coalition_mask] = float(np.sum(best_hourly))
        hourly_costs_by_hour[coalition_mask] = best_hourly

    return CoalitionWindowSolutions(
        hourly_costs,
        persistent_costs,
        hourly_masks,
        persistent_masks,
        hourly_costs_by_hour,
    )


def hourly_coalition_costs(
    capacities_gb: np.ndarray,
    fixed_power_w: np.ndarray,
    slopes_w_per_gb: np.ndarray,
    demands_gb: np.ndarray,
) -> np.ndarray:
    return coalition_window_solutions(
        capacities_gb, fixed_power_w, slopes_w_per_gb, demands_gb
    ).hourly_costs_wh


def persistent_coalition_costs(
    capacities_gb: np.ndarray,
    fixed_power_w: np.ndarray,
    slopes_w_per_gb: np.ndarray,
    demands_gb: np.ndarray,
) -> np.ndarray:
    return coalition_window_solutions(
        capacities_gb, fixed_power_w, slopes_w_per_gb, demands_gb
    ).persistent_costs_wh


def descending_capacity_hourly_policy(
    capacities_gb: np.ndarray,
    fixed_power_w: np.ndarray,
    slopes_w_per_gb: np.ndarray,
    demands_gb: np.ndarray,
) -> HourlySolution:
    """Select decreasing capacities independently at every hour."""
    capacities, fixed, slopes, demands = _validate(
        capacities_gb, fixed_power_w, slopes_w_per_gb, demands_gb
    )
    masks = np.zeros(demands.shape[1], dtype=np.int64)
    order = sorted(
        range(capacities.size), key=lambda index: (-capacities[index], index)
    )
    for hour, required in enumerate(np.sum(demands, axis=0)):
        selected: list[int] = []
        for index in order:
            selected.append(index)
            if np.sum(capacities[selected]) >= required - TOLERANCE:
                break
        masks[hour] = sum(1 << index for index in selected)
    return _evaluate_hourly_validated(
        masks, capacities, fixed, slopes, demands
    )


def descending_capacity_persistent_policy(
    capacities_gb: np.ndarray,
    fixed_power_w: np.ndarray,
    slopes_w_per_gb: np.ndarray,
    demands_gb: np.ndarray,
) -> WindowSolution:
    """Select decreasing capacities until the complete window is feasible."""
    capacities, fixed, slopes, demands = _validate(
        capacities_gb, fixed_power_w, slopes_w_per_gb, demands_gb
    )
    required = float(np.max(np.sum(demands, axis=0)))
    guardians: list[int] = []
    for index in sorted(
        range(capacities.size), key=lambda player: (-capacities[player], player)
    ):
        guardians.append(index)
        if np.sum(capacities[guardians]) >= required - TOLERANCE:
            break
    return _evaluate_persistent_validated(
        tuple(guardians), capacities, fixed, slopes, demands
    )


def proportional_hourly_policy(
    optimal_guardian_masks: np.ndarray,
    capacities_gb: np.ndarray,
    fixed_power_w: np.ndarray,
    slopes_w_per_gb: np.ndarray,
    demands_gb: np.ndarray,
) -> HourlySolution:
    """Reuse hourly-optimal guardians and allocate by capacity shares."""
    return evaluate_hourly_guardians(
        optimal_guardian_masks,
        capacities_gb,
        fixed_power_w,
        slopes_w_per_gb,
        demands_gb,
        proportional=True,
    )


def proportional_persistent_policy(
    optimal_guardians: tuple[int, ...],
    capacities_gb: np.ndarray,
    fixed_power_w: np.ndarray,
    slopes_w_per_gb: np.ndarray,
    demands_gb: np.ndarray,
) -> WindowSolution:
    """Reuse persistent-optimal guardians and allocate by capacity shares."""
    return evaluate_persistent_guardians(
        optimal_guardians,
        capacities_gb,
        fixed_power_w,
        slopes_w_per_gb,
        demands_gb,
        proportional=True,
    )


def standalone_energy(
    fixed_power_w: np.ndarray,
    slopes_w_per_gb: np.ndarray,
    demands_gb: np.ndarray,
) -> float:
    fixed = np.asarray(fixed_power_w, dtype=float)
    slopes = np.asarray(slopes_w_per_gb, dtype=float)
    demands = np.asarray(demands_gb, dtype=float)
    return (
        demands.shape[1] * float(np.sum(fixed))
        + float(np.sum(slopes[:, None] * demands))
    )
