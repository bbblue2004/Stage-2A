"""Operational policies with one guardian set fixed over an hour window."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


TOLERANCE = 1e-9


@dataclass(frozen=True)
class WindowSolution:
    energy_wh: float
    guardians: tuple[int, ...]
    allocations_gb: np.ndarray

    @property
    def num_guardians(self) -> int:
        return len(self.guardians)


@dataclass(frozen=True)
class CoalitionWindowSolutions:
    costs_wh: np.ndarray
    guardian_masks: np.ndarray
    grand_hourly_oracle_wh: float


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
    if not all(np.all(np.isfinite(values)) for values in (capacities, fixed, slopes, demands)):
        raise ValueError("all operational inputs must be finite")
    if np.any(capacities <= 0.0) or np.any(fixed <= 0.0) or np.any(slopes <= 0.0):
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
        raise ValueError("the guardian set is infeasible over the window")
    allocation = np.zeros((capacities.size, total_demand.size), dtype=float)
    if proportional:
        for guardian in guardians:
            allocation[guardian] = total_demand * capacities[guardian] / capacity
        return allocation

    remaining = total_demand.copy()
    for guardian in sorted(guardians, key=lambda index: (slopes[index], index)):
        allocation[guardian] = np.minimum(capacities[guardian], remaining)
        remaining -= allocation[guardian]
        remaining[remaining < TOLERANCE] = 0.0
    if np.max(remaining) > TOLERANCE:
        raise RuntimeError("greedy allocation left unserved demand")
    return allocation


def evaluate_guardians(
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
    if not selected or any(index < 0 or index >= capacities.size for index in selected):
        raise ValueError("guardians must be a non-empty subset of operators")
    return _evaluate_validated(
        selected, capacities, fixed, slopes, demands, proportional
    )


def _evaluate_validated(
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


def optimal_persistent_policy(
    capacities_gb: np.ndarray,
    fixed_power_w: np.ndarray,
    slopes_w_per_gb: np.ndarray,
    demands_gb: np.ndarray,
) -> WindowSolution:
    """Enumerate every non-empty guardian set and retain the cheapest one."""
    return optimal_persistent_with_oracle(
        capacities_gb, fixed_power_w, slopes_w_per_gb, demands_gb
    )[0]


def persistent_coalition_costs(
    capacities_gb: np.ndarray,
    fixed_power_w: np.ndarray,
    slopes_w_per_gb: np.ndarray,
    demands_gb: np.ndarray,
) -> np.ndarray:
    """Return the exact persistent cost of every coalition.

    Entry ``mask`` is the minimum energy of the coalition whose members are
    selected by the bits of ``mask``; entry zero is zero. The routine shares
    validated inputs and subset sums across coalitions, which is substantially
    cheaper than optimising every subset independently.
    """
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

    costs = np.full(num_masks, np.inf, dtype=float)
    costs[0] = 0.0
    for coalition_mask in range(1, num_masks):
        total_demand = total_demands[coalition_mask]
        guardian_mask = coalition_mask
        while guardian_mask:
            if np.max(total_demand) <= capacity_sums[guardian_mask] + TOLERANCE:
                remaining = total_demand.copy()
                variable_energy = 0.0
                for player in sorted(
                    members[guardian_mask], key=lambda index: (slopes[index], index)
                ):
                    allocation = np.minimum(capacities[player], remaining)
                    variable_energy += slopes[player] * float(np.sum(allocation))
                    remaining -= allocation
                    remaining[remaining < TOLERANCE] = 0.0
                if np.max(remaining) > TOLERANCE:
                    raise RuntimeError("greedy allocation left unserved demand")
                energy = horizon * fixed_sums[guardian_mask] + variable_energy
                costs[coalition_mask] = min(costs[coalition_mask], energy)
            guardian_mask = (guardian_mask - 1) & coalition_mask

        if not np.isfinite(costs[coalition_mask]):
            raise ValueError("a coalition cannot serve its own window demand")
    return costs


def persistent_coalition_solutions(
    capacities_gb: np.ndarray,
    fixed_power_w: np.ndarray,
    slopes_w_per_gb: np.ndarray,
    demands_gb: np.ndarray,
) -> CoalitionWindowSolutions:
    """Solve every coalition and the grand-coalition hourly oracle together."""
    capacities, fixed, slopes, demands = _validate(
        capacities_gb, fixed_power_w, slopes_w_per_gb, demands_gb
    )
    num_players = capacities.size
    num_masks = 1 << num_players
    grand_mask = num_masks - 1
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

    costs = np.full(num_masks, np.inf, dtype=float)
    costs[0] = 0.0
    guardian_masks = np.zeros(num_masks, dtype=np.int64)
    hourly_best = np.full(horizon, np.inf, dtype=float)
    for coalition_mask in range(1, num_masks):
        total_demand = total_demands[coalition_mask]
        best_key: tuple[float, int, tuple[int, ...]] | None = None
        guardian_mask = coalition_mask
        while guardian_mask:
            feasible_hours = (
                total_demand <= capacity_sums[guardian_mask] + TOLERANCE
            )
            if np.any(feasible_hours) and (
                coalition_mask == grand_mask or np.all(feasible_hours)
            ):
                remaining = np.where(feasible_hours, total_demand, 0.0)
                variable_by_hour = np.zeros(horizon, dtype=float)
                for player in sorted(
                    members[guardian_mask], key=lambda index: (slopes[index], index)
                ):
                    allocation = np.minimum(capacities[player], remaining)
                    variable_by_hour += slopes[player] * allocation
                    remaining -= allocation
                    remaining[remaining < TOLERANCE] = 0.0
                if np.max(remaining) > TOLERANCE:
                    raise RuntimeError("greedy allocation left unserved demand")
                hourly_cost = fixed_sums[guardian_mask] + variable_by_hour
                if coalition_mask == grand_mask:
                    candidate_hours = np.where(feasible_hours, hourly_cost, np.inf)
                    hourly_best = np.minimum(hourly_best, candidate_hours)
                if np.all(feasible_hours):
                    energy = float(np.sum(hourly_cost))
                    key = (
                        energy,
                        len(members[guardian_mask]),
                        members[guardian_mask],
                    )
                    if best_key is None or key < best_key:
                        best_key = key
                        costs[coalition_mask] = energy
                        guardian_masks[coalition_mask] = guardian_mask
            guardian_mask = (guardian_mask - 1) & coalition_mask

        if best_key is None:
            raise ValueError("a coalition cannot serve its own window demand")
    if not np.all(np.isfinite(hourly_best)):
        raise ValueError("at least one grand-coalition hour cannot be served")
    return CoalitionWindowSolutions(
        costs,
        guardian_masks,
        float(np.sum(hourly_best)),
    )


def optimal_persistent_with_oracle(
    capacities_gb: np.ndarray,
    fixed_power_w: np.ndarray,
    slopes_w_per_gb: np.ndarray,
    demands_gb: np.ndarray,
) -> tuple[WindowSolution, float]:
    """Compute the persistent optimum and hourly oracle in one enumeration."""
    capacities, fixed, slopes, demands = _validate(
        capacities_gb, fixed_power_w, slopes_w_per_gb, demands_gb
    )
    best: WindowSolution | None = None
    hourly_best = np.full(demands.shape[1], np.inf, dtype=float)
    total_demand = np.sum(demands, axis=0)
    indices = tuple(range(capacities.size))
    for size in range(1, capacities.size + 1):
        for guardians in combinations(indices, size):
            guardian_capacity = float(np.sum(capacities[list(guardians)]))
            feasible_hours = total_demand <= guardian_capacity + TOLERANCE
            feasible_demands = np.where(feasible_hours, total_demand, 0.0)
            allocation = _allocation(
                guardians, capacities, slopes, feasible_demands, False
            )
            hourly_cost = (
                float(np.sum(fixed[list(guardians)]))
                + np.sum(slopes[:, None] * allocation, axis=0)
            )
            hourly_cost[~feasible_hours] = np.inf
            hourly_best = np.minimum(hourly_best, hourly_cost)
            if not np.all(feasible_hours):
                continue
            candidate = WindowSolution(
                float(np.sum(hourly_cost)), guardians, allocation
            )
            candidate_key = (candidate.energy_wh, candidate.num_guardians, candidate.guardians)
            best_key = (
                (best.energy_wh, best.num_guardians, best.guardians)
                if best is not None
                else (float("inf"), capacities.size + 1, ())
            )
            if candidate_key < best_key:
                best = candidate
    if best is None:
        raise ValueError("no guardian set can serve the window demand")
    if not np.all(np.isfinite(hourly_best)):
        raise ValueError("at least one hour cannot be served")
    return best, float(np.sum(hourly_best))


def descending_capacity_policy(
    capacities_gb: np.ndarray,
    fixed_power_w: np.ndarray,
    slopes_w_per_gb: np.ndarray,
    demands_gb: np.ndarray,
) -> WindowSolution:
    """Activate decreasing capacities until the whole window is feasible."""
    capacities, fixed, slopes, demands = _validate(
        capacities_gb, fixed_power_w, slopes_w_per_gb, demands_gb
    )
    required = float(np.max(np.sum(demands, axis=0)))
    guardians: list[int] = []
    for index in sorted(range(capacities.size), key=lambda i: (-capacities[i], i)):
        guardians.append(index)
        if np.sum(capacities[guardians]) >= required - TOLERANCE:
            break
    return _evaluate_validated(tuple(guardians), capacities, fixed, slopes, demands)


def proportional_policy(
    optimal_guardians: tuple[int, ...],
    capacities_gb: np.ndarray,
    fixed_power_w: np.ndarray,
    slopes_w_per_gb: np.ndarray,
    demands_gb: np.ndarray,
) -> WindowSolution:
    """Reuse optimal guardians but allocate traffic proportionally to capacity."""
    return evaluate_guardians(
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


def hourly_oracle_energy(
    capacities_gb: np.ndarray,
    fixed_power_w: np.ndarray,
    slopes_w_per_gb: np.ndarray,
    demands_gb: np.ndarray,
) -> float:
    return optimal_persistent_with_oracle(
        capacities_gb, fixed_power_w, slopes_w_per_gb, demands_gb
    )[1]
