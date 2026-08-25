from __future__ import annotations

import unittest
from itertools import combinations

import numpy as np

from src.core.window_optimiser import (
    coalition_window_solutions,
    descending_capacity_hourly_policy,
    hourly_coalition_costs,
    optimal_hourly_policy,
    optimal_hourly_with_persistent,
    persistent_coalition_costs,
    proportional_hourly_policy,
    standalone_energy,
)


def _exact_single_hour(
    capacities: np.ndarray,
    fixed: np.ndarray,
    slopes: np.ndarray,
    demands: np.ndarray,
    members: list[int],
) -> tuple[float, tuple[int, ...]]:
    """Oracle simple, indépendant de l'optimiseur vectorisé testé."""
    total_demand = float(np.sum(demands[members]))
    best = (float("inf"), tuple())
    for size in range(1, len(members) + 1):
        for guardians in combinations(members, size):
            if float(np.sum(capacities[list(guardians)])) < total_demand - 1e-9:
                continue
            remaining = total_demand
            variable_cost = 0.0
            for guardian in sorted(guardians, key=lambda i: (slopes[i], i)):
                allocation = min(float(capacities[guardian]), remaining)
                variable_cost += float(slopes[guardian]) * allocation
                remaining -= allocation
            candidate = float(np.sum(fixed[list(guardians)])) + variable_cost
            if (candidate, len(guardians), guardians) < (
                best[0],
                len(best[1]),
                best[1],
            ):
                best = (candidate, guardians)
    if not best[1]:
        raise ValueError("No feasible guardian set")
    return best


class WindowOptimiserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.capacities = np.asarray([6.0, 5.0, 4.0, 3.0])
        self.fixed = np.asarray([10.0, 11.0, 12.0, 13.0])
        self.slopes = np.asarray([4.0, 1.0, 2.0, 3.0])
        self.demands = np.asarray(
            [
                [1.0, 2.0],
                [1.0, 2.0],
                [1.0, 2.0],
                [1.0, 1.0],
            ]
        )

    def test_hourly_optimum_is_no_worse_than_comparison_policies(self) -> None:
        optimum, persistent = optimal_hourly_with_persistent(
            self.capacities, self.fixed, self.slopes, self.demands
        )
        capacity = descending_capacity_hourly_policy(
            self.capacities, self.fixed, self.slopes, self.demands
        )
        proportional = proportional_hourly_policy(
            optimum.guardian_masks,
            self.capacities,
            self.fixed,
            self.slopes,
            self.demands,
        )
        autonomous = standalone_energy(
            self.fixed, self.slopes, self.demands
        )

        self.assertLessEqual(optimum.energy_wh, persistent.energy_wh)
        self.assertLessEqual(optimum.energy_wh, capacity.energy_wh)
        self.assertLessEqual(optimum.energy_wh, proportional.energy_wh)
        self.assertLessEqual(optimum.energy_wh, autonomous)
        np.testing.assert_allclose(
            np.sum(optimum.allocations_gb, axis=0),
            np.sum(self.demands, axis=0),
        )

    def test_descending_capacity_policy_is_recomputed_each_hour(self) -> None:
        solution = descending_capacity_hourly_policy(
            self.capacities, self.fixed, self.slopes, self.demands
        )
        self.assertEqual(solution.guardian_sets, ((0,), (0, 1)))

    def test_single_hour_matches_existing_exact_optimiser(self) -> None:
        demands = self.demands[:, :1]
        expected_cost, expected_guardians = _exact_single_hour(
            self.capacities,
            self.fixed,
            self.slopes,
            demands[:, 0],
            [0, 1, 2, 3],
        )
        observed = optimal_hourly_policy(
            self.capacities, self.fixed, self.slopes, demands
        )

        self.assertAlmostEqual(observed.energy_wh, expected_cost)
        self.assertEqual(
            observed.guardian_sets[0], tuple(expected_guardians)
        )

    def test_hourly_coalition_costs_sum_independent_hourly_optima(self) -> None:
        observed = hourly_coalition_costs(
            self.capacities, self.fixed, self.slopes, self.demands
        )
        self.assertEqual(observed.shape, (16,))
        self.assertEqual(observed[0], 0.0)
        for mask in range(1, 16):
            coalition = [
                index for index in range(4) if mask & (1 << index)
            ]
            expected = 0.0
            for hour in range(self.demands.shape[1]):
                expected += _exact_single_hour(
                    self.capacities,
                    self.fixed,
                    self.slopes,
                    self.demands[:, hour],
                    coalition,
                )[0]
            self.assertAlmostEqual(observed[mask], expected)

    def test_joint_solutions_recover_both_policies(self) -> None:
        observed = coalition_window_solutions(
            self.capacities, self.fixed, self.slopes, self.demands
        )
        hourly, persistent = optimal_hourly_with_persistent(
            self.capacities, self.fixed, self.slopes, self.demands
        )
        grand_mask = (1 << self.capacities.size) - 1

        self.assertAlmostEqual(
            observed.hourly_costs_wh[grand_mask], hourly.energy_wh
        )
        self.assertAlmostEqual(
            observed.persistent_costs_wh[grand_mask],
            persistent.energy_wh,
        )
        np.testing.assert_array_equal(
            observed.hourly_guardian_masks[grand_mask],
            hourly.guardian_masks,
        )
        self.assertEqual(
            observed.hourly_costs_by_hour_wh.shape,
            (1 << self.capacities.size, self.demands.shape[1]),
        )
        np.testing.assert_allclose(
            np.sum(observed.hourly_costs_by_hour_wh, axis=1),
            observed.hourly_costs_wh,
        )
        for hour in range(self.demands.shape[1]):
            np.testing.assert_allclose(
                observed.hourly_costs_by_hour_wh[:, hour],
                hourly_coalition_costs(
                    self.capacities,
                    self.fixed,
                    self.slopes,
                    self.demands[:, hour : hour + 1],
                ),
            )
        self.assertEqual(
            int(observed.persistent_guardian_masks[grand_mask]).bit_count(),
            persistent.num_guardians,
        )

    def test_persistent_costs_remain_an_upper_bound(self) -> None:
        hourly = hourly_coalition_costs(
            self.capacities, self.fixed, self.slopes, self.demands
        )
        persistent = persistent_coalition_costs(
            self.capacities, self.fixed, self.slopes, self.demands
        )
        np.testing.assert_array_less(hourly - 1e-10, persistent + 1e-10)

    def test_semi_homogeneous_formula_uses_sum_of_hourly_thresholds(self) -> None:
        capacities = np.full(4, 6.0)
        fixed = np.full(4, 2.0)
        slopes = np.full(4, 3.0)
        demands = np.asarray(
            [
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 1.0],
                [4.0, 2.0],
            ]
        )
        observed = hourly_coalition_costs(
            capacities, fixed, slopes, demands
        )

        for mask in range(1, 16):
            members = [
                player for player in range(4) if mask & (1 << player)
            ]
            totals = np.sum(demands[members], axis=0)
            expected = (
                2.0 * float(np.sum(np.ceil(totals / 6.0)))
                + 3.0 * float(np.sum(totals))
            )
            self.assertAlmostEqual(observed[mask], expected)

    def test_low_traffic_hourly_cost_allocation_satisfies_every_coalition(
        self,
    ) -> None:
        capacities = np.full(4, 20.0)
        fixed = np.asarray([1.0, 4.0, 2.0, 3.0])
        slopes = np.asarray([3.0, 1.0, 2.0, 1.5])
        demands = np.asarray(
            [
                [1.0, 4.0, 2.0],
                [2.0, 1.0, 3.0],
                [1.0, 2.0, 1.0],
                [3.0, 1.0, 2.0],
            ]
        )
        costs = hourly_coalition_costs(
            capacities, fixed, slopes, demands
        )
        hourly_grand_costs = np.asarray(
            [
                optimal_hourly_policy(
                    capacities,
                    fixed,
                    slopes,
                    demands[:, hour : hour + 1],
                ).energy_wh
                for hour in range(demands.shape[1])
            ]
        )
        total_demands = np.sum(demands, axis=0)
        net_costs = np.sum(
            demands / total_demands * hourly_grand_costs,
            axis=1,
        )

        self.assertAlmostEqual(float(np.sum(net_costs)), costs[-1])
        for mask in range(1, 16):
            members = [
                player for player in range(4) if mask & (1 << player)
            ]
            self.assertLessEqual(
                float(np.sum(net_costs[members])),
                costs[mask] + 1e-9,
            )


if __name__ == "__main__":
    unittest.main()
