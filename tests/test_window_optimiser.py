from __future__ import annotations

import unittest

import numpy as np

from src.core.generate_data import OperatorParams
from src.core.optimiser import coalition_cost_star
from src.core.window_optimiser import (
    descending_capacity_policy,
    hourly_oracle_energy,
    optimal_persistent_policy,
    optimal_persistent_with_oracle,
    persistent_coalition_costs,
    persistent_coalition_solutions,
    proportional_policy,
    standalone_energy,
)


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

    def test_persistent_optimum_is_no_worse_than_reference_policies(self) -> None:
        optimum = optimal_persistent_policy(
            self.capacities, self.fixed, self.slopes, self.demands
        )
        capacity = descending_capacity_policy(
            self.capacities, self.fixed, self.slopes, self.demands
        )
        proportional = proportional_policy(
            optimum.guardians,
            self.capacities,
            self.fixed,
            self.slopes,
            self.demands,
        )
        autonomous = standalone_energy(self.fixed, self.slopes, self.demands)

        self.assertLessEqual(optimum.energy_wh, capacity.energy_wh)
        self.assertLessEqual(optimum.energy_wh, proportional.energy_wh)
        self.assertLessEqual(optimum.energy_wh, autonomous)
        np.testing.assert_allclose(
            np.sum(optimum.allocations_gb, axis=0),
            np.sum(self.demands, axis=0),
        )

    def test_hourly_oracle_is_a_lower_bound(self) -> None:
        optimum = optimal_persistent_policy(
            self.capacities, self.fixed, self.slopes, self.demands
        )
        oracle = hourly_oracle_energy(
            self.capacities, self.fixed, self.slopes, self.demands
        )
        self.assertLessEqual(oracle, optimum.energy_wh)

    def test_descending_capacity_policy_selects_until_peak_is_feasible(self) -> None:
        solution = descending_capacity_policy(
            self.capacities, self.fixed, self.slopes, self.demands
        )
        self.assertEqual(solution.guardians, (0, 1))

    def test_single_hour_matches_existing_exact_optimiser(self) -> None:
        demands = self.demands[:, :1]
        expected_cost, expected_guardians, _ = coalition_cost_star(
            [0, 1, 2, 3],
            [
                OperatorParams(str(index), self.capacities[index], self.fixed[index], self.slopes[index])
                for index in range(4)
            ],
            {index: float(demands[index, 0]) for index in range(4)},
        )
        observed = optimal_persistent_policy(
            self.capacities, self.fixed, self.slopes, demands
        )

        self.assertAlmostEqual(observed.energy_wh, expected_cost)
        self.assertEqual(observed.guardians, tuple(expected_guardians))

    def test_all_coalition_costs_match_independent_optimisations(self) -> None:
        observed = persistent_coalition_costs(
            self.capacities, self.fixed, self.slopes, self.demands
        )
        self.assertEqual(observed.shape, (16,))
        self.assertEqual(observed[0], 0.0)
        for mask in range(1, 16):
            coalition = tuple(index for index in range(4) if mask & (1 << index))
            indices = np.asarray(coalition, dtype=int)
            expected = optimal_persistent_policy(
                self.capacities[indices],
                self.fixed[indices],
                self.slopes[indices],
                self.demands[indices],
            )
            self.assertAlmostEqual(observed[mask], expected.energy_wh)

    def test_joint_coalition_solutions_recover_grand_operational_metrics(self) -> None:
        observed = persistent_coalition_solutions(
            self.capacities, self.fixed, self.slopes, self.demands
        )
        expected, oracle = optimal_persistent_with_oracle(
            self.capacities, self.fixed, self.slopes, self.demands
        )
        grand_mask = (1 << self.capacities.size) - 1

        self.assertAlmostEqual(observed.costs_wh[grand_mask], expected.energy_wh)
        self.assertEqual(
            int(observed.guardian_masks[grand_mask]).bit_count(),
            expected.num_guardians,
        )
        self.assertAlmostEqual(observed.grand_hourly_oracle_wh, oracle)


if __name__ == "__main__":
    unittest.main()
