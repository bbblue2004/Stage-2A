"""Regression tests for the cooperative allocation procedure."""

import unittest

from src.core.game import (
    allocation_check,
    closest_stable_allocation,
    convexity_test,
    core_allocation,
    least_core_allocation,
    nucleolus_allocation,
    shapley_value,
)


class AllocationProcedureTests(unittest.TestCase):
    def test_convex_unanimity_game_selects_stable_shapley(self) -> None:
        players = [0, 1, 2]
        savings = {
            (): 0.0,
            (0,): 0.0,
            (1,): 0.0,
            (2,): 0.0,
            (0, 1): 0.0,
            (0, 2): 0.0,
            (1, 2): 0.0,
            (0, 1, 2): 1.0,
        }

        shapley = shapley_value(players, savings)

        self.assertTrue(convexity_test(players, savings).convex)
        self.assertTrue(allocation_check(players, savings, shapley).in_core)
        for value in shapley.values():
            self.assertAlmostEqual(value, 1.0 / 3.0)

    def test_empty_core_counterexample(self) -> None:
        players = [0, 1, 2]
        savings = {
            (): 0.0,
            (0,): 0.0,
            (1,): 0.0,
            (2,): 0.0,
            (0, 1): 1.0,
            (0, 2): 1.0,
            (1, 2): 1.0,
            (0, 1, 2): 1.0,
        }

        shapley = shapley_value(players, savings)
        least_core = least_core_allocation(players, savings)
        nucleolus = nucleolus_allocation(players, savings)

        self.assertFalse(core_allocation(players, savings).feasible)
        self.assertFalse(allocation_check(players, savings, shapley).in_core)
        self.assertAlmostEqual(least_core.epsilon, 1.0 / 3.0, places=7)
        self.assertEqual(nucleolus.status, "Optimal")
        for value in nucleolus.allocation.values():
            self.assertAlmostEqual(value, 1.0 / 3.0, places=7)

    def test_projection_recovers_closest_core_point(self) -> None:
        players = [0, 1, 2]
        savings = {
            (): 0.0,
            (0,): 0.0,
            (1,): 0.0,
            (2,): 0.0,
            (0, 1): 4.0,
            (0, 2): 4.0,
            (1, 2): 0.0,
            (0, 1, 2): 4.0,
        }
        shapley = shapley_value(players, savings)
        core = core_allocation(players, savings)
        projection = closest_stable_allocation(
            players,
            savings,
            shapley,
            {i: 1.0 for i in players},
            epsilon=0.0,
            initial=core.allocation,
        )

        self.assertTrue(projection.success)
        self.assertTrue(
            allocation_check(players, savings, projection.allocation).in_core
        )
        self.assertAlmostEqual(projection.allocation[0], 4.0)
        self.assertAlmostEqual(projection.allocation[1], 0.0)
        self.assertAlmostEqual(projection.allocation[2], 0.0)


if __name__ == "__main__":
    unittest.main()
