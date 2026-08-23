"""Regression tests for the cooperative allocation procedure."""

import unittest

from src.core.game import (
    allocation_check,
    closest_stable_allocation,
    convexity_test,
    core_allocation,
    least_core_allocation,
    nucleolus_allocation,
    nucleolus_allocation_fast,
    shapley_value,
)


class AllocationProcedureTests(unittest.TestCase):
    def test_fast_nucleolus_matches_reference_lp_cascade(self) -> None:
        players = [0, 1, 2, 3]
        games = (
            {
                coalition: float(len(coalition) ** 2)
                for coalition in [
                    (), (0,), (1,), (2,), (3,), (0, 1), (0, 2),
                    (0, 3), (1, 2), (1, 3), (2, 3), (0, 1, 2),
                    (0, 1, 3), (0, 2, 3), (1, 2, 3), (0, 1, 2, 3),
                ]
            },
            {
                (): 0.0,
                (0,): 0.0,
                (1,): 0.0,
                (2,): 0.0,
                (3,): 0.0,
                (0, 1): 2.0,
                (0, 2): 1.0,
                (0, 3): 3.0,
                (1, 2): 2.5,
                (1, 3): 1.5,
                (2, 3): 2.0,
                (0, 1, 2): 4.0,
                (0, 1, 3): 4.5,
                (0, 2, 3): 4.0,
                (1, 2, 3): 3.5,
                (0, 1, 2, 3): 5.0,
            },
        )
        for game in games:
            with self.subTest(grand_value=game[tuple(players)]):
                reference = nucleolus_allocation(players, game)
                fast = nucleolus_allocation_fast(players, game)
                self.assertEqual(reference.status, "Optimal")
                self.assertEqual(fast.status, "Optimal")
                for player in players:
                    self.assertAlmostEqual(
                        reference.allocation[player],
                        fast.allocation[player],
                        places=6,
                    )

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
