from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from src.experiments.coalition_stability import (
    GRAND_MASK,
    _bondareva_gap,
    _allocation_outcomes,
    _convexity_violation,
    _diagnose,
    _least_core,
    _max_excess,
    _savings_from_costs,
    _shapley_value,
)
from src.core.window_optimiser import (
    hourly_coalition_costs,
    persistent_coalition_costs,
)


class StabilityEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.capacities = np.full(4, 6.0)
        self.fixed = np.ones(4)
        self.slopes = np.ones(4)

    def _game(self, demands: list[float]) -> tuple[np.ndarray, np.ndarray]:
        demand_array = np.asarray(demands, dtype=float)[:, None]
        costs = hourly_coalition_costs(
            self.capacities, self.fixed, self.slopes, demand_array
        )
        return costs, _savings_from_costs(costs)

    def test_constant_profiles_make_hourly_and_persistent_games_equal(self) -> None:
        demands = np.repeat(
            np.asarray([1.0, 3.0, 4.0, 5.0])[:, None],
            3,
            axis=1,
        )
        hourly = hourly_coalition_costs(
            self.capacities, self.fixed, self.slopes, demands
        )
        persistent = persistent_coalition_costs(
            self.capacities, self.fixed, self.slopes, demands
        )

        np.testing.assert_allclose(hourly, persistent)

    def test_empty_core_counterexample_matches_section_five(self) -> None:
        costs, savings = self._game([1.0, 3.0, 4.0, 5.0])
        epsilon, _ = _least_core(savings)
        diagnostics = _diagnose(
            costs,
            savings,
            self.capacities,
            np.asarray([1.0, 3.0, 4.0, 5.0])[:, None],
        )

        self.assertAlmostEqual(savings[GRAND_MASK], 1.0)
        self.assertAlmostEqual(_bondareva_gap(savings), 2.0 / 3.0)
        self.assertAlmostEqual(epsilon, 2.0 / 5.0)
        self.assertEqual(diagnostics["category"], "empty_core")
        self.assertTrue(diagnostics["loo_certificate"])
        self.assertGreater(_convexity_violation(savings), 0.0)

    def test_nonempty_core_can_exclude_shapley(self) -> None:
        costs, savings = self._game([1.0, 2.0, 3.0, 4.0])
        shapley = _shapley_value(savings)
        maximum_excess, _ = _max_excess(savings, shapley)
        diagnostics = _diagnose(
            costs,
            savings,
            self.capacities,
            np.asarray([1.0, 2.0, 3.0, 4.0])[:, None],
        )

        np.testing.assert_allclose(shapley, [2.0 / 3.0, 2.0 / 3.0, 0.5, 1.0 / 6.0])
        self.assertAlmostEqual(maximum_excess, 1.0 / 6.0)
        self.assertAlmostEqual(_bondareva_gap(savings), 0.0)
        self.assertEqual(diagnostics["category"], "nonempty_shapley_out")

    def test_shapley_stable_case_skips_nucleolus_and_has_no_breakup_loss(self) -> None:
        savings = np.asarray(
            [float(mask.bit_count()) for mask in range(1 << 4)], dtype=float
        )

        with patch(
            "src.experiments.coalition_stability.nucleolus_allocation",
            side_effect=AssertionError("nucleolus should not be called"),
        ):
            outcome = _allocation_outcomes(savings, "shapley_in_core")

        self.assertEqual(outcome["selected_rule"], "Shapley")
        self.assertEqual(outcome["grand_nucleolus_computed"], 0)
        self.assertAlmostEqual(float(outcome["breakup_loss_wh"]), 0.0)
        for player in range(1, 5):
            self.assertAlmostEqual(
                float(outcome[f"breakup_loss_op{player}_wh"]), 0.0
            )


if __name__ == "__main__":
    unittest.main()
