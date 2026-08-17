from __future__ import annotations

import unittest

import numpy as np

from src.experiments.coalition_stability import (
    GRAND_MASK,
    _bondareva_gap,
    _convexity_violation,
    _diagnose,
    _least_core,
    _max_excess,
    _savings_from_costs,
    _shapley_value,
)
from src.core.window_optimiser import persistent_coalition_costs


class StabilityEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.capacities = np.full(4, 6.0)
        self.fixed = np.ones(4)
        self.slopes = np.ones(4)

    def _game(self, demands: list[float]) -> tuple[np.ndarray, np.ndarray]:
        demand_array = np.asarray(demands, dtype=float)[:, None]
        costs = persistent_coalition_costs(
            self.capacities, self.fixed, self.slopes, demand_array
        )
        return costs, _savings_from_costs(costs)

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


if __name__ == "__main__":
    unittest.main()
