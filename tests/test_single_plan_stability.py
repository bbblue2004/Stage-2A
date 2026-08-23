from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from src.cli.single_plan_stability import (
    PROPER_PARTITIONS,
    _best_stable_partition,
    _stable_allocation,
)
from src.core.window_optimiser import hourly_coalition_costs
from src.experiments.coalition_stability import (
    GRAND_MASK,
    _diagnose,
    _savings_from_costs,
)


class SinglePlanStabilityTests(unittest.TestCase):
    def test_all_four_player_proper_partitions_are_enumerated(self) -> None:
        # Bell(4) = 15, including the grand coalition itself.
        self.assertEqual(len(PROPER_PARTITIONS), 14)

    def test_nucleolus_is_not_called_when_shapley_is_stable(self) -> None:
        savings = np.asarray(
            [float(mask.bit_count()) for mask in range(1 << 4)]
        )

        with patch(
            "src.cli.single_plan_stability.nucleolus_allocation",
            side_effect=AssertionError("nucleolus should not be called"),
        ):
            selected = _stable_allocation(savings, tuple(range(4)))

        self.assertIsNotNone(selected)
        assert selected is not None
        _, rule, used_nucleolus = selected
        self.assertEqual(rule, "Shapley")
        self.assertFalse(used_nucleolus)

    def test_empty_core_falls_back_to_a_proper_partition(self) -> None:
        capacities = np.full(4, 6.0)
        fixed = np.ones(4)
        slopes = np.ones(4)
        demands = np.asarray([1.0, 3.0, 4.0, 5.0])[:, None]
        costs = hourly_coalition_costs(
            capacities, fixed, slopes, demands
        )
        savings = _savings_from_costs(costs)
        diagnostics = _diagnose(
            costs, savings, capacities, demands
        )

        allocation, partition, realised, _ = _best_stable_partition(
            savings
        )

        self.assertEqual(diagnostics["category"], "empty_core")
        self.assertIn(partition, PROPER_PARTITIONS)
        self.assertAlmostEqual(float(np.sum(allocation)), realised)
        self.assertLessEqual(realised, float(savings[GRAND_MASK]) + 1e-9)


if __name__ == "__main__":
    unittest.main()
