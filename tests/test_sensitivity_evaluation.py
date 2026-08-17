import unittest

import numpy as np

from src.experiments.parameter_sensitivity import (
    _core_status,
    _evaluate_instance,
    _shapley_value,
)


class SensitivityEvaluationTests(unittest.TestCase):
    def test_empty_core_is_detected_for_balanced_three_player_game(self) -> None:
        savings = np.zeros(8)
        savings[3] = savings[5] = savings[6] = 1.0
        savings[7] = 1.0

        self.assertEqual(_core_status(savings, 3), (False, False))

    def test_shapley_value_is_efficient(self) -> None:
        savings = np.asarray([0.0, 0.0, 0.0, 2.0, 0.0, 3.0, 4.0, 5.0])

        allocation = _shapley_value(savings, 3)

        self.assertAlmostEqual(float(np.sum(allocation)), savings[-1])

    def test_positive_sleep_power_reduces_switch_off_savings(self) -> None:
        capacities = np.asarray([10.0, 10.0])
        fixed = np.asarray([10.0, 10.0])
        slopes = np.asarray([1.0, 1.0])
        demands = np.asarray([[1.0, 2.0], [1.0, 2.0]])

        off = _evaluate_instance(capacities, fixed, slopes, demands)
        sleep = _evaluate_instance(
            capacities, fixed, slopes, demands, sleep_rate=0.10
        )

        self.assertGreater(off["savings_pct"], sleep["savings_pct"])
        self.assertEqual(off["guardians"], 1)
        self.assertEqual(sleep["guardians"], 1)
        self.assertEqual(off["core_empty"], 0)
        self.assertEqual(sleep["core_empty"], 0)


if __name__ == "__main__":
    unittest.main()
