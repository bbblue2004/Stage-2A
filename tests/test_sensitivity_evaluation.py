import unittest

import numpy as np

from src.experiments.parameter_sensitivity import (
    _core_status,
    _evaluate_instance,
    _plan_setting_summaries,
    _shapley_value,
    _summaries,
)


class SensitivityEvaluationTests(unittest.TestCase):
    def test_summaries_average_days_before_taking_plan_median(self) -> None:
        rows = []
        for site_id, values in (("a", [0.0]), ("b", [100.0, 100.0, 100.0])):
            for day, value in enumerate(values):
                rows.append(
                    {
                        "site_id": site_id,
                        "factor": "traffic_level",
                        "setting": "1.00",
                        "setting_value": 1.0,
                        "day": str(day),
                        "savings_pct": value,
                        "guardians_mean": 1.0,
                        "persistence_gap_pp": 0.0,
                        "core_empty": 0,
                        "shapley_stable": int(site_id == "b"),
                    }
                )

        plan_rows = _plan_setting_summaries(rows)
        summary = _summaries(rows, plan_rows)[0]

        self.assertEqual(len(plan_rows), 2)
        self.assertEqual(summary["plans"], 2)
        self.assertAlmostEqual(summary["savings_pct_median"], 50.0)
        self.assertAlmostEqual(summary["shapley_stable_pct_pooled"], 75.0)

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
        self.assertEqual(off["guardians_mean"], 1)
        self.assertEqual(sleep["guardians_mean"], 1)
        self.assertEqual(off["core_empty"], 0)
        self.assertEqual(sleep["core_empty"], 0)

    def test_infeasible_coalition_is_marked_out_of_domain(self) -> None:
        capacities = np.asarray([1.0, 10.0])
        fixed = np.asarray([10.0, 10.0])
        slopes = np.asarray([1.0, 1.0])
        demands = np.asarray([[1.2], [1.0]])

        result = _evaluate_instance(capacities, fixed, slopes, demands)

        self.assertEqual(result["feasible"], 0)
        self.assertTrue(np.isnan(result["savings_pct"]))
        self.assertEqual(result["core_empty"], -1)


if __name__ == "__main__":
    unittest.main()
