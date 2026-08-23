from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from src.cli.single_plan_end_section import (
    _close_traffic,
    _generic_savings,
    _operator_count_summary,
    _sensitivity_summary,
    _three_player_diagnostics,
    _threshold_summary,
)


class SinglePlanEndSectionTests(unittest.TestCase):
    def test_close_corner_copies_the_reference_shape_and_equalises_volume(self) -> None:
        reference = np.arange(1.0, 49.0).reshape(2, 24)
        population = SimpleNamespace(
            antenna_ids=np.asarray(["ref", "other"]),
            traffic_gb=np.stack((reference, 2.0 * reference)),
        )
        site = {
            "reference_id": "ref",
            "mu": float(reference.mean()),
            "n_op": 4,
            "n_days": 2,
        }

        close = _close_traffic(population, site)

        self.assertEqual(close.shape, (4, 2, 24))
        for operator in range(1, 4):
            np.testing.assert_allclose(close[operator], close[0])
        np.testing.assert_allclose(close[0], reference)

    def test_threshold_summary_conditions_epsilon_and_blocking_size(self) -> None:
        records = [
            {
                "k_h": 1,
                "shapley_in_core": 1,
                "core_nonempty": 1,
                "blocking_size": 2,
                "least_core_epsilon_normalized": float("nan"),
            },
            {
                "k_h": 2,
                "shapley_in_core": 0,
                "core_nonempty": 0,
                "blocking_size": 3,
                "least_core_epsilon_normalized": 0.04,
            },
            {
                "k_h": 2,
                "shapley_in_core": 0,
                "core_nonempty": 1,
                "blocking_size": 3,
                "least_core_epsilon_normalized": float("nan"),
            },
            {
                "k_h": 3,
                "shapley_in_core": 1,
                "core_nonempty": 1,
                "blocking_size": 2,
                "least_core_epsilon_normalized": float("nan"),
            },
            {
                "k_h": 4,
                "shapley_in_core": 1,
                "core_nonempty": 1,
                "blocking_size": 2,
                "least_core_epsilon_normalized": float("nan"),
            },
        ]

        summary = _threshold_summary(records)

        self.assertEqual(summary[1]["instances"], 2)
        self.assertEqual(summary[1]["empty_core_pct"], 50.0)
        self.assertEqual(summary[1]["shapley_in_core_pct"], 0.0)
        self.assertEqual(summary[1]["epsilon_empty_median_pct"], 4.0)
        self.assertEqual(summary[1]["blocking_size_mode"], "3 (100.0 %)")

    def test_three_player_diagnostics_detect_overlapping_pair_claims(self) -> None:
        # Each pair saves one unit, but the grand coalition also saves only
        # one: the three overlapping claims cannot all be honoured.
        costs = np.asarray([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0])
        savings = _generic_savings(costs, 3)

        diagnostics = _three_player_diagnostics(
            costs,
            savings,
            capacities=np.asarray([10.0, 10.0, 10.0]),
            demands=np.ones((3, 1)),
        )

        self.assertEqual(diagnostics["category"], "empty_core")
        self.assertEqual(diagnostics["loo_certificate"], 1)
        self.assertAlmostEqual(
            diagnostics["least_core_epsilon_wh"], 1.0 / 3.0
        )

    def test_operator_summary_keeps_denominators_explicit(self) -> None:
        records = [
            {
                "low_traffic_condition": 1,
                "shapley_in_core": 1,
                "core_nonempty": 1,
                "savings_wh": 2000.0,
                "savings_pct": 40.0,
            },
            {
                "low_traffic_condition": 0,
                "shapley_in_core": 0,
                "core_nonempty": 0,
                "savings_wh": 4000.0,
                "savings_pct": 60.0,
            },
        ]

        summary = _operator_count_summary({3: records})[0]

        self.assertEqual(summary["num_operators"], 3)
        self.assertEqual(summary["instances"], 2)
        self.assertEqual(summary["core_nonempty_pct"], 50.0)
        self.assertEqual(summary["savings_kwh_mean"], 3.0)

    def test_sensitivity_effects_are_ranges_across_settings(self) -> None:
        rows = []
        for setting, saving, guardians in (
            ("0.80", 70.0, 1.0),
            ("1.00", 72.0, 1.5),
            ("1.20", 68.0, 2.0),
        ):
            rows.append(
                {
                    "factor": "traffic",
                    "setting": setting,
                    "savings_pct": saving,
                    "guardians_mean": guardians,
                    "persistence_gap_pp": 2.0 * guardians,
                    "core_empty": 0,
                    "shapley_stable": 1,
                }
            )
        # Supply invariant settings for the other factors expected by the
        # summary routine.
        for factor in ("fixed", "variable", "sleep", "position", "duration"):
            rows.append(
                {
                    "factor": factor,
                    "setting": "central",
                    "savings_pct": 70.0,
                    "guardians_mean": 1.0,
                    "persistence_gap_pp": 0.0,
                    "core_empty": 0,
                    "shapley_stable": 1,
                }
            )

        _, effects = _sensitivity_summary(rows)
        traffic = next(row for row in effects if row["factor"] == "traffic")

        self.assertEqual(traffic["savings_range_pp"], 4.0)
        self.assertEqual(traffic["guardians_range"], 1.0)
        self.assertEqual(traffic["persistence_range_pp"], 2.0)
        self.assertEqual(traffic["shapley_range_pp"], 0.0)


if __name__ == "__main__":
    unittest.main()
