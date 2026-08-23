from __future__ import annotations

import unittest

import numpy as np

from src.experiments.operational_efficiency import (
    _descriptive_interval,
    _day_type,
    _efficiency_scenarios,
    _evaluate_demands,
)


class OperationalEfficiencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.capacities = np.asarray([12.0, 10.0, 9.0, 8.0])
        self.fixed = np.asarray([100.0, 120.0, 90.0, 110.0])
        self.slopes = np.asarray([3.0, 1.0, 2.0, 2.5])
        hours = np.arange(24, dtype=float)
        shape = 0.5 + 0.35 * (1.0 + np.cos((hours - 20.0) * np.pi / 12.0))
        self.demands = np.vstack(
            (0.9 * shape, 1.2 * shape, 1.5 * shape, 1.8 * shape)
        )

    def test_full_profile_slices_exactly_to_the_study_window(self) -> None:
        result = _evaluate_demands(
            self.capacities,
            self.fixed,
            self.slopes,
            self.demands,
            tuple(range(24)),
            tuple(range(7)),
        )
        positions = np.asarray(result["window_positions"], dtype=int)
        hourly = np.asarray(result["optimum_hourly_wh"], dtype=float)
        window = result["window_optimum"]

        self.assertAlmostEqual(float(np.sum(hourly[positions])), window.energy_wh)
        np.testing.assert_array_equal(positions, np.arange(7))
        self.assertEqual(np.asarray(result["minimum_guardians"]).shape, (24,))
        self.assertEqual(np.asarray(result["low_traffic"]).shape, (24,))

    def test_descriptive_interval_reports_the_middle_half(self) -> None:
        values = np.asarray([1.0, 2.0, 3.0, 8.0])
        summary = _descriptive_interval(values)

        self.assertEqual(summary["estimate"], 2.5)
        self.assertEqual(summary["q25"], 1.75)
        self.assertEqual(summary["q75"], 4.25)

    def test_weekday_weekend_split_matches_the_observed_week(self) -> None:
        self.assertEqual([_day_type(day) for day in range(7)].count("weekday"), 5)
        self.assertEqual([_day_type(day) for day in range(7)].count("weekend"), 2)

    def test_reference_grid_uses_comparable_fixed_power_equipment(self) -> None:
        scenarios = _efficiency_scenarios("central")

        self.assertEqual(len(scenarios), 3)
        self.assertEqual(
            {scenario.equipment_level for scenario in scenarios}, {"close"}
        )

    def test_zero_demand_keeps_the_nonempty_guardian_convention(self) -> None:
        demands = self.demands.copy()
        demands[:, 3] = 0.0
        result = _evaluate_demands(
            self.capacities,
            self.fixed,
            self.slopes,
            demands,
            tuple(range(24)),
            tuple(range(7)),
        )

        self.assertEqual(int(np.asarray(result["minimum_guardians"])[3]), 1)


if __name__ == "__main__":
    unittest.main()
