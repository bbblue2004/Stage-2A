from __future__ import annotations

import csv
from datetime import date, datetime, timedelta
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.data_processing.power_validation import (
    AntennaSeries,
    fit_affine,
    fit_nonnegative_affine,
    fit_quadratic,
    load_population,
    validate_antenna,
)


class AffinePowerModelTests(unittest.TestCase):
    def test_unconstrained_fit_recovers_an_exact_affine_model(self) -> None:
        traffic = np.asarray([0.0, 1.0, 2.0, 4.0, 7.0])
        power = 125.0 + 3.5 * traffic

        fit = fit_affine(traffic, power)

        self.assertAlmostEqual(fit.f_tilde, 125.0, places=10)
        self.assertAlmostEqual(fit.gamma_tilde, 3.5, places=10)
        self.assertAlmostEqual(fit.sse, 0.0, places=20)
        self.assertAlmostEqual(fit.r_squared, 1.0, places=12)

    def test_nonnegative_fit_uses_the_correct_boundary(self) -> None:
        traffic = np.asarray([0.0, 1.0, 2.0, 3.0])
        power = np.asarray([5.0, 4.0, 3.0, 2.0])

        free = fit_affine(traffic, power)
        constrained = fit_nonnegative_affine(traffic, power)

        self.assertLess(free.gamma_tilde, 0.0)
        self.assertAlmostEqual(constrained.gamma_tilde, 0.0, places=12)
        self.assertAlmostEqual(constrained.f_tilde, np.mean(power), places=12)

    def test_centered_quadratic_fit_is_stable_for_large_traffic_values(self) -> None:
        traffic = 1_000_000.0 + np.arange(20, dtype=float)
        centered = traffic - 1_000_000.0
        power = 500.0 + 2.0 * centered + 0.25 * centered**2

        fit = fit_quadratic(traffic, power)
        prediction = fit.predict(traffic)

        np.testing.assert_allclose(prediction, power, rtol=1e-8, atol=1e-4)
        self.assertLess(fit.sse, 1e-15)

    def test_leave_one_day_out_has_no_day_leakage(self) -> None:
        days = tuple(date(2023, 3, 20) + timedelta(days=index) for index in range(5))
        traffic = np.vstack(
            [np.arange(24, dtype=float) + 0.25 * index for index in range(5)]
        )
        power = 200.0 + 4.0 * traffic
        series = AntennaSeries("exact", days, traffic, power)

        result, folds, diagnostics = validate_antenna(series)

        self.assertEqual(result.status, "included")
        self.assertEqual(len(folds), 5)
        self.assertEqual({fold.n_train for fold in folds}, {4 * 24})
        self.assertEqual({fold.n_test for fold in folds}, {24})
        self.assertEqual({fold.held_out_day for fold in folds}, {d.isoformat() for d in days})
        self.assertAlmostEqual(result.f_tilde, 200.0, places=9)
        self.assertAlmostEqual(result.gamma_tilde, 4.0, places=9)
        self.assertAlmostEqual(result.cv_r_squared, 1.0, places=12)
        self.assertLess(result.cv_rmse_affine, 1e-10)
        self.assertIsNotNone(diagnostics)
        assert diagnostics is not None
        for day in days:
            self.assertEqual(
                int(np.sum(diagnostics["day"] == day.isoformat())),
                24,
            )

    def test_constant_traffic_is_reported_without_fitting(self) -> None:
        days = tuple(date(2023, 3, 20) + timedelta(days=index) for index in range(5))
        traffic = np.ones((5, 24))
        power = np.arange(120, dtype=float).reshape(5, 24)
        series = AntennaSeries("constant", days, traffic, power)

        result, folds, diagnostics = validate_antenna(series)

        self.assertEqual(result.status, "constant_traffic")
        self.assertEqual(folds, [])
        self.assertIsNone(diagnostics)
        self.assertTrue(np.isnan(result.cv_r_squared))

    def test_constant_leave_one_day_training_sample_is_reported(self) -> None:
        days = tuple(date(2023, 3, 20) + timedelta(days=index) for index in range(5))
        traffic = np.ones((5, 24))
        traffic[-1] = np.arange(24, dtype=float)
        power = 100.0 + traffic
        series = AntennaSeries("single-varying-day", days, traffic, power)

        result, folds, diagnostics = validate_antenna(series)

        self.assertEqual(result.status, "unidentifiable_training_fold")
        self.assertEqual(folds, [])
        self.assertIsNone(diagnostics)

    def test_loader_averages_duplicates_and_keeps_only_complete_days(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            headers = (
                "HEURE(PSDATE)",
                "SYS.NIDT",
                "DL_VOLUME_PDCP_GBYTES",
                "AVERAGE_POWER_CONSUMPTION_(W)",
            )
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file, delimiter=";")
                writer.writerow(headers)
                for day_offset in range(3):
                    day = datetime(2023, 3, 20) + timedelta(days=day_offset)
                    for hour in range(24):
                        writer.writerow(
                            (
                                (day + timedelta(hours=hour)).strftime("%Y-%m-%d %H"),
                                "A",
                                float(hour),
                                100.0 + hour,
                            )
                        )
                writer.writerow(("2023-03-20 00", "A", 2.0, 102.0))
                writer.writerow(("", "", "", ""))

            population, audit = load_population(path)

        self.assertEqual(audit.total_rows, 74)
        self.assertEqual(audit.parsed_rows, 73)
        self.assertEqual(audit.invalid_rows, 1)
        self.assertEqual(audit.duplicate_rows, 1)
        self.assertEqual(len(population), 1)
        self.assertEqual(population[0].num_days, 3)
        self.assertAlmostEqual(population[0].traffic[0, 0], 1.0)
        self.assertAlmostEqual(population[0].power[0, 0], 101.0)

    def test_loader_marks_a_network_wide_joint_zero_as_an_outage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            headers = (
                "HEURE(PSDATE)",
                "SYS.NIDT",
                "DL_VOLUME_PDCP_GBYTES",
                "AVERAGE_POWER_CONSUMPTION_(W)",
            )
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file, delimiter=";")
                writer.writerow(headers)
                for antenna in ("A", "B"):
                    for day_offset in range(5):
                        day = datetime(2023, 3, 20) + timedelta(days=day_offset)
                        for hour in range(24):
                            outage = day_offset == 4 and hour == 2
                            writer.writerow(
                                (
                                    (day + timedelta(hours=hour)).strftime(
                                        "%Y-%m-%d %H"
                                    ),
                                    antenna,
                                    0.0 if outage else float(hour + 1),
                                    0.0 if outage else 100.0 + hour,
                                )
                            )

            population, audit = load_population(path)

        self.assertEqual(audit.global_outage_timestamps, ("2023-03-24 02:00:00",))
        self.assertEqual(audit.globally_excluded_rows, 2)
        self.assertEqual({series.num_days for series in population}, {5})
        self.assertEqual({series.num_observations for series in population}, {119})
        self.assertTrue(all(np.isnan(series.traffic[4, 2]) for series in population))

    def test_leave_one_day_out_records_extrapolated_test_points(self) -> None:
        days = tuple(date(2023, 3, 20) + timedelta(days=index) for index in range(5))
        traffic = np.tile(np.arange(24, dtype=float), (5, 1))
        traffic[-1, -1] = 100.0
        power = 200.0 + 4.0 * traffic
        series = AntennaSeries("extrapolation", days, traffic, power)

        result, folds, diagnostics = validate_antenna(series)

        self.assertGreater(result.cv_extrapolation_fraction, 0.0)
        held_out = next(fold for fold in folds if fold.held_out_day == days[-1].isoformat())
        self.assertAlmostEqual(held_out.extrapolation_fraction, 1.0 / 24.0)
        self.assertIsNotNone(diagnostics)
        assert diagnostics is not None
        self.assertEqual(int(np.sum(diagnostics["is_extrapolation"])), 1)

    def test_active_fit_excludes_sleep_and_inconsistent_zero_power_rows(self) -> None:
        days = tuple(date(2023, 3, 20) + timedelta(days=index) for index in range(5))
        traffic = np.tile(np.arange(24, dtype=float), (5, 1))
        power = 200.0 + 4.0 * traffic
        traffic[0, 0], power[0, 0] = 0.0, 0.0
        traffic[1, 1], power[1, 1] = 3.0, 0.0
        series = AntennaSeries("active-only", days, traffic, power)

        result, folds, diagnostics = validate_antenna(series)

        self.assertEqual(result.status, "included")
        self.assertEqual(result.num_observations, 118)
        self.assertEqual(result.num_active_days, 5)
        self.assertAlmostEqual(result.f_tilde, 200.0, places=9)
        self.assertAlmostEqual(result.gamma_tilde, 4.0, places=9)
        self.assertEqual(sum(fold.n_test for fold in folds), 118)
        self.assertIsNotNone(diagnostics)


if __name__ == "__main__":
    unittest.main()
