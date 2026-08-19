from __future__ import annotations

import csv
from datetime import date, datetime, timedelta
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.data_processing.power_validation import (
    AntennaSeries,
    CalibratedPopulation,
    calibrate_antenna,
    fit_affine,
    load_calibrated_population,
    load_population,
    save_calibrated_population,
)


def _days(count: int = 5) -> tuple[date, ...]:
    return tuple(date(2023, 3, 20) + timedelta(days=index) for index in range(count))


def _calibrated_population(count: int = 16) -> CalibratedPopulation:
    traffic = np.arange(count * 5 * 24, dtype=float).reshape(count, 5, 24)
    return CalibratedPopulation(
        antenna_ids=np.asarray([f"A{index:02d}" for index in range(count)]),
        days=np.asarray([day.isoformat() for day in _days()]),
        traffic_gb=traffic,
        power_w=100.0 + traffic,
        p_fixed_w=np.linspace(100.0, 200.0, count),
        slope_w_per_gb=np.ones(count),
        r_squared=np.ones(count),
        normalized_rmse=np.zeros(count),
        peak_traffic_gb=np.max(traffic, axis=(1, 2)),
        traffic_group=np.repeat(np.arange(2), count // 2).astype(np.int8),
        fixed_power_group=np.repeat(np.arange(2), count // 2).astype(np.int8),
    )


class AffinePowerCalibrationTests(unittest.TestCase):
    def test_affine_fit_recovers_exact_coefficients(self) -> None:
        traffic = np.asarray([0.0, 1.0, 2.0, 4.0, 7.0])
        power = 125.0 + 3.5 * traffic

        fit = fit_affine(traffic, power)

        self.assertAlmostEqual(fit.p_fixed_w, 125.0, places=10)
        self.assertAlmostEqual(fit.slope_w_per_gb, 3.5, places=10)
        self.assertAlmostEqual(fit.r_squared, 1.0, places=12)
        self.assertAlmostEqual(fit.rmse_w, 0.0, places=12)
        self.assertAlmostEqual(fit.normalized_rmse, 0.0, places=12)

    def test_direct_calibration_uses_all_five_days(self) -> None:
        traffic = np.vstack(
            [np.arange(24, dtype=float) + day_index for day_index in range(5)]
        )
        power = 200.0 + 4.0 * traffic
        series = AntennaSeries("exact", _days(), traffic, power)

        result = calibrate_antenna(series)

        self.assertEqual(result.status, "included")
        self.assertEqual(result.num_observations, 120)
        self.assertEqual(result.num_active_observations, 120)
        self.assertAlmostEqual(result.p_fixed_w, 200.0, places=9)
        self.assertAlmostEqual(result.slope_w_per_gb, 4.0, places=9)
        self.assertAlmostEqual(result.r_squared, 1.0, places=12)

    def test_zero_power_rows_are_excluded_from_active_fit(self) -> None:
        traffic = np.tile(np.arange(24, dtype=float), (5, 1))
        power = 200.0 + 4.0 * traffic
        traffic[0, 0], power[0, 0] = 0.0, 0.0
        traffic[1, 1], power[1, 1] = 3.0, 0.0
        series = AntennaSeries("active-only", _days(), traffic, power)

        result = calibrate_antenna(series)

        self.assertEqual(result.status, "included")
        self.assertEqual(result.num_active_observations, 118)
        self.assertAlmostEqual(result.p_fixed_w, 200.0, places=9)
        self.assertAlmostEqual(result.slope_w_per_gb, 4.0, places=9)

    def test_constant_traffic_is_excluded(self) -> None:
        traffic = np.ones((5, 24))
        power = np.arange(120, dtype=float).reshape(5, 24) + 100.0
        result = calibrate_antenna(
            AntennaSeries("constant", _days(), traffic, power)
        )
        self.assertEqual(result.status, "constant_traffic")

    def test_nonpositive_slope_is_excluded(self) -> None:
        traffic = np.tile(np.arange(24, dtype=float), (5, 1))
        power = 500.0 - 2.0 * traffic
        result = calibrate_antenna(
            AntennaSeries("negative-slope", _days(), traffic, power)
        )
        self.assertEqual(result.status, "nonpositive_slope")

    def test_loader_selects_first_five_days_and_averages_duplicates(self) -> None:
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
                for day_offset in range(6):
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

        self.assertEqual(audit.selected_dates, tuple(day.isoformat() for day in _days()))
        self.assertEqual(audit.selected_rows, 121)
        self.assertEqual(audit.duplicate_selected_rows, 1)
        self.assertEqual(audit.unparsed_source_rows, 1)
        self.assertEqual(len(population), 1)
        self.assertEqual(population[0].num_observations, 120)
        self.assertAlmostEqual(population[0].traffic[0, 0], 1.0)
        self.assertAlmostEqual(population[0].power[0, 0], 101.0)

    def test_calibrated_population_cache_round_trip(self) -> None:
        expected = _calibrated_population()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "population.npz"
            save_calibrated_population(expected, path)
            observed = load_calibrated_population(path)

        np.testing.assert_array_equal(observed.antenna_ids, expected.antenna_ids)
        np.testing.assert_allclose(observed.traffic_gb, expected.traffic_gb)
        np.testing.assert_allclose(observed.p_fixed_w, expected.p_fixed_w)
        np.testing.assert_array_equal(observed.traffic_group, expected.traffic_group)

if __name__ == "__main__":
    unittest.main()
