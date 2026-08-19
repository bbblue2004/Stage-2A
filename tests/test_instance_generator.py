from __future__ import annotations

from datetime import date, timedelta
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataclasses import replace

from src.data_processing.instance_generator import (
    DEFAULT_SITE_SEED,
    ScenarioSpec,
    calibrate_protocol,
    capacities_for_site,
    campaign_b_capacities,
    generate_site_blueprints,
    load_site_blueprints,
    materialize_site,
    minimal_guardian_counts,
    protocol_scenarios,
    save_protocol_spec,
    save_site_blueprints,
    size_factors_from_quantiles,
)
from src.data_processing.power_validation import CalibratedPopulation


def _days(count: int = 5) -> tuple[date, ...]:
    return tuple(date(2023, 3, 20) + timedelta(days=index) for index in range(count))


def _synthetic_population(count: int = 64) -> CalibratedPopulation:
    hours = np.arange(24, dtype=float)
    residential = 0.4 + np.cos((hours - 20) * np.pi / 12) ** 2
    commercial = 0.3 + np.cos((hours - 12) * np.pi / 12) ** 2
    traffic = np.empty((count, 5, 24), dtype=float)
    p_fixed = np.empty(count, dtype=float)
    slope = np.empty(count, dtype=float)
    for index in range(count):
        scale = 2.0 + 0.4 * index
        profile = residential if index % 2 == 0 else commercial
        day_effects = 0.85 + 0.07 * np.arange(5)
        traffic[index] = scale * day_effects[:, None] * profile[None, :]
        p_fixed[index] = 80.0 + 12.0 * index
        slope[index] = 2.0 + 0.05 * (index % 11)
    groups = np.repeat(np.arange(4), count // 4).astype(np.int8)
    return CalibratedPopulation(
        antenna_ids=np.asarray([f"A{index:03d}" for index in range(count)]),
        days=np.asarray([day.isoformat() for day in _days()]),
        traffic_gb=traffic,
        power_w=p_fixed[:, None, None] + slope[:, None, None] * traffic,
        p_fixed_w=p_fixed,
        slope_w_per_gb=slope,
        r_squared=np.ones(count),
        normalized_rmse=np.zeros(count),
        peak_traffic_gb=np.max(traffic, axis=(1, 2)),
        traffic_group=groups,
        fixed_power_group=groups,
    )


class InstanceGeneratorTests(unittest.TestCase):
    def test_size_factors_have_mean_one_and_follow_quantiles(self) -> None:
        values = np.linspace(1.0, 9.0, 81)
        factors = size_factors_from_quantiles(values, (0.25, 0.50, 0.75, 0.90), 4)
        self.assertAlmostEqual(float(np.mean(factors)), 1.0, places=12)
        self.assertTrue(np.all(np.diff(factors) > 0.0))

    def test_blueprints_are_reproducible_and_disjoint(self) -> None:
        population = _synthetic_population()
        first = generate_site_blueprints(population, num_sites=16, seed=123)
        second = generate_site_blueprints(population, num_sites=16, seed=123)
        np.testing.assert_array_equal(first.reference_index, second.reference_index)
        np.testing.assert_array_equal(first.donor_indices, second.donor_indices)
        np.testing.assert_array_equal(first.energy_distant, second.energy_distant)
        self.assertEqual(len(set(first.site_ids)), 16)
        for site_index in range(first.num_sites):
            used = {
                int(first.reference_index[site_index]),
                *map(int, first.donor_indices[site_index]),
                *map(int, first.energy_close[site_index]),
                *map(int, first.energy_moderate[site_index]),
                *map(int, first.energy_distant[site_index]),
            }
            self.assertEqual(len(used), 17)

    def test_references_are_stratified_without_replacement(self) -> None:
        population = _synthetic_population()
        blueprints = generate_site_blueprints(population, num_sites=16, seed=7)
        _, counts = np.unique(blueprints.mean_traffic_group, return_counts=True)
        np.testing.assert_array_equal(counts, np.full(4, 4))
        self.assertEqual(len(set(blueprints.reference_index)), 16)

    def test_lambda_zero_preserves_the_reference_shape(self) -> None:
        population = _synthetic_population()
        blueprints = generate_site_blueprints(population, num_sites=8, seed=5)
        spec = ScenarioSpec("A", "close", "close", "moderate", capacity_rate=0.70)
        protocol = replace(
            calibrate_protocol(population, num_sites=8, seed=5),
            shape_lambda={"close": 0.0, "moderate": 0.35, "distant": 1.0},
        )
        site = materialize_site(blueprints, 0, population, spec, protocol)
        shapes = site.traffic_gb.reshape(4, -1)
        shapes = shapes / shapes.mean(axis=1, keepdims=True)
        correlations = np.corrcoef(shapes)
        np.testing.assert_allclose(correlations, np.ones((4, 4)), atol=1e-10)
        np.testing.assert_allclose(shapes.mean(axis=1), np.ones(4), atol=1e-10)

    def test_volume_and_equipment_axes_are_matched_across_scenarios(self) -> None:
        population = _synthetic_population()
        blueprints = generate_site_blueprints(population, num_sites=4, seed=11)
        protocol = calibrate_protocol(population, num_sites=4, seed=11)
        close = materialize_site(
            blueprints,
            0,
            population,
            ScenarioSpec("A", "close", "moderate", "close", capacity_rate=0.70),
            protocol,
        )
        far = materialize_site(
            blueprints,
            0,
            population,
            ScenarioSpec("A", "far", "moderate", "close", capacity_rate=0.70),
            protocol,
        )
        distant_equip = materialize_site(
            blueprints,
            0,
            population,
            ScenarioSpec("A", "close", "moderate", "distant", capacity_rate=0.70),
            protocol,
        )
        self.assertEqual(close.reference_id, far.reference_id)
        self.assertEqual(close.donor_ids, far.donor_ids)
        self.assertEqual(close.energy_ids, far.energy_ids)
        self.assertNotEqual(close.energy_ids, distant_equip.energy_ids)
        np.testing.assert_allclose(
            close.traffic_gb.sum(axis=(1, 2)).sum(),
            far.traffic_gb.sum(axis=(1, 2)).sum(),
            rtol=1e-10,
        )
        self.assertGreater(float(np.std(far.alpha)), float(np.std(close.alpha)))

    def test_campaign_b_peak_requires_the_target_number_of_guardians(self) -> None:
        demands = np.asarray(
            [
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
            ]
        )
        capacities = campaign_b_capacities(demands, guardian_target=2, window_peak_rate=1.0)
        np.testing.assert_allclose(capacities, np.full(4, 6.0))
        counts = minimal_guardian_counts(capacities, demands)
        np.testing.assert_array_equal(counts, np.asarray([1, 2, 2]))

    def test_campaign_a_individual_feasibility_and_capacity_formula(self) -> None:
        population = _synthetic_population()
        blueprints = generate_site_blueprints(population, num_sites=3, seed=19)
        protocol = calibrate_protocol(population, num_sites=3, seed=19)
        spec = ScenarioSpec("A", "moderate", "moderate", "moderate", capacity_rate=0.70)
        site = materialize_site(blueprints, 1, population, spec, protocol)
        demands = site.traffic_gb[:, 0, :7]
        capacities = capacities_for_site(site, demands)
        np.testing.assert_allclose(capacities, site.peak_traffic_gb / 0.70)
        self.assertTrue(np.all(np.max(demands, axis=1) <= capacities + 1e-12))

    def test_blueprint_round_trip_and_protocol_json(self) -> None:
        population = _synthetic_population()
        blueprints = generate_site_blueprints(population, num_sites=6, seed=DEFAULT_SITE_SEED)
        protocol = calibrate_protocol(
            population, num_sites=6, seed=DEFAULT_SITE_SEED
        )
        with tempfile.TemporaryDirectory() as directory:
            site_path = Path(directory) / "blueprints.csv"
            spec_path = Path(directory) / "protocol.json"
            save_site_blueprints(blueprints, population, site_path)
            save_protocol_spec(protocol, spec_path)
            loaded = load_site_blueprints(site_path, population, seed=DEFAULT_SITE_SEED)
        np.testing.assert_array_equal(loaded.reference_index, blueprints.reference_index)
        np.testing.assert_array_equal(loaded.alpha_permutation, blueprints.alpha_permutation)
        keys = [spec.key for spec in protocol_scenarios()]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(any(key.startswith("A_") for key in keys))
        self.assertTrue(any(key.startswith("B_") for key in keys))

    def test_constructed_mean_traffic_matches_mu_alpha(self) -> None:
        population = _synthetic_population()
        blueprints = generate_site_blueprints(population, num_sites=5, seed=3)
        protocol = calibrate_protocol(population, num_sites=5, seed=3)
        spec = ScenarioSpec("A", "far", "distant", "moderate", capacity_rate=1.0)
        site = materialize_site(blueprints, 2, population, spec, protocol)
        observed = site.traffic_gb.mean(axis=(1, 2))
        np.testing.assert_allclose(observed, site.mu_gb * site.alpha, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
