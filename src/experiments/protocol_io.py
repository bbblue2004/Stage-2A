"""Load frozen protocol artefacts shared by Section 6 experiments."""

from __future__ import annotations

from pathlib import Path

from src.data_processing.instance_generator import (
    ProtocolSpec,
    ScenarioSpec,
    SiteBlueprints,
    load_protocol_spec,
    load_site_blueprints,
    protocol_scenarios,
)
from src.data_processing.power_validation import (
    CalibratedPopulation,
    load_calibrated_population,
)


def load_protocol_inputs(
    calibration_dir: Path,
) -> tuple[CalibratedPopulation, SiteBlueprints, ProtocolSpec]:
    cache_path = calibration_dir / "calibrated_population.npz"
    sites_path = calibration_dir / "site_blueprints.csv"
    protocol_path = calibration_dir / "protocol_parameters.json"
    missing = [path for path in (cache_path, sites_path, protocol_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing protocol inputs: " + ", ".join(str(path) for path in missing)
        )
    population = load_calibrated_population(cache_path)
    blueprints = load_site_blueprints(sites_path, population)
    protocol = load_protocol_spec(protocol_path)
    return population, blueprints, protocol


def scenarios_for_grid(grid: str) -> tuple[ScenarioSpec, ...]:
    if grid == "central":
        return protocol_scenarios(
            include_campaign_a_rates=True,
            include_heterogeneity=False,
            include_campaign_b=False,
        )
    if grid == "full":
        return protocol_scenarios()
    if grid == "thresholds":
        return protocol_scenarios(
            include_campaign_a_rates=False,
            include_heterogeneity=False,
            include_campaign_b=True,
        )
    raise ValueError("grid must be 'central', 'full' or 'thresholds'")


def is_central_campaign_a(row: dict[str, object]) -> bool:
    return (
        str(row.get("campaign", "")) == "A"
        and str(row.get("volume_level", "")) == "moderate"
        and str(row.get("shape_level", "")) == "moderate"
        and str(row.get("equipment_level", "")) == "moderate"
    )
