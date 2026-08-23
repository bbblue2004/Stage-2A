"""Load frozen protocol artefacts shared by Section 6 experiments."""

from __future__ import annotations

from pathlib import Path

from src.data_processing.instance_generator import (
    CAMPAIGN_A_RATES,
    CENTRAL_EQUIPMENT,
    CENTRAL_SHAPE,
    CENTRAL_VOLUME,
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


SECTION_63_EQUIPMENT = CENTRAL_EQUIPMENT


def _section_63_scenarios() -> tuple[ScenarioSpec, ...]:
    return tuple(
        ScenarioSpec(
            "A",
            CENTRAL_VOLUME,
            CENTRAL_SHAPE,
            SECTION_63_EQUIPMENT,
            capacity_rate=rate,
        )
        for rate in CAMPAIGN_A_RATES
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
        return _section_63_scenarios()
    if grid == "full":
        scenarios = list(protocol_scenarios())
        keys = {spec.key for spec in scenarios}
        scenarios.extend(
            spec for spec in _section_63_scenarios() if spec.key not in keys
        )
        return tuple(scenarios)
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
        and str(row.get("volume_level", "")) == CENTRAL_VOLUME
        and str(row.get("shape_level", "")) == CENTRAL_SHAPE
        and str(row.get("equipment_level", "")) == SECTION_63_EQUIPMENT
    )
