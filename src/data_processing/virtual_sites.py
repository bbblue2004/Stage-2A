"""Build the fixed population of virtual four-antenna sites."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data_processing.power_validation import CalibratedPopulation


DEFAULT_NUM_SITES = 1_000
DEFAULT_SITE_SEED = 20_260_814
NUM_OPERATORS = 4


@dataclass(frozen=True)
class VirtualSites:
    site_ids: np.ndarray
    antenna_indices: np.ndarray
    traffic_group: np.ndarray
    fixed_power_group: np.ndarray
    seed: int

    @property
    def num_sites(self) -> int:
        return int(self.antenna_indices.shape[0])


def _largest_remainder(counts: np.ndarray, total: int) -> np.ndarray:
    """Allocate ``total`` draws proportionally, with deterministic ties."""
    quotas = total * counts.astype(float) / np.sum(counts)
    allocation = np.floor(quotas).astype(int)
    remainder = total - int(np.sum(allocation))
    if remainder:
        order = np.argsort(-(quotas - allocation), kind="mergesort")
        allocation[order[:remainder]] += 1
    return allocation


def generate_virtual_sites(
    population: CalibratedPopulation,
    num_sites: int = DEFAULT_NUM_SITES,
    seed: int = DEFAULT_SITE_SEED,
) -> VirtualSites:
    """Draw distinct four-antenna coalitions within crossed quartile groups."""
    if num_sites <= 0:
        raise ValueError("num_sites must be positive")
    groups = np.column_stack(
        (population.traffic_group, population.fixed_power_group)
    )
    unique_groups, inverse, counts = np.unique(
        groups, axis=0, return_inverse=True, return_counts=True
    )
    allocation = _largest_remainder(counts, num_sites)
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    row_groups: list[np.ndarray] = []

    for group_index, target in enumerate(allocation):
        if target == 0:
            continue
        candidates = np.flatnonzero(inverse == group_index)
        if candidates.size < NUM_OPERATORS:
            raise ValueError(
                "A selected crossed group contains fewer than four antennas: "
                f"{tuple(unique_groups[group_index])}"
            )
        if math.comb(int(candidates.size), NUM_OPERATORS) < int(target):
            raise ValueError(
                "Not enough distinct coalitions in crossed group "
                f"{tuple(unique_groups[group_index])}"
            )
        seen: set[tuple[int, ...]] = set()
        while len(seen) < target:
            draw = tuple(
                sorted(
                    int(index)
                    for index in rng.choice(
                        candidates, size=NUM_OPERATORS, replace=False
                    )
                )
            )
            seen.add(draw)
        for draw in sorted(seen):
            rows.append(np.asarray(draw, dtype=np.int32))
            row_groups.append(unique_groups[group_index])

    antenna_indices = np.stack(rows)
    site_groups = np.stack(row_groups)
    permutation = rng.permutation(num_sites)
    antenna_indices = antenna_indices[permutation]
    site_groups = site_groups[permutation]
    site_ids = np.asarray(
        [f"site_{index:04d}" for index in range(1, num_sites + 1)], dtype=str
    )
    return VirtualSites(
        site_ids=site_ids,
        antenna_indices=antenna_indices,
        traffic_group=site_groups[:, 0].astype(np.int8),
        fixed_power_group=site_groups[:, 1].astype(np.int8),
        seed=seed,
    )


def save_virtual_sites(
    sites: VirtualSites,
    population: CalibratedPopulation,
    path: Path,
) -> Path:
    """Export the frozen site list in a human-readable form."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            (
                "site_id",
                "traffic_group",
                "fixed_power_group",
                "antenna_1",
                "antenna_2",
                "antenna_3",
                "antenna_4",
            )
        )
        for site_id, traffic_group, fixed_group, indices in zip(
            sites.site_ids,
            sites.traffic_group,
            sites.fixed_power_group,
            sites.antenna_indices,
            strict=True,
        ):
            writer.writerow(
                (
                    site_id,
                    int(traffic_group),
                    int(fixed_group),
                    *(population.antenna_ids[indices]),
                )
            )
    return path


def load_virtual_sites(
    path: Path,
    population: CalibratedPopulation,
    seed: int = DEFAULT_SITE_SEED,
) -> VirtualSites:
    """Load and validate a previously frozen virtual-site list."""
    index_by_id = {
        str(antenna_id): index
        for index, antenna_id in enumerate(population.antenna_ids)
    }
    site_ids: list[str] = []
    antenna_indices: list[list[int]] = []
    traffic_groups: list[int] = []
    fixed_groups: list[int] = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            identifiers = [row[f"antenna_{index}"] for index in range(1, 5)]
            if len(set(identifiers)) != NUM_OPERATORS:
                raise ValueError(f"{row['site_id']}: antenna identifiers are not distinct")
            try:
                indices = [index_by_id[identifier] for identifier in identifiers]
            except KeyError as error:
                raise ValueError(f"unknown antenna in {row['site_id']}: {error.args[0]}") from error
            traffic_group = int(row["traffic_group"])
            fixed_group = int(row["fixed_power_group"])
            if any(population.traffic_group[index] != traffic_group for index in indices):
                raise ValueError(f"{row['site_id']}: inconsistent traffic group")
            if any(population.fixed_power_group[index] != fixed_group for index in indices):
                raise ValueError(f"{row['site_id']}: inconsistent fixed-power group")
            site_ids.append(row["site_id"])
            antenna_indices.append(indices)
            traffic_groups.append(traffic_group)
            fixed_groups.append(fixed_group)
    if not site_ids or len(site_ids) != len(set(site_ids)):
        raise ValueError("the site file must contain unique site identifiers")
    return VirtualSites(
        site_ids=np.asarray(site_ids, dtype=str),
        antenna_indices=np.asarray(antenna_indices, dtype=np.int32),
        traffic_group=np.asarray(traffic_groups, dtype=np.int8),
        fixed_power_group=np.asarray(fixed_groups, dtype=np.int8),
        seed=seed,
    )
