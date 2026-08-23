"""Hourly stability and sharing on the Section 6 walkthrough plan.

The pilot keeps the central plan of Sections 6.1--6.3 and evaluates the
seven hours of H, the seven observed days and the three capacity rates.
Shapley is retained whenever it is in the core. The nucleolus is computed
only when the core is non-empty and Shapley is outside it. If the grand
coalition has an empty core, the counterfactual fallback is the proper
partition with the largest total saving among partitions whose blocks admit
a stable internal allocation.
"""

from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.cli.single_plan_efficiency import (
    CACHE,
    CENTRAL_RATE,
    RATES,
    WINDOW,
    _build_site,
)
from src.core.game import (
    allocation_check,
    core_allocation,
    nucleolus_allocation,
    shapley_value,
)
from src.core.window_optimiser import hourly_coalition_costs
from src.data_processing.antenna_metrics import DEFAULT_ELECTRICITY_PRICE_PER_KWH
from src.data_processing.power_validation import load_calibrated_population
from src.experiments.coalition_stability import (
    GRAND_MASK,
    _diagnose,
    _shapley_value,
    _savings_from_costs,
)


RESULTS = Path("results") / "single_plan_stability"
FIGURES = Path("figures") / "coalition_stability"
DAYS = ("lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim.")
CATEGORY_ORDER = (
    "shapley_in_core",
    "nonempty_shapley_out",
    "empty_core",
)
CATEGORY_LABELS = {
    "shapley_in_core": "Shapley stable",
    "nonempty_shapley_out": "Autre partage stable",
    "empty_core": "Grande coalition instable",
}
CATEGORY_COLORS = {
    "shapley_in_core": "#2F8F5B",
    "nonempty_shapley_out": "#E3A13B",
    "empty_core": "#C0504D",
}


def _mask(players: tuple[int, ...]) -> int:
    return sum(1 << player for player in players)


def _game_for_players(
    savings: np.ndarray, players: tuple[int, ...]
) -> dict[tuple[int, ...], float]:
    return {
        coalition: float(savings[_mask(coalition)])
        for size in range(len(players) + 1)
        for coalition in combinations(players, size)
    }


def _stable_allocation(
    savings: np.ndarray,
    players: tuple[int, ...],
    *,
    core_known_nonempty: bool = False,
) -> tuple[dict[int, float], str, bool] | None:
    """Return a stable allocation, avoiding the nucleolus when Shapley works."""
    game = _game_for_players(savings, players)
    player_list = list(players)
    shapley = shapley_value(player_list, game)
    if allocation_check(player_list, game, shapley).in_core:
        return shapley, "Shapley", False

    if not core_known_nonempty:
        core = core_allocation(player_list, game)
        if not core.feasible:
            return None

    scale = max(1.0, abs(game[players]))
    scaled_game = {
        coalition: value / scale for coalition, value in game.items()
    }
    result = nucleolus_allocation(player_list, scaled_game)
    if result.status != "Optimal":
        raise RuntimeError(f"nucleolus failed on coalition {players}: {result.status}")
    allocation = {
        player: result.allocation[player] * scale for player in player_list
    }
    residual = game[players] - sum(allocation.values())
    allocation[max(allocation, key=allocation.get)] += residual
    if not allocation_check(player_list, game, allocation).in_core:
        raise RuntimeError(f"nucleolus is not stable on coalition {players}")
    return allocation, "nucléole", True


def _set_partitions(
    players: tuple[int, ...],
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Enumerate set partitions in a deterministic canonical order."""
    if not players:
        return ((),)
    first = players[0]
    partitions: set[tuple[tuple[int, ...], ...]] = set()
    for smaller in _set_partitions(players[1:]):
        partitions.add(tuple(sorted(((first,), *smaller))))
        for index, block in enumerate(smaller):
            joined = tuple(sorted((first, *block)))
            candidate = list(smaller)
            candidate[index] = joined
            partitions.add(tuple(sorted(candidate)))
    return tuple(sorted(partitions, key=lambda item: (len(item), item)))


PROPER_PARTITIONS = tuple(
    partition
    for partition in _set_partitions(tuple(range(4)))
    if partition != (tuple(range(4)),)
)


def _best_stable_partition(
    savings: np.ndarray,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...], float, int]:
    """Choose the most efficient proper partition with stable internal blocks."""
    candidates: list[
        tuple[float, int, tuple[tuple[int, ...], ...], np.ndarray, int]
    ] = []
    for partition in PROPER_PARTITIONS:
        allocation = np.zeros(4, dtype=float)
        total = 0.0
        nucleolus_calls = 0
        feasible = True
        for block in partition:
            selected = _stable_allocation(savings, block)
            if selected is None:
                feasible = False
                break
            block_allocation, _, used_nucleolus = selected
            for player, value in block_allocation.items():
                allocation[player] = value
            total += float(savings[_mask(block)])
            nucleolus_calls += int(used_nucleolus)
        if feasible:
            candidates.append(
                (total, -len(partition), partition, allocation, nucleolus_calls)
            )
    if not candidates:
        raise RuntimeError("no internally stable proper partition found")
    _, _, partition, allocation, nucleolus_calls = max(
        candidates, key=lambda item: (item[0], item[1], item[2])
    )
    return allocation, partition, float(np.sum(allocation)), nucleolus_calls


def _select_grand_allocation(
    savings: np.ndarray,
    category: str,
) -> tuple[np.ndarray, str, bool]:
    if category == "shapley_in_core":
        return _shapley_value(savings), "Shapley", False
    if category != "nonempty_shapley_out":
        raise ValueError(f"no stable grand-coalition allocation for {category}")
    selected = _stable_allocation(
        savings, tuple(range(4)), core_known_nonempty=True
    )
    if selected is None:
        raise RuntimeError("diagnosed non-empty core has no stable allocation")
    allocation, rule, used_nucleolus = selected
    return (
        np.asarray([allocation[player] for player in range(4)], dtype=float),
        rule,
        used_nucleolus,
    )


def _partition_label(partition: tuple[tuple[int, ...], ...]) -> str:
    return "|".join(
        "{" + ",".join(str(player + 1) for player in block) + "}"
        for block in partition
    )


def _hourly_rows(site: dict) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rate in RATES:
        capacities = site["peaks"] / rate
        for day in range(site["n_days"]):
            for hour in WINDOW:
                demands = site["traffic"][:, day, hour : hour + 1]
                costs = hourly_coalition_costs(
                    capacities, site["p_fixed"], site["slope"], demands
                )
                savings = _savings_from_costs(costs)
                diagnostics = _diagnose(costs, savings, capacities, demands)
                category = str(diagnostics["category"])
                shapley = _shapley_value(savings)

                if category == "empty_core":
                    selected, partition, realised_total, fallback_nucleoli = (
                        _best_stable_partition(savings)
                    )
                    rule = "meilleure partition propre"
                    partition_text = _partition_label(partition)
                    breakup_loss = shapley - selected
                    total_loss = float(savings[GRAND_MASK] - realised_total)
                    if not np.isclose(
                        float(np.sum(breakup_loss)), total_loss, atol=1e-6
                    ):
                        raise RuntimeError("operator breakup losses do not add up")
                    grand_nucleolus = False
                else:
                    selected, rule, grand_nucleolus = _select_grand_allocation(
                        savings, category
                    )
                    partition_text = "{1,2,3,4}"
                    realised_total = float(savings[GRAND_MASK])
                    breakup_loss = np.zeros(4, dtype=float)
                    total_loss = 0.0
                    fallback_nucleoli = 0

                standalone = float(
                    sum(costs[1 << player] for player in range(4))
                )
                row: dict[str, object] = {
                    "rate": float(rate),
                    "day": day,
                    "day_label": DAYS[day],
                    "hour": hour,
                    "category": category,
                    "standalone_wh": standalone,
                    "grand_savings_wh": float(savings[GRAND_MASK]),
                    "grand_savings_pct": (
                        100.0 * float(savings[GRAND_MASK]) / standalone
                    ),
                    "selected_rule": rule,
                    "selected_partition": partition_text,
                    "grand_nucleolus_computed": int(grand_nucleolus),
                    "fallback_nucleolus_calls": fallback_nucleoli,
                    "realised_savings_wh": realised_total,
                    "breakup_loss_wh": total_loss,
                }
                for player in range(4):
                    row[f"grand_shapley_op{player + 1}_wh"] = float(shapley[player])
                    row[f"realised_op{player + 1}_wh"] = float(selected[player])
                    row[f"breakup_loss_op{player + 1}_wh"] = float(
                        breakup_loss[player]
                    )
                rows.append(row)
    return rows


def _category_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for rate in RATES:
        selected = [row for row in rows if float(row["rate"]) == rate]
        summary: dict[str, object] = {
            "rate": rate,
            "instances": len(selected),
        }
        for category in CATEGORY_ORDER:
            count = sum(row["category"] == category for row in selected)
            summary[f"{category}_count"] = count
            summary[f"{category}_pct"] = 100.0 * count / len(selected)
        summaries.append(summary)
    return summaries


def _operator_summary(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    central = [row for row in rows if float(row["rate"]) == CENTRAL_RATE]
    daily: list[dict[str, object]] = []
    for day in range(7):
        selected = [row for row in central if int(row["day"]) == day]
        record: dict[str, object] = {"day": day}
        for player in range(4):
            record[f"gain_op{player + 1}_wh"] = sum(
                float(row[f"realised_op{player + 1}_wh"]) for row in selected
            )
            record[f"loss_op{player + 1}_wh"] = sum(
                float(row[f"breakup_loss_op{player + 1}_wh"])
                for row in selected
            )
        daily.append(record)

    empty = [row for row in central if row["category"] == "empty_core"]
    summaries: list[dict[str, object]] = []
    for player in range(4):
        gain_wh = float(
            np.mean([row[f"gain_op{player + 1}_wh"] for row in daily])
        )
        loss_wh = float(
            np.mean([row[f"loss_op{player + 1}_wh"] for row in daily])
        )
        conditional_loss_wh = (
            float(
                np.mean(
                    [
                        row[f"breakup_loss_op{player + 1}_wh"]
                        for row in empty
                    ]
                )
            )
            if empty
            else 0.0
        )
        summaries.append(
            {
                "operator": player + 1,
                "mean_gain_kwh_per_night": gain_wh / 1000.0,
                "mean_gain_eur_per_night": (
                    gain_wh
                    / 1000.0
                    * DEFAULT_ELECTRICITY_PRICE_PER_KWH
                ),
                "mean_breakup_loss_kwh_per_night": loss_wh / 1000.0,
                "mean_breakup_loss_eur_per_night": (
                    loss_wh
                    / 1000.0
                    * DEFAULT_ELECTRICITY_PRICE_PER_KWH
                ),
                "conditional_loss_kwh_per_empty_hour": (
                    conditional_loss_wh / 1000.0
                ),
            }
        )
    return summaries


def _figure(rows: list[dict[str, object]]) -> Path:
    values = [
        np.asarray(
            [
                float(row["grand_savings_wh"]) / 1000.0
                for row in rows
                if row["category"] == category
            ],
            dtype=float,
        )
        for category in CATEGORY_ORDER
    ]
    labels = [
        f"{CATEGORY_LABELS[category]}\n(n={len(value)})"
        for category, value in zip(CATEGORY_ORDER, values, strict=True)
    ]
    figure, axis = plt.subplots(figsize=(7.4, 3.8))
    boxes = axis.boxplot(
        values,
        tick_labels=labels,
        whis=(5, 95),
        showfliers=False,
        patch_artist=True,
    )
    for box, category in zip(boxes["boxes"], CATEGORY_ORDER, strict=True):
        box.set_facecolor(CATEGORY_COLORS[category])
        box.set_alpha(0.82)
    axis.set_ylabel(
        "énergie évitée par la grande coalition\npar créneau (kWh)"
    )
    axis.set_title("Valeur horaire selon le diagnostic de stabilité")
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", labelsize=8.5)
    figure.tight_layout()

    FIGURES.mkdir(parents=True, exist_ok=True)
    output = FIGURES / "single_plan_stability.pdf"
    figure.savefig(output, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    population = load_calibrated_population(CACHE)
    site = _build_site(population)
    rows = _hourly_rows(site)
    categories = _category_summary(rows)
    operators = _operator_summary(rows)
    figure = _figure(rows)

    RESULTS.mkdir(parents=True, exist_ok=True)
    _write_csv(RESULTS / "hourly_instances.csv", rows)
    _write_csv(RESULTS / "category_summary.csv", categories)
    _write_csv(RESULTS / "operator_summary.csv", operators)
    analysis = {
        "reference_id": site["reference_id"],
        "capacity_rates": list(RATES),
        "window_hours": list(WINDOW),
        "instances": len(rows),
        "categories": {
            category: sum(row["category"] == category for row in rows)
            for category in CATEGORY_ORDER
        },
        "grand_nucleolus_computations": sum(
            int(row["grand_nucleolus_computed"]) for row in rows
        ),
        "operator_summary_at_central_rate": operators,
        "electricity_price_eur_per_kwh": DEFAULT_ELECTRICITY_PRICE_PER_KWH,
        "figure": figure.as_posix(),
    }
    (RESULTS / "analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(
        f">> Plan {site['reference_id']}: {len(rows)} jeux horaires sur H"
    )
    for summary in categories:
        print(
            f"   r={summary['rate']:.2f}: "
            f"Shapley stable {summary['shapley_in_core_count']}, "
            f"autre partage stable {summary['nonempty_shapley_out_count']}, "
            f"grande coalition instable {summary['empty_core_count']}"
        )
    print(
        "   nucléole calculé pour la grande coalition : "
        f"{sum(int(row['grand_nucleolus_computed']) for row in rows)} fois"
    )
    print(">> r=0,90, moyennes par nuit")
    for row in operators:
        print(
            f"   op. {row['operator']}: "
            f"gain {row['mean_gain_kwh_per_night']:.3f} kWh "
            f"({row['mean_gain_eur_per_night']:.3f} EUR), "
            f"perte de rupture {row['mean_breakup_loss_kwh_per_night']:+.4f} kWh"
        )
    print(f">> Figure : {figure.resolve()}")
    print(f">> Résultats : {RESULTS.resolve()}")


if __name__ == "__main__":
    main()
