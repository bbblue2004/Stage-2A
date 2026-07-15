"""Console report with progressive output."""

from __future__ import annotations

from typing import Any

from src.core.generate_data import Scenario
from src.core.simulation import evaluate_day


def _emit(msg: str = "") -> None:
    print(msg, flush=True)


def _print_header(scenario: Scenario) -> None:
    source = "radio_sites.csv" if scenario.data_source == "csv" else "hard-coded fallback"

    _emit("=" * 78)
    _emit("RAN sharing - cooperative game evaluation")
    _emit("=" * 78)
    _emit(f"Site (operator 1) : {scenario.antenna_id}")
    _emit(f"Traffic source    : {source}")
    if scenario.data_source == "csv":
        _emit("Cost coeffs (beta, K): derived from CSV power-vs-rho regression")
    else:
        _emit("Cost coeffs (beta, K): synthetic fallback")
    _emit(f"Horizon           : {scenario.horizon_label()} (hourly resolution)")
    _emit("Allocation        : least-core once at end (rules 1-3 disabled)")
    _emit()

    _emit(f"{'Op':<4} {'epsilon':>8} {'c':>6} {'beta':>6} {'K':>6}")
    _emit("-" * 36)
    for i, op in enumerate(scenario.operators):
        _emit(
            f"{i + 1:<4} {op.capacity_epsilon:>8.1f} {op.c:>6.2f} "
            f"{op.beta:>6.2f} {op.K:>6.2f}"
        )
    _emit()


def _print_hourly_header() -> None:
    _emit("Hourly load, surplus, LC gain share and guardians (^ = config change)")
    _emit("-" * 78)
    _emit(
        f"{'Hour':>4} | {'rho':>6} | {'Surplus':>8} | {'Gain LC':>8} | "
        f"{'Chg':>3} | Gardiens"
    )
    _emit("-" * 78)


def _print_hour_row(row: dict[str, Any]) -> None:
    chg = "^" if row["guardians_changed"] else ""
    _emit(
        f"{row['hour']:02d}:00 | {row['rho_mean']:>6.3f} | "
        f"{row['surplus']:>+8.2f} | {row['lc_gain']:>+8.2f} | "
        f"{chg:>3} | {row['guardians_label']}"
    )


def run_report(scenario: Scenario) -> dict[str, Any]:
    """Print progressively and return evaluation results."""
    _print_header(scenario)

    def on_progress(current: int, total: int) -> None:
        _emit(f">> hour {current}/{total} computed")

    def on_phase(msg: str) -> None:
        _emit()
        _emit(f">> {msg}")

    peak: dict[str, Any] | None = None

    def on_hour(row: dict[str, Any]) -> None:
        nonlocal peak
        if peak is None or row["surplus"] > peak["surplus"]:
            peak = row
        _print_hour_row(row)

    _emit(">> Computing hourly v* and guardians...")
    result = evaluate_day(
        scenario,
        on_hour=None,
        on_phase=on_phase,
        on_progress=on_progress,
    )

    _print_hourly_header()
    for row in result["hourly"]:
        on_hour(row)

    _emit("-" * 78)
    if peak:
        _emit(
            f"Peak surplus at {peak['hour']:02d}:00 (+{peak['surplus']:.2f}); "
            f"gardiens {peak['guardians_label']}"
        )
    _emit(
        f"Guardian configuration changes: {result['guardian_changes']} "
        f"(over {len(result['hourly']) - 1} possible transitions)"
    )
    _emit(f"Total LC gain (daily): {result['total_lc_gain']:+.2f}")
    _emit()

    summary = result["least_core_summary"]
    if summary["feasible"]:
        _emit(
            f"Least-core LP: feasible optimal solution found "
            f"(status: {summary['status']}, epsilon = {summary['epsilon']:.4f})."
        )
    else:
        _emit(
            f"Least-core LP: no optimal solution (status: {summary['status']}). "
            "The core may be empty; no theoretical guarantee of feasibility."
        )
    _emit()

    coalition = scenario.coalition
    payoffs = result["payoffs"]
    standalone = payoffs["standalone"]

    _emit("Daily summary per operator")
    _emit("-" * 60)
    _emit(f"{'Op':<4} {'Standalone':>12} {'Least core':>12} {'Gain LC':>10}")
    _emit("-" * 60)
    for i in coalition:
        gain_lc = payoffs["least_core"][i] - standalone[i]
        _emit(
            f"{i + 1:<4} {standalone[i]:>12.2f} "
            f"{payoffs['least_core'][i]:>12.2f} {gain_lc:>+10.2f}"
        )
    _emit("-" * 60)
    total_surplus = sum(r["surplus"] for r in result["hourly"])
    _emit(
        f"{'Tot':<4} {sum(standalone.values()):>12.2f} "
        f"{sum(payoffs['least_core'].values()):>12.2f} "
        f"{result['total_lc_gain']:>+10.2f}"
    )
    _emit(f"Sum of hourly surpluses: {total_surplus:+.2f}")

    return result
