"""Run every numerical experiment in dependency order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STAGES = (
    (
        "power calibration",
        "src.experiments.power_calibration",
        "--rebuild-cache",
    ),
    (
        "protocol walkthrough figure",
        "src.experiments.plan_walkthrough",
        None,
    ),
    (
        "instance diagnostics",
        "src.experiments.instance_diagnostics",
        "--rebuild",
    ),
    (
        "operational efficiency",
        "src.experiments.operational_efficiency",
        "--rebuild",
    ),
    (
        "coalition stability",
        "src.experiments.coalition_stability",
        "--rebuild",
    ),
    (
        "threshold mechanisms",
        "src.experiments.threshold_mechanisms",
        "--rebuild",
    ),
    (
        "parameter sensitivity",
        "src.experiments.parameter_sensitivity",
        "--rebuild",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="force every stage to ignore a valid existing cache",
    )
    args = parser.parse_args()

    for label, module, rebuild_flag in STAGES:
        command = [sys.executable, "-m", module, "--quiet"]
        if args.rebuild and rebuild_flag is not None:
            command.append(rebuild_flag)
        print(f">> Starting {label}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    print(">> All numerical experiments completed", flush=True)


if __name__ == "__main__":
    main()
