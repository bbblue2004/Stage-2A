from dataclasses import dataclass
from src.data_processing.antenna_metrics import compute_antenna_metrics

DEFAULT_STEP_MINUTES = 15


@dataclass
class OperatorParams:
    """
    Parameters for a single mobile network operator.

    Attributes:
        name: Identifier for the operator
        capacity_epsilon: Maximum traffic capacity (ε_i)
        c: Revenue per unit of traffic
        beta: Variable cost coefficient related to load (β_i)
        K: Fixed costs (typically negative)
    """
    name: str
    capacity_epsilon: float
    c: float
    beta: float
    K: float


@dataclass
class Scenario:
    """Coherent bundle of operators, traffic series, and time-horizon metadata."""
    name: str
    operators: list[OperatorParams]
    traffic: dict[int, list[float]]
    horizon_hours: float
    step_minutes: int

    @property
    def num_steps(self) -> int:
        return len(next(iter(self.traffic.values())))

    @property
    def coalition(self) -> list[int]:
        return list(range(len(self.operators)))

    def sample_steps(self, count: int = 7) -> list[int]:
        if self.num_steps <= 1 or count <= 1:
            return [0]
        return sorted({
            int(round(i * (self.num_steps - 1) / (count - 1)))
            for i in range(count)
        })

    def horizon_label(self) -> str:
        if self.horizon_hours > 1:
            return f"{int(self.horizon_hours)}h ({self.step_minutes}min steps)"
        return f"{int(self.horizon_hours * 60)} min ({self.step_minutes}min steps)"

    def step_label(self, t: int) -> str:
        minutes = t * self.step_minutes
        if self.horizon_hours > 1:
            return f"t={minutes / 60.0:.1f}h (step {t})"
        return f"t={minutes} min (step {t})"


def get_example_operators() -> list[OperatorParams]:
    return [
        OperatorParams(name="Operator1", capacity_epsilon=1600, c=0.041, beta=4.20, K=-55),
        OperatorParams(name="Operator2", capacity_epsilon=1400, c=0.036, beta=3.80, K=-42),
        OperatorParams(name="Operator3", capacity_epsilon=1400, c=0.042, beta=3.80, K=-42),
        OperatorParams(name="Operator4", capacity_epsilon=1550, c=0.031, beta=4.50, K=-52),
    ]


def get_realistic_example_operators(beta, k, epsilon) -> list[OperatorParams]:
    """
    The Operators should represent approximately a radio site of Orange, SFR, Bouygues and Free (in order).
    """
    

    return [
        OperatorParams(name="Operator1", capacity_epsilon=1600, c=0.041, beta=4.20, K=-55),
        OperatorParams(name="Operator2", capacity_epsilon=1400, c=0.036, beta=3.80, K=-42),
        OperatorParams(name="Operator3", capacity_epsilon=1400, c=0.042, beta=3.80, K=-42),
        OperatorParams(name="Operator4", capacity_epsilon=1550, c=0.031, beta=4.50, K=-52),
    ]


def get_example_traffic(
    num_steps: int | None = None,
    horizon_hours: float = 1.0,
    step_minutes: int = DEFAULT_STEP_MINUTES,
) -> dict[int, list[float]]:
    import math

    if num_steps is None:
        num_steps = int(horizon_hours * 60 / step_minutes)

    traffic: dict[int, list[float]] = {0: [], 1: [], 2: [], 3: []}

    for t in range(num_steps):
        t_norm = t / num_steps if num_steps > 1 else 0.0
        wave = 0.6 + 0.8 * math.sin(math.pi * t_norm) ** 2
        traffic[0].append(20.0 + 50.0 * wave)

        early_wave = 0.5 + 0.9 * math.sin(math.pi * (t_norm + 0.1)) ** 2
        traffic[1].append(15.0 + 40.0 * early_wave)

        counter_wave = 1.0 - 0.6 * math.sin(math.pi * t_norm) ** 2
        traffic[2].append(10.0 + 30.0 * counter_wave)

        traffic[3].append(10.0 + 35.0 * t_norm + 5.0 * math.sin(2 * math.pi * t_norm))

    return traffic


def get_realistic_example_traffic(
    operators: list[OperatorParams],
    num_steps: int | None = None,
    horizon_hours: float = 24.0,
    step_minutes: int = DEFAULT_STEP_MINUTES,
    mean_load_factor: float = 0.3,
    noise_std: float = 0.08,
    seed: int | None = 42,
) -> dict[int, list[float]]:
    import math
    import random

    if num_steps is None:
        num_steps = int(horizon_hours * 60 / step_minutes)

    if seed is not None:
        random.seed(seed)

    n = len(operators)
    ti_moy = [mean_load_factor * op.capacity_epsilon for op in operators]
    traffic: dict[int, list[float]] = {i: [] for i in range(n)}

    for t in range(num_steps):
        t_hour = t * step_minutes / 60.0
        psi = max(
            0.1,
            1
            - 0.7 * math.cos(math.pi * (t_hour - 4) / 12)
            + 0.4 * math.sin(math.pi * (t_hour - 12) / 6),
        )
        for i in range(n):
            noise = 1.0 + random.gauss(0, noise_std)
            raw = ti_moy[i] * psi * noise
            traffic[i].append(min(max(0.0, raw), operators[i].capacity_epsilon))

    return traffic


def load_scenario(name: str, step_minutes: int = DEFAULT_STEP_MINUTES) -> Scenario:
    if name == "example":
        operators = get_example_operators()
        traffic = get_example_traffic(step_minutes=step_minutes)
        return Scenario(
            name="example",
            operators=operators,
            traffic=traffic,
            horizon_hours=1.0,
            step_minutes=step_minutes,
        )
    if name == "realistic":
        operators = get_realistic_example_operators()
        traffic = get_realistic_example_traffic(operators, step_minutes=step_minutes)
        return Scenario(
            name="realistic",
            operators=operators,
            traffic=traffic,
            horizon_hours=24.0,
            step_minutes=step_minutes,
        )
    raise ValueError(f"Unknown scenario: {name!r}. Use 'example' or 'realistic'.")


def _plot_traffic_comparison(out_path: str | None = None) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    example = load_scenario("example")
    realistic = load_scenario("realistic")
    palette = ["#2563eb", "#dc2626", "#16a34a", "#f59e0b"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ex_traffic = example.traffic[i]
        rl_traffic = realistic.traffic[i]
        t_example_h = np.arange(len(ex_traffic)) * example.step_minutes / 60.0
        t_realistic_h = np.arange(len(rl_traffic)) * realistic.step_minutes / 60.0

        ax.plot(t_example_h, ex_traffic, color=palette[i], lw=1.8, ls="--", label="example (1h)")
        ax.plot(t_realistic_h, rl_traffic, color=palette[i], lw=1.2, alpha=0.85, label="realistic (24h)")
        ax.set_title(f"Operator {i + 1}")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Traffic")
        ax.legend(fontsize=8)

    fig.suptitle("Example vs Realistic Traffic (absolute values)", fontweight="bold")
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    import os

    out_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(out_dir, exist_ok=True)
    _plot_traffic_comparison(os.path.join(out_dir, "traffic_comparison.png"))
