"""Diagnose why the affine power model fits some antennas poorly.

Takes one antenna, separates the low-power cluster from the main cloud, and
checks whether the split follows the hour of the day, the weekday, or the
traffic level.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.data_processing.power_validation import load_calibrated_population

CACHE = Path("data") / "processed" / "calibrated_population_7d.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--antenna", default="00003663U1")
    args = parser.parse_args()

    pop = load_calibrated_population(CACHE)
    ids = [str(a) for a in pop.antenna_ids]
    k = ids.index(args.antenna)

    traffic = pop.traffic_gb[k]
    power = pop.power_w[k]
    n_days = traffic.shape[0]
    hours = np.tile(np.arange(24), (n_days, 1))
    days = np.repeat(np.arange(n_days)[:, None], 24, axis=1)
    active = power > 0.0

    print(f">> Antenne {args.antenna}")
    print(f"   P_fixe = {pop.p_fixed_w[k]:.1f} W, pente = {pop.slope_w_per_gb[k]:.2f} W/Go")
    print(f"   R^2 = {pop.r_squared[k]:.3f}, RMSE normalisée = {pop.normalized_rmse[k]:.3f}")
    print(f"   heures actives {active.sum()} / {active.size}")
    print(f"   trafic : min {traffic[active].min():.3f}, "
          f"médiane {np.median(traffic[active]):.3f}, max {traffic[active].max():.3f} Go/h")
    print(f"   puissance : min {power[active].min():.1f}, "
          f"médiane {np.median(power[active]):.1f}, max {power[active].max():.1f} W")

    p = power[active]
    d = traffic[active]
    h = hours[active]
    day = days[active]
    split = 0.5 * (p.min() + np.median(p))
    low = p < split
    print(f"\n>> Séparation au seuil {split:.0f} W")
    print(f"   groupe bas  : {low.sum():>4} heures, "
          f"puissance {p[low].mean():.0f} W, trafic {d[low].mean():.3f} Go/h")
    print(f"   groupe haut : {(~low).sum():>4} heures, "
          f"puissance {p[~low].mean():.0f} W, trafic {d[~low].mean():.3f} Go/h")

    print("\n>> Répartition du groupe bas par heure de la journée")
    for start in range(0, 24, 12):
        row = "  ".join(
            f"{hour:02d}h:{int(np.sum(low & (h == hour))):>2}"
            for hour in range(start, start + 12)
        )
        print(f"   {row}")
    print("\n>> Répartition du groupe bas par jour (0 = lundi)")
    print("   " + "  ".join(
        f"j{j}:{int(np.sum(low & (day == j))):>3}" for j in range(n_days)
    ))

    print("\n>> Ajustement en excluant le groupe bas")
    if (~low).sum() >= 3 and np.ptp(d[~low]) > 0:
        slope, intercept = np.polyfit(d[~low], p[~low], 1)
        resid = p[~low] - (intercept + slope * d[~low])
        rmse = float(np.sqrt(np.mean(resid**2)) / p[~low].mean())
        ss = 1.0 - np.sum(resid**2) / np.sum((p[~low] - p[~low].mean()) ** 2)
        print(f"   P_fixe = {intercept:.1f} W (contre {pop.p_fixed_w[k]:.1f})")
        print(f"   pente  = {slope:.2f} W/Go (contre {pop.slope_w_per_gb[k]:.2f})")
        print(f"   R^2    = {ss:.3f}, RMSE normalisée = {rmse:.3f}")

    print("\n>> Comparaison de population : amplitude du trafic et qualité du fit")
    span = pop.traffic_gb.reshape(len(ids), -1).max(axis=1)
    rmse_all = pop.normalized_rmse
    for label, lo, hi in (("q0-q25", 0.00, 0.25), ("q25-q50", 0.25, 0.50),
                          ("q50-q75", 0.50, 0.75), ("q75-q100", 0.75, 1.00)):
        a, b = np.quantile(span, (lo, hi))
        sel = (span >= a) & (span <= b)
        print(f"   pic dans {label} [{a:6.1f}, {b:6.1f}] Go/h : "
              f"RMSE médiane {np.median(rmse_all[sel]):.3f}, "
              f"R^2 médian {np.median(pop.r_squared[sel]):.3f}")
    print(f"\n   corrélation entre log(pic) et RMSE normalisée : "
          f"{np.corrcoef(np.log(span), rmse_all)[0, 1]:+.3f}")

    print("\n>> Combien d'antennes ont un régime nocturne distinct ?")
    n = len(ids)
    d_all = pop.traffic_gb.reshape(n, -1)
    p_all = pop.power_w.reshape(n, -1)
    hour_all = np.tile(np.arange(24), (n, pop.days.size))
    fitted = pop.p_fixed_w[:, None] + pop.slope_w_per_gb[:, None] * d_all
    resid = p_all - fitted
    live = p_all > 0.0
    night = live & (hour_all <= 5)
    day_mask = live & (hour_all >= 8)

    night_bias = np.full(n, np.nan)
    for k2 in range(n):
        if night[k2].sum() >= 10 and day_mask[k2].sum() >= 10:
            scale = max(p_all[k2][live[k2]].mean(), 1e-9)
            night_bias[k2] = (resid[k2][night[k2]].mean()
                              - resid[k2][day_mask[k2]].mean()) / scale
    ok = ~np.isnan(night_bias)
    values = night_bias[ok]
    q = np.quantile(values, (0.05, 0.25, 0.50, 0.75, 0.95))
    print(f"   antennes évaluables : {ok.sum()}")
    print("   écart de résidu nuit (0-5 h) moins jour (8-23 h), "
          "en part de la puissance moyenne :")
    print(f"      q05 {q[0]:+.3f}  q25 {q[1]:+.3f}  médiane {q[2]:+.3f}  "
          f"q75 {q[3]:+.3f}  q95 {q[4]:+.3f}")
    for threshold in (-0.05, -0.10, -0.20):
        print(f"      part des antennes sous {threshold:+.2f} : "
              f"{np.mean(values < threshold):.1%}")
    print("\n>> Biais sur la fenêtre nocturne 0-7 h")
    window = live & (hour_all <= 6)
    over = np.full(n, np.nan)
    for k2 in range(n):
        if window[k2].sum() >= 10:
            observed = p_all[k2][window[k2]].mean()
            predicted = fitted[k2][window[k2]].mean()
            if observed > 0:
                over[k2] = predicted / observed
    valid = ~np.isnan(over)
    qq = np.quantile(over[valid], (0.25, 0.50, 0.75, 0.95))
    print("   puissance prédite par l'ajustement, divisée par la puissance "
          "réellement mesurée")
    print(f"      q25 {qq[0]:.3f}  médiane {qq[1]:.3f}  q75 {qq[2]:.3f}  "
          f"q95 {qq[3]:.3f}")
    print(f"      part des antennes surestimées de plus de 10 % : "
          f"{np.mean(over[valid] > 1.10):.1%}")

    strong = ok & (night_bias < -0.10)
    print(f"\n   parmi les antennes à régime nocturne marqué ({strong.sum()}) :")
    print(f"      RMSE normalisée médiane {np.median(rmse_all[strong]):.3f} "
          f"contre {np.median(rmse_all[ok & ~strong]):.3f} pour les autres")
    print(f"      corrélation entre écart nocturne et RMSE : "
          f"{np.corrcoef(values, rmse_all[ok])[0, 1]:+.3f}")


if __name__ == "__main__":
    main()
