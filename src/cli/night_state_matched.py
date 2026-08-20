"""Is the night level driven by the clock or simply by low traffic?

Compares night and day hours restricted to a common traffic range. If the
power gap survives at matched traffic, the level shift is a state of the
equipment and not a point further down the same affine relation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data_processing.power_validation import load_calibrated_population  # noqa: E402

CACHE = Path("data") / "processed" / "calibrated_population_7d.npz"
FIGURES = Path("figures") / "diagnostics"
NIGHT_END = 6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--antenna", default="00003663U1")
    args = parser.parse_args()

    pop = load_calibrated_population(CACHE)
    ids = [str(a) for a in pop.antenna_ids]
    n = len(ids)
    traffic = pop.traffic_gb.reshape(n, -1)
    power = pop.power_w.reshape(n, -1)
    hours = np.tile(np.arange(24), (n, pop.days.size))
    live = power > 0.0
    night = hours < NIGHT_END

    k = ids.index(args.antenna)
    keep = live[k]
    d, p, is_night = traffic[k][keep], power[k][keep], night[k][keep]
    day = ~is_night

    print(f">> Antenne {args.antenna}")
    print(f"   heures éteintes, puissance nulle : {int((~keep).sum())} / {keep.size}")
    print(f"   nuit : {int(is_night.sum())} h, trafic "
          f"[{d[is_night].min():.3f}, {d[is_night].max():.3f}] Go/h, "
          f"puissance [{p[is_night].min():.0f}, {p[is_night].max():.0f}] W")
    print(f"   jour : {int(day.sum())} h, trafic "
          f"[{d[day].min():.3f}, {d[day].max():.3f}] Go/h, "
          f"puissance [{p[day].min():.0f}, {p[day].max():.0f}] W")

    print("\n>> Dispersion de la puissance à trafic quasi constant, la nuit")
    span = p[is_night].max() - p[is_night].min()
    print(f"   étendue {span:.0f} W soit {span / p[is_night].mean():.1%} "
          f"de la moyenne nocturne")
    print(f"   écart-type {p[is_night].std():.1f} W")

    ceiling = d[is_night].max()
    overlap = day & (d <= ceiling)
    print(f"\n>> Heures de jour dont le trafic ne dépasse pas le maximum nocturne "
          f"({ceiling:.3f} Go/h)")
    print(f"   nombre : {int(overlap.sum())}")
    if overlap.any():
        print(f"   puissance moyenne de jour  : {p[overlap].mean():.0f} W "
              f"(trafic moyen {d[overlap].mean():.3f} Go/h)")
        print(f"   puissance moyenne de nuit  : {p[is_night].mean():.0f} W "
              f"(trafic moyen {d[is_night].mean():.3f} Go/h)")
        print(f"   écart : {p[overlap].mean() - p[is_night].mean():+.0f} W")
    print("\n   Heure par heure, puissance moyenne et trafic moyen :")
    hour_of = hours[k][keep]
    for hour in range(24):
        sel = hour_of == hour
        if sel.any():
            tag = "nuit" if hour < NIGHT_END else "jour"
            print(f"      {hour:02d} h  {tag}  trafic {d[sel].mean():6.3f} Go/h   "
                  f"puissance {p[sel].mean():6.1f} W")

    print("\n>> Même test sur toute la population, à trafic apparié")
    ratios = []
    gaps = []
    for j in range(n):
        keep_j = live[j]
        dj, pj = traffic[j][keep_j], power[j][keep_j]
        nj = night[j][keep_j]
        if nj.sum() < 6 or (~nj).sum() < 12:
            continue
        top = dj[nj].max()
        bottom = dj[nj].min()
        sel = (~nj) & (dj <= top) & (dj >= bottom)
        if sel.sum() < 6:
            continue
        night_power = pj[nj].mean()
        if night_power <= 0:
            continue
        ratios.append(pj[sel].mean() / night_power)
        gaps.append(pj[sel].mean() - night_power)
    ratios = np.asarray(ratios)
    gaps = np.asarray(gaps)
    print(f"   antennes avec un recouvrement de trafic exploitable : {ratios.size}")
    print("   puissance de jour divisée par puissance de nuit, à trafic comparable :")
    print(f"      médiane {np.median(ratios):.3f}   q25 {np.quantile(ratios, 0.25):.3f}"
          f"   q75 {np.quantile(ratios, 0.75):.3f}")
    print(f"   écart absolu médian : {np.median(gaps):+.0f} W")
    print(f"   part des antennes où le jour consomme au moins 10 % de plus : "
          f"{np.mean(ratios > 1.10):.1%}")

    print("\n>> L'état bas coïncide-t-il avec un trafic nul ?")
    night_live = live & night
    zero = night_live & (traffic <= 0.0)
    print(f"   heures nocturnes actives : {int(night_live.sum())}")
    print(f"   dont trafic exactement nul : {int(zero.sum())} "
          f"({zero.sum() / max(night_live.sum(), 1):.2%})")
    share = np.array([
        traffic[j][night[j] & live[j]].mean()
        / max(traffic[j][(~night[j]) & live[j]].mean(), 1e-9)
        for j in range(n)
        if (night[j] & live[j]).any() and ((~night[j]) & live[j]).any()
    ])
    print(f"   trafic nocturne moyen rapporté au trafic diurne moyen : "
          f"médiane {np.median(share):.3f}")

    print("\n>> Dans l'état bas, la puissance dépend-elle encore du trafic ?")
    night_slopes, day_slopes, spans = [], [], []
    for j in range(n):
        keep_j = live[j]
        dj, pj = traffic[j][keep_j], power[j][keep_j]
        nj = night[j][keep_j]
        if nj.sum() < 12 or (~nj).sum() < 24:
            continue
        if np.ptp(dj[nj]) <= 0 or np.ptp(dj[~nj]) <= 0:
            continue
        # Amplitude de trafic disponible la nuit, pour juger de l'identifiabilité.
        low = dj[nj]
        if low.min() <= 0:
            continue
        spans.append(low.max() / low.min())
        night_slopes.append(np.polyfit(low, pj[nj], 1)[0])
        day_slopes.append(np.polyfit(dj[~nj], pj[~nj], 1)[0])
    night_slopes = np.asarray(night_slopes)
    day_slopes = np.asarray(day_slopes)
    spans = np.asarray(spans)
    print(f"   antennes évaluables : {night_slopes.size}")
    print(f"   amplitude de trafic nocturne, max/min : médiane {np.median(spans):.1f}")
    print(f"   pente de nuit  : médiane {np.median(night_slopes):7.2f} W/Go")
    print(f"   pente de jour  : médiane {np.median(day_slopes):7.2f} W/Go")
    ratio_slope = night_slopes / day_slopes
    ratio_slope = ratio_slope[np.isfinite(ratio_slope)]
    print(f"   pente de nuit / pente de jour : médiane {np.median(ratio_slope):.3f}")
    print(f"   part des antennes à pente nocturne négative : "
          f"{np.mean(night_slopes < 0):.1%}")

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))

    axis = axes[0]
    mean_power = np.array([p[hour_of == h].mean() for h in range(24)])
    mean_traffic = np.array([d[hour_of == h].mean() for h in range(24)])
    axis.plot(range(24), mean_power, color="#2A6BB0", marker="o", markersize=4,
              linewidth=1.4, label="puissance")
    axis.axvspan(-0.5, NIGHT_END - 0.5, color="#C0504D", alpha=0.10)
    axis.set_xlabel("heure de la journée")
    axis.set_ylabel("puissance moyenne (W)", color="#2A6BB0")
    axis.set_ylim(bottom=0.0)
    axis.set_xlim(-0.5, 23.5)
    twin = axis.twinx()
    twin.plot(range(24), mean_traffic, color="#7F7F7F", marker="s",
              markersize=3, linewidth=1.2, linestyle="--", label="trafic")
    twin.set_ylabel("trafic moyen (Go/h)", color="#7F7F7F")
    twin.set_ylim(bottom=0.0)
    axis.set_title("(a) la bascule suit l'horloge, pas le trafic :\n"
                   "entre 5 h et 6 h le trafic double, la puissance saute de "
                   "70 %", fontsize=9)
    axis.grid(alpha=0.25, linewidth=0.5)

    axis = axes[1]
    for sel, colour, label in ((day, "#2A6BB0", "6 h-24 h"),
                               (is_night, "#C0504D", "0 h-6 h")):
        axis.scatter(d[sel], p[sel], s=20, alpha=0.7, color=colour,
                     edgecolors="none", label=label)
    axis.set_xscale("log")
    axis.set_xlabel("trafic descendant (Go/h), échelle logarithmique")
    axis.set_ylabel("puissance moyenne (W)")
    axis.set_ylim(bottom=0.0)
    axis.set_title("(b) en échelle logarithmique, les points de nuit\n"
                   "se déplient : le trafic n'y est pas nul", fontsize=9)
    axis.grid(alpha=0.25, linewidth=0.5)
    axis.legend(fontsize=8, loc="lower right")

    axis = axes[2]
    clipped = ratios[(ratios > 0.5) & (ratios < 2.5)]
    axis.hist(clipped, bins=60, color="#6E8B3D", alpha=0.8)
    axis.axvline(1.0, color="black", linewidth=1.2)
    axis.axvline(float(np.median(ratios)), color="black", linewidth=1.2,
                 linestyle="--")
    axis.set_xlabel("puissance de jour / puissance de nuit, à trafic comparable")
    axis.set_ylabel("antennes")
    axis.set_title(f"(c) à trafic égal, le jour consomme\n"
                   f"{np.median(ratios):.2f} fois plus, sur {ratios.size} antennes",
                   fontsize=9)
    axis.grid(alpha=0.25, linewidth=0.5)

    figure.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    output = FIGURES / "night_state_matched.png"
    figure.savefig(output, dpi=140)
    plt.close(figure)
    print(f"\n>> Figure : {output.resolve()}")


if __name__ == "__main__":
    main()
