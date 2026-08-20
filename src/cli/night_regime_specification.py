"""Compare three specifications of the affine power model.

The seven-day cloud of many antennas splits into a night level and a day
level. This script quantifies what that split does to the calibrated pair
(F, gamma) under three fits:

  all    : one intercept, one slope, every active hour  (current calibration)
  day    : one intercept, one slope, hours 6 h-24 h only
  dummy  : two intercepts (day and night), one shared slope, every active hour
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data_processing.power_validation import load_calibrated_population  # noqa: E402

CACHE = Path("data") / "processed" / "calibrated_population_7d.npz"
FIGURES = Path("figures") / "diagnostics"
NIGHT_END = 6
EXAMPLE = "00003663U1"


def _fit(design: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    rmse = float(np.sqrt(np.mean(residual**2)) / target.mean())
    return coefficients, rmse


def main() -> None:
    pop = load_calibrated_population(CACHE)
    n = len(pop.antenna_ids)
    traffic = pop.traffic_gb.reshape(n, -1)
    power = pop.power_w.reshape(n, -1)
    hours = np.tile(np.arange(24), (n, pop.days.size))
    live = power > 0.0
    night = hours < NIGHT_END

    columns = ("f_all", "g_all", "r_all",
               "f_day", "g_day", "r_day",
               "f_dum", "g_dum", "r_dum", "gap")
    out = {name: np.full(n, np.nan) for name in columns}

    for k in range(n):
        keep = live[k]
        d, p, is_night = traffic[k][keep], power[k][keep], night[k][keep]
        if p.size < 24 or np.ptp(d) <= 0:
            continue

        ones = np.ones_like(d)
        (f_all, g_all), r_all = _fit(np.column_stack([ones, d]), p)

        day = ~is_night
        if day.sum() >= 12 and np.ptp(d[day]) > 0:
            (f_day, g_day), r_day = _fit(
                np.column_stack([ones[day], d[day]]), p[day])
            out["f_day"][k], out["g_day"][k], out["r_day"][k] = f_day, g_day, r_day

        if is_night.sum() >= 6 and day.sum() >= 12:
            design = np.column_stack([ones, is_night.astype(float), d])
            (f_dum, gap, g_dum), r_dum = _fit(design, p)
            out["f_dum"][k], out["g_dum"][k], out["r_dum"][k] = f_dum, g_dum, r_dum
            out["gap"][k] = gap

        out["f_all"][k], out["g_all"][k], out["r_all"][k] = f_all, g_all, r_all

    ok = np.isfinite(out["f_dum"]) & np.isfinite(out["f_day"])
    print(f">> Antennes retenues pour la comparaison : {ok.sum()} / {n}\n")

    print(">> Qualité d'ajustement, RMSE normalisée")
    for tag, key in (("toutes heures ", "r_all"), ("jour seul     ", "r_day"),
                     ("indicatrice   ", "r_dum")):
        v = out[key][ok]
        print(f"   {tag} : médiane {np.median(v):.4f}   "
              f"q75 {np.quantile(v, 0.75):.4f}   q95 {np.quantile(v, 0.95):.4f}")

    print("\n>> Décalage nocturne estimé par l'indicatrice (W, négatif = nuit plus basse)")
    gap = out["gap"][ok]
    print(f"   médiane {np.median(gap):+.1f}   q05 {np.quantile(gap, 0.05):+.1f}   "
          f"q95 {np.quantile(gap, 0.95):+.1f}")
    print(f"   en part de F_indicatrice : médiane "
          f"{np.median(gap / out['f_dum'][ok]):+.3f}")

    print("\n>> Effet sur les paramètres, rapport à la spécification 'toutes heures'")
    for tag, key, base in (("F  indicatrice / F  toutes ", "f_dum", "f_all"),
                           ("gamma indicatrice / gamma toutes", "g_dum", "g_all"),
                           ("F  jour / F  toutes        ", "f_day", "f_all"),
                           ("gamma jour / gamma toutes  ", "g_day", "g_all")):
        ratio = out[key][ok] / out[base][ok]
        ratio = ratio[np.isfinite(ratio)]
        print(f"   {tag} : médiane {np.median(ratio):.3f}   "
              f"q25 {np.quantile(ratio, 0.25):.3f}   q75 {np.quantile(ratio, 0.75):.3f}")

    print("\n>> Niveau nocturne F + décalage, rapporté à la spécification actuelle")
    f_night = out["f_dum"][ok] + out["gap"][ok]
    print(f"   (F + décalage) / F toutes heures : médiane "
          f"{np.median(f_night / out['f_all'][ok]):.3f}")
    print("   rapport F / gamma, qui gouverne l'intérêt d'éteindre :")
    for tag, num, den in (("toutes heures      ", out["f_all"][ok], out["g_all"][ok]),
                          ("indicatrice, jour  ", out["f_dum"][ok], out["g_dum"][ok]),
                          ("indicatrice, nuit  ", f_night, out["g_dum"][ok])):
        ratio = num / den
        ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
        print(f"      {tag} : médiane {np.median(ratio):8.2f} Go/h")

    print("\n>> Part fixe F / (F + gamma * trafic moyen), sur heures actives")
    mean_traffic = np.array([
        traffic[k][live[k]].mean() if live[k].any() else np.nan for k in range(n)
    ])
    for tag, fk, gk in (("toutes heures", "f_all", "g_all"),
                        ("indicatrice  ", "f_dum", "g_dum")):
        share = out[fk][ok] / (out[fk][ok] + out[gk][ok] * mean_traffic[ok])
        print(f"   {tag} : médiane {np.median(share):.3f}")

    print("\n>> Le décalage est-il un état, ou une courbure de la relation ?")
    extrapolation = np.full(n, np.nan)
    curvature = np.full(n, np.nan)
    for k in range(n):
        if not np.isfinite(out["f_day"][k]):
            continue
        keep = live[k]
        d, p, is_night = traffic[k][keep], power[k][keep], night[k][keep]
        day = ~is_night
        if is_night.sum() >= 6:
            predicted = out["f_day"][k] + out["g_day"][k] * d[is_night]
            observed = p[is_night].mean()
            if observed > 0:
                extrapolation[k] = predicted.mean() / observed
        # Courbure : pente sur la moitié basse contre la moitié haute du
        # trafic, en restant dans le seul régime de jour.
        if day.sum() >= 24:
            dd, pp = d[day], p[day]
            cut = np.median(dd)
            low, high = dd <= cut, dd > cut
            if low.sum() >= 6 and high.sum() >= 6 and np.ptp(dd[low]) > 0 \
                    and np.ptp(dd[high]) > 0:
                s_low = np.polyfit(dd[low], pp[low], 1)[0]
                s_high = np.polyfit(dd[high], pp[high], 1)[0]
                if abs(s_low) > 1e-9:
                    curvature[k] = s_high / s_low

    v = extrapolation[np.isfinite(extrapolation)]
    print("   droite ajustée de jour, extrapolée aux heures de nuit,")
    print("   divisée par la puissance nocturne réellement mesurée :")
    print(f"      médiane {np.median(v):.3f}   q25 {np.quantile(v, 0.25):.3f}   "
          f"q75 {np.quantile(v, 0.75):.3f}")
    print(f"      part des antennes surestimées de plus de 10 % : "
          f"{np.mean(v > 1.10):.1%}")
    c = curvature[np.isfinite(curvature)]
    print("\n   dans le seul régime de jour, pente de la moitié haute du trafic")
    print("   divisée par celle de la moitié basse :")
    print(f"      médiane {np.median(c):.3f}   q25 {np.quantile(c, 0.25):.3f}   "
          f"q75 {np.quantile(c, 0.75):.3f}")

    print("\n>> Signe des paramètres")
    for tag, key in (("gamma toutes heures", "g_all"), ("gamma indicatrice  ", "g_dum"),
                     ("F toutes heures    ", "f_all"), ("F indicatrice      ", "f_dum")):
        v = out[key][ok]
        print(f"   {tag} : {np.mean(v <= 0):.2%} de valeurs négatives ou nulles")

    _figure(pop, traffic, power, live, night, out, ok)


def _figure(pop, traffic, power, live, night, out, ok) -> None:
    ids = [str(a) for a in pop.antenna_ids]
    k = ids.index(EXAMPLE)
    keep = live[k]
    d, p, is_night = traffic[k][keep], power[k][keep], night[k][keep]

    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.3))

    axis = axes[0]
    for sel, colour, label in ((~is_night, "#2A6BB0", "6 h-24 h"),
                               (is_night, "#C0504D", "0 h-6 h")):
        axis.scatter(d[sel], p[sel], s=18, alpha=0.7, color=colour,
                     edgecolors="none", label=label)
    grid = np.linspace(0.0, float(d.max()) * 1.02, 100)
    axis.plot(grid, out["f_all"][k] + out["g_all"][k] * grid, color="black",
              linewidth=1.6, label="une droite, 168 h")
    axis.plot(grid, out["f_day"][k] + out["g_day"][k] * grid, color="#2A6BB0",
              linewidth=1.4, linestyle="--", label="jour seul")
    axis.plot(grid, out["f_dum"][k] + out["gap"][k] + out["g_dum"][k] * grid,
              color="#C0504D", linewidth=1.4, linestyle="--",
              label="indicatrice, niveau nuit")
    axis.set_xlabel("trafic descendant (Go/h)")
    axis.set_ylabel("puissance moyenne (W)")
    axis.set_title(f"(a) antenne {EXAMPLE}\nune seule droite traverse "
                   "les deux états", fontsize=9)
    axis.set_ylim(bottom=0.0)
    axis.set_xlim(left=0.0)
    axis.grid(alpha=0.25, linewidth=0.5)
    axis.legend(fontsize=7.5, loc="lower right")

    for column, (key, base, colour, name) in enumerate(
        ((("f_dum", "f_all", "#2A6BB0", "F")),
         (("g_dum", "g_all", "#C0504D", r"$\gamma$"))), start=1
    ):
        ratio = out[key][ok] / out[base][ok]
        ratio = ratio[np.isfinite(ratio)]
        ratio = ratio[(ratio > 0.0) & (ratio < 3.0)]
        axis = axes[column]
        axis.hist(ratio, bins=60, color=colour, alpha=0.75)
        axis.axvline(1.0, color="black", linewidth=1.2)
        axis.axvline(float(np.median(ratio)), color="black", linewidth=1.2,
                     linestyle="--")
        axis.set_xlabel(f"{name} avec indicatrice / {name} sur 168 h")
        axis.set_ylabel("antennes")
        axis.set_title(f"({'bc'[column - 1]}) effet sur {name}, "
                       f"médiane {np.median(ratio):.2f}", fontsize=9)
        axis.grid(alpha=0.25, linewidth=0.5)

    figure.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    output = FIGURES / "night_regime_specification.png"
    figure.savefig(output, dpi=140)
    plt.close(figure)
    print(f"\n>> Figure : {output.resolve()}")


if __name__ == "__main__":
    main()
