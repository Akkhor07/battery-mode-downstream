"""Figure 1: do degradation modes improve capacity forecasts? Ratio of the mode-based to the capacity-only SOH error,
with the 95 % cluster-bootstrap interval, for every pre-specified forecasting setup (sources: prognostics
data/step7_reviewer.txt and data/step9_michigan.txt)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from common import BLUE, COL2, GREY, INK, MUTED, ORANGE, PROG, VERMILION, comparisons, panel, save

kir = comparisons(PROG / "step7_reviewer.txt")
mich = comparisons(PROG / "step9_michigan.txt", "C/20 discharge", "C/20 charge")

# (group, label, row) ; groups are drawn as blocks from top to bottom
REF = [
    ("Kirkaldy NMC, 40 cells", "sqrt extrapolation", kir["NMC 40, sqrt, modes_ref"]),
    ("Kirkaldy NMC, 40 cells", "power-law extrapolation", kir["NMC 40, power, modes_ref"]),
    ("Kirkaldy NMC, 40 cells", "half-cell, after Lui et al.", kir["NMC Yu half-cell, modes_ref"]),
    ("Kirkaldy NMC, 40 cells", "cross-cell, leave-one-cell-out", kir["LOCO + ref LLI, LAM_PE, LAM_NE"]),
    ("Kirkaldy NMC, 40 cells", "cross-cell, leave-one-expt-out", kir["LOEO + ref LLI, LAM_PE, LAM_NE"]),
    ("Michigan NMC, 19 cells", "mode map, sqrt", mich["primary k=5 sqrt modes_map (n=19)"]),
    ("Michigan NMC, 19 cells", "mode map, power law", mich["primary k=5 power modes_map (n=19)"]),
    ("Michigan NMC, 19 cells", "after Lui et al., sqrt", mich["primary k=5 sqrt modes_yu (n=19)"]),
    ("Michigan NMC, 19 cells", "after Lui et al., power law", mich["primary k=5 power modes_yu (n=19)"]),
    ("Wheeler LFP, 20 cells", "sqrt extrapolation", kir["LFP 20, 600 cyc, sqrt, modes_ref"]),
    ("Wheeler LFP, 20 cells", "power-law extrapolation", kir["LFP 20, 600 cyc, power, modes_ref"]),
    ("Wheeler LFP, 20 cells", "half-cell, after Lui et al.", kir["LFP Yu half-cell, modes_ref"]),
]
EST = [
    ("Kirkaldy NMC, 40 cells", "sqrt extrapolation", kir["NMC 40, sqrt, modes_pinn_lm"]),
    ("Kirkaldy NMC, 40 cells", "power-law extrapolation", kir["NMC 40, power, modes_pinn_lm"]),
    ("Kirkaldy NMC, 40 cells", "half-cell, after Lui et al.", kir["NMC Yu half-cell, modes_pinn_lm"]),
    ("Kirkaldy NMC, 40 cells", "cross-cell, leave-one-cell-out", kir["LOCO + est LLI, LAM_PE"]),
    ("Kirkaldy NMC, 40 cells", "cross-cell, leave-one-expt-out", kir["LOEO + est LLI, LAM_PE"]),
    ("Wheeler LFP, 20 cells", "sqrt extrapolation", kir["LFP 20, 600 cyc, sqrt, modes_pinn_enc"]),
    ("Wheeler LFP, 20 cells", "power-law extrapolation", kir["LFP 20, 600 cyc, power, modes_pinn_enc"]),
    ("Wheeler LFP, 20 cells", "half-cell, after Lui et al.", kir["LFP Yu half-cell, modes_pinn_enc"]),
]


def positions(rows):
    y, ys, heads, prev = 0.0, [], [], None
    for g, _, _ in rows:
        if g != prev:
            y -= 0.6 if prev is not None else 0
            heads.append((g, y))
            y -= 1.0
            prev = g
        ys.append(y)
        y -= 1.0
    return np.array(ys), heads


def forest(ax, rows, colour, xlim, ticks):
    ys, heads = positions(rows)
    ax.axvspan(0.9, 1.1, color=GREY, alpha=0.15, lw=0, zorder=0)
    ax.axvline(1.0, color=MUTED, lw=0.6, zorder=1)
    for y, (_, label, r) in zip(ys, rows):
        lo, hi, mid = 1 + r["lo"] / 100, 1 + r["hi"] / 100, r["modes"] / r["cap"]
        clip_lo, clip_hi = max(lo, xlim[0]), min(hi, xlim[1])
        ax.plot([clip_lo, clip_hi], [y, y], color=colour, lw=1.2, solid_capstyle="round", zorder=2)
        for edge, clip, x in ((lo, clip_lo, xlim[0]), (hi, clip_hi, xlim[1])):
            if edge != clip:
                ax.annotate("", xy=(x, y), xytext=(x * (1.12 if x < 1 else 0.89), y),
                            arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.0, mutation_scale=6))
        ax.plot(mid, y, "o", ms=4.2, mfc=colour, mec="white", mew=0.8, zorder=3)
        ax.text(xlim[1] * 1.06, y, f"{mid:.2f}", va="center", ha="left", fontsize=6.5, color=INK)
        ax.text(xlim[0] / 1.06, y, label, va="center", ha="right", fontsize=6.5, color=INK)
    for g, y in heads:
        ax.text(xlim[0] / 1.06, y, g, va="center", ha="right", fontsize=6.8, fontweight="bold", color=INK)
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("SOH forecast error, modes / capacity only")
    return ys


fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 4.0), gridspec_kw=dict(wspace=1.15, width_ratios=[1.05, 1]))
ya = forest(a, REF, BLUE, (0.3, 2.5), [0.5, 0.9, 1.1, 2])
a.set_title("Reference (whole-curve) modes", loc="left", pad=10)
panel(a, "a")
yb = forest(b, EST, VERMILION, (0.8, 30), [1, 2, 5, 10, 20])
b.set_title("Modes estimated from partial windows", loc="left", pad=10)
panel(b, "b")
for ax in (a, b):
    ax.text(1.0, 1.0, "ratio", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5, color=MUTED)
lo = min(ya.min(), yb.min()) - 0.8
for ax in (a, b):
    ax.set_ylim(lo, 1.0)
b.text(0.03, 0.12, "Michigan: check-ups have no partial windows,\nso the estimator cannot be applied",
       transform=b.transAxes, fontsize=6.3, color=MUTED, va="bottom")
a.text(1.0, 0.95, "±10 %", ha="center", va="top", fontsize=6, color=MUTED)
save(fig, "fig1_forecast")
