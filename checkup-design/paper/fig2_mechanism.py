"""Figure 2: why modes do not beat capacity. (a-c) Within each cell, capacity fade follows the reference LLI almost
exactly, so the modes carry little forecast information beyond capacity itself. (d) Positive control: with exact modes
a forecasting gain exists, but it is gone at a mode noise of ~1 pp per check-up; the measured scatter of the reference
modes is at that level and the estimates are far above it (sources: prognostics data/*.parquet, step5_halfcell.txt,
PAPER_OUTLINE robustness section for the scatter values)."""
from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import BLUE, COL2, GREEN, GREY, INK, MUTED, ORANGE, PROG, VERMILION, YELLOW, panel, save


def michigan() -> pd.DataFrame:
    """Same construction as step9_michigan.soh_table('discharge'): valid cells, from the first full check-up."""
    r = pd.read_parquet(PROG / "michigan" / "modes.parquet")
    m = r[r.kind == "discharge"]
    good = m.groupby("cell").filter(lambda c: c.ok.all()).cell.unique()
    t = m[m.cell.isin(good)].sort_values(["cell", "rpt"]).copy()
    g = t.groupby("cell")
    t["SOH"] = t.Q / g.Q.transform("first")
    t["LLI_ref"] = 1 - t.QLi / g.QLi.transform("first")
    return t


def within_r2(t, cell):
    r2 = []
    for _, c in t.groupby(cell):
        if len(c) >= 3 and np.ptp(c.LLI_ref) > 0:
            r2.append(np.corrcoef(c.LLI_ref, 1 - c.SOH)[0, 1] ** 2)
    return np.median(r2)


kir = pd.read_parquet(PROG / "capacity_nmc.parquet").assign(cid=lambda d: d.set + d.cell)
lfp = pd.read_parquet(PROG / "checkups_lfp.parquet")
mich = michigan()
SETS = [("Kirkaldy NMC (40 cells)", kir, "cid", BLUE), ("Michigan NMC (21 cells)", mich, "cell", GREEN),
        ("Wheeler LFP (20 cells)", lfp, "cell", ORANGE)]

fig = plt.figure(figsize=(COL2, 2.25))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.45], wspace=0.45)
for i, (name, t, cell, colour) in enumerate(SETS):
    ax = fig.add_subplot(gs[i])
    for _, c in t.groupby(cell):
        c = c.sort_values("rpt")
        ax.plot(100 * c.LLI_ref, 100 * (1 - c.SOH), "-", color=colour, lw=0.6, alpha=0.55)
    hi = 100 * max(t.LLI_ref.max(), (1 - t.SOH).max())
    ax.plot([0, hi], [0, hi], ":", color=MUTED, lw=0.7)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_aspect("equal")
    ax.set_title(name, fontsize=7, loc="left")
    ax.text(0.04, 0.96, f"within-cell $R^2$\nmedian {within_r2(t, cell):.3f}", transform=ax.transAxes, va="top",
            fontsize=6.3, color=INK)
    ax.set_xlabel("reference LLI (%)")
    if i == 0:
        ax.set_ylabel("capacity fade, 1 − SOH (%)")
    panel(ax, "abc"[i])

# (d) positive control
txt = (PROG / "step5_halfcell.txt").read_text(encoding="utf-8")
ax = fig.add_subplot(gs[3])
ax.axhspan(-10, 10, color=GREY, alpha=0.15, lw=0)
ax.axhline(0, color=MUTED, lw=0.6)
for label, colour, marker in (("all cells", INK, "o"), ("no knee", BLUE, "s"), ("knee cells", VERMILION, "^")):
    line = next(l for l in txt.splitlines() if l.strip().startswith(label))
    pts = [(float(sd), float(ch)) for sd, ch in re.findall(r"modes sd ([\d.]+):\s+[\d.]+ \(([+-]\d+)%\)", line)]
    sd, ch = np.array(pts).T
    n = re.search(r"n=\s*(\d+)", line)[1]
    ax.plot(100 * sd, ch, "-", marker=marker, ms=3.5, color=colour, mec="white", mew=0.5, label=f"{label} (n={n})")
# measured check-up scatter around a smooth per-cell curve (upper bound on noise), from step4/step12
ax.axvspan(0.39, 1.07, color=YELLOW, alpha=0.35, lw=0)
ax.text(0.73, 0.03, "reference\nLLI scatter", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=5.8, color=INK)
ax.text(1.9, 240, "no knee at 4 pp:\n+598 % (off scale)", ha="right", va="top", fontsize=5.6, color=BLUE)
ax.set_xscale("symlog", linthresh=0.5)
ax.set_xticks([0, 0.5, 1, 2, 4])
ax.set_xticklabels(["0", "0.5", "1", "2", "4"])
ax.minorticks_off()
ax.set_ylim(-100, 250)
ax.set_xlabel("mode noise per check-up, s.d. (pp)")
ax.set_ylabel("forecast error vs capacity (%)")
ax.set_title("Positive control, 60 synthetic cells", fontsize=7, loc="left")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.83), handlelength=1.6)
panel(ax, "d")
save(fig, "fig2_mechanism")
