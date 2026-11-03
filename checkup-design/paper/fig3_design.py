"""Figure 3: choosing the partial windows of a check-up. (a) The Cramer-Rao bound does not rank the 50 grid designs
(16 development cells); it is 10-100x below the realised error. (b) The label-free fit misfit does, within a window
size. (c) Pre-specified confirmation on 24 held-out cells: the misfit rule against random 35-50 % pairs and against
the best fixed design of the development cells (sources: results/nmc_design_misfit_summary.csv,
step3_confirm_analyse.log, step4_fixed.log)."""
from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

import pandas as pd

from common import BLUE, COL2, DESIGN, GREY, INK, MUTED, ORANGE, VERMILION, panel, save

d = pd.read_csv(DESIGN / "nmc_design_misfit_summary.csv")
SIZE_STYLE = {0.2: (ORANGE, "o", "20 % windows (2.4 h)"), 0.3: (BLUE, "s", "30 % windows (3.6 h)")}

fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.35), gridspec_kw=dict(wspace=0.62, width_ratios=[1, 1, 1.15]))

a = axes[0]
for size, (c, m, lab) in SIZE_STYLE.items():
    s = d[d["size"].round(1) == size]
    a.plot(s.crb, s.rmse, m, ms=3.5, color=c, mec="white", mew=0.4, label=lab)
lim = (0.08, 30)
a.plot(lim, lim, ":", color=MUTED, lw=0.7)
a.text(3.2, 3.0, "error = bound", fontsize=5.8, color=MUTED, ha="left", va="center")
a.set_xscale("log")
a.set_yscale("log")
a.set_xlim(*lim)
a.set_ylim(2.5, 70)
a.set_xticks([0.1, 1, 10])
a.set_xticklabels(["0.1", "1", "10"])
a.set_yticks([2, 5, 10, 20])
a.set_yticklabels(["2", "5", "10", "20"])
a.minorticks_off()
a.set_xlabel("Cramér–Rao bound (pp)")
a.set_ylabel("realised RMSE of LLI, LAM$_{PE}$ (pp)")
a.text(0.04, 0.96, f"Spearman {spearmanr(d.crb, d.rmse)[0]:+.2f}", transform=a.transAxes, va="top", fontsize=6.3)
a.legend(loc="upper left", bbox_to_anchor=(0, 0.9), handletextpad=0.3, borderaxespad=0.2)
a.set_title("Design by Fisher bound", loc="left", fontsize=7)
panel(a, "a")

b = axes[1]
for size, (c, m, lab) in SIZE_STYLE.items():
    s = d[d["size"].round(1) == size]
    b.plot(s.misfit, s.rmse, m, ms=3.5, color=c, mec="white", mew=0.4)
    rho = spearmanr(s.misfit, s.rmse)[0]
    b.text(0.97, 0.25 - 0.08 * (size == 0.3), f"{size:.0%}: Spearman {rho:+.2f}", transform=b.transAxes, va="top", ha="right",
           fontsize=6.3, color=c)
b.text(0.97, 0.09, f"pooled: {spearmanr(d.misfit, d.rmse)[0]:+.2f}", transform=b.transAxes, va="top", ha="right", fontsize=6.3)
b.set_xlabel("fit misfit, median (mV)")
b.set_ylabel("realised RMSE (pp)")
b.set_ylim(0, None)
b.set_title("Design by fit misfit (label-free)", loc="left", fontsize=7)
panel(b, "b")

# (c) confirmation
conf = (DESIGN / "step3_confirm_analyse.log").read_text(encoding="utf-8")
fix = (DESIGN / "step4_fixed.log").read_text(encoding="utf-8")
rule_p = float(re.search(r"rule: misfit-chosen 30% design\s+RMSE ([\d.]+)", conf)[1])
rand_p = float(re.search(r"random 35-50 % pairs\s+RMSE ([\d.]+)", conf)[1])
fix_p = float(re.search(r"F30 .*?RMSE ([\d.]+) pp", fix)[1])
per = {e: (float(r), float(q)) for e, r, q in re.findall(r"(expt\d): rule ([\d.]+)\s+random ([\d.]+)", conf)}
perf = {e: float(f) for e, f in re.findall(r"(expt\d): rule [\d.]+\s+fixed ([\d.]+)", fix.split("Fany")[0])}
n = {e: int(k) for e, k in re.findall(r"(expt\d): .*?\((\d+) cells\)", conf)}
rows = [(f"Expt {e[-1]}, {n[e]} cells", per[e][0], per[e][1], perf[e]) for e in sorted(per)]
rows.append(("pooled, 24", rule_p, rand_p, fix_p))

c = axes[2]
STRAT = [("misfit rule, 30 %  (3.6 h)", 1, VERMILION, "o"), ("random 35–50 % pairs  (5.1 h)", 2, GREY, "D"),
         ("best fixed design of dev. cells  (3.6 h)", 3, BLUE, "s")]
ys = np.arange(len(rows))[::-1].astype(float)
ys[-1] -= 0.0
ys[:-1] += 0.4  # gap above the pooled row
for (lab, *vals), y in zip(rows, ys):
    lo, hi = min(vals), max(vals)
    c.plot([lo, hi], [y, y], color=GREY, lw=0.6, alpha=0.6, zorder=1)
    for (slab, i, col, mk) in STRAT:
        c.plot(vals[i - 1], y, mk, ms=4.6 if i == 1 else 3.8, color=col, mec="white", mew=0.5, zorder=3,
               label=slab if y == ys[0] else None)
c.set_yticks(ys)
c.set_yticklabels([r[0] for r in rows])
c.get_yticklabels()[-1].set_fontweight("bold")
c.tick_params(axis="y", length=0)
c.spines["left"].set_visible(False)
c.set_xlim(2, 10)
c.set_ylim(ys.min() - 0.7, ys.max() + 2.2)
c.set_xlabel("RMSE of LLI, LAM$_{PE}$ (pp)")
c.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.02), handletextpad=0.3, labelspacing=0.3)
c.set_title("Held-out confirmation", loc="left", fontsize=7)
panel(c, "c")
save(fig, "fig3_design")
