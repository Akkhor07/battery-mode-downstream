"""Figure 4: how good are the reference modes, and does it matter? (a) Ground truth (Dubarry & Anseán 2022 synthetic
C/25 curves) fitted with the real-data pipelines: worst |error| for truth <= 10 %. (b) Forecast ratio when the
references are perturbed with those error patterns. (c) Window-rule non-inferiority under the same perturbations
(sources: prognostics data/dubarry/fits.parquet, step12_refsens.txt; results/step5_sensitivity.log)."""
from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from common import BLUE, COL2, DESIGN, GREEN, GREY, INK, MUTED, PROG, VERMILION, panel, save

# ---- (a) ground-truth error map, same metric as step11_dubarry.part_eval
r = pd.read_parquet(PROG / "dubarry" / "fits.parquet")
ROWS = [("LLI", "LLI", "LLI"), ("LAM$_{PE}$, delithiated", "LAMPE", "LAMPE"), ("LAM$_{PE}$, lithiated", "LAMLiPE", "LAMLiPE"),
        ("LAM$_{NE}$ (graphite), delith.", "LAMNE", "LAMGIC"), ("LAM$_{NE}$ (graphite), lith.", "LAMliNE", "LAMliGIC"),
        ("LAM$_{NE}$ (Si), delith.", None, "LAMSi"), ("LAM$_{NE}$ (Si), lith.", None, "LAMliSi"),
        ("NE ohmic resistance", "ORINE", "ORINE"), ("NE rate degradation", "RDFNE", "RDFNE"),
        ("PE rate degradation", "RDFPE", "RDFPE"), ("combined path", "Example", "Example")]
CHEMS = ["GIC-NMC532", "(GIC,Si)-NMC811"]


def worst(chem, name):
    t = r[(r.chem == chem) & (r.set == name)].sort_values("level")
    f = t.iloc[0]
    est = np.c_[100 * (1 - t.QLi / f.QLi), 100 * (1 - t.Qp / f.Qp), 100 * (1 - t.Qn / f.Qn)]
    tru = t[["t_LLI", "t_LAM_PE", "t_LAM_NE"]].to_numpy()
    small = tru.max(axis=1) <= 10
    return np.abs(est - tru)[small].max(axis=0)


M = np.full((len(ROWS), 6), np.nan)
for i, (_, n532, n811) in enumerate(ROWS):
    for j, name in enumerate((n532, n811)):
        if name is not None:
            M[i, 3 * j:3 * j + 3] = worst(CHEMS[j], name)

fig = plt.figure(figsize=(COL2, 3.1))
gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1, 1], wspace=0.35, hspace=0.8)
a = fig.add_subplot(gs[:, 0])
cmap = LinearSegmentedColormap.from_list("seq", ["#F2F6FB", "#9CC3E4", BLUE, "#08306B"])
norm = LogNorm(0.2, 60)
im = a.imshow(np.where(np.isnan(M), np.nan, np.clip(M, 0.2, 60)), cmap=cmap, norm=norm, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        if np.isnan(v):
            a.text(j, i, "–", ha="center", va="center", fontsize=6, color=MUTED)
            continue
        a.text(j, i, f"{v:.1f}" if v < 10 else f"{v:.0f}", ha="center", va="center", fontsize=5.8,
               color="white" if v > 6 else INK, fontweight="bold" if v > 2 else "normal")
a.set_xticks(range(6))
a.set_xticklabels(["LLI", "LAM$_{PE}$", "LAM$_{NE}$"] * 2, fontsize=6.3)
a.set_yticks(range(len(ROWS)))
a.set_yticklabels([x[0] for x in ROWS], fontsize=6.3)
a.tick_params(length=0)
for s in a.spines.values():
    s.set_visible(False)
a.axvline(2.5, color="white", lw=2.5)
a.axhline(len(ROWS) - 1.5, color="white", lw=2.5)
for j, lab in ((1, "graphite | NMC532"), (4, "Si-graphite | NMC811")):
    a.text(j, -0.85, lab, ha="center", va="bottom", fontsize=6.5, fontweight="bold")
cb = fig.colorbar(im, ax=a, orientation="horizontal", fraction=0.04, pad=0.08, aspect=30)
cb.set_ticks([0.5, 2, 10, 50])
cb.set_ticklabels(["0.5", "2", "10", "50"])
cb.ax.tick_params(labelsize=6)
cb.set_label("worst |error| for truth ≤ 10 % (pp); bold > 2 pp", fontsize=6.3)
a.set_title("Ground truth: synthetic curves", loc="left", fontsize=7, pad=14)
panel(a, "a")

# ---- (b) forecasting under perturbed references
SC = ["none", "N1", "N2", "SC", "LK", "KD", "ALL"]
SC_LAB = {"none": "unperturbed", "N1": "noise 1 pp", "N2": "noise 2 pp", "SC": "scale ±20 %", "LK": "LLI→LAM$_{PE}$ leak",
          "KD": "kinetic drift", "ALL": "all combined"}
txt = (PROG / "step12_refsens.txt").read_text(encoding="utf-8")
b = fig.add_subplot(gs[0, 1])
b.axvspan(0.9, 1.1, color=GREY, alpha=0.15, lw=0)
b.axvline(1, color=MUTED, lw=0.6)
for k, (block, colour, dy) in enumerate((("Michigan", GREEN, -0.15), ("Kirkaldy", BLUE, 0.15))):
    part = txt.split(block)[1].split("Kirkaldy")[0] if block == "Michigan" else txt.split(block)[1]
    unp = float(re.search(r"unperturbed ratio ([\d.]+)", part)[1])
    vals = {"none": (unp, unp, unp)}
    for s, med, lo, hi in re.findall(r"(\w+)\s+median ([\d.]+) \[5-95 %: ([\d.]+), ([\d.]+)\]", part):
        vals[s] = (float(med), float(lo), float(hi))
    for i, s in enumerate(SC):
        med, lo, hi = vals[s]
        b.plot([lo, hi], [i + dy] * 2, color=colour, lw=1.1)
        b.plot(med, i + dy, "o", ms=3.4, color=colour, mec="white", mew=0.4, label=block if i == 0 else None)
b.set_yticks(range(len(SC)))
b.set_yticklabels([SC_LAB[s] for s in SC], fontsize=6.2)
b.invert_yaxis()
b.tick_params(axis="y", length=0)
b.set_xlabel("forecast error, mode map / capacity (median, 5–95 %)", fontsize=6.5)
b.legend(loc="lower right", handletextpad=0.2, borderaxespad=0.1)
b.set_title("Forecasting with perturbed references", loc="left", fontsize=7)
panel(b, "b")

# ---- (c) window rule under perturbed references
log = (DESIGN / "step5_sensitivity.log").read_text(encoding="utf-8")
c = fig.add_subplot(gs[1, 1])
rows = {m[0]: (float(m[1]), float(m[2])) for m in
        re.findall(r"^\s+(\w+)\s+rule - random:.*?non-inferior in\s+([\d.]+) %.*?non-inferior in\s+([\d.]+) %", log, re.M)}
y = np.arange(len(SC))
h = 0.38
rand_ = [rows[s][0] for s in SC]
fixd = [rows[s][1] for s in SC]
c.barh(y - h / 2, rand_, h, color=GREY, label="rule vs random pairs")
c.barh(y + h / 2, fixd, h, color=VERMILION, label="rule vs best fixed design")
for yi, v in zip(y, fixd):
    if v < 90:
        inside = v > 20
        c.text(v - 2 if inside else v + 2, yi + h / 2, f"{v:.1f} %", va="center", ha="right" if inside else "left",
               fontsize=5.8, color="white" if inside else VERMILION)
c.axvline(90, color=INK, lw=0.6, ls="--")
c.text(90, -0.9, "90 % (pre-specified)", ha="center", va="bottom", fontsize=5.8)
c.set_yticks(y)
c.set_yticklabels([SC_LAB[s] for s in SC], fontsize=6.2)
c.invert_yaxis()
c.tick_params(axis="y", length=0)
c.set_xlim(0, 105)
c.set_xlabel("bootstrap draws non-inferior (%)", fontsize=6.5)
c.legend(loc="upper left", bbox_to_anchor=(0, -0.36), ncol=2, handlelength=1, columnspacing=0.8)
c.set_title("Window rule with perturbed references", loc="left", fontsize=7, pad=10)
panel(c, "c")
save(fig, "fig4_references")
