"""Figure 5: resistance growth is where modes looked useful, but the gain depends on how the reference modes are
defined. Change in the MAE of 0.1 s resistance growth when LLI and LAM_PE are added to capacity fade, with the
95 % cluster-bootstrap interval; filled = significant after Holm correction (source: prognostics data/step8_refit.txt)."""
from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np

from common import BLUE, COL2, GREY, INK, MUTED, PROG, VERMILION, panel, save

txt = (PROG / "step8_refit.txt").read_text(encoding="utf-8")
BLOCKS = [("published", "published modes, all cells"), ("published, chg_gr subset", "published, charge-refit cells"),
          ("chg_gr", "own refit, C/10 charge"), ("published, dis_gr subset", "published, discharge-refit cells"),
          ("dis_gr", "own refit, C/10 discharge")]
LINE = re.compile(r"leave-one-(cell|experiment)-out \+ (est|ref) LLI, LAM_PE\s+([\d.]+) vs\s+([\d.]+)\s+cluster 95% "
                  r"\[([+-][\d.]+),([+-][\d.]+)\]")
HOLM = re.compile(r"Holm-adjusted p: (.*)")


def block(key):
    head = "=== reference = published, all cells ===" if key == "published" else f"=== reference = {key} ==="
    part = txt.split(head)[1].split("===")[0]
    part = part.split("B. NMC 0.1 s resistance")[1].split("C. Cross-cell")[0]
    out, split = {}, None
    for line in part.splitlines():
        m = LINE.search(line)
        if m:
            split = m[1]
            out[(split, m[2])] = [float(m[3]) - float(m[4]), float(m[5]), float(m[6]), None]
        h = HOLM.search(line)
        if h:
            for name, p in re.findall(r"\+ (est|ref) LLI, LAM_PE ([\d.]+)", h[1]):
                out[(split, name)][3] = float(p)
    return out


data = {k: block(k) for k, _ in BLOCKS}
fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.3), sharey=True, gridspec_kw=dict(wspace=0.08))
ys = np.arange(len(BLOCKS))[::-1].astype(float)
for ax, (split, title) in zip(axes, (("cell", "leave one cell out"), ("experiment", "leave one experiment out"))):
    ax.axvline(0, color=MUTED, lw=0.6)
    for y, (key, _) in zip(ys, BLOCKS):
        for kind, colour, dy in (("ref", BLUE, 0.15), ("est", VERMILION, -0.15)):
            d, lo, hi, p = data[key][(split, kind)]
            ax.plot([lo, hi], [y + dy] * 2, color=colour, lw=1.1)
            sig = p is not None and p < 0.05
            ax.plot(d, y + dy, "o", ms=4, mec=colour, mew=1.0, mfc=colour if sig else "white", zorder=3)
            ax.text(hi + 0.08, y + dy, (f"p={p:.2f}" if p >= 0.01 else "p<0.01"), va="center", fontsize=5.5, color=colour)
    ax.set_xlim(-3.2, 4.2)
    ax.set_xlabel("Δ MAE of resistance growth vs capacity only (pp of $R_0$)", fontsize=6.5)
    ax.set_title(title, loc="left", fontsize=7)
    ax.tick_params(axis="y", length=0)
    ax.text(-3.1, ys.max() + 0.75, "← modes help", fontsize=5.8, color=MUTED, va="center")
axes[0].set_yticks(ys)
axes[0].set_yticklabels([lab for _, lab in BLOCKS])
axes[0].spines["left"].set_visible(False)
axes[1].spines["left"].set_visible(False)
axes[0].set_ylim(ys.min() - 0.6, ys.max() + 1.0)
h = [plt.Line2D([], [], marker="o", ls="-", color=BLUE, ms=4, label="+ reference LLI, LAM$_{PE}$"),
     plt.Line2D([], [], marker="o", ls="-", color=VERMILION, ms=4, label="+ estimated LLI, LAM$_{PE}$"),
     plt.Line2D([], [], marker="o", ls="", mfc=GREY, mec=GREY, ms=4, label="Holm p < 0.05"),
     plt.Line2D([], [], marker="o", ls="", mfc="white", mec=GREY, ms=4, label="not significant")]
fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.55, -0.12), handlelength=1.6, columnspacing=1.2)
panel(axes[0], "a")
panel(axes[1], "b")
save(fig, "fig5_resistance")
