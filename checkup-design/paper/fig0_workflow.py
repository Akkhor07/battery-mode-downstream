"""Figure 1 (workflow): data, the two kinds of modes, the three analyses and their outcomes. Drawn, not data."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import BLUE, COL2, GREEN, GREY, INK, MUTED, ORANGE, VERMILION, save

TINT = {GREY: "#F0F0F0", BLUE: "#E3EEF7", ORANGE: "#FBF0DC", GREEN: "#DFF2EB", VERMILION: "#FAE6DC"}

fig = plt.figure(figsize=(COL2, 3.25))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 44)
ax.axis("off")


def box(x, y, w, h, title, body, colour):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=TINT[colour],
                                ec=colour, lw=0.8))
    ax.text(x + 0.9, y + h - 0.8, title, ha="left", va="top", fontsize=6.9, fontweight="bold", color=INK)
    ax.text(x + 0.9, y + h - 3.3, body, ha="left", va="top", fontsize=6.1, color=INK, linespacing=1.25)
    return x, y, w, h


def arrow(p, q, colour=MUTED):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=colour,
                                 shrinkA=0, shrinkB=1))


def line(xs, ys, colour=MUTED):
    ax.plot(xs, ys, color=colour, lw=0.8, solid_capstyle="butt")


mid = lambda b: b[1] + b[3] / 2  # noqa: E731

for x, t in ((1, "Data"), (24, "Degradation modes"), (47, "Analyses"), (77, "Outcomes")):
    ax.text(x, 42.8, t, ha="left", va="center", fontsize=7.4, fontweight="bold", color=MUTED)

# data
d1 = box(1, 32.5, 18.5, 8, "Kirkaldy et al.", "LG M50T, NMC811/Gr–SiO$_x$\n40 cells, 5 experiments", GREY)
d2 = box(1, 22.5, 18.5, 8, "Wheeler et al.", "A123, graphite/LFP\n20 cells, 7 protocols", GREY)
d3 = box(1, 12.5, 18.5, 8, "Mohtat et al.", "NMC532/graphite pouch\n19 cells (replication)", GREY)
d4 = box(1, 2.0, 18.5, 8, "Dubarry (synthetic)", "C/25 curves with known\nmodes, 20 degradation sets", GREY)

# modes
m0 = box(24, 32.5, 19, 8, "Capacity only", "SOH at every check-up\n(baseline)", GREY)
m1 = box(24, 22.5, 19, 8, "Reference modes", "fits of whole low-rate\ncurves (published, refitted)", BLUE)
m2 = box(24, 12.5, 19, 8, "Estimated modes", "two partial windows, two\nC-rates; simulation-trained", VERMILION)

# analyses
a1 = box(47, 29.5, 26, 11, "1  Life forecasting",
         "modes vs capacity extrapolation: mode\nmap, half-cell and faithful Lui et al.,\ncross-cell, mechanism switch; knee\nindicator", ORANGE)
a2 = box(47, 16, 26, 11, "2  Check-up window design",
         "50 window pairs: Cramér–Rao and Bayesian\nbounds vs label-free misfit; rule fixed\non 16 cells, confirmed on 24 held-out\ncells", ORANGE)
a3 = box(47, 2.0, 26, 11.5, "3  Reference audit",
         "fits of curves with known modes; forecasts\nand window rule repeated with references\nperturbed by the errors found\n(6 scenarios $\\times$ 200 draws)", ORANGE)

# outcomes
o1 = box(77, 29.5, 22, 11, "No forecasting gain",
         "modes at most 1 % better on\naverage; 10 % gain excluded in\n9 of 12 setups; estimates worse", GREEN)
o2 = box(77, 16, 22, 11, "Bounds fail, misfit rule holds",
         "bounds do not rank designs;\nrule non-inferior to random wide\nwindows, 30 % less test time", GREEN)
o3 = box(77, 2.0, 22, 11.5, "LLI recovered, LAM$_\\mathrm{NE}$ not",
         "LLI within about 2 pp; errors of\nthat size leave the conclusions\nunchanged", GREEN)

# data -> modes: one collector
xc = 21.5
line([19.8, xc], [mid(d1)] * 2), line([19.8, xc], [mid(d2)] * 2), line([19.8, xc], [mid(d3)] * 2)
line([xc, xc], [mid(d3), mid(d1)])
arrow((xc, mid(m1)), (23.7, mid(m1)))
arrow((xc, mid(m0)), (23.7, mid(m0)))
arrow((xc, mid(m2)), (23.7, mid(m2)))
# modes -> analyses: one collector
xm = 45
for m in (m0, m1, m2):
    line([43.3, xm], [mid(m)] * 2)
line([xm, xm], [mid(a3) + 2, mid(m0)])
for a, dy in ((a1, 0), (a2, 0), (a3, 2)):
    arrow((xm, mid(a) + dy), (46.7, mid(a) + dy))
# synthetic data -> audit
arrow((19.8, mid(d4)), (46.7, mid(d4)))
# analyses -> outcomes
for a, o in ((a1, o1), (a2, o2), (a3, o3)):
    arrow((73.4, mid(a)), (76.6, mid(o)), colour=INK)

ax.text(1, 0.15, "All comparisons: same capacity-only baseline, 10 % equivalence margin, two-stage cluster bootstrap, "
        "criteria fixed in advance (Table 2).", ha="left", va="bottom", fontsize=6.1, color=MUTED, style="italic")
save(fig, "fig0_workflow")
