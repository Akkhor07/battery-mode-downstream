"""Shared paths, style and result parsing for the paper figures (one script per figure, run from this folder).

    "F:/Research New/battery-pinn-degradation/.venv/Scripts/python.exe" figN_*.py
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PROG = HERE.parent.parent / "battery-pinn-prognostics" / "data"
DESIGN = HERE.parent / "results"
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)

# Okabe-Ito, colour-blind safe; fixed order, never cycled
BLUE, ORANGE, GREEN, VERMILION, SKY, PURPLE, YELLOW = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7", "#F0E442"
GREY, INK, MUTED = "#8C8C8C", "#222222", "#666666"

# Elsevier: single column 90 mm, 1.5 column 140 mm, double column 190 mm
MM = 1 / 25.4
COL1, COL15, COL2 = 90 * MM, 140 * MM, 190 * MM

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7.5,
    "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.2,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "savefig.dpi": 600, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def panel(ax, letter: str):
    ax.text(-0.02, 1.02, f"({letter})", transform=ax.transAxes, ha="right", va="bottom", fontweight="bold", fontsize=8.5)


def save(fig, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", FIGS / f"{name}.pdf")


_ROW = re.compile(r"^\s*(?P<label>.+?)\s+capacity\s+(?P<cap>[\d.]+)\s+modes\s+(?P<modes>[\d.]+)\s+change\s+(?P<chg>[+-]\d+)%"
                  r".*clusters: 95% \[(?P<lo>[+-]?\d+)%,(?P<hi>[+-]?\d+)%\] best gain (?P<gain>[\d.]+)%")


def comparisons(path: Path, start: str | None = None, stop: str | None = None) -> dict[str, dict]:
    """Rows 'capacity X modes Y change Z% | ... | clusters: 95% [lo,hi] best gain G%' of a result file, by label.
    Only lines between the first line containing ``start`` and the next containing ``stop`` are read."""
    out, on = {}, start is None
    for line in path.read_text(encoding="utf-8").splitlines():
        if start is not None and start in line:
            on = True
        elif on and stop is not None and stop in line:
            break
        m = _ROW.match(line) if on else None
        if m:
            out[m["label"].strip()] = {k: float(m[k]) for k in ("cap", "modes", "chg", "lo", "hi", "gain")}
    return out


def write(df: pd.DataFrame, name: str, note: str = ""):
    df.to_csv(FIGS / f"{name}.csv", index=False)
    cells = lambda r: "| " + " | ".join("–" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |"  # noqa: E731
    md = "\n".join([cells(df.columns), "|" + "---|" * len(df.columns)] + [cells(r) for r in df.itertuples(index=False)])
    (FIGS / f"{name}.md").write_text(md + ("\n\n" + note if note else "") + "\n", encoding="utf-8")
    print(f"--- {name}\n{md}\n")
