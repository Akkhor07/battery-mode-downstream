"""Step 12: could reference-mode errors of the size found in step 11 (Dubarry ground truth) hide a forecasting gain?

Pre-specified in battery-pinn-checkup-design/PLAN.md, step 5. The reference LLI and LAM_PE are perturbed with the
error patterns of step 11 and the mode map is re-run; capacity is measured directly and is not perturbed.
  Michigan: discharge references, primary design (k = 5), sqrt.   Kirkaldy: published references, last k per set, sqrt.

    "F:/Research New/battery-pinn-degradation/.venv/Scripts/python.exe" -m src.step12_refsens
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import step1_gonogo as s1
from src import step3_nmc_forecast as s3
from src import step9_michigan as s9

SCENARIOS = ("N1", "N2", "SC", "LK", "KD", "ALL")
N_DRAWS = 200


def delta(cell, prog, lli, lam, scen, rng):
    """Reference error for LLI and LAM_PE (fractions); zero at each cell's first check-up (prog == 0)."""
    cells = np.unique(cell)
    idx = np.searchsorted(cells, cell)
    d = np.zeros((len(cell), 2))
    if scen in ("N1", "ALL"):
        d += rng.normal(0, 0.01, d.shape)
    if scen == "N2":
        d += rng.normal(0, 0.02, d.shape)
    if scen in ("SC", "ALL"):
        d += np.c_[lli, lam] * rng.uniform(-0.2, 0.2, (len(cells), 2))[idx]
    if scen == "LK":
        d[:, 1] += 0.6 * lli
    if scen in ("KD", "ALL"):
        d += (rng.uniform(0, 0.05, len(cells))[idx] * prog)[:, None]
    d[prog == 0] = 0
    return d


def michigan(t, rng, scen):
    t = t.copy()
    if scen != "none":
        prog = (t.rpt / t.groupby("cell").rpt.transform("max")).to_numpy()
        d = delta(t.cell.to_numpy(), prog, t.LLI.to_numpy(), t.LAM_PE.to_numpy(), scen, rng)
        t["LLI"] += d[:, 0]
        t["LAM_PE"] += d[:, 1]
    p = s9.forecasts(t, s9.K_PRIMARY).pivot(index="cell", columns="method", values="rmse").dropna()
    return p.modes_map.mean() / p.capacity.mean()


def kirkaldy(t, cap, rng, scen):
    t = t.copy()
    if scen != "none":
        prog = (t.rpt / t.groupby("cid").rpt.transform("max")).to_numpy()
        d = delta(t.cid.to_numpy(), prog, t.LLI_ref.to_numpy(), t.LAM_PE_ref.to_numpy(), scen, rng)
        t["LLI_ref"] += d[:, 0]
        t["LAM_PE_ref"] += d[:, 1]
    f = s3.forecasts(t, cap)
    f = f[f.k == f.set.map(f.groupby("set").k.max())]
    p = f.pivot(index="cid", columns="method", values="soh_rmse_pp")
    return p.modes_ref.mean() / p.capacity.mean()


def main():
    s1.CURVE = "sqrt"
    s3.EST = ("ref",)  # only the reference modes are perturbed
    mt = s9.soh_table("discharge")
    kt, kcap = s3.load()
    lines = ["STEP 12: mode map / capacity RMSE ratio under perturbed reference modes (sqrt; < 1 = modes better)"]
    for name, fn in (("Michigan k=5", lambda rng, s: michigan(mt, rng, s)),
                     ("Kirkaldy 40", lambda rng, s: kirkaldy(kt, kcap, rng, s))):
        lines.append(f"  {name}: unperturbed ratio {fn(None, 'none'):.3f}")
        for scen in SCENARIOS:
            rng = np.random.default_rng(12)
            r = np.array([fn(rng, scen) for _ in range(N_DRAWS)])
            lines.append(f"    {scen:4s} median {np.median(r):.3f} [5-95 %: {np.percentile(r, 5):.3f}, {np.percentile(r, 95):.3f}]"
                         f", draws with >= 10 % gain {100 * (r <= 0.9).mean():.1f} %")
            print(lines[-1], flush=True)
    text = "\n".join(lines)
    Path("data/step12_refsens.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
