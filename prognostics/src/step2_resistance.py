"""Step 2 (Gap 3 go/no-go): do mode estimates predict resistance growth on real NMC beyond capacity fade?

Target: relative growth of the 0.1 s pulse resistance from the Kirkaldy performance summaries, dR = R/R0 - 1,
at every check-up after the first (40 cells, Expt 1-5). Leave-one-cell-out linear regression on
  capacity       1 - SOH
  + est modes    LLI, LAM_PE (PINN + LM, relative to the first check-up) and the change of log Dn, log Dp
  + ref modes    LLI, LAM_PE, LAM_NE from the whole-curve reference fit (offline upper bound)
Paired cluster bootstrap over cells for the MAE difference to capacity. Reported with and without Expt 5 cell A,
whose resistance grows by +511 % (all other cells <= +43 %).

    "F:/Research New/battery-pinn-degradation/.venv/Scripts/python.exe" -m src.step2_resistance
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from src.stage1_export import KIRKALDY_ALL as KIRKALDY, OUT, STAGE1

FEATURES = {
    "capacity": ["fade"],
    "capacity + est modes": ["fade", "LLI_est", "LAM_PE_est", "logDn_est", "logDp_est"],
    "capacity + ref modes": ["fade", "LLI_ref", "LAM_PE_ref", "LAM_NE_ref"],
}
OUTLIER = "expt5A"


def resistance() -> pd.DataFrame:
    rows = []
    for sname, folder in KIRKALDY.items():
        for f in sorted(glob.glob(str(STAGE1 / "data" / "kirkaldy" / folder / "Summary Data" / "Performance Summary" / "*.csv"))):
            r = pd.read_csv(f)["0.1s Resistance [Ohms]"].to_numpy()
            rows.append(pd.DataFrame({"set": sname, "cell": Path(f).name.split(" cell ")[1][0],
                                      "rpt": np.arange(len(r)), "dR": r / r[0] - 1}))
    return pd.concat(rows, ignore_index=True)


def table() -> pd.DataFrame:
    c = pd.read_parquet(OUT / "checkups_nmc.parquet").merge(resistance(), on=["set", "cell", "rpt"])
    c = c.sort_values(["set", "cell", "rpt"])
    c["cid"] = c.set + c.cell
    g = c.groupby("cid")
    for m in ("LLI", "LAM_PE"):
        c[f"{m}_est"] = 1 - (1 - c[f"{m}_pinn_lm"]) / (1 - g[f"{m}_pinn_lm"].transform("first"))
    for m in ("logDn", "logDp"):
        c[f"{m}_est"] = c[f"{m}_pinn_lm"] - g[f"{m}_pinn_lm"].transform("first")
    c["fade"] = 1 - c.SOH
    return c[(c.rpt > 0) & c.dR.notna()]


def loco_abs_error(c: pd.DataFrame, cols: list[str]) -> pd.Series:
    out = []
    for cid in c.cid.unique():
        tr, te = c[c.cid != cid], c[c.cid == cid]
        w = np.linalg.lstsq(np.c_[np.ones(len(tr)), tr[cols]], tr.dR, rcond=None)[0]
        out.append(pd.Series(np.abs(np.c_[np.ones(len(te)), te[cols]] @ w - te.dR.to_numpy()), index=te.cid))
    return pd.concat(out)


def main():
    rng = np.random.default_rng(0)
    lines = []
    full = table()
    for label, c in ((f"all {full.cid.nunique()} cells", full), (f"without {OUTLIER}", full[full.cid != OUTLIER])):
        cells = c.cid.unique()
        lines.append(f"{label}: {len(c)} check-ups, corr(dR, 1 - SOH) = {np.corrcoef(c.dR, c.fade)[0, 1]:.2f}")
        errs = {k: loco_abs_error(c, v) for k, v in FEATURES.items()}
        base = errs["capacity"].groupby(level=0).mean().loc[cells].to_numpy()
        idx = rng.integers(0, len(cells), (2000, len(cells)))
        for k, e in errs.items():
            per = e.groupby(level=0).mean().loc[cells].to_numpy()
            lo, hi = np.percentile(100 * (per[idx].mean(1) - base[idx].mean(1)), [2.5, 97.5])
            lines.append(f"  {k:22s} LOCO MAE {100 * e.mean():5.1f} % of R0 (vs capacity {lo:+.1f}..{hi:+.1f})")
    text = "\n".join(lines)
    print(text)
    (OUT / "step2_resistance.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
