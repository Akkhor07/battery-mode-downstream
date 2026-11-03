"""Step 10: a third lab on the same cell as Kirkaldy - Stanford LG INR21700-M50T, real driving profiles.

Pozzato, Allam, Onori (2022), Data in Brief 41:107995, OSF qsabn (CC BY 4.0): 10 cells at 23 degC, check-ups with a
C/20 discharge (2.5-4.2 V). Only the capacity tests are used (data/stanford/raw/Diag_*/). The JES estimator needs
C/10 + C/2 windows, which these check-ups do not have, so - as for Michigan - only reference modes are tested.

Reference modes: graphite-only quasi-OCV fit of the C/20 discharge (src.step8_refit, variant dis_gr: same model and
quality rule). Ageing axis: days since the cell's first check-up (the capacity-test files carry no throughput;
all cells ran the same driving-profile ageing between check-ups, so time stands in for throughput - an assumption).

Pre-specified test (as for Michigan, written before any forecast): observe check-ups 0-5 (primary) or 0-3
(secondary), forecast the later ones; capacity vs mode map vs Yu; clusters = cell group letter (G, V, W).

    "F:/Research New/battery-pinn-degradation/.venv/Scripts/python.exe" -m src.step10_stanford [--part fit|eval]
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src import step1_gonogo as s1
from src import step7_reviewer as s7
from src import step9_michigan as s9
from src.step8_refit import fit_curve, u_gr, u_p

ROOT = Path("data/stanford")
V_MIN, V_MAX = 2.5, 4.2


def read(f: str):
    x = pd.read_excel(f, sheet_name=None, usecols=lambda c: c in {"Date_Time", "Step_Index", "Current(A)", "Voltage(V)"})
    x = next(v for k, v in x.items() if k.startswith("Channel"))
    d = x[(x["Current(A)"] < -0.2) & (x["Current(A)"] > -0.3)]
    if len(d) < 1000:
        return None
    step = d.Step_Index.mode()[0]  # the C/20 discharge step
    d = d[d.Step_Index == step]
    t = (pd.to_datetime(d.Date_Time) - pd.to_datetime(d.Date_Time.iloc[0])).dt.total_seconds().to_numpy()
    i = d["Current(A)"].to_numpy()
    q = np.cumsum(np.abs(i) * np.diff(t, prepend=t[0])) / 3600
    g = np.linspace(0, q[-1], 400)
    return pd.to_datetime(x.Date_Time.iloc[0]), g, np.interp(g, q, d["Voltage(V)"].to_numpy()), float(np.median(i))


def part_fit():
    rows = []
    for f in sorted(glob.glob(str(ROOT / "raw" / "Diag_*" / "*.xlsx"))):
        diag = int(re.search(r"Diag_(\d+)", f)[1])
        cell = re.search(r"0_05C_(\w+?)_Channel", f)[1]
        r = read(f)
        if r is None or r[1][-1] < 2.0:
            print("skip", diag, cell)
            continue
        date, q, v, cur = r
        rows.append({"cell": cell, "diag": diag, "date": date, "Q": q[-1], **fit_curve(q, v, cur, u_gr, 5.2)})
        print(diag, cell, round(q[-1], 3), round(rows[-1]["rmse_mV"], 1), flush=True)
    r = pd.DataFrame(rows).sort_values(["cell", "diag"])
    r["ok"] = (r.rmse_mV <= 20) & (r.R > 0) & r.Qn.between(2.05, 11.95) & r.Qp.between(2.05, 11.95)
    r.to_parquet(ROOT / "modes.parquet")


def ocv_capacity(qn, qp, qli):
    def ocv(x):
        return u_p(np.clip((qli - x * qn) / qp, 0, 1)) - u_gr(np.clip(x, 0, 1))

    lo, hi = max(0.0, (qli - qp) / qn) + 1e-6, min(1.0, qli / qn) - 1e-6
    try:
        return (brentq(lambda x: ocv(x) - V_MAX, lo, hi) - brentq(lambda x: ocv(x) - V_MIN, lo, hi)) * qn
    except ValueError:
        return np.nan


def table() -> pd.DataFrame:
    r = pd.read_parquet(ROOT / "modes.parquet")
    good = r.groupby("cell").filter(lambda c: c.ok.all()).cell.unique()
    t = r[r.cell.isin(good)].sort_values(["cell", "diag"]).copy()
    t["rpt"] = t.groupby("cell").cumcount()
    g = t.groupby("cell")
    t["SOH"] = t.Q / g.Q.transform("first")
    t["x"] = (t.date - g.date.transform("first")).dt.total_seconds() / 86400 / 100  # 100 days
    for mode, col in (("LLI", "QLi"), ("LAM_PE", "Qp"), ("LAM_NE", "Qn")):
        t[mode] = 1 - t[col] / g[col].transform("first")
    return t


def part_eval():
    r = pd.read_parquet(ROOT / "modes.parquet")
    t = table()
    lines = [f"Stanford M50T: {r.cell.nunique()} cells, {len(r)} C/20 check-ups, median fit {r.rmse_mV.median():.1f} mV, "
             f"failing {(~r.ok).sum()}; cells with every check-up valid {t.cell.nunique()}",
             "  check-ups per valid cell: " + ", ".join(f"{c} {n}" for c, n in t.groupby('cell').size().items()),
             f"  within-cell R^2 of fade on LLI: "
             f"{t.groupby('cell').apply(lambda c: np.corrcoef(c.LLI, 1 - c.SOH)[0, 1] ** 2).median():.3f} (median)"]
    for k, tag in ((5, "primary"), (3, "secondary")):
        for curve in ("sqrt", "power"):
            s1.CURVE = curve
            f = s9.forecasts(t, k, ocv_capacity)
            if f.empty:
                lines.append(f"  {tag} k={k}: no cell has enough check-ups")
                continue
            p = f.pivot(index="cell", columns="method", values="rmse").dropna()
            grp = p.index.str[0].to_series(index=p.index)
            for m in ("modes_map", "modes_yu"):
                s7.report(lines, f"{tag} k={k} {curve} {m} (n={len(p)})", p[m], p.capacity, grp)
        s1.CURVE = "sqrt"
    text = "\n".join(lines)
    Path("data/step10_stanford.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("fit", "eval"), default="fit")
    (part_fit if ap.parse_args().part == "fit" else part_eval)()
