"""Step 3: the step 1 forecasting test on the second chemistry (40 Kirkaldy NMC cells, Expt 1-5; Expt 1-3 added in step 6).

Per-cell extrapolation in charge throughput (the two experiments cycle differently) with the same two-parameter
curve as step 1, y(x) = c1 sqrt(x) + c2 x, from the check-ups up to RPT k:
  capacity       fit 1 - SOH on the check-ups that also have mode estimates (every second RPT)
  capacity_all   fit 1 - SOH on every RPT up to k (capacity is measured at every RPT)
  modes_<est>    fit LLI and LAM_PE, map to 1 - SOH with a linear map learned on the other cells, anchored at the
                 cell's own observed fade (as in step 1)
Score: SOH RMSE over all later RPTs. Paired cluster bootstrap over cells against 'capacity'.

    "F:/Research New/battery-pinn-degradation/.venv/Scripts/python.exe" -m src.step3_nmc_forecast
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.step1_gonogo import fit_curve, paired_boot

K_OBS = {"expt5": (4, 6), "expt4": (4,), "expt1": (6,), "expt2": (6,), "expt3": (6,)}  # last observed RPT
EST = ("pinn_lm", "cnn_dr", "ref")
MODES = ("LLI", "LAM_PE")


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    t = pd.read_parquet("data/checkups_nmc.parquet").sort_values(["set", "cell", "rpt"])
    t["cid"] = t.set + t.cell
    g = t.groupby("cid")
    for est in ("pinn_lm", "cnn_dr"):  # estimates relative to the first check-up, as in the JES paper
        for m in MODES:
            t[f"{m}_{est}"] = 1 - (1 - t[f"{m}_{est}"]) / (1 - g[f"{m}_{est}"].transform("first"))
    cap = pd.read_parquet("data/capacity_nmc.parquet")
    cap["cid"] = cap.set + cap.cell
    return t, cap


def forecasts(t: pd.DataFrame, cap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sname, ks in K_OBS.items():
        for k in ks:
            for cid in t[t.set == sname].cid.unique():
                c, cc = t[t.cid == cid], cap[cap.cid == cid]
                obs, obs_all = c[c.rpt <= k], cc[cc.rpt <= k]
                fut = cc[cc.rpt > k]
                x_obs, x_fut = obs.throughput_Ah / 1000, fut.throughput_Ah.to_numpy() / 1000
                preds = {"capacity": fit_curve(x_obs, 1 - obs.SOH)(x_fut),
                         "capacity_all": fit_curve(obs_all.throughput_Ah / 1000, 1 - obs_all.SOH)(x_fut)}
                for est in EST:
                    cols = [f"{m}_{est}" for m in MODES]
                    if obs[cols].isna().any().any():  # cnn estimates exist for Expt 5 and 4 only
                        continue
                    other = t[t.cid != cid].dropna(subset=cols)
                    X = np.column_stack([other[f"{m}_{est}"] for m in MODES] + [np.ones(len(other))])
                    w, *_ = np.linalg.lstsq(X, 1 - other.SOH.to_numpy(), rcond=None)
                    mc = {m: fit_curve(x_obs, obs[f"{m}_{est}"]) for m in MODES}
                    fade_obs = np.column_stack([mc[m](x_obs) for m in MODES]) @ w[:2]
                    off = np.mean((1 - obs.SOH.to_numpy()) - fade_obs)
                    preds[f"modes_{est}"] = np.column_stack([mc[m](x_fut) for m in MODES]) @ w[:2] + off
                for name, p in preds.items():
                    rows.append({"set": sname, "k": k, "cid": cid, "method": name, "n_points": len(obs),
                                 "soh_rmse_pp": 100 * np.sqrt(np.mean((1 - p - fut.SOH.to_numpy()) ** 2))})
    return pd.DataFrame(rows)


def main():
    t, cap = load()
    f = forecasts(t, cap)
    f.to_csv("data/step3_nmc_forecast.csv", index=False)
    lines = ["NMC per-cell extrapolation in throughput: mean SOH RMSE [pp] over later RPTs (vs capacity, 95 % CI)"]
    for (sname, k), d in f.groupby(["set", "k"], sort=False):
        p = d.pivot(index="cid", columns="method", values="soh_rmse_pp")
        lines.append(f"\n  {sname}, observed up to RPT {k} ({d.n_points.iloc[0]} estimate check-ups, {len(p)} cells)")
        for m in p.columns:
            lo, hi = paired_boot(p[m], p["capacity"])
            lines.append(f"    {m:14s} {p[m].mean():5.2f}" + ("" if m == "capacity" else f"  ({lo:+.2f}..{hi:+.2f})"))
    text = "\n".join(lines)
    print(text)
    open("data/step3_nmc_forecast.txt", "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
