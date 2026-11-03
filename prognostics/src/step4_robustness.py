"""Step 4: robustness of the "modes add nothing beyond capacity" results.

equivalence  For every comparison, the largest improvement over capacity that the data still allow: the
             one-sided 95 % bound of MAE(modes) - MAE(capacity) from the paired cluster bootstrap over cells,
             reported as a share of the capacity error. Below MARGIN (10 %) = no useful gain, at 95 %.
curves       Every forecast repeated with a power law a x^b (0.2 <= b <= 3, can bend either way), besides the
             default c1 sqrt(x) + c2 x.
mechanism    How much of the capacity fade the dominant reference mode explains on its own, pooled and within
             cells; if one mode carries the fade, capacity already contains what the modes add.

    "F:/Research New/battery-pinn-degradation/.venv/Scripts/python.exe" -m src.step4_robustness
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import step1_gonogo as s1
from src import step2_resistance as s2
from src import step3_nmc_forecast as s3

MARGIN = 0.10
RNG = np.random.default_rng(4)


def best_gain(a, b, n=4000):
    """Largest plausible reduction of mean |a| below mean |b| (one-sided 95 %), as a share of mean |b|."""
    a, b = np.abs(np.asarray(a, float)), np.abs(np.asarray(b, float))
    idx = RNG.integers(0, len(a), (n, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return max(0.0, -np.percentile(d, 5)) / b.mean(), d.mean() / b.mean()


def verdict(g):
    return "no useful gain" if g < MARGIN else "gain not excluded"


def forecasting(lines, rows):
    lfp = pd.read_parquet("data/checkups_lfp.parquet")
    nmc, cap = s3.load()
    for curve in ("sqrt", "power"):
        s1.CURVE = curve
        ex = s1.extrapolation(lfp)
        lines.append(f"\nLFP extrapolation, curve '{curve}' (20 cells): mean change vs capacity / best plausible gain")
        for n_obs, d in ex.groupby("n_obs"):
            p = d.pivot(index="cell", columns="method")
            for m in ("modes_ref", "modes_pinn_enc"):
                for metric in ("err_80", "soh_rmse_pp"):
                    g, mean = best_gain(p[metric][m], p[metric]["capacity"])
                    cap_mae = p[metric]["capacity"].abs().mean()
                    rows.append(dict(data="LFP", curve=curve, obs=n_obs, method=m, metric=metric, cap=cap_mae,
                                     mean_change=mean, best_gain=g))
                    lines.append(f"  N_obs {n_obs:4d} {m:15s} {metric:11s} capacity {cap_mae:7.1f}  "
                                 f"change {mean:+7.0%}  best gain {g:5.1%}  -> {verdict(g)}")
        f = s3.forecasts(nmc, cap)
        lines.append(f"\nNMC extrapolation, curve '{curve}' (6-9 cells per experiment): SOH RMSE")
        for (sname, k), d in f.groupby(["set", "k"], sort=False):
            p = d.pivot(index="cid", columns="method", values="soh_rmse_pp")
            for m in ("modes_ref", "modes_pinn_lm"):
                g, mean = best_gain(p[m], p["capacity"])
                rows.append(dict(data=f"NMC {sname}", curve=curve, obs=k, method=m, metric="soh_rmse_pp",
                                 cap=p.capacity.mean(), mean_change=mean, best_gain=g))
                lines.append(f"  {sname} to RPT {k} {m:15s} capacity {p.capacity.mean():5.2f} pp  "
                             f"change {mean:+7.0%}  best gain {g:5.1%}  -> {verdict(g)}")
        # all experiments together, each at its longest history, for a tighter bound
        p = f[f.k == f.set.map(f.groupby("set").k.max())].pivot(index="cid", columns="method", values="soh_rmse_pp")
        for m in ("modes_ref", "modes_pinn_lm"):
            g, mean = best_gain(p[m], p["capacity"])
            rows.append(dict(data="NMC pooled", curve=curve, obs=-1, method=m, metric="soh_rmse_pp",
                             cap=p.capacity.mean(), mean_change=mean, best_gain=g))
            lines.append(f"  pooled ({len(p)})    {m:15s} capacity {p.capacity.mean():5.2f} pp  "
                         f"change {mean:+7.0%}  best gain {g:5.1%}  -> {verdict(g)}")
    s1.CURVE = "sqrt"


def resistance(lines, rows):
    c = s2.table()
    c = c[c.cid != s2.OUTLIER]
    errs = {k: s2.loco_abs_error(c, v).groupby(level=0).mean() for k, v in s2.FEATURES.items()}
    lines.append(f"\nNMC 0.1 s resistance growth ({c.cid.nunique()} cells, without {s2.OUTLIER})")
    for k in ("capacity + est modes", "capacity + ref modes"):
        g, mean = best_gain(errs[k], errs["capacity"].loc[errs[k].index])
        rows.append(dict(data="NMC resistance", curve="linear", obs=0, method=k, metric="abs err",
                         cap=errs["capacity"].mean(), mean_change=mean, best_gain=g))
        lines.append(f"  {k:22s} change {mean:+6.0%}  best gain {g:5.1%}  -> {verdict(g)}")


def mechanism(lines):
    def r2(y, X):
        A = np.column_stack([X, np.ones(len(y))])
        res = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
        return 1 - res.var() / y.var()

    lfp = pd.read_parquet("data/checkups_lfp.parquet")
    nmc = pd.read_parquet("data/capacity_nmc.parquet").dropna(subset=["LLI_ref"])
    lines.append("\nShare of capacity fade (1 - SOH) explained by the reference modes, pooled R^2 [within-cell median]")
    for name, d, modes in (("LFP", lfp, ("LLI", "LAM_NE")), ("NMC", nmc, ("LLI", "LAM_PE", "LAM_NE"))):
        y = 1 - d.SOH.to_numpy()
        key = "uid" if "uid" in d else ["set", "cell"]
        parts = []
        for ms in [(m,) for m in modes] + [modes]:
            X = d[[f"{m}_ref" for m in ms]].to_numpy()
            within = [r2(1 - g.SOH.to_numpy(), g[[f"{m}_ref" for m in ms]].to_numpy())
                      for _, g in d.groupby(key) if len(g) > len(ms) + 1]
            parts.append(f"{'+'.join(ms)} {r2(y, X):.3f} [{np.median(within):.3f}]")
        lines.append(f"  {name}: " + "; ".join(parts))


def main():
    lines, rows = [f"Equivalence margin: a useful gain is >= {MARGIN:.0%} of the capacity error"], []
    forecasting(lines, rows)
    resistance(lines, rows)
    mechanism(lines)
    text = "\n".join(lines)
    print(text)
    open("data/step4_robustness.txt", "w", encoding="utf-8").write(text + "\n")
    pd.DataFrame(rows).to_csv("data/step4_robustness.csv", index=False)


if __name__ == "__main__":
    main()
