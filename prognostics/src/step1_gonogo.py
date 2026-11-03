"""Step 1 (go/no-go): do label-free mode estimates from partial windows improve LFP capacity forecasts over
capacity alone? Simple, deliberately untuned forecasters on the 20 Wheeler cells.

A. Per-cell extrapolation from the check-ups up to N_obs cycles, all with the same two-parameter curve
   y(N) = c1 sqrt(N) + c2 N (linear least squares; works for noisy estimates that start near or below zero):
     capacity      fit 1 - SOH directly
     modes_<est>   fit LLI and LAM_NE, map to 1 - SOH with a linear map learned on the other cells
   (A hybrid that regresses fade on the capacity basis plus the mode curves is identical to 'capacity' by
   the normal equations, because the mode curves lie in the same basis; it is therefore not reported.)
   Scores: SOH RMSE over all later check-ups, error of the cycle at which SOH crosses 0.8 and 0.7.
B. Leave-one-cell-out ridge regression of log cycles-to-0.8 / 0.7 from features at N_obs
   (capacity: SOH and fade slope; + modes: estimated LLI, LAM_NE and their slopes).

Estimators: pinn_enc (JES decision rule for LFP), cnn_dr (data-driven comparison), ref (whole-curve reference
fit, an offline upper bound). Paired cluster bootstrap over cells for the differences to the capacity baseline.

    "F:/Research New/battery-pinn-degradation/.venv/Scripts/python.exe" -m src.step1_gonogo
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_OBS = (400, 600, 800)
THRESH = (0.8, 0.7)
EST = ("pinn_enc", "cnn_dr", "ref")
MODES = ("LLI", "LAM_NE")
N_MAX = 20000  # forecasts that never cross a threshold are capped here
RNG = np.random.default_rng(0)


CURVE = "sqrt"  # "sqrt": c1 sqrt(n) + c2 n; "power": a n^b with 0.2 <= b <= 3 (can bend either way)


def basis(n):
    n = np.asarray(n, float)
    return np.stack([np.sqrt(n), n], -1)


def fit_curve(n, y):
    """Fitted curve as a callable n -> y, through the origin, of the family set by CURVE."""
    n, y = np.asarray(n, float), np.asarray(y, float)
    if CURVE == "power":
        from scipy.optimize import curve_fit

        f = lambda x, a, b: a * np.power(np.asarray(x, float), b)  # noqa: E731
        try:
            (a, b), _ = curve_fit(f, n / n.max(), y, p0=(y[-1] if y[-1] else 1e-3, 0.7), bounds=([-np.inf, 0.2], [np.inf, 3.0]),
                                  maxfev=20000)
            return lambda x, a=a, b=b, s=n.max(): f(np.asarray(x, float) / s, a, b)
        except RuntimeError:
            pass  # fall back to the default family
    c, *_ = np.linalg.lstsq(basis(n), y, rcond=None)
    return lambda x, c=c: basis(x) @ c


def crossing(n, soh, thr):
    """First cycle at which SOH falls below thr, linearly interpolated; nan if never."""
    below = np.nonzero(soh < thr)[0]
    if not len(below):
        return np.nan
    i = below[0]
    if i == 0:
        return n[0]
    return n[i - 1] + (soh[i - 1] - thr) * (n[i] - n[i - 1]) / (soh[i - 1] - soh[i])


def forecast_crossing(fade_fn, thr):
    grid = np.arange(0, N_MAX + 1, 10.0)
    return crossing(grid, 1 - fade_fn(grid), thr) if np.isfinite(crossing(grid, 1 - fade_fn(grid), thr)) else N_MAX


def col(est, m):
    return f"{m}_ref" if est == "ref" else f"{m}_{est}"


def extrapolation(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cells = t.uid.unique()
    for n_obs in N_OBS:
        for cell in cells:
            c = t[t.uid == cell].sort_values("cycles")
            obs, fut = c[c.cycles <= n_obs], c[c.cycles > n_obs]
            truth = {thr: crossing(c.cycles.values, c.SOH.values, thr) for thr in THRESH}
            preds = {}
            cap = fit_curve(obs.cycles, 1 - obs.SOH)
            preds["capacity"] = cap
            other = t[t.uid != cell]
            for est in EST:
                # map modes -> capacity fade, learned on all check-ups of the other cells
                X = np.column_stack([other[col(est, m)] for m in MODES] + [np.ones(len(other))])
                w, *_ = np.linalg.lstsq(X, 1 - other.SOH.values, rcond=None)
                mc = {m: fit_curve(obs.cycles, obs[col(est, m)]) for m in MODES}
                # anchor at the observed check-ups: the map's offset is replaced by the cell's own early residual
                fade_obs = np.column_stack([mc[m](obs.cycles) for m in MODES]) @ w[:2]
                off = np.mean((1 - obs.SOH.values) - fade_obs)
                preds[f"modes_{est}"] = lambda n, mc=mc, w=w, off=off: (
                    np.column_stack([mc[m](n) for m in MODES]) @ w[:2] + off)
            for name, f in preds.items():
                r = {"n_obs": n_obs, "cell": cell, "method": name, "n_points": len(obs),
                     "soh_rmse_pp": 100 * np.sqrt(np.mean((1 - f(fut.cycles.values) - fut.SOH.values) ** 2))}
                for thr in THRESH:
                    r[f"err_{int(thr * 100)}"] = forecast_crossing(f, thr) - truth[thr]
                rows.append(r)
    return pd.DataFrame(rows)


def early_regression(t: pd.DataFrame, alpha: float = 1.0) -> pd.DataFrame:
    rows = []
    for n_obs in N_OBS:
        feats = []
        for cell, c in t.groupby("uid"):
            c = c.sort_values("cycles")
            obs = c[c.cycles <= n_obs]
            f = {"cell": cell, "soh": obs.SOH.iloc[-1], "soh_slope": np.polyfit(obs.cycles, obs.SOH, 1)[0] * 1000}
            for est in EST:
                for m in MODES:
                    f[f"{est}_{m}"] = obs[col(est, m)].iloc[-1]
                    f[f"{est}_{m}_slope"] = np.polyfit(obs.cycles, obs[col(est, m)], 1)[0] * 1000
            for thr in THRESH:
                f[f"y_{int(thr * 100)}"] = np.log(crossing(c.cycles.values, c.SOH.values, thr))
            feats.append(f)
        F = pd.DataFrame(feats)
        sets = {"capacity": ["soh", "soh_slope"]}
        for est in EST:
            sets[f"cap+modes_{est}"] = sets["capacity"] + [f"{est}_{m}{s}" for m in MODES for s in ("", "_slope")]
        for name, fs in sets.items():
            for thr in THRESH:
                y = F[f"y_{int(thr * 100)}"].values
                for i in range(len(F)):
                    tr = np.arange(len(F)) != i
                    X = F[fs].values
                    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
                    Xs = (X - mu) / sd
                    A = np.column_stack([Xs[tr], np.ones(tr.sum())])
                    w = np.linalg.solve(A.T @ A + alpha * np.diag([1] * len(fs) + [0]), A.T @ y[tr])
                    pred = np.r_[Xs[i], 1] @ w
                    rows.append({"n_obs": n_obs, "cell": F.cell[i], "method": name, "thr": thr,
                                 "err": np.exp(pred) - np.exp(y[i])})
    return pd.DataFrame(rows)


def paired_boot(a, b, n=2000):
    """95% interval of mean(|a|) - mean(|b|), resampling cells."""
    a, b = np.abs(np.asarray(a)), np.abs(np.asarray(b))
    idx = RNG.integers(0, len(a), (n, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return np.percentile(d, [2.5, 97.5])


def main():
    t = pd.read_parquet("data/checkups_lfp.parquet")
    ex = extrapolation(t)
    ex.to_csv("data/step1_extrapolation.csv", index=False)
    print("A. per-cell extrapolation (20 LFP cells): mean SOH RMSE [pp] and MAE of crossing cycle")
    for n_obs in N_OBS:
        d = ex[ex.n_obs == n_obs]
        base = d[d.method == "capacity"].set_index("cell")
        print(f"\n  N_obs = {n_obs} cycles ({d.n_points.min()}-{d.n_points.max()} check-ups)")
        for m, g in d.groupby("method", sort=False):
            g = g.set_index("cell")
            b = base.loc[g.index]  # hybrid needs > 4 check-ups, so it covers fewer cells at small N_obs
            line = f"    {m:16s} n={len(g):2d} SOH RMSE {g.soh_rmse_pp.mean():5.1f} (cap {b.soh_rmse_pp.mean():4.1f})"
            for thr in (80, 70):
                e = g[f"err_{thr}"]
                line += f" | {thr}%: MAE {e.abs().mean():5.0f} cyc"
                if m != "capacity":
                    lo, hi = paired_boot(e, b[f"err_{thr}"])
                    line += f" (vs cap {lo:+.0f}..{hi:+.0f})"
            print(line)

    rg = early_regression(t)
    rg.to_csv("data/step1_regression.csv", index=False)
    print("\nB. leave-one-cell-out ridge, MAE of cycles to threshold")
    for n_obs in N_OBS:
        print(f"\n  N_obs = {n_obs} cycles")
        for thr in THRESH:
            d = rg[(rg.n_obs == n_obs) & (rg.thr == thr)]
            base = d[d.method == "capacity"].set_index("cell").err
            line = f"    {int(thr * 100)}%:"
            for m, g in d.groupby("method", sort=False):
                e = g.set_index("cell").err.loc[base.index]
                line += f"  {m} {e.abs().mean():.0f}"
                if m != "capacity":
                    lo, hi = paired_boot(e, base)
                    line += f" [{lo:+.0f},{hi:+.0f}]"
            print(line)


if __name__ == "__main__":
    main()
