"""Step 7: the three remaining reviewer points.

clusters   The 40 NMC cells come from 5 experiments and the 20 LFP cells from 7 protocols; cells within one share
           conditions. Every key comparison is redone with a two-stage bootstrap (experiments/protocols, then
           cells within them) and, where a model is trained across cells, with leave-one-experiment-out.
margins    Verdicts at equivalence margins of 5, 10 and 20 % of the capacity error.
early      Cross-cell early-life prediction on NMC, the setting of Li et al. (2023): features from the check-ups
           up to RPT 4 (about 3 kAh; three estimate check-ups) -> capacity fade at 6.3 kAh (reached by all 40
           cells). Ridge on standardised features; capacity (fade and fade rate) vs + estimated or reference
           modes (state and rate), leave-one-cell-out and leave-one-experiment-out.

    "F:/Research New/battery-pinn-degradation/.venv/Scripts/python.exe" -m src.step7_reviewer
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import step1_gonogo as s1
from src import step2_resistance as s2
from src import step3_nmc_forecast as s3
from src import step5_halfcell as s5

MARGINS = (0.05, 0.10, 0.20)
X_TARGET = 6300.0  # A.h, reached by every NMC cell
K_EARLY = 4  # last observed RPT for early-life features
RNG = np.random.default_rng(7)


def hier_diff(a: pd.Series, b: pd.Series, group: pd.Series, n: int = 4000) -> np.ndarray:
    """Bootstrap of mean|a| - mean|b| resampling groups, then cells within each drawn group."""
    a, b = a.abs(), b.abs()
    groups = {g: np.flatnonzero(group.to_numpy() == g) for g in group.unique()}
    keys = list(groups)
    av, bv, out = a.to_numpy(), b.to_numpy(), np.empty(n)
    for i in range(n):
        idx = np.concatenate([RNG.choice(groups[g], len(groups[g])) for g in RNG.choice(keys, len(keys))])
        out[i] = av[idx].mean() - bv[idx].mean()
    return out


def cell_diff(a: pd.Series, b: pd.Series, n: int = 4000) -> np.ndarray:
    av, bv = a.abs().to_numpy(), b.abs().to_numpy()
    idx = RNG.integers(0, len(av), (n, len(av)))
    return av[idx].mean(1) - bv[idx].mean(1)


def report(lines, label, a, b, group):
    """Mean change, best plausible gain (one-sided 95 %) under cell and cluster bootstrap, margin verdicts."""
    base = b.abs().mean()
    parts = [f"  {label:34s} capacity {base:6.2f}  modes {a.abs().mean():6.2f}  change {(a.abs().mean() / base - 1):+5.0%}"]
    for name, d in (("cells", cell_diff(a, b)), ("clusters", hier_diff(a, b, group))):
        gain = max(0.0, -np.percentile(d, 5)) / base
        lo, hi = np.percentile(d, [2.5, 97.5]) / base
        ok = "/".join("Y" if gain < m else "n" for m in MARGINS)
        parts.append(f"{name}: 95% [{lo:+.0%},{hi:+.0%}] best gain {gain:4.1%} no-useful-gain@5/10/20% {ok}")
    lines.append(" | ".join(parts))


def forecasting(lines):
    lines.append("A. Per-cell forecasting, SOH RMSE [pp]; bootstrap over cells and over experiments/protocols")
    nmc, cap = s3.load()
    for curve in ("sqrt", "power"):
        s1.CURVE = curve
        f = s3.forecasts(nmc, cap)
        f = f[f.k == f.set.map(f.groupby("set").k.max())]
        p = f.pivot(index="cid", columns="method", values="soh_rmse_pp")
        grp = p.index.str[:5].to_series(index=p.index)
        for m in ("modes_ref", "modes_pinn_lm"):
            report(lines, f"NMC 40, {curve}, {m}", p[m], p.capacity, grp)
        lfp = pd.read_parquet("data/checkups_lfp.parquet")
        ex = s1.extrapolation(lfp)
        ex = ex[ex.n_obs == 600].pivot(index="cell", columns="method", values="soh_rmse_pp")
        grp = ex.index.str.extract(r"lfp_(\d)", expand=False).to_series(index=ex.index)
        for m in ("modes_ref", "modes_pinn_enc"):
            report(lines, f"LFP 20, 600 cyc, {curve}, {m}", ex[m], ex.capacity, grp)
    s1.CURVE = "sqrt"
    d = s5.datasets()
    for chem in ("nmc", "lfp"):
        r = s5.yu_table(chem, d[chem])
        grp = (r.index.str[:5] if chem == "nmc" else r.index.str[0]).to_series(index=r.index)
        for m in r.columns.drop("capacity"):
            report(lines, f"{chem.upper()} Yu half-cell, {m}", r[m], r.capacity, grp)


def loeo_abs_error(c: pd.DataFrame, cols: list[str], by: str) -> pd.Series:
    """Leave-one-group-out linear regression of dR; mean absolute error per cell."""
    out = []
    for g in c[by].unique():
        tr, te = c[c[by] != g], c[c[by] == g]
        w = np.linalg.lstsq(np.c_[np.ones(len(tr)), tr[cols]], tr.dR, rcond=None)[0]
        out.append(pd.Series(np.abs(np.c_[np.ones(len(te)), te[cols]] @ w - te.dR.to_numpy()), index=te.cid))
    return pd.concat(out).groupby(level=0).mean()


def resistance(lines):
    lines.append("\nB. NMC 0.1 s resistance growth (39 cells without expt5A), MAE [% of R0]; Holm over the 2 primary tests")
    c = s2.table()
    c = c[c.cid != s2.OUTLIER]
    feats = {"capacity": ["fade"], "+ est LLI, LAM_PE": ["fade", "LLI_est", "LAM_PE_est"],
             "+ ref LLI, LAM_PE": ["fade", "LLI_ref", "LAM_PE_ref"]}
    for split in ("cid", "set"):
        e = {k: loeo_abs_error(c, v, split) for k, v in feats.items()}
        grp = e["capacity"].index.str[:5].to_series(index=e["capacity"].index)
        pvals = {}
        for k in list(feats)[1:]:
            d = hier_diff(e[k], e["capacity"], grp)
            pvals[k] = min(1.0, 2 * min((d >= 0).mean(), (d <= 0).mean()))
            lo, hi = np.percentile(d, [2.5, 97.5])
            lines.append(f"  leave-one-{'cell' if split == 'cid' else 'experiment'}-out {k:18s} "
                         f"{100 * e[k].mean():5.2f} vs {100 * e['capacity'].mean():5.2f}  "
                         f"cluster 95% [{100 * lo:+.2f},{100 * hi:+.2f}]  p={pvals[k]:.3f}")
        ranked = sorted(pvals.items(), key=lambda kv: kv[1])
        holm = {k: min(1.0, p * (len(ranked) - i)) for i, (k, p) in enumerate(ranked)}
        lines.append("    Holm-adjusted p: " + ", ".join(f"{k} {p:.3f}" for k, p in holm.items()))


def early_features() -> pd.DataFrame:
    nmc, cap = s3.load()
    cap = cap.sort_values(["cid", "rpt"])
    rows = []
    for cid, cc in cap.groupby("cid"):
        e = nmc[(nmc.cid == cid) & (nmc.rpt <= K_EARLY)]
        early = cc[cc.rpt <= K_EARLY]
        x = early.throughput_Ah.to_numpy() / 1000
        r = {"cid": cid, "set": cid[:5], "fade": 1 - early.SOH.iloc[-1],
             "fade_rate": np.polyfit(x, 1 - early.SOH, 1)[0],
             "target": 1 - np.interp(X_TARGET, cc.throughput_Ah, cc.SOH)}
        for m in ("LLI", "LAM_PE", "LAM_NE"):
            r[f"{m}_ref"] = early[f"{m}_ref"].iloc[-1]
            r[f"{m}_ref_rate"] = np.polyfit(x, early[f"{m}_ref"], 1)[0]
        xe = e.throughput_Ah.to_numpy() / 1000
        for m in ("LLI", "LAM_PE"):
            r[f"{m}_est"] = e[f"{m}_pinn_lm"].iloc[-1]
            r[f"{m}_est_rate"] = np.polyfit(xe, e[f"{m}_pinn_lm"], 1)[0]
        rows.append(r)
    return pd.DataFrame(rows).set_index("cid")


def ridge_cv(F: pd.DataFrame, cols: list[str], by: str, alpha: float = 1.0) -> pd.Series:
    err = pd.Series(index=F.index, dtype=float)
    keys = F.index if by == "cid" else F[by]
    for g in pd.unique(keys):
        te = (F.index == g) if by == "cid" else (F[by] == g).to_numpy()
        X, y = F[cols].to_numpy(), F.target.to_numpy()
        mu, sd = X[~te].mean(0), X[~te].std(0) + 1e-12
        Xs = (X - mu) / sd
        A = np.c_[Xs[~te], np.ones((~te).sum())]
        w = np.linalg.solve(A.T @ A + alpha * np.diag([1.0] * len(cols) + [0.0]), A.T @ y[~te])
        err[te] = np.c_[Xs[te], np.ones(te.sum())] @ w - y[te]
    return 100 * err


def early(lines):
    F = early_features()
    lines.append(f"\nC. Cross-cell early-life prediction (NMC, {len(F)} cells): fade at {X_TARGET / 1000:.1f} kAh from "
                 f"check-ups up to RPT {K_EARLY}; target fade {100 * F.target.min():.1f}-{100 * F.target.max():.1f} %; "
                 "MAE [pp]")
    sets = {"capacity": ["fade", "fade_rate"],
            "+ est LLI, LAM_PE": ["fade", "fade_rate", "LLI_est", "LLI_est_rate", "LAM_PE_est", "LAM_PE_est_rate"],
            "+ ref LLI, LAM_PE, LAM_NE": ["fade", "fade_rate"] + [f"{m}_ref{s}" for m in ("LLI", "LAM_PE", "LAM_NE")
                                                                 for s in ("", "_rate")]}
    for by in ("cid", "set"):
        e = {k: ridge_cv(F, v, by) for k, v in sets.items()}
        for k in list(sets)[1:]:
            report(lines, f"{'LOCO' if by == 'cid' else 'LOEO'} {k}", e[k], e["capacity"], F.set)


def main():
    lines = []
    forecasting(lines)
    resistance(lines)
    early(lines)
    text = "\n".join(lines)
    print(text)
    open("data/step7_reviewer.txt", "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
