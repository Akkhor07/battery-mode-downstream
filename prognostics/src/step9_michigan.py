"""Step 9: an independent dataset with knees - University of Michigan NMC/graphite pouch cells.

Mohtat, Siegel, Stefanopoulou, Lee (2021), Deep Blue doi:10.7302/7tw1-kc35 (CC BY 4.0): 21 cells, 7 cycling
conditions x 3 temperatures, full C/20 charge and discharge at every check-up (about every 5 % capacity loss),
aged to about 70 % (several cells with a late knee). Only *reference* modes can be tested here: the JES estimator
was trained for the LG M50T and is not applicable to this cell.

Reference modes: quasi-OCV fit (src.step8_refit.fit_curve) of each C/20 curve with the OCPs that PyBaMM's
Mohtat2020 set fits to this very cell; same quality rule as step 8 (<= 20 mV, R > 0, no capacity at a bound);
a cell is used only if every check-up passes. Two independent versions: C/20 charge and C/20 discharge.

Pre-specified test (written before any forecast was run):
    primary    observe check-ups 0-5 (every cell has >= 7), forecast all later check-ups (SOH RMSE)
    secondary  observe check-ups 0-3
    methods    capacity extrapolation vs (a) linear map fade ~ LLI + LAM_PE learned on the other cells (step 3) and
               (b) Yu et al.: extrapolate Q_n, Q_p, Q_Li separately, capacity from the same OCV model, anchored
               to the last observed SOH
    statistics cell and cluster bootstrap (clusters = 7 cycling conditions), margins 5/10/20 % (step 7)
    knee cells (descriptive) future fade rate >= 1.5 x observed fade rate, from measured capacity only

    "F:/Research New/battery-pinn-degradation/.venv/Scripts/python.exe" -m src.step9_michigan [--part fit|eval]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src import step1_gonogo as s1
from src import step7_reviewer as s7
from src.step8_refit import fit_curve

ROOT = Path("data/michigan")
V_MIN, V_MAX = 3.0, 4.2
QN0, QP0 = 5.97, 5.80  # Mohtat2020 electrode capacities [A.h]
K_PRIMARY, K_SECONDARY = 5, 3


def u_n(sto):
    """Graphite OCP, PyBaMM Mohtat2020 (graphite_ocp_PeymanMPM)."""
    return (0.063 + 0.8 * np.exp(-75 * (sto + 0.001)) - 0.0120 * np.tanh((sto - 0.127) / 0.016)
            - 0.0118 * np.tanh((sto - 0.155) / 0.016) - 0.0035 * np.tanh((sto - 0.220) / 0.020)
            - 0.0095 * np.tanh((sto - 0.190) / 0.013) - 0.0145 * np.tanh((sto - 0.490) / 0.020)
            - 0.0800 * np.tanh((sto - 1.030) / 0.055))


def u_p(sto):
    """NMC OCP, PyBaMM Mohtat2020 (NMC_ocp_PeymanMPM)."""
    return (4.3452 - 1.6518 * sto + 1.6225 * sto ** 2 - 2.0843 * sto ** 3 + 3.5146 * sto ** 4
            - 2.2166 * sto ** 5 - 0.5623e-4 * np.exp(109.451 * sto - 100.006))


def condition(cell: int) -> int:
    return (cell - 1) // 3  # Table 1 of the dataset READ_ME: three temperatures per cycling condition


def segments(cell: int):
    """C/20 charge and discharge (constant-current parts) of every check-up: (rpt, throughput, kind, q, v, I)."""
    d = pd.read_csv(ROOT / "data" / f"{cell:02d}" / "OCV_wExpansion.csv")
    d.columns = ["t", "I", "V", "E", "T", "Q", "Cap", "cyc", "tp"]
    out = []
    for rpt, (cyc, c) in enumerate(d.groupby("cyc", sort=True)):
        for kind, sel in (("charge", (c.I > 200) & (c.I < 300) & (c.V < 4.199)),
                          ("discharge", (c.I < -200) & (c.I > -300))):
            s = c[sel]
            if len(s) < 100:
                continue
            # one contiguous block (the C/20 step)
            brk = np.flatnonzero(np.diff(s.index.to_numpy()) > 5)
            s = s.iloc[brk[-1] + 1:] if kind == "discharge" and len(brk) else s.iloc[:(brk[0] + 1) if len(brk) else None]
            dt = np.diff(s.t.to_numpy(), prepend=s.t.iloc[0])
            q = np.cumsum(np.abs(s.I.to_numpy()) * dt) / 3.6e6
            g = np.linspace(0, q[-1], 400)
            out.append({"cell": cell, "rpt": rpt, "cyc": cyc, "tp": c.tp.max(), "kind": kind, "q": g,
                        "v": np.interp(g, q, s.V.to_numpy()), "I": np.sign(s.I.median()) * np.abs(s.I).median() / 1000})
    return out


def part_fit():
    rows = []
    for cell in range(1, 22):
        for s in segments(cell):
            f = fit_curve(s["q"], s["v"], s["I"], u_n, QN0, up=u_p, qp_starts=(5.3, 6.3),
                          ends=((0.002, 0.9), (0.02, 0.85)))
            rows.append({"cell": cell, "rpt": s["rpt"], "cyc": s["cyc"], "tp": s["tp"], "kind": s["kind"],
                         "Q": s["q"][-1], **f})
        print("cell", cell, flush=True)
    r = pd.DataFrame(rows).sort_values(["kind", "cell", "rpt"])
    r["ok"] = (r.rmse_mV <= 20) & (r.R > 0) & r.Qn.between(2.05, 11.95) & r.Qp.between(2.05, 11.95)
    g = r.groupby(["kind", "cell"])
    for m, col in (("LLI", "QLi"), ("LAM_PE", "Qp"), ("LAM_NE", "Qn")):
        r[m] = 1 - r[col] / g[col].transform("first")
    r.to_parquet(ROOT / "modes.parquet")
    return r


def ocv_capacity(qn, qp, qli):
    """Capacity between V_MIN and V_MAX at open circuit for electrode capacities and lithium inventory [A.h]."""
    def ocv(x):
        return u_p(np.clip((qli - x * qn) / qp, 0, 1)) - u_n(np.clip(x, 0, 1))

    lo, hi = max(0.0, (qli - qp) / qn) + 1e-6, min(1.0, qli / qn) - 1e-6
    try:
        x100 = brentq(lambda x: ocv(x) - V_MAX, lo, hi)
        x0 = brentq(lambda x: ocv(x) - V_MIN, lo, hi)
    except ValueError:
        return np.nan
    return (x100 - x0) * qn


def soh_table(kind: str) -> pd.DataFrame:
    """Per check-up: throughput, SOH from the C/20 discharge capacity, modes of one fit version (valid cells only)."""
    r = pd.read_parquet(ROOT / "modes.parquet")
    cap = r[r.kind == "discharge"][["cell", "rpt", "tp", "Q"]].sort_values(["cell", "rpt"])
    m = r[r.kind == kind]
    good = m.groupby("cell").filter(lambda c: c.ok.all()).cell.unique()
    t = cap.merge(m[["cell", "rpt", "Qn", "Qp", "QLi"]], on=["cell", "rpt"])
    t = t[t.cell.isin(good)].sort_values(["cell", "rpt"])
    # the first check-up of the 45 degC cells (3, 6, ..., 21) has no C/20 discharge (the file stops after the
    # charge): every cell is referenced to, and counted from, its first check-up that has both curves
    t["rpt"] = t.groupby("cell").cumcount()
    g = t.groupby("cell")
    t["SOH"] = t.Q / g.Q.transform("first")
    for mode, col in (("LLI", "QLi"), ("LAM_PE", "Qp"), ("LAM_NE", "Qn")):
        t[mode] = 1 - t[col] / g[col].transform("first")
    # ageing axis from the same first check-up (fix 2026-10-28: the 45 degC cells start at tp > 0, and the fitted
    # curves pass through the origin)
    t["x"] = (t.tp - g.tp.transform("first")) / 1000
    return t


def forecasts(t: pd.DataFrame, k: int, cap_fn=None) -> pd.DataFrame:
    """Per cell, observing check-ups 0..k: SOH RMSE [pp] of capacity, the mode map and the Yu forecast.
    ``cap_fn(Q_n, Q_p, Q_Li)`` is the OCV capacity model (default: this cell's)."""
    ocv_capacity_ = cap_fn or ocv_capacity
    rows = []
    for cell, c in t.groupby("cell"):
        obs, fut = c[c.rpt <= k], c[c.rpt > k]
        if len(obs) <= k or fut.empty:
            continue
        xo, xf, y = obs.x.to_numpy(), fut.x.to_numpy(), fut.SOH.to_numpy()
        p = {"capacity": 1 - s1.fit_curve(xo, 1 - obs.SOH.to_numpy())(xf)}
        other = t[t.cell != cell]
        X = np.c_[other.LLI, other.LAM_PE, np.ones(len(other))]
        w = np.linalg.lstsq(X, 1 - other.SOH.to_numpy(), rcond=None)[0]
        mc = {m: s1.fit_curve(xo, obs[m].to_numpy()) for m in ("LLI", "LAM_PE")}
        off = np.mean((1 - obs.SOH.to_numpy()) - np.c_[mc["LLI"](xo), mc["LAM_PE"](xo)] @ w[:2])
        p["modes_map"] = 1 - (np.c_[mc["LLI"](xf), mc["LAM_PE"](xf)] @ w[:2] + off)
        # Yu et al.: extrapolate the relative losses of Q_n, Q_p, Q_Li, capacity from the OCV model
        fits = {col: s1.fit_curve(xo, 1 - obs[col].to_numpy() / obs[col].iloc[0]) for col in ("Qn", "Qp", "QLi")}
        q0 = {col: obs[col].iloc[0] for col in fits}

        def cap_at(x):
            return np.array([ocv_capacity_(*(q0[col] * (1 - fits[col](np.array([xx]))[0]) for col in ("Qn", "Qp", "QLi")))
                             for xx in np.atleast_1d(x)])

        last = cap_at(xo[-1:])[0]
        p["modes_yu"] = cap_at(xf) / last * obs.SOH.iloc[-1]
        early = np.polyfit(xo, obs.SOH, 1)[0]
        late = np.polyfit(np.r_[xo[-1], xf], np.r_[obs.SOH.iloc[-1], y], 1)[0]
        for name, v in p.items():
            ok = np.isfinite(v)
            rows.append({"cell": cell, "k": k, "method": name, "knee": late <= 1.5 * early,  # rates are negative
                         "rmse": 100 * np.sqrt(np.mean((v[ok] - y[ok]) ** 2)) if ok.any() else np.nan})
    return pd.DataFrame(rows)


def part_eval():
    r = pd.read_parquet(ROOT / "modes.parquet")
    lines = ["Michigan pouch cells: quasi-OCV fits with the Mohtat2020 OCPs"]
    for kind, rv in r.groupby("kind"):
        good = rv.groupby("cell").filter(lambda c: c.ok.all()).cell.nunique()
        lines.append(f"  {kind}: {len(rv)} check-ups, median fit {rv.rmse_mV.median():.1f} mV, failing {(~rv.ok).sum()}, "
                     f"cells with every check-up valid {good}/21")
    for kind in ("discharge", "charge"):
        t = soh_table(kind)
        lines.append(f"\n=== reference modes from the C/20 {kind} ({t.cell.nunique()} cells) ===")
        lines.append(f"  within-cell R^2 of fade on LLI: "
                     f"{t.groupby('cell').apply(lambda c: np.corrcoef(c.LLI, 1 - c.SOH)[0, 1] ** 2).median():.3f} (median)")
        for k, tag in ((K_PRIMARY, "primary"), (K_SECONDARY, "secondary")):
            for curve in ("sqrt", "power"):
                s1.CURVE = curve
                f = forecasts(t, k)
                p = f.pivot(index="cell", columns="method", values="rmse").dropna()
                knee = f.groupby("cell").knee.first().loc[p.index]
                grp = p.index.map(condition).to_series(index=p.index)
                for m in ("modes_map", "modes_yu"):
                    s7.report(lines, f"{tag} k={k} {curve} {m} (n={len(p)})", p[m], p.capacity, grp)
                lines.append(f"    knee cells ({knee.sum()}): " + ", ".join(f"{c} {p[knee][c].mean():.2f}" for c in p.columns)
                             + f" | no knee ({(~knee).sum()}): " + ", ".join(f"{c} {p[~knee][c].mean():.2f}" for c in p.columns))
            s1.CURVE = "sqrt"
    text = "\n".join(lines)
    Path("data/step9_michigan.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("fit", "eval", "test"), default="fit")
    a = ap.parse_args()
    if a.part == "test":
        for s in segments(1)[:4] + segments(21)[-2:]:
            f = fit_curve(s["q"], s["v"], s["I"], u_n, QN0, up=u_p, qp_starts=(5.3, 6.3), ends=((0.002, 0.9), (0.02, 0.85)))
            print(s["cell"], s["rpt"], s["kind"], round(s["I"], 3), round(s["q"][-1], 3), {k: round(float(x), 3) for k, x in f.items()})
    elif a.part == "fit":
        part_fit()
    else:
        part_eval()
