"""Step 8: are the conclusions robust to how the NMC reference modes are defined?

The published Kirkaldy modes are OCV fits to the C/10 *discharge*, which the estimator also reads. Here the
modes are refitted independently, several ways, and the forecasting, resistance and early-life comparisons of
step 7 are repeated with each version:

    chg_gr      C/10 *charge* (never read by the estimator), graphite negative electrode
    (a 10-90 % charge-only fit was tried and dropped: without the curve ends Q_n is not identifiable)
    dis_gr      C/10 discharge, graphite negative electrode

A graphite + silicon blend (as in Stage 1 and the published fit) was tried first and rejected: with the stock
NMC811 OCP the quasi-OCV fit cannot reproduce the curves (25-35 mV at the global optimum, found by differential
evolution, against 8-16 mV for graphite only). Kirkaldy et al. used measured half-cell OCVs, and the JES decoder
needed full SPMe dynamics. LAM_NE here is therefore a graphite-equivalent loss, a different definition from the
published one, which is the point of the check.

Fit: quasi-OCV model V(Q) = U_p(y) - U_n(x) +/- I R with x, y linear in the charge passed; free parameters are the
starting stoichiometries, the electrode capacities Q_n, Q_p and a lumped resistance R (which also absorbs
hysteresis). Lithium inventory Q_Li = x Q_n + y Q_p. Modes are losses relative to the cell's first check-up.
OCPs: Chen2020 / ORegan2022 graphite and NMC811 fits, silicon from PyBaMM Chen2020_composite (as in Stage 1).

    "F:/Research New/battery-pinn-degradation/.venv/Scripts/python.exe" -m src.step8_refit [--part fit|eval]
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from src.stage1_export import KIRKALDY_ALL, STAGE1

OUT = Path("data/refit")
SI_POLY = (-51.02, 161.3, -205.7, 140.2, -58.76, 16.87, -3.792, 0.9937)  # silicon delithiation OCP, x^7 .. x^0
VARIANTS = {"chg_gr": ("charge", 1.0), "dis_gr": ("discharge", 1.0)}
MODES = ("LLI", "LAM_PE", "LAM_NE")


def u_gr(x):
    return (1.9793 * np.exp(-39.3631 * x) + 0.2482 - 0.0909 * np.tanh(29.8538 * (x - 0.1234))
            - 0.04478 * np.tanh(14.9159 * (x - 0.2769)) - 0.0205 * np.tanh(30.4444 * (x - 0.6103)))


def u_p(y):
    return (-0.8090 * y + 4.4875 - 0.0428 * np.tanh(18.5138 * (y - 0.5542))
            - 17.7326 * np.tanh(15.7890 * (y - 0.3117)) + 17.5842 * np.tanh(15.9308 * (y - 0.3120)))


def blended_un(gr_frac: float):
    """Negative-electrode OCP vs total stoichiometry for a graphite + silicon blend at a common potential."""
    s = np.linspace(0.0, 1.0, 20001)
    if gr_frac >= 1.0:
        return u_gr
    ug = np.minimum.accumulate(u_gr(s))
    us = np.minimum.accumulate(np.polyval(SI_POLY, s))
    u = np.linspace(ug[-1], ug[0], 4001)
    x = gr_frac * np.interp(u, ug[::-1], s[::-1]) + (1 - gr_frac) * np.interp(u, us[::-1], s[::-1], left=1.0, right=0.0)
    o = np.argsort(x)
    x, u = x[o], u[o]
    keep = np.r_[True, np.diff(x) > 1e-12]
    x, u = x[keep], u[keep]
    return lambda z: np.interp(z, x, u)


def curve(f: str, n: int = 400):
    """Constant-current part of a C/10 curve: charge passed [A.h] from its start, voltage, current [A]."""
    d = pd.read_csv(f)
    i = d["Current (mA)"].to_numpy() / 1000
    sgn = np.sign(np.median(i[np.abs(i) > 0.05]))
    cc = np.abs(i) > 0.9 * np.median(np.abs(i[np.abs(i) > 0.05]))
    d = d[cc]
    q = np.abs(d["Charge (mA.h)"].to_numpy() / 1000)
    q = q - q[0]
    v = d["Voltage (V)"].to_numpy()
    keep = np.r_[True, np.diff(q) > 1e-5]
    q, v = q[keep], v[keep]
    g = np.linspace(0, q[-1], n)
    return g, np.interp(g, q, v), sgn * np.median(np.abs(i[cc]))


def fit_curve(q, v, current, un, qn_fresh: float = 5.2, mid: bool = False, up=u_p, qp_starts=(6.5, 7.5),
              ends=((0.02, 0.95), (0.05, 0.9))):
    """Least-squares quasi-OCV fit; returns Q_n, Q_p, Q_Li, R and the fit RMSE [mV].

    Stoichiometries are kept inside [0, 1] over the whole curve by a penalty (otherwise the OCP fits are
    extrapolated and degenerate solutions with tiny electrode capacities appear). Starts on a small grid around
    the ORegan2022 fresh cell (Q_n 5.2 A.h graphite, scaled for silicon; Q_p 7.2 A.h).
    """
    chg = current > 0
    sgn = 1.0 if chg else -1.0  # charging: lithium moves from the positive to the negative electrode

    def resid(p):
        x0, y0, qn, qp, r = p
        x, y = x0 + sgn * q / qn, y0 - sgn * q / qp
        pen = 10 * np.r_[np.clip(x - 1, 0, None).max(), np.clip(-x, 0, None).max(),
                         np.clip(y - 1, 0, None).max(), np.clip(-y, 0, None).max()]
        e = up(np.clip(y, 0, 1)) - un(np.clip(x, 0, 1)) + current * r - v
        return np.r_[e * w, pen]

    w = ((q > 0.1 * q[-1]) & (q < 0.9 * q[-1])).astype(float) if mid else np.ones_like(q)
    best = None
    qe = q[-1]
    for qn in (0.9 * qn_fresh, 1.1 * qn_fresh):
        for qp in qp_starts:
            for xe, ye in ends:  # stoichiometries at the discharged end
                x0, y0 = (xe, ye) if chg else (xe + qe / qn, ye - qe / qp)
                p0 = np.clip([x0, y0, qn, qp, 0.02], [0, 0, 2, 2, -0.1], [1, 1, 12, 12, 0.3])
                r = least_squares(resid, p0, bounds=([0, 0, 2, 2, -0.1], [1, 1, 12, 12, 0.3]))
                if best is None or r.cost < best.cost:
                    best = r
    x0, y0, qn, qp, r = best.x
    return {"Qn": qn, "Qp": qp, "QLi": x0 * qn + y0 * qp, "R": r,
            "rmse_mV": 1000 * np.sqrt(np.sum(best.fun[:len(q)] ** 2) / w.sum())}


def part_fit():
    rows = []
    uns = {v: blended_un(g) for v, (_, g) in VARIANTS.items()}
    for sname, folder in KIRKALDY_ALL.items():
        root = STAGE1 / "data" / "kirkaldy" / folder / "Processed Timeseries Data" / "0.1C Voltage Curves"
        # Expt 2-4 publish only the C/10 discharge, so the charge refit covers Expt 1 and 5
        for f in sorted(glob.glob(str(root / "*" / "*0.1C discharge data.csv"))):
            m = re.search(r"cell (\w) - RPT(\d+)", Path(f).name)
            cell, rpt = m[1], int(m[2])
            for var, (kind, _) in VARIANTS.items():
                ff = f if kind == "discharge" else f.replace("discharge data", "charge data")
                if not Path(ff).exists():
                    continue
                try:
                    q, v, cur = curve(ff)
                    if q[-1] < 2.0:  # truncated curve
                        continue
                    rows.append({"set": sname, "cell": cell, "rpt": rpt, "variant": var,
                                 **fit_curve(q, v, cur, uns[var], 5.2 / VARIANTS[var][1], var.endswith("_mid"))})
                except Exception as e:  # noqa: BLE001  (malformed file: skip the check-up)
                    print("skip", sname, cell, rpt, var, e)
        print(sname, "done", flush=True)
    r = pd.DataFrame(rows).sort_values(["variant", "set", "cell", "rpt"])
    # quality rule, fixed before any downstream result was looked at: fit within 20 mV, positive resistance,
    # no electrode capacity at a bound. Expt 1 (silicon degradation) fails it often: a graphite-only model
    # cannot describe those cells.
    r["ok"] = (r.rmse_mV <= 20) & (r.R > 0) & r.Qn.between(2.05, 11.95) & r.Qp.between(2.05, 11.95)
    g = r.groupby(["variant", "set", "cell"])
    for m, col in (("LLI", "QLi"), ("LAM_PE", "Qp"), ("LAM_NE", "Qn")):
        r[m] = 1 - r[col] / g[col].transform("first")
    OUT.mkdir(parents=True, exist_ok=True)
    r.to_parquet(OUT / "refit_modes.parquet")
    return r


def valid(r: pd.DataFrame, var: str) -> pd.DataFrame:
    """Fits of one variant for the cells whose every fitted check-up passes the quality rule (RPT 0 included)."""
    rv = r[r.variant == var]
    good = rv.groupby(["set", "cell"]).filter(lambda c: c.ok.all() and (c.rpt == 0).any())
    return good


def agreement(r: pd.DataFrame, lines: list[str]):
    cap = pd.read_parquet("data/capacity_nmc.parquet")
    lines.append("Quality rule (fit <= 20 mV, R > 0, no capacity at a bound); a cell is used only if all its check-ups pass")
    for var, rv in r.groupby("variant"):
        n = rv.groupby("set").apply(lambda s: f"{s.cell.nunique() - valid(r[r.variant == var], var).query('set == @s.name').cell.nunique()}"
                                              f"/{s.cell.nunique()}")
        lines.append(f"  {var}: check-ups failing {(~rv.ok).sum()}/{len(rv)}; cells excluded per set " + ", ".join(f"{k} {v}" for k, v in n.items()))
    lines.append("Refitted reference modes vs published Kirkaldy modes (valid cells, check-ups after the first)")
    for var in r.variant.unique():
        rv = valid(r, var)
        x = rv.merge(cap, on=["set", "cell", "rpt"], suffixes=("", "_pub"))
        x = x[x.rpt > 0]
        parts = [f"  {var:9s} {rv.groupby(['set', 'cell']).ngroups} cells n={len(x):3d} fit {rv.rmse_mV.median():4.1f} mV"]
        for m in MODES:
            e = 100 * (x[m] - x[f"{m}_ref"])
            parts.append(f"{m} corr {np.corrcoef(x[m], x[f'{m}_ref'])[0, 1]:.2f} rmse {np.sqrt((e ** 2).mean()):.1f} pp")
        parts.append(f"fade~LLI R2 {np.corrcoef(x.LLI, 1 - x.SOH)[0, 1] ** 2:.3f}")
        lines.append(" | ".join(parts))


def variant_tables(r: pd.DataFrame, var: str, published: bool = False):
    """checkups_nmc / capacity_nmc restricted to the valid cells of one variant and to check-ups with a refit,
    with the *_ref columns replaced by the refit (or, with ``published``, the published modes on the same rows,
    so that every comparison is made on identical cells and check-ups)."""
    rv = valid(r, var)[["set", "cell", "rpt", *MODES]].rename(columns={m: f"{m}_ref" for m in MODES})
    out = {}
    for name in ("checkups_nmc", "capacity_nmc"):
        t = pd.read_parquet(f"data/{name}.parquet")
        if published:
            out[name] = t.merge(rv[["set", "cell", "rpt"]], on=["set", "cell", "rpt"])
        else:
            out[name] = t.drop(columns=[f"{m}_ref" for m in MODES]).merge(rv, on=["set", "cell", "rpt"])
    return out


def part_eval():
    from src import step3_nmc_forecast as s3
    from src import step5_halfcell as s5
    from src import step7_reviewer as s7
    from src import step1_gonogo as s1

    r = pd.read_parquet(OUT / "refit_modes.parquet")
    lines = []
    agreement(r, lines)
    real_read = pd.read_parquet
    runs = [("published, all cells", None)]
    for v in VARIANTS:
        runs += [(f"published, {v} subset", variant_tables(r, v, published=True)), (v, variant_tables(r, v))]
    for var, tabs in runs:

        def read(path, *a, **k):
            name = Path(str(path)).stem
            return tabs[name].copy() if tabs and name in tabs else real_read(path, *a, **k)

        pd.read_parquet = read
        try:
            lines.append(f"\n=== reference = {var} ===")
            nmc, cap = s3.load()
            for curve in ("sqrt", "power"):
                s1.CURVE = curve
                f = s3.forecasts(nmc, cap)  # check-ups without a refit are skipped inside (NaN modes)
                f = f[f.k == f.set.map(f.groupby("set").k.max())]
                p = f.pivot(index="cid", columns="method", values="soh_rmse_pp").dropna(subset=["modes_ref"])
                s7.report(lines, f"forecast {curve}, modes_ref", p.modes_ref, p.capacity, p.index.str[:5].to_series(index=p.index))
            s1.CURVE = "sqrt"
            y = s5.yu_table("nmc", s5.datasets()["nmc"])
            s7.report(lines, "Yu half-cell, modes_ref", y.modes_ref, y.capacity, y.index.str[:5].to_series(index=y.index))
            s7.resistance(lines)
            s7.early(lines)
        finally:
            pd.read_parquet = real_read
        print("\n".join(lines[-12:]), flush=True)
    text = "\n".join(lines)
    (Path("data") / "step8_refit.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("fit", "eval", "test"), default="fit")
    a = ap.parse_args()
    if a.part == "test":
        root = STAGE1 / "data" / "kirkaldy" / KIRKALDY_ALL["expt5"] / "Processed Timeseries Data" / "0.1C Voltage Curves" / "Cell D"
        for rpt in (0, 8, 15):
            for kind, var in (("charge", "chg_gr"), ("discharge", "dis_gr")):
                q, v, cur = curve(str(root / f"Expt 5 - cell D - RPT{rpt} - 0.1C {kind} data.csv"))
                print(rpt, var, round(cur, 3), {k: round(x, 4) for k, x in fit_curve(q, v, cur, blended_un(VARIANTS[var][1]), 5.2, var.endswith("_mid")).items()})
    elif a.part == "fit":
        part_fit()
    else:
        part_eval()
