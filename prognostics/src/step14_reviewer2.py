"""Step 14: checks requested by the simulated journal review of the paper-2 manuscript (2026-11-03).
Criteria written here before running; all are secondary analyses and are reported whatever they show.

A. (Major 3) The LAM_NE references are unreliable (step 11), yet the half-cell, faithful Lui et al. and knee tests use
   them. Repeat them without LAM_NE information: LAM_NE held at its first-check-up value (relative LAM_NE = 0), and
   the knee test with the LAM_PE:LLI ratio instead of LAM_NE:LLI. The conclusion is called unchanged if the verdict
   of every row (10 % gain excluded or not; ratio useful or not) is the same as with LAM_NE.

B. (Major 1) A forecaster that uses the modes to detect a change of mechanism. At the end of observation, the
   recent rate (slope over the last three observed check-ups) of LAM (LAM_PE for NMC; LAM_NE for LFP, where LAM_PE is
   fixed) is compared with that of LLI. If LAM is growing faster, capacity is extrapolated with the power law, which
   can accelerate (b up to 3); otherwise with the sqrt curve. Fair comparator: the same switch driven by capacity
   alone (recent fade rate above the mean fade rate over the observation window). Also reported against the
   capacity sqrt baseline of Table 3. For NMC the switch is also run with LAM_NE (descriptive, LAM_NE unreliable).
   Statistics as step 7 (two-stage bootstrap, best plausible gain, 10 % margin).

C. (Major 5) Few clusters: for the reference-mode rows of Table 3, the change recomputed leaving out one cluster at
   a time (range), and an exact sign-flip test on the cluster-mean differences (all 2^G sign patterns).

D. (Major 1) Observation and forecast horizons per dataset.

    "F:/Research New/battery-pinn-degradation/.venv/Scripts/python.exe" -m src.step14_reviewer2
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from src import step1_gonogo as s1
from src import step3_nmc_forecast as s3
from src import step5_halfcell as s5
from src import step7_reviewer as s7
from src import step9_michigan as s9
from src import step13_lui_ratio as s13


def datasets():
    data = [s13.kirkaldy(), s13.wheeler(), s13.michigan()]
    for _, t, _, _ in data:
        t["x"] = t.x - t.groupby("cid").x.transform("first")
    return data


def part_a(lines, data):
    lines.append("A. Without LAM_NE information (relative LAM_NE fixed at 0)")
    for name, t, last_obs, soh_of in data:
        t0 = t.copy()
        t0["LAM_NE"] = 0.0
        s13.part_a(lines, name + " [no LAM_NE]", t0, last_obs, soh_of)
    lines.append("\nA2. Knee test with the LAM_PE:LLI ratio")
    for name, t, last_obs, _ in data:
        t2 = t.copy()
        t2["LAM_NE"] = t2["LAM_PE"]  # part_b reads LAM_NE / LLI
        s13.part_b(lines, name + " [LAM_PE:LLI]", t2, last_obs)


def slope(x, y):
    return np.polyfit(x, y, 1)[0] if len(x) >= 2 else np.nan


def part_b(lines, data):
    lines.append("\nB. Mechanism-switching forecast (power law if LAM grows faster than LLI over the last 3 check-ups)")
    for name, t, last_obs, _ in data:
        lam_cols = ["LAM_NE"] if "LFP" in name else ["LAM_PE", "LAM_NE"]
        for lam in lam_cols:
            rows = []
            for cid, c in t.groupby("cid"):
                k = last_obs(c)
                obs, fut = c[c.rpt <= k], c[c.rpt > k]
                if len(obs) < 4 or fut.empty:
                    continue
                xo, xf, y = obs.x.to_numpy(), fut.x.to_numpy(), fut.SOH.to_numpy()
                fade = 1 - obs.SOH.to_numpy()
                r = slice(-3, None)
                flag_modes = slope(xo[r], obs[lam].to_numpy()[r]) > slope(xo[r], obs.LLI.to_numpy()[r])
                flag_cap = slope(xo[r], fade[r]) > slope(xo, fade)
                f = {}
                for curve in ("sqrt", "power"):
                    s1.CURVE = curve
                    f[curve] = 1 - s1.fit_curve(xo, fade)(xf)
                s1.CURVE = "sqrt"
                pred = {"capacity": f["sqrt"], "switch_modes": f["power"] if flag_modes else f["sqrt"],
                        "switch_capacity": f["power"] if flag_cap else f["sqrt"]}
                for m, v in pred.items():
                    rows.append({"cid": cid, "grp": str(c.grp.iloc[0]), "method": m, "flag_m": flag_modes,
                                 "flag_c": flag_cap, "rmse": 100 * np.sqrt(np.mean((v - y) ** 2))})
            r = pd.DataFrame(rows)
            p = r.pivot(index="cid", columns="method", values="rmse").dropna()
            grp = r.drop_duplicates("cid").set_index("cid").grp.loc[p.index]
            fl = r.drop_duplicates("cid").set_index("cid").loc[p.index]
            lines.append(f"  {name}, switch on {lam}: {len(p)} cells; switched by modes {int(fl.flag_m.sum())}, "
                         f"by capacity {int(fl.flag_c.sum())}, both {int((fl.flag_m & fl.flag_c).sum())}")
            s7.report(lines, f"   modes switch vs capacity sqrt", p.switch_modes, p.capacity, grp)
            s7.report(lines, f"   modes switch vs capacity switch", p.switch_modes, p.switch_capacity, grp)


def cluster_checks(lines, label, a, b, grp):
    d = (a.abs() - b.abs()).groupby(grp).mean()
    base = b.abs().groupby(grp).mean()
    change = a.abs().mean() / b.abs().mean() - 1
    loo = [a.abs()[grp != g].mean() / b.abs()[grp != g].mean() - 1 for g in d.index]
    obs = d.mean()
    flips = [np.mean(np.array(s) * d.to_numpy()) for s in itertools.product((1, -1), repeat=len(d))]
    p_gain = np.mean(np.array(flips) <= obs)  # one-sided: modes better (negative difference)
    lines.append(f"  {label:38s} change {change:+.0%}; leave-one-cluster-out {min(loo):+.0%} to {max(loo):+.0%} "
                 f"({len(d)} clusters); exact sign-flip p(modes better) = {p_gain:.2f}")


def part_c(lines):
    lines.append("\nC. Few-cluster checks for the reference-mode rows of Table 3")
    nmc, cap = s3.load()
    for curve in ("sqrt", "power"):
        s1.CURVE = curve
        f = s3.forecasts(nmc, cap)
        f = f[f.k == f.set.map(f.groupby("set").k.max())]
        p = f.pivot(index="cid", columns="method", values="soh_rmse_pp")
        cluster_checks(lines, f"Kirkaldy {curve}, mode map", p.modes_ref, p.capacity, p.index.str[:5].to_series(index=p.index))
        lfp = pd.read_parquet("data/checkups_lfp.parquet")
        ex = s1.extrapolation(lfp)
        ex = ex[ex.n_obs == 600].pivot(index="cell", columns="method", values="soh_rmse_pp")
        g = ex.index.str.extract(r"lfp_(\d)", expand=False).to_series(index=ex.index)
        cluster_checks(lines, f"Wheeler {curve}, mode map", ex.modes_ref, ex.capacity, g)
    s1.CURVE = "sqrt"
    d = s5.datasets()
    for chem in ("nmc", "lfp"):
        r = s5.yu_table(chem, d[chem])
        g = (r.index.str[:5] if chem == "nmc" else r.index.str[0]).to_series(index=r.index)
        cluster_checks(lines, f"{chem.upper()} half-cell after Lui", r.modes_ref, r.capacity, g)
    t = s9.soh_table("discharge")
    for curve in ("sqrt", "power"):
        s1.CURVE = curve
        f = s9.forecasts(t, s9.K_PRIMARY)
        p = f.pivot(index="cell", columns="method", values="rmse").dropna()
        g = p.index.map(s9.condition).to_series(index=p.index)
        cluster_checks(lines, f"Michigan {curve}, mode map", p.modes_map, p.capacity, g)
        cluster_checks(lines, f"Michigan {curve}, after Lui", p.modes_yu, p.capacity, g)
    s1.CURVE = "sqrt"


def part_d(lines, data):
    lines.append("\nD. Observation and forecast horizons (median over cells)")
    for name, t, last_obs, _ in data:
        rows = []
        for cid, c in t.groupby("cid"):
            k = last_obs(c)
            obs, fut = c[c.rpt <= k], c[c.rpt > k]
            if fut.empty:
                continue
            rows.append((len(obs), len(fut), obs.x.max() / c.x.max(), 1 - obs.SOH.iloc[-1], 1 - c.SOH.iloc[-1]))
        a = np.array(rows)
        lines.append(f"  {name}: observed {np.median(a[:, 0]):.0f} check-ups, forecast {np.median(a[:, 1]):.0f}; "
                     f"observation covers {np.median(a[:, 2]):.0%} of the test; fade at end of observation "
                     f"{100 * np.median(a[:, 3]):.1f} %, at end of test {100 * np.median(a[:, 4]):.1f} %")


def main():
    data = datasets()
    lines = []
    part_a(lines, data)
    part_b(lines, data)
    part_c(lines)
    part_d(lines, data)
    text = "\n".join(lines)
    print(text)
    open("data/step14_reviewer2.txt", "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
