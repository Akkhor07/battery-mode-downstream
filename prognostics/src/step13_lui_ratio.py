"""Step 13: two secondary checks requested after reading the full texts (2026-10-28).

A. Faithful Lui et al. (2021, J. Power Sources 485:229327) forecast. Their Sections 2.3-2.4 and Table 1, as read:
   modes are extrapolated one by one, coefficients are bounded to +-50 % of a best fit on the *other cells of the
   same group*, and the extrapolated modes are mapped to capacity by the half-cell model with no anchoring to the last
   measured capacity. The capacity-only baseline gets the same model and the same bounds.
       config 3 (primary)   Model II, loss = a x^b, a and b within +-50 % of the pooled fit on the other cells of the
                            group (Kirkaldy: experiment; Wheeler: protocol; Michigan: cycling condition)
       config 2 (secondary) Model I, y = a exp(b x) + c (1 - exp(d x)) on the remaining fraction, same bounds
       unbounded power      our earlier per-cell fit, same pipeline (bridge to steps 5 and 9)
   Each is reported unanchored (as in Lui et al.) and anchored to the last observed SOH (as in steps 5 and 9).
   Reference modes only, made relative to each cell's first check-up. Observation points as in steps 5 and 9
   (Kirkaldy: last RPT of steps 5/7; Wheeler: 600 cycles; Michigan: check-ups 0-5, C/20 discharge fits).
   Statistics: step 7 (cell and cluster bootstrap, best plausible gain, margins 5/10/20 %).
   Criterion (fixed before running): a useful gain is not excluded if the cluster-level best plausible gain is
   >= 10 % for config 3 unanchored. Secondary check, reported whatever it shows.

B. LAM_NE:LLI ratio as an early warning of a knee (Baure & Dubarry 2019, Batteries 5:42: ratio > 1 precedes the
   accelerated, plating stage). At the same observation points as A, per cell:
       score     LAM_NE / LLI at the last observed check-up (reference modes, relative to the first check-up;
                 LLI floored at 0.1 pp)
       rule      score > 1
       label     knee = later fade rate >= 1.5 x the observed fade rate, from measured capacity only (step 9)
       baseline  capacity-only acceleration: fade rate over the last 3 observed check-ups / over all observed ones
   Criterion (fixed before running): the ratio is a useful early warning if its AUC has a 95 % cell-bootstrap lower
   bound > 0.5 *and* the paired AUC difference (ratio - capacity baseline) has a lower bound > 0. Sensitivity and
   specificity of the rule are reported descriptively.

    "F:/Research New/battery-pinn-degradation/.venv/Scripts/python.exe" -m src.step13_lui_ratio
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from src import step1_gonogo as s1
from src import step5_halfcell as s5
from src import step7_reviewer as s7
from src import step9_michigan as s9

warnings.filterwarnings("ignore")
MODES = ("LLI", "LAM_PE", "LAM_NE")
RNG = np.random.default_rng(13)


def power(x, a, b):
    return a * np.power(np.maximum(np.asarray(x, float), 0), b)


def dexp(x, a, b, c, d):
    x = np.asarray(x, float)
    return a * np.exp(b * x) + c * (1 - np.exp(d * x))


def _fit(f, x, y, p0, bounds=(-np.inf, np.inf)):
    try:
        return curve_fit(f, x, y, p0=p0, bounds=bounds, maxfev=20000)[0]
    except (RuntimeError, ValueError):
        return None


def _box(c, rel=0.5):
    lo, hi = c - rel * np.abs(c), c + rel * np.abs(c)
    return lo, np.maximum(hi, lo + 1e-9)


def bounded_forecast(config, x_tr, y_tr, x_obs, y_obs):
    """Fitted callable for loss (config 3, unbounded) or remaining fraction (config 2)."""
    if config == "unbounded":
        return s1.fit_curve(x_obs, y_obs)  # s1.CURVE = "power"
    if np.all(np.abs(y_tr) < 1e-9) and np.all(np.abs(y_obs) < 1e-9):
        return lambda x: np.zeros_like(np.asarray(x, float)) + (1.0 if config == "c2" else 0.0)
    if config == "c3":
        c0 = _fit(power, x_tr, y_tr, (max(y_tr.max(), 1e-4), 0.7), ([-np.inf, 0.05], [np.inf, 5.0]))
        f = power
    else:
        c0 = _fit(dexp, x_tr, y_tr, (1.0, -0.01, 0.0, -0.1))
        f = dexp
    if c0 is None:
        return None
    lo, hi = _box(c0)
    c = _fit(f, x_obs, y_obs, np.clip(c0, lo, hi), (lo, hi))
    return None if c is None else (lambda x, c=c: f(x, *c))


def forecast_cell(config, train, obs, xf, soh_of, anchor):
    """SOH forecast at xf for capacity and for the three modes mapped by ``soh_of(lli, lpe, lne)``."""
    tgt = (lambda y: 1 - y) if config == "c2" else (lambda y: y)  # model I works on the remaining fraction
    back = tgt
    out = {}
    fc = bounded_forecast(config, train.x.to_numpy(), tgt(1 - train.SOH.to_numpy()), obs.x.to_numpy(),
                          tgt(1 - obs.SOH.to_numpy()))
    out["capacity"] = np.full(len(xf), np.nan) if fc is None else 1 - back(fc(xf))
    fm = [bounded_forecast(config, train.x.to_numpy(), tgt(train[m].to_numpy()), obs.x.to_numpy(),
                           tgt(obs[m].to_numpy())) for m in MODES]
    if any(f is None for f in fm):
        out["modes"] = np.full(len(xf), np.nan)
        return out
    fut = soh_of(*(back(f(xf)) for f in fm))
    if anchor:
        xl = obs.x.to_numpy()[-1:]
        fut = fut * obs.SOH.iloc[-1] / soh_of(*(back(f(xl)) for f in fm))[0]
    out["modes"] = fut
    return out


def relative(t, cols):
    g = t.groupby("cid")
    for c in cols:
        t[c] = 1 - (1 - t[c]) / (1 - g[c].transform("first"))
    return t


def kirkaldy():
    cap = pd.read_parquet("data/capacity_nmc.parquet").dropna(subset=["LLI_ref"]).sort_values(["set", "cell", "rpt"])
    cap["cid"], cap["grp"] = cap.set + cap.cell, cap.set
    cap["x"] = cap.throughput_Ah / 1000
    cap = relative(cap.rename(columns={f"{m}_ref": m for m in MODES}), list(MODES))
    hc = s5.HalfCell("nmc")
    k = s5.datasets()["nmc"]["obs"]
    return "Kirkaldy NMC", cap, lambda c: k[c.set.iloc[0]], hc.soh


def wheeler():
    t = pd.read_parquet("data/checkups_lfp.parquet").sort_values(["cell", "rpt"])
    t["cid"], t["grp"], t["x"] = t.cell, t.protocol, t.cycles / 1000
    t = relative(t.rename(columns={f"{m}_ref": m for m in MODES}), list(MODES))
    t["LAM_PE"] = t.LAM_PE.fillna(0.0)
    return "Wheeler LFP", t.dropna(subset=["LLI", "LAM_NE"]), lambda c: 6, s5.HalfCell("lfp").soh


def michigan():
    t = s9.soh_table("discharge")
    t["cid"], t["grp"] = t.cell, t.cell.map(s9.condition)
    q0 = t.groupby("cell")[["Qn", "Qp", "QLi"]].first()
    soh_fns = {}
    for cell, q in q0.iterrows():
        c0 = s9.ocv_capacity(q.Qn, q.Qp, q.QLi)

        def soh_of(lli, lpe, lne, q=q, c0=c0):
            b = np.broadcast_arrays(np.asarray(lli, float), np.asarray(lpe, float), np.asarray(lne, float))
            return np.array([s9.ocv_capacity(q.Qn * (1 - n), q.Qp * (1 - p), q.QLi * (1 - li))
                             for li, p, n in zip(*b)]) / c0
        soh_fns[cell] = soh_of
    return "Michigan NMC", t, lambda c: s9.K_PRIMARY, soh_fns


def part_a(lines, name, t, last_obs, soh_of):
    lines.append(f"\n{name}: {t.cid.nunique()} cells")
    s1.CURVE = "power"
    for config in ("c3", "c2", "unbounded"):
        for anchor in (False, True):
            rows = []
            for cid, c in t.groupby("cid"):
                k = last_obs(c)
                obs, fut = c[c.rpt <= k], c[c.rpt > k]
                if len(obs) < 3 or fut.empty:
                    continue
                train = t[(t.grp == c.grp.iloc[0]) & (t.cid != cid)]
                if train.cid.nunique() < 1:
                    continue
                fn = soh_of[cid] if isinstance(soh_of, dict) else soh_of
                p = forecast_cell(config, train, obs, fut.x.to_numpy(), fn, anchor)
                y = fut.SOH.to_numpy()
                for m, v in p.items():
                    ok = np.isfinite(v)
                    rows.append({"cid": cid, "grp": c.grp.iloc[0], "method": m,
                                 "rmse": 100 * np.sqrt(np.mean((v[ok] - y[ok]) ** 2)) if ok.sum() == len(v) else np.nan})
            r = pd.DataFrame(rows)
            p = r.pivot(index="cid", columns="method", values="rmse")
            n_all = len(p)
            p = p.dropna()
            grp = r.drop_duplicates("cid").set_index("cid").grp.loc[p.index].astype(str)
            tag = {"c3": "config 3 (bounded power)", "c2": "config 2 (bounded dbl-exp)", "unbounded": "unbounded power"}[config]
            s7.report(lines, f"{tag}, {'anchored' if anchor else 'unanchored'} (n={len(p)}/{n_all})",
                      p.modes, p.capacity, grp)
    s1.CURVE = "sqrt"


def auc(score, label):
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return (np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))


def part_b(lines, name, t, last_obs):
    rows = []
    for cid, c in t.groupby("cid"):
        k = last_obs(c)
        obs, fut = c[c.rpt <= k], c[c.rpt > k]
        if len(obs) < 4 or fut.empty:
            continue
        xo, so = obs.x.to_numpy(), obs.SOH.to_numpy()
        early = np.polyfit(xo, so, 1)[0]
        late = np.polyfit(np.r_[xo[-1], fut.x], np.r_[so[-1], fut.SOH], 1)[0]
        recent = np.polyfit(xo[-3:], so[-3:], 1)[0]
        lli, lne = obs.LLI.iloc[-1], obs.LAM_NE.iloc[-1]
        rows.append({"cid": cid, "knee": late <= 1.5 * early, "ratio": lne / max(lli, 1e-3),
                     "accel": recent / early if early < 0 else np.nan})
    r = pd.DataFrame(rows).dropna()
    lab = r.knee.to_numpy()
    if lab.all() or not lab.any():
        lines.append(f"  {name}: {len(r)} cells, {lab.sum()} knee -> not testable (one class only); median ratio "
                     f"{r.ratio.median():.2f}, ratio > 1 in {(r.ratio > 1).sum()} cells")
        return
    sr, sa = r.ratio.to_numpy(), r.accel.to_numpy()
    a_r, a_a = auc(sr, lab), auc(sa, lab)
    boot = []
    for _ in range(4000):
        i = RNG.integers(0, len(r), len(r))
        boot.append((auc(sr[i], lab[i]), auc(sa[i], lab[i])))
    b = np.array(boot, float)
    b = b[np.isfinite(b).all(1)]
    rule = sr > 1
    tp, fn_, fp, tn = (rule & lab).sum(), (~rule & lab).sum(), (rule & ~lab).sum(), (~rule & ~lab).sum()
    lr, hr = np.percentile(b[:, 0], [2.5, 97.5])
    ld, hd = np.percentile(b[:, 0] - b[:, 1], [2.5, 97.5])
    useful = lr > 0.5 and ld > 0
    lines.append(f"  {name}: {len(r)} cells, {lab.sum()} knee | ratio AUC {a_r:.2f} [{lr:.2f},{hr:.2f}] | capacity "
                 f"acceleration AUC {a_a:.2f} | difference [{ld:+.2f},{hd:+.2f}] | rule ratio>1: TP {tp} FN {fn_} FP {fp} "
                 f"TN {tn} | median ratio knee {np.median(sr[lab]) if lab.any() else np.nan:.2f}, no knee "
                 f"{np.median(sr[~lab]) if (~lab).any() else np.nan:.2f} -> {'useful' if useful else 'not useful'}")


def main():
    data = [kirkaldy(), wheeler(), michigan()]
    for _, t, _, _ in data:  # ageing axis from each cell's first check-up (curves pass through the origin)
        t["x"] = t.x - t.groupby("cid").x.transform("first")
    lines = ["A. Faithful Lui et al. forecast (reference modes), SOH RMSE [pp] over later check-ups"]
    for name, t, last_obs, soh_of in data:
        part_a(lines, name, t, last_obs, soh_of)
    lines.append("\nB. LAM_NE:LLI ratio (Baure & Dubarry) as knee early warning, at the same observation points")
    for name, t, last_obs, _ in data:
        part_b(lines, name, t, last_obs)
    text = "\n".join(lines)
    print(text)
    open("data/step13_lui_ratio.txt", "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
