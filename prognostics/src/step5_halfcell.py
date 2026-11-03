"""Step 5: the published mode-based forecast (Yu et al. 2021, J. Power Sources) and a positive control.

Half-cell capacity model: PyBaMM's electrode-SOH solver (ORegan2022 for the LG M50T, 2.5-4.2 V; Prada2013 for the
A123 LFP, 2.0-3.6 V) maps LLI, LAM_PE, LAM_NE (fractions of the initial lithium inventory and electrode
capacities) to the cell capacity, relative to the unaged cell.

validate   Reference modes -> modelled SOH against measured SOH, every check-up.
yu         Forecast as in Yu et al.: extrapolate each mode separately (same curve family as steps 1 and 3), then
           compute capacity with the half-cell model, anchored to the last observed SOH (ratio). Reference modes
           and estimated modes, against extrapolating capacity directly.
control    Positive control: 60 synthetic NMC cells (half-cell model as ground truth) whose anode loss
           accelerates, so some develop a capacity knee late in life that capacity alone cannot foresee. The same
           forecasting pipeline must show a clear gain here with exact modes, and the mode noise at which the gain
           disappears is the accuracy an estimator would need.

    "F:/Research New/battery-pinn-degradation/.venv/Scripts/python.exe" -m src.step5_halfcell
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.step1_gonogo import fit_curve
from src.step4_robustness import best_gain, verdict

CHEM = {"nmc": ("ORegan2022", 2.5, 4.2), "lfp": ("Prada2013", 2.0, 3.6)}
RNG = np.random.default_rng(5)


class HalfCell:
    """Relative capacity Q(LLI, LAM_PE, LAM_NE) / Q(0, 0, 0) from the electrode-SOH equations."""

    def __init__(self, chem: str):
        import pybamm

        name, vmin, vmax = CHEM[chem]
        P = pybamm.ParameterValues(name)
        P.update({"Lower voltage cut-off [V]": vmin, "Upper voltage cut-off [V]": vmax})
        p = pybamm.lithium_ion.LithiumIonParameters() if hasattr(pybamm.lithium_ion, "LithiumIonParameters") \
            else pybamm.LithiumIonParameters()
        self.q0 = (P.evaluate(p.n.Q_init), P.evaluate(p.p.Q_init), P.evaluate(p.Q_Li_particles_init))
        self.solver = pybamm.lithium_ion.ElectrodeSOHSolver(P, p)
        self.c0 = self._q(0.0, 0.0, 0.0)

    def _q(self, lli, lpe, lne):
        qn, qp, qli = self.q0
        try:
            return float(self.solver.solve({"Q_n": qn * (1 - lne), "Q_p": qp * (1 - lpe), "Q_Li": qli * (1 - lli)})["Q"])
        except Exception:  # no consistent state (e.g. more lithium than sites): treat as dead
            return np.nan

    def soh(self, lli, lpe, lne):
        b = np.broadcast_arrays(np.asarray(lli, float), np.asarray(lpe, float), np.asarray(lne, float))
        return np.array([self._q(*v) for v in zip(*(x.ravel() for x in b))]).reshape(b[0].shape) / self.c0


def rel(t, cols, key):
    g = t.groupby(key)
    for c in cols:
        t[c] = 1 - (1 - t[c]) / (1 - g[c].transform("first"))
    return t


def datasets():
    """Per chemistry: check-up table with x (ageing axis), SOH, modes per estimator; truth table for scoring."""
    nmc = pd.read_parquet("data/checkups_nmc.parquet").sort_values(["set", "cell", "rpt"])
    nmc["cid"], nmc["x"] = nmc.set + nmc.cell, nmc.throughput_Ah / 1000
    nmc = rel(nmc, [f"{m}_pinn_lm" for m in ("LLI", "LAM_PE", "LAM_NE")], "cid")
    cap = pd.read_parquet("data/capacity_nmc.parquet")
    cap["cid"], cap["x"] = cap.set + cap.cell, cap.throughput_Ah / 1000
    lfp = pd.read_parquet("data/checkups_lfp.parquet").sort_values(["cell", "rpt"])
    lfp["cid"], lfp["x"] = lfp.cell, lfp.cycles / 1000
    lfp = rel(lfp, ["LLI_pinn_enc", "LAM_NE_pinn_enc"], "cid")
    lfp["LAM_PE_pinn_enc"] = 0.0  # cathode loss is unidentifiable on LFP (JES paper); the reference fixes it at 0
    return {
        "nmc": dict(t=nmc, truth=cap, est="pinn_lm",
                    obs={"expt5": 6, "expt4": 4, "expt1": 6, "expt2": 6, "expt3": 6}),  # last observed RPT
        "lfp": dict(t=lfp, truth=lfp, est="pinn_enc", obs={"": 6}),  # 600 cycles
    }


def validate(lines):
    lines.append("Half-cell model with reference modes vs measured SOH (all check-ups)")
    for chem, d in datasets().items():
        hc = HalfCell(chem)
        t = d["truth"].dropna(subset=["LLI_ref"])
        s = hc.soh(t.LLI_ref, t.LAM_PE_ref, t.LAM_NE_ref)
        ok = np.isfinite(s)
        e = s[ok] - t.SOH.to_numpy()[ok]
        r2 = 1 - np.var(e) / np.var(t.SOH.to_numpy()[ok])
        lines.append(f"  {chem}: n={ok.sum()}, RMSE {100 * np.sqrt(np.mean(e ** 2)):.2f} pp, bias {100 * e.mean():+.2f} pp,"
                     f" R^2 {r2:.3f}")


def forecast_modes(hc, x_obs, modes_obs, soh_last, x_fut):
    """Extrapolate each mode, map through the half-cell model, anchor to the last observed SOH."""
    fits = [fit_curve(x_obs, m) for m in modes_obs]
    fut = hc.soh(*(f(x_fut) for f in fits))
    last = hc.soh(*(f(np.array([x_obs.max()])) for f in fits))[0]
    return fut * soh_last / last


def yu_table(chem: str, d: dict) -> pd.DataFrame:
    """Per cell: SOH RMSE [pp] of capacity and of the Yu-style forecast (reference and estimated modes)."""
    hc, t, truth, est = HalfCell(chem), d["t"], d["truth"], d["est"]
    res = []
    for cid, c in t.groupby("cid"):
        k = d["obs"][c.set.iloc[0] if chem == "nmc" else ""]
        obs = c[c.rpt <= k]
        fut = truth[(truth.cid == cid) & (truth.rpt > k)]
        if len(obs) < 3 or fut.empty:
            continue
        xo, xf, y = obs.x.to_numpy(), fut.x.to_numpy(), fut.SOH.to_numpy()
        p = {"capacity": 1 - fit_curve(xo, 1 - obs.SOH.to_numpy())(xf)}
        for name, cols in (("modes_ref", [f"{m}_ref" for m in ("LLI", "LAM_PE", "LAM_NE")]),
                           (f"modes_{est}", [f"{m}_{est}" for m in ("LLI", "LAM_PE", "LAM_NE")])):
            p[name] = forecast_modes(hc, xo, [obs[c_].to_numpy() for c_ in cols], obs.SOH.iloc[-1], xf)
        for name, v in p.items():
            ok = np.isfinite(v)
            res.append({"cid": cid, "method": name,
                        "rmse": 100 * np.sqrt(np.mean((v[ok] - y[ok]) ** 2)) if ok.any() else np.nan})
    return pd.DataFrame(res).pivot(index="cid", columns="method", values="rmse").dropna()


def yu(lines):
    lines.append("\nYu-style forecast (modes extrapolated separately -> half-cell capacity), SOH RMSE [pp] over later "
                 "check-ups; best plausible gain vs capacity (one-sided 95 %)")
    for chem, d in datasets().items():
        r = yu_table(chem, d)
        lines.append(f"  {chem} ({len(r)} cells): capacity {r.capacity.mean():.2f}")
        for m in r.columns.drop("capacity"):
            g, mean = best_gain(r[m], r.capacity)
            lines.append(f"    {m:15s} {r[m].mean():6.2f}  change {mean:+6.0%}  best gain {g:5.1%}  -> {verdict(g)}")


def synthetic_cells(hc, n=60, n_rpt=16, dx=0.6):
    x = dx * np.arange(n_rpt)
    cells = []
    for i in range(n):
        lli = RNG.uniform(0.03, 0.06) * np.sqrt(x) + RNG.uniform(0, 0.005) * x
        lpe = RNG.uniform(0.002, 0.008) * x
        lne = RNG.uniform(0.002, 0.01) * x ** RNG.uniform(1.0, 2.0)  # accelerating anode loss
        cells.append(pd.DataFrame({"cid": i, "rpt": np.arange(n_rpt), "x": x, "LLI": lli, "LAM_PE": lpe,
                                   "LAM_NE": lne, "SOH": hc.soh(lli, lpe, lne)}))
    return pd.concat(cells, ignore_index=True).dropna()


def control(lines, k=6, noises=(0.0, 0.005, 0.01, 0.02, 0.04)):
    hc = HalfCell("nmc")
    s = synthetic_cells(hc)
    # knee: late-life fade rate at least twice the early rate
    rate = s.groupby("cid").apply(lambda c: np.polyfit(c.x[c.rpt >= 10], c.SOH[c.rpt >= 10], 1)[0]
                                  / np.polyfit(c.x[c.rpt <= k], c.SOH[c.rpt <= k], 1)[0])
    knee = rate[rate > 2].index
    lines.append(f"\nPositive control: {s.cid.nunique()} synthetic NMC cells (half-cell truth), observe RPT 0-{k}, forecast "
                 f"RPT {k + 1}-15; {len(knee)} cells with a late knee (fade rate x2). SOH RMSE [pp]")
    res = []
    for cid, c in s.groupby("cid"):
        obs, fut = c[c.rpt <= k], c[c.rpt > k]
        xo, xf, y = obs.x.to_numpy(), fut.x.to_numpy(), fut.SOH.to_numpy()
        row = {"cid": cid, "knee": cid in knee,
               "capacity": 100 * np.sqrt(np.mean((1 - fit_curve(xo, 1 - obs.SOH.to_numpy())(xf) - y) ** 2))}
        for sd in noises:
            m = [obs[c_].to_numpy() + RNG.normal(0, sd, len(obs)) for c_ in ("LLI", "LAM_PE", "LAM_NE")]
            v = forecast_modes(hc, xo, m, obs.SOH.iloc[-1], xf)
            ok = np.isfinite(v)
            row[f"modes sd {sd:.3f}"] = 100 * np.sqrt(np.mean((v[ok] - y[ok]) ** 2)) if ok.any() else np.nan
        res.append(row)
    r = pd.DataFrame(res).dropna()
    for label, sub in (("all cells", r), ("knee cells", r[r.knee]), ("no knee", r[~r.knee])):
        if sub.empty:
            continue
        line = f"  {label:10s} (n={len(sub):2d}) capacity {sub.capacity.mean():5.2f}"
        for sd in noises:
            col = f"modes sd {sd:.3f}"
            g, mean = best_gain(sub[col], sub.capacity)
            line += f" | modes sd {sd:.3f}: {sub[col].mean():5.2f} ({mean:+.0%})"
        lines.append(line)


def main():
    lines = []
    validate(lines)
    yu(lines)
    control(lines)
    text = "\n".join(lines)
    print(text)
    open("data/step5_halfcell.txt", "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
