"""Step 11: ground-truth check of the reference-mode fits on synthetic curves with known degradation modes.

Dubarry & Ansean (2022), Mendeley 4k8f4t352y (CC BY 4.0), alawa toolbox, C/25. Map files vary one mechanism at a
time (0-30 %); the RDF maps change only kinetics, so the true LLI/LAM are zero. Example files hold a 5-point
combined path (LLI up to 15 %, LAM_PE and LAM_NE up to 10 %).

The synthetic electrodes are alawa stock data, not the OCPs our fits assume - the same situation as a real cell.
Each chemistry is fitted with the exact pipeline used on real data (discharge curve, same starts, same rule):
  GIC-NMC532         -> Michigan pipeline (Mohtat2020 OCPs, src.step9_michigan)
  (GIC,Si)-NMC811    -> Kirkaldy/Stanford pipeline (Chen2020 graphite-only + NMC811, src.step8_refit)
100 % of the synthetic capacity is mapped to 5.0 A.h, discharged at C/25.

Pass criteria (written before running): for true values <= 10 %, |estimate - truth| <= 2 pp for the varied mode
and <= 2 pp for the modes that should stay at zero (crosstalk); kinetics-only (RDF) curves must give |mode| <= 2 pp.
LLI is reported as Li-inventory loss (our definition) and, as a check on the convention, as loss in units of the
fresh cell capacity.

    "F:/Research New/battery-pinn-degradation/.venv/Scripts/python.exe" -m src.step11_dubarry [--part fit|eval]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

from src import step8_refit as s8
from src import step9_michigan as s9

ROOT = Path("data/dubarry")
C0 = 5.0
CHEM = {
    "GIC-NMC532": dict(un=s9.u_n, qn_fresh=s9.QN0, up=s9.u_p, qp_starts=(5.3, 6.3), ends=((0.002, 0.9), (0.02, 0.85))),
    "(GIC,Si)-NMC811": dict(un=s8.u_gr, qn_fresh=5.2, up=s8.u_p, qp_starts=(6.5, 7.5),
                            ends=((0.02, 0.95), (0.05, 0.9))),
}


def curves(chem: str):
    """(set, level, true LLI/LAM_PE/LAM_NE in %, q [A.h], v) for every discharge curve used."""
    m = sio.loadmat(ROOT / f"{chem}_Maps.mat")
    qgrid = m["Qmd"].ravel()
    for j in range(m["Mapindex"].shape[1]):
        name = str(m["Mapindex"][0, j][0])
        P, V = m["Params"][0, j], m["Vmd"][0, j]
        for k in range(0, P.shape[0], 2):  # every other level: 0, 1.2, ... 30 %
            yield name, P[k, 0], P[k, 1:4], qgrid, V[:, k]
    e = sio.loadmat(ROOT / f"{chem}_Example.mat")
    for k in range(e["Params"].shape[0]):
        yield "Example", e["Params"][k, 0], e["Params"][k, 1:4], e["Qd"].ravel(), e["Vd"][:, k]


def part_fit():
    rows = []
    for chem, kw in CHEM.items():
        for name, level, truth, qg, v in curves(chem):
            ok = ~np.isnan(v)
            q = np.abs(qg[ok] - qg[ok][0]) / 100 * C0
            g = np.linspace(0, q[-1], 400)
            f = s8.fit_curve(g, np.interp(g, q, v[ok]), -C0 / 25, kw["un"], kw["qn_fresh"], up=kw["up"],
                             qp_starts=kw["qp_starts"], ends=kw["ends"])
            rows.append({"chem": chem, "set": name, "level": level, "Q": q[-1], "t_LLI": truth[0],
                         "t_LAM_PE": truth[1], "t_LAM_NE": truth[2], **f})
            print(chem, name, level, round(q[-1], 3), round(f["rmse_mV"], 1), round(f["R"], 3), flush=True)
    r = pd.DataFrame(rows)
    r["ok"] = (r.rmse_mV <= 20) & (r.R > 0) & r.Qn.between(2.05, 11.95) & r.Qp.between(2.05, 11.95)
    r.to_parquet(ROOT / "fits.parquet")


def part_eval():
    r = pd.read_parquet(ROOT / "fits.parquet")
    lines = []
    for (chem, name), t in r.groupby(["chem", "set"], sort=False):
        t = t.sort_values("level")
        f = t.iloc[0]  # the fresh curve of this set
        est = pd.DataFrame({"LLI": 100 * (1 - t.QLi / f.QLi), "LAM_PE": 100 * (1 - t.Qp / f.Qp),
                            "LAM_NE": 100 * (1 - t.Qn / f.Qn), "LLI_cap": 100 * (f.QLi - t.QLi) / f.Q})
        tru = t[["t_LLI", "t_LAM_PE", "t_LAM_NE"]].set_axis(["LLI", "LAM_PE", "LAM_NE"], axis=1)
        err = est[["LLI", "LAM_PE", "LAM_NE"]] - tru
        small = (tru.max(axis=1) <= 10).to_numpy()
        worst = err[small].abs().max()
        verdict = "PASS" if (worst <= 2).all() else "FAIL"
        lines.append(f"{chem:16s} {name:8s} fits ok {t.ok.sum()}/{len(t)}, rmse {t.rmse_mV.median():.1f} mV, "
                     f"R {t.R.min():.3f}..{t.R.max():.3f} | max |err| (truth<=10%): "
                     + ", ".join(f"{k} {v:.1f}" for k, v in worst.items()) + f" pp -> {verdict}")
        for k in range(len(t)):
            if k % 3 == 0 or name == "Example":
                lines.append(f"    level {t.level.iloc[k]:5.1f}  truth " + "/".join(f"{x:4.1f}" for x in tru.iloc[k])
                             + "  est " + "/".join(f"{x:5.1f}" for x in est.iloc[k, :3])
                             + f"  (LLI per cell cap {est.LLI_cap.iloc[k]:5.1f})  R {t.R.iloc[k]:.3f}"
                             + ("" if t.ok.iloc[k] else "  [fails rule]"))
    text = "\n".join(lines)
    Path("data/step11_dubarry.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("fit", "eval"), default="fit")
    (part_fit if ap.parse_args().part == "fit" else part_eval)()
