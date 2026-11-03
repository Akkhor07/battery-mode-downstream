"""Gap 2, step 6 (reviewer check, 2026-11-03): does a Bayesian bound that includes the estimator's prior rank the
window designs, where the classical Cramer-Rao bound did not?

The estimator is a MAP fit regularised towards the encoder output with a Gaussian prior of s.d. 3 in the logit
units of its outputs (scaled sigmoids over the training ranges). Mapped to physical units at the median ageing
state, that prior has s.d. range * sigma(1 - sigma) * 3 for each latent quantity (modes, log-diffusivities, window
starts). Its precision is added to the Fisher information of every grid design (a van Trees-type bound with the
prior linearised at the median state) and the resulting bound is compared with the realised error, exactly as the
classical bound in step 1. Criterion (fixed before running): the Bayesian bound is a usable design criterion if its
Spearman correlation with the realised RMSE has a 95 % cell-bootstrap interval above zero.

    "F:/Research New/battery-pinn-degradation/.venv/Scripts/python.exe" -m gap2.step6_bayes_crb
"""
from __future__ import annotations

from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from gap2.step1_design_grid import MODES, OUT, Q_NOM, RATES, USED, grid_designs, relative

PRIOR_SD = 3.0
# training ranges of the real-NMC estimator (latent outputs): modes as fractions, diffusivity scalings log-uniform
RANGES = {"LLI": (-0.06, 0.25), "LAM_PE": (-0.10, 0.20), "LAM_NE": (-0.05, 0.35),
          "logDn": (np.log(1.0), np.log(8.0)), "logDp": (np.log(0.15), np.log(2.0))}


def prior_sd(value, lo, hi):
    s = np.clip((value - lo) / (hi - lo), 0.05, 0.95)
    return (hi - lo) * s * (1 - s) * PRIOR_SD


def main():
    import sys
    from gap2.step1_design_grid import STAGE1
    sys.path.insert(0, str(STAGE1))
    from src.analysis.identifiability_real import MODELS, sensitivities, simulate_family
    from src.simulation import pybamm_generator as gen

    m = MODELS["nmc"]
    state, (dn, dp) = m["states"]["median"], (m["dn"], m["dp"])
    sd_lat = [prior_sd(state[0], *RANGES["LLI"]), prior_sd(state[1], *RANGES["LAM_PE"]),
              prior_sd(state[2], *RANGES["LAM_NE"]), prior_sd(np.log(dn), *RANGES["logDn"]),
              prior_sd(np.log(dp), *RANGES["logDp"])]
    print("prior s.d. in physical units (modes in pp, log-diffusivities):",
          [round(100 * v, 1) for v in sd_lat[:3]], [round(v, 2) for v in sd_lat[3:]])

    fams = {}
    with Pool(3, initializer=gen._init_worker_with_ref, initargs=(0.5, m["param_set"], "discharge")) as pool:
        for c in RATES:
            fams[c] = simulate_family(state, (dn, dp), c, pool)
    rows = []
    for d in grid_designs():
        cols_all, starts, sd_start = [], [], []
        ok = True
        for c, size, f in ((RATES[0], d["size_a"], d["fa"]), (RATES[1], d["size_b"], d["fb"])):
            fam, steps = fams[c]
            lo, hi = max(q[0] for q, _ in fam.values()), min(q[-1] for q, _ in fam.values())
            dq = size * Q_NOM
            q0 = lo + f * (hi - lo - dq - 1e-3)
            w = sensitivities(fam, steps, q0, dq)
            if w is None:
                ok = False
                break
            cols_all.append(w)
            sd_start.append(prior_sd(q0, lo, hi - dq))
        if not ok:
            continue
        blocks = []
        for i, (cols, dvdq) in enumerate(cols_all):
            st = [dvdq if j == i else np.zeros_like(dvdq) for j in range(2)]
            blocks.append(np.column_stack(cols[:5] + st))
        J = np.vstack(blocks)
        F = J.T @ J / 0.002 ** 2
        P = np.diag(1 / np.array(sd_lat + sd_start) ** 2)
        crb = 100 * np.sqrt(np.diag(np.linalg.pinv(F)))[:3]
        bayes = 100 * np.sqrt(np.diag(np.linalg.inv(F + P)))[:3]
        rows.append({**d, **{f"crb_{k}": v for k, v in zip(MODES, crb)},
                     **{f"bcrb_{k}": v for k, v in zip(MODES, bayes)}})
    b = pd.DataFrame(rows).set_index("design")

    p = relative(pd.read_parquet(OUT / "nmc_design_preds.parquet"))
    e = p.groupby(["set", "cell", "rpt", "design", "kind"])[[f"{k}_rel" for k in MODES] + [f"{k}_true" for k in MODES]].mean().reset_index()
    e = e[e.kind == "grid"]
    e["se"] = sum((e[f"{k}_rel"] - e[f"{k}_true"]) ** 2 for k in USED) / len(USED)
    rmse = lambda d: d.groupby("design").se.mean().pow(0.5).mul(100)  # noqa: E731
    b["bcrb_used"] = np.sqrt((b[[f"bcrb_{k}" for k in USED]] ** 2).mean(1))
    b["crb_used"] = np.sqrt((b[[f"crb_{k}" for k in USED]] ** 2).mean(1))
    rm = rmse(e).to_frame("rmse").join(b)
    rng = np.random.default_rng(6)
    cells = e[["set", "cell"]].drop_duplicates().to_numpy()
    lines = [f"Grid designs: {len(rm)}; realised RMSE of LLI, LAM_PE {rm.rmse.min():.2f}-{rm.rmse.max():.2f} pp",
             f"classical CRB {rm.crb_used.min():.3f}-{rm.crb_used.max():.3f} pp; Bayesian bound "
             f"{rm.bcrb_used.min():.3f}-{rm.bcrb_used.max():.3f} pp; max relative change "
             f"{(1 - rm.bcrb_used / rm.crb_used).max():.1%}"]
    for col in ("crb_used", "bcrb_used"):
        r = spearmanr(rm[col], rm.rmse)[0]
        boots = []
        for _ in range(500):
            pick = cells[rng.integers(0, len(cells), len(cells))]
            sub = pd.concat([e[(e.set == s) & (e.cell == c)] for s, c in pick])
            boots.append(spearmanr(rm[col], rmse(sub).loc[rm.index])[0])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        lines.append(f"Spearman({col}, realised RMSE) = {r:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    best = rm.bcrb_used.idxmin()
    lines.append(f"Bayesian-bound-best design {best}: realised {rm.loc[best, 'rmse']:.2f} pp "
                 f"(median {rm.rmse.median():.2f}, best {rm.rmse.min():.2f})")
    text = "\n".join(lines)
    print(text)
    (OUT / "step6_bayes_crb.log").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
