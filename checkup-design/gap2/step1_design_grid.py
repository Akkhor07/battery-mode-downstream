"""Gap 2, step 1 (go/no-go): does the Cramer-Rao bound of a window-pair design predict its error on real cells?

For every Kirkaldy check-up with both a C/10 and a C/2 discharge (16 cells, Expt 5 and 4), the frozen Stage-1
PINN + LM (3 seeds, 8 LM steps, prior s.d. 3, as in the JES paper) is run on a fixed grid of designs:
window a on the C/10 discharge and window b on the C/2 discharge, both of size 20 % or 30 % of 5 A.h, each
starting at 0, 25, 50, 75 or 100 % of its feasible range (50 grid designs), plus 10 random pairs of 35-50 %
(drawn per check-up) as the "large random" reference. Estimates are taken relative to the cell's first check-up
estimated with the same design. The CRB of every grid design comes from the JES Fisher engine
(src.analysis.identifiability_real, ORegan2022 at the median state).

    py = "F:/Research New/battery-pinn-degradation/.venv/Scripts/python.exe"
    $py -m gap2.step1_design_grid --part eval      # ~15-20 min CPU
    $py -m gap2.step1_design_grid --part crb       # a few min (PyBaMM)
    $py -m gap2.step1_design_grid --part analyse
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STAGE1 = Path(os.environ.get("STAGE1_ROOT", r"F:\Research New\battery-pinn-degradation"))
sys.path.insert(0, str(STAGE1))

OUT = Path("results")
SEEDS = (0, 1, 2)
FRACS = (0.0, 0.25, 0.5, 0.75, 1.0)
SIZES = (0.2, 0.3)
N_RANDOM, RANDOM_SIZE = 10, (0.35, 0.50)
Q_NOM = 5.0
RATES = (0.1, 0.5)
MODES = ("LLI", "LAM_PE", "LAM_NE")
USED = ("LLI", "LAM_PE")  # identifiable and unbiased on real NMC (JES paper)


def grid_designs():
    return [dict(design=f"s{s:.1f}_a{fa:.2f}_b{fb:.2f}", kind="grid", size_a=s, size_b=s, fa=fa, fb=fb)
            for s in SIZES for fa in FRACS for fb in FRACS]


def build_inputs(store, rng):
    """Window pairs for every design and check-up: arrays of voltage/temperature/scalars plus metadata."""
    from src.data_loader import _sample, window_bounds

    states = store.meta.groupby("state_id")
    vs, ts, ss, ys, info = [], [], [], [], []
    for state, g in states:
        if g.C_rate.nunique() < 2:
            continue
        sid = dict(zip(g.C_rate.round(2), g.sim_id))
        ca, cb = store.curves[sid[RATES[0]]], store.curves[sid[RATES[1]]]
        designs = grid_designs() + [dict(design=f"r{i}", kind="random", size_a=rng.uniform(*RANDOM_SIZE),
                                         size_b=rng.uniform(*RANDOM_SIZE), fa=rng.uniform(), fb=rng.uniform())
                                    for i in range(N_RANDOM)]
        for d in designs:
            wa = window_bounds(ca, d["size_a"] * Q_NOM, d["fa"])
            wb = window_bounds(cb, d["size_b"] * Q_NOM, d["fb"])
            if wa is None or wb is None:
                continue
            sa, sb = _sample(ca, *wa, store_spec, rng), _sample(cb, *wb, store_spec, rng)
            vs.append(np.stack([sa[0], sb[0]])), ts.append(np.stack([sa[1], sb[1]]))
            ss.append(np.stack([sa[2], sb[2]])), ys.append(ca.y)
            info.append({"state_id": state, "cell": g.cell.iloc[0] if "cell" in g else str(state),
                         "rpt": int(g.rpt.iloc[0]) if "rpt" in g else 0,
                         "temp_group": int(g.temp_group.iloc[0]) if "temp_group" in g else -1,
                         **d, "q0_a": wa[0], "q0_b": wb[0],
                         "hours": d["size_a"] / RATES[0] + d["size_b"] / RATES[1]})
    return np.array(vs), np.array(ts), np.array(ss), np.array(ys), pd.DataFrame(info)


store_spec = None


def part_eval(args):
    global store_spec
    import torch
    from torch.utils.data import TensorDataset

    from src.data_loader import WindowSpec, _to_tensors
    from src.real_data import EXPT4, EXPT5, TARGETS, KirkaldyStore, load_model
    from src.train import predict_pinn

    torch.set_num_threads(args.threads)
    rows = []
    for sname, root in (("expt5", EXPT5), ("expt4", EXPT4)):
        store = KirkaldyStore(STAGE1 / root)
        store_spec = WindowSpec(dsoc_min=0.15, dsoc_max=1.0, q_nominal=Q_NOM)  # n_points 128, v_noise 2 mV
        v, t, s, y, info = build_inputs(store, np.random.default_rng(args.seed))
        print(f"{sname}: {info.state_id.nunique()} check-ups, {len(info)} window pairs", flush=True)
        for seed in SEEDS:
            model, norm, spec, _ = load_model(STAGE1 / "results" / "runs_real" / f"pinn_pair_wide_seed{seed}")
            seq, scal, yt = _to_tensors(norm, v, t, s, y)
            yn, _ = predict_pinn(model, TensorDataset(seq, scal, yt), batch=512, refine_steps=8, prior_sd=3.0)
            pred = norm.y_inverse(yn)
            out = info.assign(set=sname, seed=seed)
            for j, k in enumerate(TARGETS):
                out[f"{k}_pred"] = pred[:, j]
                out[f"{k}_true"] = y[:, j]
            rows.append(out)
            print(f"  seed {seed} done", flush=True)
    OUT.mkdir(exist_ok=True)
    pd.concat(rows, ignore_index=True).to_parquet(OUT / "nmc_design_preds.parquet", index=False)


def part_crb(args):
    from multiprocessing import Pool

    from src.analysis.identifiability_real import MODELS, bound, sensitivities, simulate_family
    from src.simulation import pybamm_generator as gen

    m = MODELS["nmc"]
    fams = {}
    with Pool(args.workers, initializer=gen._init_worker_with_ref, initargs=(0.5, m["param_set"], "discharge")) as pool:
        for c in RATES:
            fams[c] = simulate_family(m["states"]["median"], (m["dn"], m["dp"]), c, pool)
    rows = []
    for d in grid_designs():
        ws = []
        for c, size, f in ((RATES[0], d["size_a"], d["fa"]), (RATES[1], d["size_b"], d["fb"])):
            fam, steps = fams[c]
            # range covered by every perturbed curve, so the finite differences exist across the whole window
            lo, hi = max(q[0] for q, _ in fam.values()), min(q[-1] for q, _ in fam.values())
            dq = size * Q_NOM
            q0 = lo + f * (hi - lo - dq - 1e-3)
            ws.append(sensitivities(fam, steps, q0, dq))
        if any(w is None for w in ws):
            continue
        b = bound(ws, sigma=0.002)
        rows.append({**d, **{f"crb_{k}": v for k, v in zip(MODES, b)}})
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_parquet(OUT / "nmc_design_crb.parquet", index=False)
    print(pd.DataFrame(rows).describe().round(3).to_string())


def relative(p: pd.DataFrame) -> pd.DataFrame:
    """Relative to the cell's first check-up: grid designs with the same design, random pairs with their mean."""
    p = p.copy()
    p["anchor"] = np.where(p.kind == "grid", p.design, "random")
    first = p[p.rpt == p.groupby(["set", "cell"]).rpt.transform("min")]
    base = first.groupby(["set", "cell", "seed", "anchor"])[[f"{k}_pred" for k in MODES]].mean()
    p = p.join(base, on=["set", "cell", "seed", "anchor"], rsuffix="_0")
    for k in MODES:
        p[f"{k}_rel"] = 1 - (1 - p[f"{k}_pred"]) / (1 - p[f"{k}_pred_0"])
    return p[p.rpt > p.groupby(["set", "cell"]).rpt.transform("min")]


def part_analyse(args):
    from scipy.stats import spearmanr

    p = relative(pd.read_parquet(OUT / "nmc_design_preds.parquet"))
    crb = pd.read_parquet(OUT / "nmc_design_crb.parquet").set_index("design")
    # seed ensemble per design and check-up
    e = p.groupby(["set", "cell", "rpt", "design", "kind"])[[f"{k}_rel" for k in MODES] + [f"{k}_true" for k in MODES]
                                                         + ["hours", "size_a"]].mean().reset_index()
    for k in MODES:
        e[f"{k}_se"] = (e[f"{k}_rel"] - e[f"{k}_true"]) ** 2
    rng = np.random.default_rng(1)
    cells = e[["set", "cell"]].drop_duplicates().to_numpy()

    def design_rmse(d):
        return d.groupby("design")[[f"{k}_se" for k in MODES]].mean().pow(0.5).mul(100)

    grid = e[e.kind == "grid"]
    rm = design_rmse(grid).join(crb[[f"crb_{k}" for k in MODES] + ["size_a"]])
    print("Grid designs: realised RMSE [pp] vs CRB [pp] (seed ensemble, both experiments, 16 cells)")
    print(rm.describe().round(2).to_string())
    for k in USED:
        r, _ = spearmanr(rm[f"crb_{k}"], rm[f"{k}_se"])
        boots = []
        for _ in range(500):
            pick = cells[rng.integers(0, len(cells), len(cells))]
            sub = pd.concat([grid[(grid.set == s) & (grid.cell == c)] for s, c in pick])
            rb = design_rmse(sub).join(crb[f"crb_{k}"])
            boots.append(spearmanr(rb[f"crb_{k}"], rb[f"{k}_se"])[0])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"\n{k}: Spearman(CRB, realised RMSE) over {len(rm)} designs = {r:.2f} [{lo:.2f}, {hi:.2f}]")
        for size in SIZES:
            sub = rm[rm.size_a == size]
            best = sub[f"crb_{k}"].idxmin()
            print(f"  size {size:.0%}: CRB-best design {best}: RMSE {sub.loc[best, f'{k}_se']:.2f} "
                  f"(grid median {sub[f'{k}_se'].median():.2f}, grid best {sub[f'{k}_se'].min():.2f}, "
                  f"worst {sub[f'{k}_se'].max():.2f})")
        rnd = e[e.kind == "random"]
        print(f"  random 35-50 % pairs: RMSE {100 * np.sqrt(rnd[f'{k}_se'].mean()):.2f}, "
              f"mean duration {rnd.hours.mean():.1f} h (grid 20 %: {0.2 / RATES[0] + 0.2 / RATES[1]:.1f} h)")
    rm.to_csv(OUT / "nmc_design_summary.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", choices=["eval", "crb", "analyse"], required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=40_000)
    args = ap.parse_args()
    {"eval": part_eval, "crb": part_crb, "analyse": part_analyse}[args.part](args)


if __name__ == "__main__":
    main()
