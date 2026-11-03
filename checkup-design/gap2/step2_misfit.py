"""Gap 2, step 2: can a label-free signal pick check-up windows on real cells, where the CRB fails?

real  For every NMC check-up and design of step 1 (same windows, same seed), the frozen PINN + LM (3 seeds,
      double precision as in src.analysis.model_error) gives the estimate and the voltage RMS misfit of the refined
      decoder fit on each window (and at the encoder output).
sim   Control on the simulated test set of the Stage-1 run (150 ageing states, true modes known, model correct up
      to its approximation): same 50 grid designs plus random 35-50 % pairs, seed 0.
analyse
      Design level: Spearman of realised error with the CRB and with the misfit (real and simulated).
      Procedure: each cell's first check-up (a full characterisation at beginning of life) gives the misfit of every
      grid design; the lowest-misfit design (per cell, or pooled over the other cells) is then used for all later
      check-ups. Compared with the CRB-best design, the median design, random 35-50 % pairs and the oracle.

    py = "F:/Research New/battery-pinn-degradation/.venv/Scripts/python.exe"
    $py -m gap2.step2_misfit --part real      # ~25 min CPU
    $py -m gap2.step2_misfit --part sim       # ~10 min CPU
    $py -m gap2.step2_misfit --part analyse
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from gap2 import step1_design_grid as s1
from gap2.step1_design_grid import MODES, OUT, Q_NOM, SEEDS, STAGE1, USED


def evaluate(model, norm, v, t, s, y, steps=8, prior_sd=3.0, batch=200):
    """Refined estimates (physical units) and per-window voltage RMS [mV] after LM and at the encoder output."""
    import torch

    from src.data_loader import _to_tensors
    from src.models.pinn import _refine_lm

    seq, scal, _ = _to_tensors(norm, v, t, s, y)
    preds, rms, rms0 = [], [], []
    for i in range(0, len(seq), batch):
        sq, sc = seq[i:i + batch].double(), scal[i:i + batch].double()
        with torch.no_grad():
            vv, temp, dq, c_rate = model.raw_inputs(sq, sc)
            z0 = model.encode(sq, sc)
            sig = torch.exp(0.5 * (model.log_var["voltage"] + model.log_var_v_slope * torch.log2(c_rate)))
            zr = _refine_lm(model, z0, vv, temp, dq, c_rate, sig, prior_sd, steps)
            for z, store in ((zr, rms), (z0, rms0)):
                vh = model.reconstruct(model.physical(z, dq), dq, c_rate, temp)["v"]
                store.append((1000 * ((vh - vv) ** 2).mean(-1).sqrt()).numpy())
            preds.append(model.labels_normalised(model.physical(zr, dq)).numpy())
    return norm.y_inverse(np.concatenate(preds)), np.concatenate(rms), np.concatenate(rms0)


def run(store, runs, tag, args):
    import torch

    from src.data_loader import WindowSpec
    from src.real_data import TARGETS, load_model

    torch.set_num_threads(args.threads)
    s1.store_spec = WindowSpec(dsoc_min=0.15, dsoc_max=1.0, q_nominal=Q_NOM)
    v, t, s, y, info = s1.build_inputs(store, np.random.default_rng(args.seed))
    print(f"{tag}: {info.state_id.nunique()} states, {len(info)} window pairs", flush=True)
    rows = []
    for seed, run_dir in runs:
        model, norm, _, _ = load_model(run_dir)
        pred, rms, rms0 = evaluate(model.double(), norm, v, t, s, y, batch=args.batch)
        out = info.assign(set=tag, seed=seed, rms_a_mV=rms[:, 0], rms_b_mV=rms[:, 1], rms0_a_mV=rms0[:, 0],
                          rms0_b_mV=rms0[:, 1])
        for j, k in enumerate(TARGETS):
            out[f"{k}_pred"], out[f"{k}_true"] = pred[:, j], y[:, j]
        rows.append(out)
        print(f"  seed {seed} done", flush=True)
    return pd.concat(rows, ignore_index=True)


def part_real(args):
    from src.real_data import EXPT4, EXPT5, KirkaldyStore

    runs = [(sd, STAGE1 / "results" / "runs_real" / f"pinn_pair_wide_seed{sd}") for sd in SEEDS]
    df = pd.concat([run(KirkaldyStore(STAGE1 / root), runs, name, args)
                    for name, root in (("expt5", EXPT5), ("expt4", EXPT4))], ignore_index=True)
    df.to_parquet(OUT / "nmc_design_misfit.parquet", index=False)


CONFIRM = {"expt1": ("Expt 1 - Si-based Degradation", "ABDEFJKLM"),
           "expt2": ("Expt 2,2 - C-based Degradation 2", "ABCDEF"),
           "expt3": ("Expt 3 - Cathode Degradation and Li-Plating", "ABCDEFGHI")}


def part_confirm(args):
    """Pre-specified confirmation (PLAN.md, step 3) on the 24 cells of Expt 1, 2,2 and 3."""
    import src.real_data as rd
    from src.real_data import KirkaldyStore

    rd.CELLS = "ABCDEFGHIJKLM"  # Expt 1 has cells J-M; state ids come from the position in this list
    runs = [(sd, STAGE1 / "results" / "runs_real" / f"pinn_pair_wide_seed{sd}") for sd in SEEDS]
    df = pd.concat([run(KirkaldyStore(STAGE1 / "data" / "kirkaldy" / folder, cells=cells), runs, name, args)
                    for name, (folder, cells) in CONFIRM.items()], ignore_index=True)
    df.to_parquet(OUT / "nmc_design_misfit_confirm.parquet", index=False)


def part_confirm_analyse(args, size=0.3, margin=0.10, n_boot=4000):
    from scipy.stats import spearmanr

    raw = pd.read_parquet(OUT / "nmc_design_misfit_confirm.parquet")
    raw["misfit"] = raw[["rms_a_mV", "rms_b_mV"]].mean(axis=1)
    p = s1.relative(raw)
    keys = ["set", "cell", "rpt", "design", "kind", "size_a", "hours"]
    e = p.groupby(keys)[[f"{k}_rel" for k in USED] + [f"{k}_true" for k in USED]].mean().reset_index()
    e["se"] = sum((e[f"{k}_rel"] - e[f"{k}_true"]) ** 2 for k in USED) / len(USED)
    e["cid"] = e["set"] + e["cell"]
    # the rule, exactly as pre-specified
    first = raw[(raw.rpt == raw.groupby(["set", "cell"]).rpt.transform("min")) & raw.design.str.startswith(f"s{size:.1f}_")]
    first = first.groupby(["set", "cell", "design"]).misfit.mean().reset_index()
    choice = first.loc[first.groupby(["set", "cell"]).misfit.idxmin()].assign(cid=lambda d: d.set + d.cell)
    choice = choice.set_index("cid").design
    grid = e[e.kind == "grid"]
    rule = grid[grid.design == grid.cid.map(choice)].groupby("cid").se.mean()
    rand = e[e.kind == "random"].groupby("cid").se.mean()
    cells = rule.index.intersection(rand.index)
    rule, rand = rule.loc[cells], rand.loc[cells]
    sets = cells.str[:5]
    rmse = lambda s: 100 * np.sqrt(s.mean())  # noqa: E731
    rng = np.random.default_rng(3)
    groups = {g: np.flatnonzero(sets == g) for g in np.unique(sets)}
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(groups[g], len(groups[g])) for g in rng.choice(list(groups), len(groups))])
        diffs[i] = rmse(rule.iloc[idx]) - rmse(rand.iloc[idx])
    upper, bound = np.percentile(diffs, 95), margin * rmse(rand)
    hours_rule = grid[grid.design.isin(choice)].hours.mean()
    print(f"CONFIRMATION (pre-specified), {len(cells)} cells, {e[e.kind == 'random'].groupby('cid').rpt.nunique().sum()} "
          f"later check-ups")
    print(f"  rule: misfit-chosen {size:.0%} design   RMSE {rmse(rule):.2f} pp, {hours_rule:.1f} h")
    print(f"  random 35-50 % pairs              RMSE {rmse(rand):.2f} pp, {e[e.kind == 'random'].hours.mean():.1f} h")
    print(f"  difference {rmse(rule) - rmse(rand):+.2f} pp; one-sided 95 % upper bound {upper:+.2f} pp; "
          f"margin {bound:.2f} pp -> {'NON-INFERIOR (confirmed)' if upper <= bound else 'NOT confirmed'}")
    for g in np.unique(sets):
        m = sets == g
        print(f"    {g}: rule {rmse(rule[m]):.2f}  random {rmse(rand[m]):.2f}  ({m.sum()} cells)")
    # secondary, descriptive
    d = grid.groupby("design").agg(rmse=("se", lambda x: 100 * np.sqrt(x.mean())), size=("size_a", "first"))
    d["misfit"] = raw[raw.kind == "grid"].groupby("design").misfit.median()
    for s_ in (0.2, 0.3):
        sub = d[d["size"].round(1) == s_]
        print(f"  secondary: size {s_:.0%} Spearman(misfit, RMSE) {spearmanr(sub.misfit, sub.rmse)[0]:+.2f}; "
              f"median design {sub.rmse.median():.2f}, oracle {sub.rmse.min():.2f} pp")


def part_fixed(args, size=0.3, margin=0.10, n_boot=4000):
    """Step 4 (PLAN.md): the rule against one fixed design calibrated on the 16 labelled development cells."""
    def errors(raw):
        p = s1.relative(raw)
        keys = ["set", "cell", "rpt", "design", "kind", "size_a", "hours"]
        e = p.groupby(keys)[[f"{k}_rel" for k in USED] + [f"{k}_true" for k in USED]].mean().reset_index()
        e["se"] = sum((e[f"{k}_rel"] - e[f"{k}_true"]) ** 2 for k in USED) / len(USED)
        e["cid"] = e["set"] + e["cell"]
        return e

    dev = errors(pd.read_parquet(OUT / "nmc_design_misfit.parquet"))
    dd = dev[dev.kind == "grid"].groupby("design").agg(se=("se", "mean"), size=("size_a", "first"))
    fixed = {"F30 (best 30 % on dev cells)": dd[dd["size"].round(1) == size].se.idxmin(),
             "Fany (best any size on dev cells)": dd.se.idxmin()}

    raw = pd.read_parquet(OUT / "nmc_design_misfit_confirm.parquet")
    raw["misfit"] = raw[["rms_a_mV", "rms_b_mV"]].mean(axis=1)
    e = errors(raw)
    first = raw[(raw.rpt == raw.groupby(["set", "cell"]).rpt.transform("min")) & raw.design.str.startswith(f"s{size:.1f}_")]
    first = first.groupby(["set", "cell", "design"]).misfit.mean().reset_index()
    choice = first.loc[first.groupby(["set", "cell"]).misfit.idxmin()].assign(cid=lambda d: d.set + d.cell)
    choice = choice.set_index("cid").design
    grid = e[e.kind == "grid"]
    rule = grid[grid.design == grid.cid.map(choice)].groupby("cid").se.mean()
    sets = rule.index.str[:5]
    groups = {g: np.flatnonzero(sets == g) for g in np.unique(sets)}
    rmse = lambda s: 100 * np.sqrt(s.mean())  # noqa: E731
    hours = grid.groupby("design").hours.first()
    print(f"STEP 4, best fixed design (chosen on 16 dev cells) vs the misfit rule, {len(rule)} confirmation cells")
    print(f"  rule (misfit, {size:.0%})                    RMSE {rmse(rule):.2f} pp, {hours.loc[choice].mean():.1f} h")
    for name, dsg in fixed.items():
        comp = grid[grid.design == dsg].groupby("cid").se.mean().loc[rule.index]
        rng = np.random.default_rng(3)
        diffs = np.empty(n_boot)
        for i in range(n_boot):
            idx = np.concatenate([rng.choice(groups[g], len(groups[g])) for g in rng.choice(list(groups), len(groups))])
            diffs[i] = rmse(rule.iloc[idx]) - rmse(comp.iloc[idx])
        upper, bound = np.percentile(diffs, 95), margin * rmse(comp)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        print(f"  {name:36s} {dsg}: RMSE {rmse(comp):.2f} pp, {hours.loc[dsg]:.1f} h "
              f"(dev-cell RMSE {100 * np.sqrt(dd.se.loc[dsg]):.2f})")
        print(f"      rule - fixed {rmse(rule) - rmse(comp):+.2f} pp, 95 % CI [{lo:+.2f}, {hi:+.2f}], one-sided upper "
              f"{upper:+.2f} vs margin {bound:.2f} -> {'NON-INFERIOR' if upper <= bound else 'not shown non-inferior'}")
        for g in np.unique(sets):
            m = sets == g
            print(f"        {g}: rule {rmse(rule[m]):.2f}  fixed {rmse(comp[m]):.2f}")
    cd = grid.groupby("design").se.mean()
    cd30 = cd[cd.index.str.startswith(f"s{size:.1f}_")]
    print(f"  descriptive: oracle fixed design on the confirmation cells {cd.idxmin()} {100 * np.sqrt(cd.min()):.2f} pp; "
          f"oracle 30 % {cd30.idxmin()} {100 * np.sqrt(cd30.min()):.2f} pp; dev-chosen rank among 30 % designs here: "
          f"{int(cd30.rank().loc[fixed['F30 (best 30 % on dev cells)']])}/{len(cd30)}")


def part_sim(args):
    from src.data_loader import SyntheticStore
    from src.real_data import load_model

    run_dir = STAGE1 / "results" / "runs_real" / "pinn_pair_wide_seed0"
    targs = load_model(run_dir)[3]
    store = SyntheticStore(STAGE1 / targs["data"])
    test = store.split_states(seed=targs["split_seed"])["test"]
    store.meta = store.meta[store.meta.sim_id.isin(test)]
    store.meta = store.meta[store.meta.C_rate.round(2).isin(s1.RATES)]
    df = run(store, [(0, run_dir)], "sim", args)
    df.to_parquet(OUT / "sim_design_eval.parquet", index=False)


def boot_spearman(e, x, y, groups, n=500, seed=1):
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    g = e[groups].drop_duplicates().to_numpy()
    out = []
    for _ in range(n):
        pick = pd.DataFrame(g[rng.integers(0, len(g), len(g))], columns=groups)
        sub = pick.merge(e, on=groups)
        d = sub.groupby("design")[[x, y]].mean()
        out.append(spearmanr(d[x], d[y])[0])
    return np.percentile(out, [2.5, 97.5])


def part_analyse(args):
    from scipy.stats import spearmanr

    crb = pd.read_parquet(OUT / "nmc_design_crb.parquet").set_index("design")
    crb["crb_used"] = np.sqrt((crb[[f"crb_{k}" for k in USED]] ** 2).mean(1))

    # ---- simulated control: absolute estimates against the true modes
    sim = pd.read_parquet(OUT / "sim_design_eval.parquet")
    sim["misfit"] = sim[["rms_a_mV", "rms_b_mV"]].mean(1)
    for k in USED:
        sim[f"{k}_se"] = (sim[f"{k}_pred"] - sim[f"{k}_true"]) ** 2
    sim["se"] = sim[[f"{k}_se" for k in USED]].mean(1)
    sg = sim[sim.kind == "grid"].join(crb["crb_used"], on="design")
    d = sg.groupby("design").agg(rmse=("se", lambda x: 100 * np.sqrt(x.mean())), crb=("crb_used", "first"),
                                 misfit=("misfit", "median"))
    print("SIMULATED test set (150 states, seed 0): design-level Spearman with RMSE of LLI/LAM_PE")
    print(f"  CRB    {spearmanr(d.crb, d.rmse)[0]:+.2f}   misfit {spearmanr(d.misfit, d.rmse)[0]:+.2f}   "
          f"(RMSE across designs {d.rmse.min():.2f}-{d.rmse.max():.2f} pp; random 35-50 %: "
          f"{100 * np.sqrt(sim[sim.kind == 'random'].se.mean()):.2f} pp)")

    # ---- real NMC: relative estimates, seed ensemble
    p = s1.relative(pd.read_parquet(OUT / "nmc_design_misfit.parquet"))
    p["misfit"] = p[["rms_a_mV", "rms_b_mV"]].mean(1)
    keys = ["set", "cell", "rpt", "design", "kind", "size_a", "hours"]
    e = p.groupby(keys)[[f"{k}_rel" for k in USED] + [f"{k}_true" for k in USED] + ["misfit"]].mean().reset_index()
    e["se"] = sum((e[f"{k}_rel"] - e[f"{k}_true"]) ** 2 for k in USED) / len(USED)
    g = e[e.kind == "grid"].join(crb["crb_used"], on="design")
    d = g.groupby("design").agg(rmse=("se", lambda x: 100 * np.sqrt(x.mean())), crb=("crb_used", "first"),
                                misfit=("misfit", "median"), size=("size_a", "first"))
    print("\nREAL NMC (104 check-ups, 16 cells): design-level Spearman with RMSE of LLI/LAM_PE [cell bootstrap]")
    g["cid"] = g["set"] + g["cell"]
    for x in ("crb", "misfit"):
        col = "crb_used" if x == "crb" else "misfit"
        lo, hi = boot_spearman(g.assign(**{x: g[col]}), x, "se", ["cid"])
        print(f"  {x:7s} {spearmanr(d[x], d.rmse)[0]:+.2f} [{lo:+.2f}, {hi:+.2f}]")

    # ---- procedure: choose the design from the first (beginning-of-life) check-up's misfit, score later check-ups
    first = p[p.rpt == p.groupby(["set", "cell"]).rpt.transform("min")]
    first = first[first.kind == "grid"].groupby(["set", "cell", "design"]).misfit.mean().reset_index()
    first = first.join(crb["crb_used"], on="design")
    later = e  # already restricted to check-ups after the first by s1.relative
    later_grid = later[later.kind == "grid"].set_index(["set", "cell", "design"]).se
    rnd = later[later.kind == "random"].groupby(["set", "cell"]).se.mean()

    def score(choice):  # choice: Series (set, cell) -> design
        return pd.Series({sc: later_grid.loc[sc + (dsg,)].mean() for sc, dsg in choice.items()})

    cells = first[["set", "cell"]].drop_duplicates().apply(tuple, axis=1).tolist()
    strategies = {
        "misfit, own first check-up": first.loc[first.groupby(["set", "cell"]).misfit.idxmin()].set_index(["set", "cell"]).design,
        "misfit, other cells' first check-ups": pd.Series({
            c: first[(first.set + first.cell) != c[0] + c[1]].groupby("design").misfit.median().idxmin() for c in cells}),
        "CRB-best": pd.Series({c: crb.crb_used.idxmin() for c in cells}),
        "median design": pd.Series({c: d.rmse.sub(d.rmse.median()).abs().idxmin() for c in cells}),
        "oracle (best on labels)": pd.Series({c: d.rmse.idxmin() for c in cells}),
    }
    # the misfit favours short windows (they fit better but estimate worse), so also choose within one size
    for size in s1.SIZES:
        f = first[first.design.str.startswith(f"s{size:.1f}_")]
        ds = d[d["size"] == size].rmse
        strategies[f"misfit, own first, {size:.0%} only"] = f.loc[f.groupby(["set", "cell"]).misfit.idxmin()].set_index(
            ["set", "cell"]).design
        strategies[f"median {size:.0%} design"] = pd.Series({c: ds.sub(ds.median()).abs().idxmin() for c in cells})
    print("\nPROCEDURE on real NMC: RMSE [pp] of LLI/LAM_PE over later check-ups (16 cells; 95 % cell bootstrap)")
    rng = np.random.default_rng(2)
    base = rnd.loc[cells]
    for name, ch in strategies.items():
        sc = score(ch).loc[cells]
        idx = rng.integers(0, len(cells), (2000, len(cells)))
        diff = 100 * (np.sqrt(sc.values[idx].mean(1)) - np.sqrt(base.values[idx].mean(1)))
        lo, hi = np.percentile(diff, [2.5, 97.5])
        hours = d.loc[ch.loc[cells].values].index.map(lambda x: e[e.design == x].hours.iloc[0]).to_numpy().mean()
        print(f"  {name:38s} {100 * np.sqrt(sc.mean()):5.2f} pp, {hours:.1f} h  (vs random large {lo:+.2f}..{hi:+.2f})")
    print(f"  {'random 35-50 % pairs':38s} {100 * np.sqrt(base.mean()):5.2f} pp, "
          f"{later[later.kind == 'random'].hours.mean():.1f} h")
    d.to_csv(OUT / "nmc_design_misfit_summary.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", choices=["real", "sim", "analyse", "confirm", "confirm-analyse", "fixed"], required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=40_000)
    ap.add_argument("--batch", type=int, default=200, help="LM batch size (lower = less memory)")
    args = ap.parse_args()
    {"real": part_real, "sim": part_sim, "analyse": part_analyse, "confirm": part_confirm,
     "confirm-analyse": part_confirm_analyse, "fixed": part_fixed}[args.part](args)


if __name__ == "__main__":
    main()
