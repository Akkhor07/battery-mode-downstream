"""Stage-1 check-up tables for Gap 1: per cell and check-up, the degradation modes estimated by the frozen
estimators of the JES paper (battery-pinn-degradation, three seeds), their uncertainty, the reference labels and
the ageing context (cycles, charge throughput, capacity).

Nothing is retrained: the per-window-pair predictions saved by the Stage-1 runs are averaged per check-up and over
seeds. Run from this project's root with the Stage-1 environment:

    "F:/Research New/battery-pinn-degradation/.venv/Scripts/python.exe" -m src.stage1_export

Outputs (``data/``):
    checkups_nmc.parquet   16 Kirkaldy cells, check-ups with both C/10 and C/2 discharges (8 in Expt 5, 5 in Expt 4)
    checkups_lfp.parquet   20 Wheeler cells, 26-66 check-ups each
    capacity_nmc.parquet   C/10 capacity and published modes at every Kirkaldy RPT (16 / 10), for capacity baselines
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

STAGE1 = Path(os.environ.get("STAGE1_ROOT", r"F:\Research New\battery-pinn-degradation"))
OUT = Path("data")
SEEDS = (0, 1, 2)
MODES = ("LLI", "LAM_PE", "LAM_NE")
# estimator -> (run prefix, prediction-column tag); the JES decision rule uses pinn_lm on NMC and pinn_enc on LFP
MODELS = {"pinn_enc": ("pinn_pair_wide", ""), "pinn_lm": ("pinn_pair_wide", "_refined"),
          "cnn": ("cnn_lstm_pair_wide", ""), "cnn_dr": ("cnn_lstm_pair_wide_dr", "")}
SETS = {"expt5": ("runs_real", ""), "expt4": ("runs_real", "_expt4"), "lfp": ("runs_lfp", "")}
KIRKALDY = {"expt5": "Expt 5 - Standard Cycle Aging (Control)", "expt4": "Expt 4 - Drive Cycle Aging (Control)"}
# added in step 6 (estimates from src.step6_more_nmc, not from Stage-1 run files)
KIRKALDY_NEW = {"expt1": "Expt 1 - Si-based Degradation", "expt2": "Expt 2,2 - C-based Degradation 2",
                "expt3": "Expt 3 - Cathode Degradation and Li-Plating"}
KIRKALDY_ALL = {**KIRKALDY, **KIRKALDY_NEW}


def checkup_means(sname: str) -> pd.DataFrame:
    """Seed-mean (and seed s.d.) of the per-check-up mean prediction of every estimator, plus the reference."""
    root, suf = SETS[sname]
    key = ["cell", "rpt"]
    out = None
    for model, (prefix, tag) in MODELS.items():
        per_seed = []
        for s in SEEDS:
            p = pd.read_parquet(STAGE1 / "results" / root / f"{prefix}_seed{s}" / f"real_predictions{suf}.parquet")
            cols = {f"{m}_pred{tag}_rel": f"{m}_{model}" for m in MODES}
            if model.startswith("pinn"):  # diffusivity scalings (nuisance in the JES paper, never validated)
                cols.update({f"Dn_scale_pred{tag}": f"logDn_{model}", f"Dp_scale_pred{tag}": f"logDp_{model}"})
            q = p.rename(columns=cols)
            for c in (f"logDn_{model}", f"logDp_{model}"):
                if c in q:
                    q[c] = np.log(q[c])
            g = q.groupby(key)[list(cols.values())].mean()
            if out is None and s == SEEDS[0]:
                extra = [c for c in ("temp_group", "protocol") if c in p]
                ref = p.groupby(key)[[f"{m}_true" for m in MODES] + extra].first()
                out = ref.rename(columns={f"{m}_true": f"{m}_ref" for m in MODES})
            per_seed.append(g)
        stack = pd.concat(per_seed, keys=SEEDS)
        out = out.join(stack.groupby(level=key).mean())
        out = out.join(stack[[f"{m}_{model}" for m in MODES]].groupby(level=key).std(ddof=0).add_suffix("_seedsd"))
    return out.reset_index()


def laplace_sd(chem: str) -> pd.DataFrame:
    """Per check-up Laplace+ensemble s.d. of the relative modes (refined PINN), raw and calibrated on the other
    labelled group, as in Section 3.6 of the JES paper (NMC: Expt 4 <-> 5; LFP: protocols 1-3 <-> 4-7)."""
    name = "real" if chem == "nmc" else "lfp"
    u = pd.read_parquet(STAGE1 / "results" / f"uncertainty_{name}.pairs.parquet")
    cal = json.load(open(STAGE1 / "results" / f"uncertainty_{name}.json"))["calibration"]
    other = {"expt5": "expt4", "expt4": "expt5"}
    for m in MODES:
        ab = u["set"].map(lambda s: (cal[other[s]][m]["a"], cal[other[s]][m]["b_pp"] / 100))
        a, b = np.array([x[0] for x in ab]), np.array([x[1] for x in ab])
        u[f"{m}_sdcal"] = np.sqrt((a * u[f"{m}_sd"]) ** 2 + b ** 2)
    cols = [f"{m}_sd" for m in MODES] + [f"{m}_sdcal" for m in MODES]
    # set = experiment (NMC) or protocol group (LFP); cell letters repeat across the two NMC experiments
    g = u.groupby(["set", "cell", "rpt"])[cols].apply(lambda d: np.sqrt((d ** 2).mean())).reset_index()
    return g.rename(columns={f"{m}_sd": f"{m}_sdraw" for m in MODES})


def kirkaldy_context(sets: dict[str, str] = KIRKALDY) -> pd.DataFrame:
    rows = []
    for sname, folder in sets.items():
        for f in sorted(glob.glob(str(STAGE1 / "data" / "kirkaldy" / folder / "Summary Data" / "Performance Summary" / "*.csv"))):
            d = pd.read_csv(f)
            cell = Path(f).name.split(" cell ")[1][0]
            q = d["C/10 Capacity [mA h]"].to_numpy() / 1000
            rows.append(pd.DataFrame({
                "set": sname, "cell": cell, "rpt": np.arange(len(d)), "cycles": d["Ageing Cycles"].to_numpy(),
                "throughput_Ah": d["Charge Throughput [A h]"].to_numpy(), "Q_Ah": q, "SOH": q / q[0],
                **{f"{m}_ref": d[c].to_numpy() for m, c in (("LLI", "LLI"), ("LAM_PE", "LAM PE"), ("LAM_NE", "LAM NE_tot"))}}))
    return pd.concat(rows, ignore_index=True)


def main():
    OUT.mkdir(exist_ok=True)
    ctx = kirkaldy_context()
    ctx.to_parquet(OUT / "capacity_nmc.parquet", index=False)

    nmc = []
    for sname in ("expt5", "expt4"):
        t = checkup_means(sname).assign(set=sname)
        # the cell letters repeat across experiments; keep them apart
        t = t.merge(ctx[ctx.set == sname].drop(columns=[f"{m}_ref" for m in MODES]), on=["set", "cell", "rpt"])
        nmc.append(t)
    nmc = pd.concat(nmc, ignore_index=True)
    nmc = nmc.merge(laplace_sd("nmc"), on=["set", "cell", "rpt"], how="left")
    nmc["uid"] = nmc["set"] + "_" + nmc["cell"]
    nmc.to_parquet(OUT / "checkups_nmc.parquet", index=False)

    lfp = checkup_means("lfp")
    dva = pd.read_csv(STAGE1 / "results" / "lfp_dva_lam_ne.csv")[["cell", "rpt", "LAM_NE_dva"]]
    sd = laplace_sd("lfp").rename(columns={"set": "cal_group"})
    lfp = lfp.merge(dva, on=["cell", "rpt"], how="left").merge(sd, on=["cell", "rpt"], how="left")
    # check-ups every 100 cycles (rpt = cycle // 100, src/real_lfp.py); capacity from the C/25 discharge
    lfp["cycles"] = 100 * lfp["rpt"]
    disc = STAGE1 / "data" / "wheeler_lfp" / "discharges"
    lfp["Q_Ah"] = [float(np.load(disc / f"cell{c}_c{n:04d}.npz")["q_25"][-1]) for c, n in zip(lfp.cell, lfp.cycles)]
    lfp = lfp.sort_values(["cell", "rpt"])
    lfp["SOH"] = lfp["Q_Ah"] / lfp.groupby("cell")["Q_Ah"].transform("first")
    lfp["uid"] = "lfp_" + lfp["cell"]
    lfp.to_parquet(OUT / "checkups_lfp.parquet", index=False)

    for name, t in (("NMC", nmc), ("LFP", lfp)):
        n = t.groupby("uid").size()
        print(f"{name}: {t.uid.nunique()} cells, {len(t)} check-ups ({n.min()}-{n.max()} per cell); "
              f"missing cycles {t.cycles.isna().sum()}, missing calibrated sd {t.LLI_sdcal.isna().sum()}")
    print("NMC capacity table:", len(ctx), "RPTs")


if __name__ == "__main__":
    main()
