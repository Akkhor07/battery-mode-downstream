"""Step 6: 24 more NMC cells (Kirkaldy et al. Expt 1, 2,2 and 3; same LG M50T cell and check-up protocol).

eval    The frozen JES PINN (pinn_pair_wide, 3 seeds; encoder and 8 LM steps, prior s.d. 3) on one C/10 + one C/2
        window pair per draw, 40 draws per check-up, exactly as src.real_data does for Expt 5 and 4. Stage-1 code
        is used read-only; its cell list (A-H) is widened for Expt 1 (cells J-M). Predictions go to
        data/stage1_preds/<set>_seed<s>.parquet.
export  Adds the new cells to the check-up tables used by steps 2-5 (the 16-cell versions are kept as
        checkups_nmc16.parquet and capacity_nmc16.parquet).

    "F:/Research New/battery-pinn-degradation/.venv/Scripts/python.exe" -m src.step6_more_nmc --part eval
    "F:/Research New/battery-pinn-degradation/.venv/Scripts/python.exe" -m src.step6_more_nmc --part export
"""
from __future__ import annotations

import argparse
import shutil
import sys

import numpy as np
import pandas as pd

from src.stage1_export import KIRKALDY_NEW, MODES, OUT, SEEDS, STAGE1, kirkaldy_context

PREDS = OUT / "stage1_preds"
CELLS = {"expt1": "ABDEFJKLM", "expt2": "ABCDEF", "expt3": "ABCDEFGHI"}


def part_eval(args):
    import torch

    import src
    # both projects call their package 'src'; let this one also find the Stage-1 modules (no names overlap)
    src.__path__.append(str(STAGE1 / "src"))
    sys.path.insert(0, str(STAGE1))
    from src.data_loader import FixedPairDataset
    from src.real_data import TARGETS, KirkaldyStore, load_model, relative_to_first
    import src.real_data as rd
    from src.train import PINN_EXTRA, predict_pinn

    torch.set_num_threads(args.threads)
    rd.CELLS = "ABCDEFGHIJKLM"  # state ids are built from the position in this list
    PREDS.mkdir(parents=True, exist_ok=True)
    for sname, folder in KIRKALDY_NEW.items():
        store = KirkaldyStore(STAGE1 / "data" / "kirkaldy" / folder, cells=CELLS[sname])
        both = store.meta.groupby("state_id")["C_rate"].transform("nunique").eq(2)
        for seed in SEEDS:
            out = PREDS / f"{sname}_seed{seed}.parquet"
            if out.exists():
                continue
            model, norm, spec, targs = load_model(STAGE1 / "results" / "runs_real" / f"pinn_pair_wide_seed{seed}")
            ds = FixedPairDataset(store, sorted(store.meta.loc[both, "sim_id"]), spec, norm, 40, seed=30_000,
                                  mode=targs["pair_mode"])
            meta = store.meta.set_index("sim_id")
            pred = ds.info.copy()
            pred[["cell", "rpt", "temp_group"]] = meta.loc[pred["sim_id"], ["cell", "rpt", "temp_group"]].to_numpy()
            for j, k in enumerate(TARGETS):
                pred[k + "_true"] = norm.y_inverse(ds.y.numpy())[:, j]
            for tag, steps in (("", 0), ("_refined", 8)):
                yn, lat = predict_pinn(model, ds, batch=args.batch, refine_steps=steps, prior_sd=3.0)
                for k in PINN_EXTRA:
                    pred[k + "_pred" + tag] = lat[k]
                for j, k in enumerate(TARGETS):
                    pred[k + "_pred" + tag] = norm.y_inverse(yn)[:, j]
                pred = relative_to_first(pred, suffix="_pred" + tag)
            pred.to_parquet(out, index=False)
            print(f"{sname} seed {seed}: {pred.state_id.nunique()} check-ups, {len(pred)} pairs", flush=True)


def part_export(args):
    rows = []
    for sname in KIRKALDY_NEW:
        per_seed = []
        for seed in SEEDS:
            p = pd.read_parquet(PREDS / f"{sname}_seed{seed}.parquet")
            cols = {}
            for est, tag in (("pinn_enc", ""), ("pinn_lm", "_refined")):
                cols.update({f"{m}_pred{tag}_rel": f"{m}_{est}" for m in MODES})
                cols.update({f"Dn_scale_pred{tag}": f"logDn_{est}", f"Dp_scale_pred{tag}": f"logDp_{est}"})
            q = p.rename(columns=cols)
            for c in [c for c in cols.values() if c.startswith("log")]:
                q[c] = np.log(q[c])
            per_seed.append(q.groupby(["cell", "rpt"])[list(cols.values())].mean())
            if seed == SEEDS[0]:
                ref = p.groupby(["cell", "rpt"])[[f"{m}_true" for m in MODES] + ["temp_group"]].first()
        stack = pd.concat(per_seed, keys=SEEDS)
        t = ref.rename(columns={f"{m}_true": f"{m}_ref" for m in MODES}).join(stack.groupby(level=["cell", "rpt"]).mean())
        rows.append(t.reset_index().assign(set=sname))
    ctx = kirkaldy_context(KIRKALDY_NEW)
    new = pd.concat(rows, ignore_index=True).merge(ctx.drop(columns=[f"{m}_ref" for m in MODES]), on=["set", "cell", "rpt"])
    for name, extra in (("checkups_nmc", new), ("capacity_nmc", ctx)):
        keep = OUT / f"{name}16.parquet"
        if not keep.exists():
            shutil.copy(OUT / f"{name}.parquet", keep)
        old = pd.read_parquet(keep)
        pd.concat([old, extra], ignore_index=True).to_parquet(OUT / f"{name}.parquet", index=False)
    print(f"added {new.groupby('set').cell.nunique().to_dict()} cells, {len(new)} check-ups with estimates")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", choices=["eval", "export"], required=True)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    {"eval": part_eval, "export": part_export}[args.part](args)


if __name__ == "__main__":
    main()
