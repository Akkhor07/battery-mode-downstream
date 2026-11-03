"""Gap 2, step 5 (PLAN.md): does the window-design conclusion survive reference-mode errors of the size step 11
(Dubarry ground truth) found? The references are perturbed with each error pattern and the step-3/step-4
comparisons are redone. The rule itself is label-free (misfit only), so its choice of design does not change.

    "F:/Research New/battery-pinn-degradation/.venv/Scripts/python.exe" -m gap2.step5_sensitivity
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gap2 import step1_design_grid as s1
from gap2.step1_design_grid import OUT, USED

SCENARIOS = ("N1", "N2", "SC", "LK", "KD", "ALL")
N_DRAWS, N_BOOT, MARGIN, SIZE = 200, 500, 0.10, 0.3


def errors(raw):
    p = s1.relative(raw)
    keys = ["set", "cell", "rpt", "design", "kind", "size_a"]
    e = p.groupby(keys)[[f"{k}_rel" for k in USED] + [f"{k}_true" for k in USED]].mean().reset_index()
    e["cid"] = e["set"] + e["cell"]
    return e


def delta(e: pd.DataFrame, scen: str, rng) -> np.ndarray:
    """Reference error (n_rows x 2: LLI, LAM_PE), constant over designs of one check-up."""
    ck = e[["cid", "rpt"]].drop_duplicates().reset_index(drop=True)
    ck["prog"] = ck.rpt / ck.groupby("cid").rpt.transform("max")
    cells = ck.cid.unique()
    tru = e.drop_duplicates(["cid", "rpt"]).set_index(["cid", "rpt"])[[f"{k}_true" for k in USED]]
    t = tru.loc[list(zip(ck.cid, ck.rpt))].to_numpy()
    d = np.zeros((len(ck), 2))
    if scen in ("N1", "ALL"):
        d += rng.normal(0, 0.01, d.shape)
    if scen == "N2":
        d += rng.normal(0, 0.02, d.shape)
    if scen in ("SC", "ALL"):
        a = dict(zip(cells, rng.uniform(-0.2, 0.2, (len(cells), 2))))
        d += t * np.stack([a[c] for c in ck.cid])
    if scen == "LK":
        d[:, 1] += 0.6 * t[:, 0]
    if scen in ("KD", "ALL"):
        dd = dict(zip(cells, rng.uniform(0, 0.05, len(cells))))
        d += (np.array([dd[c] for c in ck.cid]) * ck.prog.to_numpy())[:, None]
    key = pd.MultiIndex.from_frame(ck[["cid", "rpt"]])
    return d[key.get_indexer(pd.MultiIndex.from_frame(e[["cid", "rpt"]]))]


def se(e, d):
    return sum((e[f"{k}_rel"].to_numpy() - e[f"{k}_true"].to_numpy() - d[:, i]) ** 2 for i, k in enumerate(USED)) / len(USED)


def main():
    dev = errors(pd.read_parquet(OUT / "nmc_design_misfit.parquet"))
    raw = pd.read_parquet(OUT / "nmc_design_misfit_confirm.parquet")
    raw["misfit"] = raw[["rms_a_mV", "rms_b_mV"]].mean(axis=1)
    e = errors(raw)
    first = raw[(raw.rpt == raw.groupby(["set", "cell"]).rpt.transform("min")) & raw.design.str.startswith(f"s{SIZE:.1f}_")]
    first = first.groupby(["set", "cell", "design"]).misfit.mean().reset_index()
    choice = first.loc[first.groupby(["set", "cell"]).misfit.idxmin()].assign(cid=lambda d: d.set + d.cell)
    choice = choice.set_index("cid").design
    is_rule = ((e.kind == "grid") & (e.design == e.cid.map(choice))).to_numpy()
    is_rand = (e.kind == "random").to_numpy()
    cells = np.array(sorted(choice.index))
    sets = np.array([c[:5] for c in cells])
    groups = [np.flatnonzero(sets == g) for g in np.unique(sets)]
    rng0 = np.random.default_rng(5)
    boots = [np.concatenate([rng0.choice(groups[g], len(groups[g])) for g in rng0.integers(0, len(groups), len(groups))])
             for _ in range(N_BOOT)]
    dev_grid = (dev.kind == "grid").to_numpy()
    dev30 = dev.size_a.round(1).to_numpy() == SIZE

    def per_cell(s, mask, extra=None):
        m = mask if extra is None else mask & extra
        return pd.Series(s[m]).groupby(e.cid.to_numpy()[m]).mean().reindex(cells).to_numpy()

    def compare(a, b):
        rm = lambda x: 100 * np.sqrt(x.mean())  # noqa: E731
        diffs = np.array([rm(a[i]) - rm(b[i]) for i in boots])
        return rm(a) - rm(b), np.percentile(diffs, 95) <= MARGIN * rm(b)

    print(f"STEP 5, window design under reference errors: {len(cells)} cells, {N_DRAWS} draws per scenario")
    base = se(e, np.zeros((len(e), 2)))
    out = []
    for scen in ("none",) + SCENARIOS:
        rng = np.random.default_rng(11)
        res = []
        for _ in range(1 if scen == "none" else N_DRAWS):
            s = base if scen == "none" else se(e, delta(e, scen, rng))
            sd = se(dev, np.zeros((len(dev), 2)) if scen == "none" else delta(dev, scen, rng))
            dsel = dev_grid & dev30
            fx = pd.Series(sd[dsel]).groupby(dev.design.to_numpy()[dsel]).mean().idxmin()
            rule = per_cell(s, is_rule)
            rand = per_cell(s, is_rand)
            fixed = per_cell(s, (e.design == fx).to_numpy())
            (d1, ok1), (d2, ok2) = compare(rule, rand), compare(rule, fixed)
            res.append((d1, ok1, d2, ok2, fx))
        r = pd.DataFrame(res, columns=["d_rand", "ni_rand", "d_fixed", "ni_fixed", "fx"])
        line = (f"  {scen:5s} rule - random: median {r.d_rand.median():+.2f} pp [{r.d_rand.quantile(.05):+.2f}, "
                f"{r.d_rand.quantile(.95):+.2f}], non-inferior in {100 * r.ni_rand.mean():5.1f} % | rule - fixed: median "
                f"{r.d_fixed.median():+.2f} pp [{r.d_fixed.quantile(.05):+.2f}, {r.d_fixed.quantile(.95):+.2f}], "
                f"non-inferior in {100 * r.ni_fixed.mean():5.1f} % (fixed design changed in {100 * (r.fx != 's0.3_a0.25_b0.75').mean():.0f} %)")
        print(line, flush=True)
        out.append(line)
    (OUT / "step5_sensitivity.log").write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
