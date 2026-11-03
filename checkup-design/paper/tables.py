"""Tables 1, 2, 4 and supplementary S1-S4 (Table 3 = table3_procedures.py). Every number is parsed from
the logged results or recomputed from the saved fit tables; dataset facts in Table 1 come from the dataset papers
(cited in the README of each analysis repo). Writes figs/table*.md and .csv.

    "F:/Research New/battery-pinn-degradation/.venv/Scripts/python.exe" tables.py
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from common import DESIGN, PROG, write


# ------------------------------------------------------------------ Table 1: datasets
kir = pd.read_parquet(PROG / "capacity_nmc.parquet")
kmodes = pd.read_parquet(PROG / "checkups_nmc.parquet")
lfp = pd.read_parquet(PROG / "checkups_lfp.parquet")
mich = pd.read_parquet(PROG / "michigan" / "modes.parquet")
md_ = mich[mich.kind == "discharge"]
mich_soh = (md_.Q / md_.groupby("cell").Q.transform("first")).min()
stan = (PROG / "step10_stanford.txt").read_text(encoding="utf-8").splitlines()[0]
dub = pd.read_parquet(PROG / "dubarry" / "fits.parquet")
t1 = pd.DataFrame([
    {"dataset": "Kirkaldy et al. 2024 (Zenodo 10.5281/zenodo.10637534)",
     "cell": "LG M50T 21700, 5 Ah, NMC811 / graphite-SiOx",
     "cells, conditions": f"{kir.cid.nunique() if 'cid' in kir else (kir.set + kir.cell).nunique()} cells, 5 experiments "
                          "(1: Si-driven, 2,2: graphite-driven, 3: cathode + plating, 4: drive cycle, 5: standard cycle); 10/25/40 °C",
     "check-ups used": f"{len(kir)} capacity; {len(kmodes)} with C/10 + C/2 discharge windows",
     "min SOH": f"{kir.SOH.min():.2f}",
     "reference modes": "published OCV fits of the C/10 discharge; own refits (C/10 charge and discharge)",
     "used for": "forecasting; window design (dev.: expt 4, 5; confirmation: expt 1-3); resistance"},
    {"dataset": "Wheeler et al. 2025 (doi 10.57745/OLBXKT)",
     "cell": "A123 APR18650M1, 1.1 Ah, graphite / LFP",
     "cells, conditions": f"{lfp.cell.nunique()} cells, 7 cycling protocols, 50 °C",
     "check-ups used": f"{len(lfp)} (every 100 cycles, to {lfp.cycles.max()} cycles), C/25 + C/3 discharge",
     "min SOH": f"{lfp.SOH.min():.2f}",
     "reference modes": "own whole-curve physics fit of C/25 + C/3 (LAM_PE fixed, not identifiable)",
     "used for": "forecasting"},
    {"dataset": "Mohtat et al. 2021 (doi 10.7302/7tw1-kc35)",
     "cell": "NMC532 / graphite pouch, ~5 Ah (measured)",
     "cells, conditions": f"{md_.cell.nunique()} cells, 7 cycling conditions",
     "check-ups used": f"{len(md_)} C/20 discharges ({md_.groupby('cell').rpt.count().min()}-"
                       f"{md_.groupby('cell').rpt.count().max()} per cell)",
     "min SOH": f"{mich_soh:.2f}",
     "reference modes": "own quasi-OCV fit of the C/20 discharge, OCPs fitted to this cell (Mohtat2020)",
     "used for": "forecasting replication, reference modes only (19 cells with >= 7 check-ups in the primary test)"},
    {"dataset": "Pozzato et al. 2022 (OSF qsabn)",
     "cell": "LG M50T 21700, 5 Ah",
     "cells, conditions": "10 cells, driving profiles, 23 °C",
     "check-ups used": re.search(r"(\d+) C/20 check-ups", stan)[1] + " C/20 discharges",
     "min SOH": "–",
     "reference modes": "own quasi-OCV fit; " + re.search(r"cells with every check-up valid (\d+)", stan)[1] + " of 10 cells pass the quality rule",
     "used for": "not usable (limitation)"},
    {"dataset": "Dubarry & Anseán 2022 (Mendeley 4k8f4t352y)",
     "cell": "synthetic: graphite / NMC532 and Si-graphite / NMC811",
     "cells, conditions": f"{dub.groupby(['chem', 'set']).ngroups} degradation sets (18 single-mechanism, 2 combined paths)",
     "check-ups used": f"{len(dub)} C/25 curves",
     "min SOH": "–",
     "reference modes": "known (simulation ground truth)",
     "used for": "accuracy of the reference-mode fits"},
])
write(t1, "table1_datasets", "SOH relative to each cell's first check-up. Michigan and Stanford check-ups have no "
      "partial windows, so only reference modes can be tested there.")

# ------------------------------------------------------------------ Table 2: forecasting numbers behind Fig. 1
ROW = re.compile(r"^\s*(?P<label>.+?)\s+capacity\s+(?P<cap>[\d.]+)\s+modes\s+(?P<modes>[\d.]+)\s+change\s+(?P<chg>[+-]\d+)%"
                 r"\s+\| cells: 95% \[(?P<clo>[+-]?\d+)%,(?P<chi>[+-]?\d+)%\] best gain (?P<cg>[\d.]+)% no-useful-gain@5/10/20% (?P<cv>\S+)"
                 r"\s+\| clusters: 95% \[(?P<klo>[+-]?\d+)%,(?P<khi>[+-]?\d+)%\] best gain (?P<kg>[\d.]+)% no-useful-gain@5/10/20% (?P<kv>\S+)")


def rows(path, start=None, stop=None):
    out, on = {}, start is None
    for line in path.read_text(encoding="utf-8").splitlines():
        if start and start in line:
            on = True
        elif on and stop and stop in line:
            break
        m = ROW.match(line) if on else None
        if m:
            out[m["label"].strip()] = m
    return out


k7 = rows(PROG / "step7_reviewer.txt")
m9 = rows(PROG / "step9_michigan.txt", "C/20 discharge", "C/20 charge")
SPEC = [("Kirkaldy NMC", 40, "sqrt extrapolation", k7, "NMC 40, sqrt, modes_ref", "NMC 40, sqrt, modes_pinn_lm"),
        ("Kirkaldy NMC", 40, "power-law extrapolation", k7, "NMC 40, power, modes_ref", "NMC 40, power, modes_pinn_lm"),
        ("Kirkaldy NMC", 40, "half-cell, after Lui et al.", k7, "NMC Yu half-cell, modes_ref", "NMC Yu half-cell, modes_pinn_lm"),
        ("Kirkaldy NMC", 40, "cross-cell, leave-one-cell-out (MAE)", k7, "LOCO + ref LLI, LAM_PE, LAM_NE", "LOCO + est LLI, LAM_PE"),
        ("Kirkaldy NMC", 40, "cross-cell, leave-one-expt-out (MAE)", k7, "LOEO + ref LLI, LAM_PE, LAM_NE", "LOEO + est LLI, LAM_PE"),
        ("Michigan NMC", 19, "mode map, sqrt", m9, "primary k=5 sqrt modes_map (n=19)", None),
        ("Michigan NMC", 19, "mode map, power law", m9, "primary k=5 power modes_map (n=19)", None),
        ("Michigan NMC", 19, "after Lui et al., sqrt", m9, "primary k=5 sqrt modes_yu (n=19)", None),
        ("Michigan NMC", 19, "after Lui et al., power law", m9, "primary k=5 power modes_yu (n=19)", None),
        ("Wheeler LFP", 20, "sqrt extrapolation, 600 cycles", k7, "LFP 20, 600 cyc, sqrt, modes_ref", "LFP 20, 600 cyc, sqrt, modes_pinn_enc"),
        ("Wheeler LFP", 20, "power-law extrapolation, 600 cycles", k7, "LFP 20, 600 cyc, power, modes_ref", "LFP 20, 600 cyc, power, modes_pinn_enc"),
        ("Wheeler LFP", 20, "half-cell, after Lui et al.", k7, "LFP Yu half-cell, modes_ref", "LFP Yu half-cell, modes_pinn_enc")]
out = []
for data, n, method, src, ref, est in SPEC:
    for kind, key in (("reference", ref), ("estimated", est)):
        if key is None:
            continue
        m = src[key]
        verdict = lambda v: {"Y": "excluded", "n": "not excluded"}[v.split("/")[1]]  # noqa: E731
        out.append({"data": data, "cells": n, "method": method, "modes": kind, "capacity only (pp)": m["cap"],
                    "with modes (pp)": m["modes"], "change": f"{int(m['chg']):+d} %",
                    "95 % CI, cells": f"[{m['clo']}, {m['chi']}] %", "95 % CI, clusters": f"[{m['klo']}, {m['khi']}] %",
                    "best plausible gain (cells / clusters)": f"{m['cg']} / {m['kg']} %",
                    "10 % gain (clusters)": verdict(m["kv"])})
write(pd.DataFrame(out), "table2_forecasting",
      "SOH RMSE over the check-ups after the observation window (cross-cell rows: MAE of fade at 6.3 kAh). Change and "
      "intervals: paired bootstrap over cells, and two-stage over experiments/protocols/conditions then cells. Best "
      "plausible gain: one-sided 95 % bound on the improvement. '10 % gain excluded' = no useful gain at a 10 % "
      "equivalence margin. Michigan: no partial windows, so no estimated modes.")

# ------------------------------------------------------------------ S1: Dubarry ground truth, per set
NICE = {"LLI": "LLI", "LAMPE": "LAM_PE delithiated", "LAMLiPE": "LAM_PE lithiated", "LAMNE": "LAM_NE delithiated",
        "LAMliNE": "LAM_NE lithiated", "LAMGIC": "LAM graphite delithiated", "LAMliGIC": "LAM graphite lithiated",
        "LAMSi": "LAM Si delithiated", "LAMliSi": "LAM Si lithiated", "ORINE": "NE ohmic resistance",
        "RDFNE": "NE rate degradation", "RDFPE": "PE rate degradation", "Example": "combined path"}
s1 = []
for (chem, name), t in dub.groupby(["chem", "set"], sort=False):
    t = t.sort_values("level")
    f = t.iloc[0]
    est = np.c_[100 * (1 - t.QLi / f.QLi), 100 * (1 - t.Qp / f.Qp), 100 * (1 - t.Qn / f.Qn)]
    tru = t[["t_LLI", "t_LAM_PE", "t_LAM_NE"]].to_numpy()
    small = tru.max(axis=1) <= 10
    w = np.abs(est - tru)[small].max(axis=0)
    s1.append({"chemistry": chem, "degradation set": NICE[name], "fits passing rule": f"{t.ok.sum()}/{len(t)}",
               "median fit RMSE (mV)": round(t.rmse_mV.median(), 1),
               "worst abs. error LLI (pp)": round(w[0], 1), "LAM_PE": round(w[1], 1), "LAM_NE": round(w[2], 1),
               "pass (all <= 2 pp)": "yes" if (w <= 2).all() else "no"})
write(pd.DataFrame(s1), "tableS1_dubarry",
      "Worst absolute error over levels with true loss <= 10 %, all fits included (the quality rule does not remove "
      "the wrong estimates reliably). Pre-specified pass: every mode within 2 pp.")

# ------------------------------------------------------------------ S2: sensitivity to reference errors
LAB = {"N1": "noise, s.d. 1 pp", "N2": "noise, s.d. 2 pp", "SC": "per-cell scale ±20 %", "LK": "LLI leaking into LAM_PE (0.6)",
       "KD": "kinetic drift, 0-5 pp", "ALL": "scale + drift + 1 pp noise"}
ref = (PROG / "step12_refsens.txt").read_text(encoding="utf-8")
mi, ki = ref.split("Kirkaldy")[0], ref.split("Kirkaldy")[1]
f_re = re.compile(r"(\w+)\s+median ([\d.]+) \[5-95 %: ([\d.]+), ([\d.]+)\], draws with >= 10 % gain ([\d.]+) %")
fm = {m[0]: m for m in f_re.findall(mi)}
fk = {m[0]: m for m in f_re.findall(ki)}
w_re = re.compile(r"^\s+(\w+)\s+rule - random: median ([+-][\d.]+) pp.*?non-inferior in\s+([\d.]+) % \| rule - fixed: "
                  r"median ([+-][\d.]+) pp.*?non-inferior in\s+([\d.]+) % \(fixed design changed in (\d+) %\)", re.M)
win = {m[0]: m for m in w_re.findall((DESIGN / "step5_sensitivity.log").read_text(encoding="utf-8"))}
s2 = []
for s in LAB:
    s2.append({"scenario": LAB[s],
               "Michigan ratio, median [5-95 %]": f"{fm[s][1]} [{fm[s][2]}, {fm[s][3]}]", "Michigan draws >= 10 % gain": f"{fm[s][4]} %",
               "Kirkaldy ratio, median [5-95 %]": f"{fk[s][1]} [{fk[s][2]}, {fk[s][3]}]", "Kirkaldy draws >= 10 % gain": f"{fk[s][4]} %",
               "rule - random (pp), median": win[s][1], "non-inferior vs random": f"{win[s][2]} %",
               "rule - fixed (pp), median": win[s][3], "non-inferior vs fixed": f"{win[s][4]} %",
               "fixed design changed": f"{win[s][5]} %"})
write(pd.DataFrame(s2), "tableS2_sensitivity",
      "200 draws per scenario. Forecast ratio = mode map / capacity SOH RMSE (unperturbed: Michigan "
      + re.search(r"Michigan.*?unperturbed ratio ([\d.]+)", ref, re.S)[1] + ", Kirkaldy "
      + re.search(r"Kirkaldy.*?unperturbed ratio ([\d.]+)", ref, re.S)[1]
      + "). Pre-specified robustness: >= 10 % gain in at most 10 % of draws; non-inferiority in >= 90 % of draws.")

# ------------------------------------------------------------------ S3: resistance, every reference definition
r8 = (PROG / "step8_refit.txt").read_text(encoding="utf-8")
BL = [("published, all cells", "published modes, all cells"), ("published, chg_gr subset", "published, charge-refit cells"),
      ("chg_gr", "own refit, C/10 charge"), ("published, dis_gr subset", "published, discharge-refit cells"),
      ("dis_gr", "own refit, C/10 discharge")]
LINE = re.compile(r"leave-one-(cell|experiment)-out \+ (est|ref) LLI, LAM_PE\s+([\d.]+) vs\s+([\d.]+)\s+cluster 95% "
                  r"\[([+-][\d.]+),([+-][\d.]+)\]\s+p=([\d.]+)")
s3 = []
for key, lab in BL:
    part = r8.split(f"=== reference = {key} ===")[1].split("===")[0].split("C. Cross-cell")[0]
    split = None
    holm = {}
    for line in part.splitlines():
        m = LINE.search(line)
        if m:
            split = m[1]
            s3.append({"reference": lab, "split": f"leave one {m[1]} out", "features": f"+ {m[2]} LLI, LAM_PE",
                       "MAE with modes (% of R0)": m[3], "capacity only": m[4], "difference 95 % CI": f"[{m[5]}, {m[6]}]",
                       "p": m[7], "Holm p": None})
        h = re.search(r"Holm-adjusted p: (.*)", line)
        if h:
            for name, p in re.findall(r"\+ (est|ref) LLI, LAM_PE ([\d.]+)", h[1]):
                for r_ in s3:
                    if r_["reference"] == lab and r_["split"] == f"leave one {split} out" and r_["features"].startswith(f"+ {name}"):
                        r_["Holm p"] = p
write(pd.DataFrame(s3), "tableS3_resistance",
      "0.1 s resistance growth, Kirkaldy NMC without the outlier cell expt 5A, linear model on capacity fade +/- modes. "
      "Cluster bootstrap (experiments, then cells); Holm over the two feature sets of each split.")

# ------------------------------------------------------------------ S4: reference fits and the quality rule
q8 = r8.splitlines()
s4 = []
for v, lab in (("chg_gr", "Kirkaldy, own refit, C/10 charge"), ("dis_gr", "Kirkaldy, own refit, C/10 discharge")):
    fail = next(l for l in q8 if l.strip().startswith(f"{v}: check-ups failing"))
    agr = next(l for l in q8 if l.strip().startswith(v) and "cells n=" in l)
    s4.append({"reference fit": lab, "check-ups failing": re.search(r"failing (\d+/\d+)", fail)[1],
               "cells used": re.search(r"(\d+) cells", agr)[1], "median fit (mV)": re.search(r"fit\s+([\d.]+) mV", agr)[1],
               "agreement with published modes (corr / RMSE)":
                   "; ".join(f"{k} {a} / {b} pp" for k, a, b in re.findall(r"(LLI|LAM_PE|LAM_NE) corr ([\d.]+) rmse ([\d.]+)", agr))})
m9t = (PROG / "step9_michigan.txt").read_text(encoding="utf-8")
for kind in ("discharge", "charge"):
    m = re.search(rf"{kind}: (\d+) check-ups, median fit ([\d.]+) mV, failing (\d+), cells with every check-up valid (\d+)/(\d+)", m9t)
    s4.append({"reference fit": f"Michigan, C/20 {kind}", "check-ups failing": f"{m[3]}/{m[1]}", "cells used": f"{m[4]} of {m[5]}",
               "median fit (mV)": m[2], "agreement with published modes (corr / RMSE)": "no published modes"})
m = re.search(r"(\d+) cells, (\d+) C/20 check-ups, median fit ([\d.]+) mV, failing (\d+); cells with every check-up valid (\d+)", stan)
s4.append({"reference fit": "Stanford, C/20 discharge", "check-ups failing": f"{m[4]}/{m[2]}", "cells used": f"{m[5]} of {m[1]}",
           "median fit (mV)": m[3], "agreement with published modes (corr / RMSE)": "no published modes"})
write(pd.DataFrame(s4), "tableS4_reference_fits",
      "Quality rule, fixed before any downstream result: fit RMSE <= 20 mV, lumped resistance R > 0, electrode capacities "
      "not at a bound; a cell is used only if all its check-ups pass. Agreement on valid cells, check-ups after the first.")

# ------------------------------------------------------------------ S5: faithful Lui et al. forecast (step 13 A)
s13 = PROG / "step13_lui_ratio.txt"
s5 = []
for data, start, stop in (("Kirkaldy NMC", "Kirkaldy NMC:", "Wheeler LFP:"), ("Wheeler LFP", "Wheeler LFP:", "Michigan NMC:"),
                          ("Michigan NMC", "Michigan NMC:", "B. LAM_NE")):
    for label, m in rows(s13, start, stop).items():
        s5.append({"data": data, "configuration": label.split(" (n=")[0], "cells": label.split("(n=")[1].split("/")[0],
                   "capacity only (pp)": m["cap"], "with modes (pp)": m["modes"], "change": f"{int(m['chg']):+d} %",
                   "95 % CI, clusters": f"[{m['klo']}, {m['khi']}] %", "best plausible gain (clusters)": f"{m['kg']} %",
                   "10 % gain (clusters)": {"Y": "excluded", "n": "not excluded"}[m["kv"].split("/")[1]]})
write(pd.DataFrame(s5), "tableS5_lui_faithful",
      "Secondary check (criteria fixed in the step-13 script header before running). Lui et al. (2021): coefficients "
      "bounded to +-50 % of a pooled fit on the other cells of the same experiment / protocol / condition; capacity-only "
      "baseline treated identically. 'Unanchored' as in Lui et al.; 'anchored' scales to the last observed SOH as in the "
      "main analysis. 'Unbounded power' = per-cell fit, same pipeline. Reference modes only. Config 2 on Michigan: 5 of "
      "19 cells could not be fitted.")

# ------------------------------------------------------------------ S6: LAM_NE:LLI ratio as knee warning (step 13 B)
B = re.compile(r"^\s+(?P<d>[\w ]+): (?P<n>\d+) cells, (?P<k>\d+) knee \| ratio AUC (?P<a>[\d.]+) \[(?P<lo>[\d.]+),(?P<hi>[\d.]+)\] "
               r"\| capacity acceleration AUC (?P<c>[\d.]+) \| difference \[(?P<dlo>[+-][\d.]+),(?P<dhi>[+-][\d.]+)\] \| rule ratio>1: "
               r"TP (?P<tp>\d+) FN (?P<fn>\d+) FP (?P<fp>\d+) TN (?P<tn>\d+).*-> (?P<v>.+)$")
s6 = []
for line in s13.read_text(encoding="utf-8").splitlines():
    if (m := B.match(line)):
        s6.append({"data": m["d"], "cells": m["n"], "knee cells": m["k"], "ratio AUC [95 % CI]": f"{m['a']} [{m['lo']}, {m['hi']}]",
                   "capacity-acceleration AUC": m["c"], "AUC difference [95 % CI]": f"[{m['dlo']}, {m['dhi']}]",
                   "ratio > 1: knees caught": f"{m['tp']} of {int(m['tp']) + int(m['fn'])}",
                   "ratio > 1: false alarms": f"{m['fp']} of {int(m['fp']) + int(m['tn'])}", "verdict": m["v"]})
    elif "not testable" in line:
        d, rest = line.strip().split(": ", 1)
        s6.append({"data": d, "cells": rest.split(" cells")[0], "knee cells": "0", "verdict": "not testable (no knee cells)"})
write(pd.DataFrame(s6), "tableS6_ratio",
      "Baure & Dubarry (2019) LAM_NE:LLI ratio at the last observed check-up (same observation points as Table 2), "
      "reference modes. Knee: later fade rate >= 1.5 x observed rate, from measured capacity. Criterion fixed before "
      "running: ratio AUC lower bound > 0.5 and AUC difference vs capacity acceleration lower bound > 0.")

# ------------------------------------------------------------------ S7: verdicts at 5, 10 and 20 % margins (Table 2 rows)
s7 = []
yes = {"Y": "excluded", "n": "not excluded"}
for data, n, method, src, ref, est in SPEC:
    for kind, key in (("reference", ref), ("estimated", est)):
        if key is None:
            continue
        m = src[key]
        s7.append({"data": data, "method": method, "modes": kind,
                   **{f"cells {mg}": yes[v] for mg, v in zip(("5 %", "10 %", "20 %"), m["cv"].split("/"))},
                   **{f"clusters {mg}": yes[v] for mg, v in zip(("5 %", "10 %", "20 %"), m["kv"].split("/"))}})
write(pd.DataFrame(s7), "tableS7_margins",
      "Whether a gain of 5, 10 or 20 % of the capacity-only error is excluded (one-sided 95 % bound of the gain below "
      "the margin), with cell-level and two-stage bootstrap. The main text uses 10 % at the two-stage level.")

# ------------------------------------------------------------------ S8: estimator implementation
s8 = [
    ("Training data", "1000 ageing states per chemistry, Latin-hypercube sampling; PyBaMM SPMe; split 70/15/15 by state"),
    ("NMC model", "O'Regan et al. LG M50T parameters, lumped thermal; C/10 and C/2 discharges to 3.0 V"),
    ("NMC sampling ranges", "LLI (extra) -0.06 to 0.25, LAM_PE -0.10 to 0.20, LAM_NE -0.05 to 0.35; D_n scaling 1-8, D_p 0.15-2 (log-uniform)"),
    ("LFP model", "Prada et al. A123 parameters, capacity 1.1 Ah, isothermal 50 C; C/25 and C/3 discharges to 2.5 V"),
    ("LFP sampling ranges", "LLI (extra) -0.05 to 0.50, LAM_PE -0.90 to 0, LAM_NE 0.05 to 0.65; D_n scaling 1-16, D_p 1-400 (log-uniform)"),
    ("Windows", "two per estimate, one per C-rate; size 15-100 % of nominal capacity, unknown start; 128 points; noise 2 mV, 0.5 C"),
    ("Encoder", "per window: Conv1d(2-32, k7), Conv1d(32-32, k5), max-pool, Conv1d(32-64, k5), max-pool, bidirectional LSTM (64); dense 128; GELU; dropout 0.1; 117 k parameters"),
    ("Latent outputs", "LLI, LAM_PE, LAM_NE, log D_n, log D_p through scaled sigmoids over the sampling ranges; window starts as sigmoid fractions of the feasible range"),
    ("Decoder", "electrode balancing; finite-volume particle diffusion (20 shells, 32 + 48 time steps, 3 Picard iterations for D(c)); Butler-Volmer kinetics; ohmic term; lumped electrolyte polarisation calibrated per C-rate on training curves"),
    ("Training", "label MSE + Gaussian voltage likelihood with learned variance; Adam, lr 1e-3, batch 256, 10 000 pairs per epoch, <= 60 epochs, early stopping (patience 12), gradient clipping at 5; seeds 0, 1, 2"),
    ("Refinement", "8 Levenberg-Marquardt iterations, central-difference Jacobian, Gaussian prior around the encoder output (s.d. 3 in logit units); NMC only"),
    ("Evidence for reported modes", "NMC: LAM_NE correlation with published reference 0.1-0.5, bias -10 to -15 pp (not used). LFP: refinement increased the error of LLI and LAM_NE for every seed; decoder misfit 12 mV on real vs 2 mV on simulated windows (encoder used). LFP LAM_PE: two-window CRB 9.5 pp vs label spread 26 pp (not identifiable)"),
]
write(pd.DataFrame(s8, columns=["item", "setting"]), "tableS8_estimator",
      "The estimator was trained only on simulations and used unchanged in every analysis of this paper.")

# ------------------------------------------------------------------ S9: checks added in response to review
s9 = [
    ("Mechanism-switching forecast vs capacity-driven switch", "Kirkaldy, switch on LAM_PE (40 cells)", "+1 %", "[-3, +5] %", "2.5 %", "no gain"),
    ("", "Kirkaldy, switch on LAM_NE (descriptive)", "-1 %", "[-5, +4] %", "4.3 %", "no gain"),
    ("", "Wheeler, switch on LAM_NE (20 cells)", "+10 %", "[+0, +23] %", "0.0 %", "worse"),
    ("", "Michigan, switch on LAM_PE (19 cells)", "+2 %", "[+0, +10] %", "0.0 %", "worse"),
    ("Faithful Lui et al. without LAM_NE (bounded power, unanchored)", "Kirkaldy", "+89 %", "[+38, +132] %", "0.0 %", "unchanged (excluded)"),
    ("", "Wheeler", "-8 %", "[-35, +21] %", "30.6 %", "unchanged (not excluded)"),
    ("", "Michigan", "+1 %", "[-33, +34] %", "27.8 %", "unchanged (not excluded)"),
    ("Same, scaled to the last SOH (secondary)", "Kirkaldy", "-5 %", "[-21, +7] %", "18.5 %", "changed (was excluded)"),
    ("Half-cell after Lui et al. without LAM_NE (power, scaled)", "Kirkaldy", "+3 %", "[-5, +12] %", "4.1 %", "excluded"),
    ("", "Michigan", "+16 %", "[-12, +45] %", "7.2 %", "excluded (was not excluded)"),
    ("", "Wheeler", "-7 %", "[-35, +24] %", "30.4 %", "not excluded (was excluded)"),
    ("Knee test with LAM_PE:LLI ratio", "Kirkaldy, 7 knee cells", "AUC 0.36", "[0.15, 0.62]", "capacity 0.72", "not useful"),
    ("", "Michigan, 2 knee cells", "AUC 0.06", "[0.00, 0.19]", "capacity 1.00", "not useful"),
    ("Bayesian bound (prior included) vs realised design error", "16 development cells, 50 designs", "Spearman -0.05", "[-0.12, +0.01]", "bound within 2.9 % of CRB", "does not rank designs"),
    ("Leave-one-cluster-out, reference-mode rows of Table 3", "Kirkaldy sqrt / power / half-cell", "-1 / +4 / +9 %", "-3..+2 / +2..+6 / +6..+12 %", "sign-flip p 0.53 / 0.84 / 1.00", "consistent"),
    ("", "Michigan map sqrt / power; Lui sqrt / power", "+4 / +8; +19 / +13 %", "+3..+6 / +2..+12; +11..+24 / +2..+22 %", "sign-flip p 0.92 / 0.75; 0.98 / 0.71", "consistent"),
    ("", "Wheeler sqrt / power / half-cell", "+5 / +4 / +39 %", "-1..+10 / -18..+31 / +19..+54 %", "sign-flip p 0.74 / 0.41 / 0.93", "consistent"),
]
write(pd.DataFrame(s9, columns=["check", "data", "change", "95 % CI or range", "best plausible gain / other", "verdict"]),
      "tableS9_review_checks",
      "Secondary checks added in response to review, with criteria written in the analysis script before running "
      "(battery-pinn-prognostics/src/step14_reviewer2.py; checkup-design gap2/step6_bayes_crb.py). Changes are relative "
      "to the comparator named in the first column; sign-flip p is the exact one-sided p-value for a gain from the modes "
      "over all sign patterns of the cluster-mean differences.")
