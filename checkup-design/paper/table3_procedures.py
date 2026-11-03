"""Table 3: window-selection procedures, development (16 cells) and pre-specified confirmation (24 held-out cells).
RMSE of the relative LLI and LAM_PE estimates over the check-ups after the first; writes figs/table3_procedures.md
and .csv (sources: results/step2_analyse.log, step3_confirm_analyse.log, step4_fixed.log)."""
from __future__ import annotations

import re

import pandas as pd

from common import DESIGN, FIGS

dev = (DESIGN / "step2_analyse.log").read_text(encoding="utf-8").split("PROCEDURE")[1]
conf = (DESIGN / "step3_confirm_analyse.log").read_text(encoding="utf-8")
fix = (DESIGN / "step4_fixed.log").read_text(encoding="utf-8").split("Fany")[0]
full = (DESIGN / "step4_fixed.log").read_text(encoding="utf-8")


def dev_row(name):
    m = re.search(re.escape(name) + r"\s+([\d.]+) pp, ([\d.]+) h(?:\s+\(vs random large ([+-][\d.]+)\.\.([+-][\d.]+)\))?", dev)
    ci = f"[{m[3]}, {m[4]}]" if m[3] else ""
    return float(m[1]), float(m[2]), ci


rows = [  # name, labels needed, dev log key, confirmation value
    ("Misfit rule, 30 % windows (pre-specified)", "no", "misfit, own first, 30% only",
     float(re.search(r"rule: misfit-chosen 30% design\s+RMSE ([\d.]+)", conf)[1])),
    ("Misfit, any window size", "no", "misfit, own first check-up", None),
    ("Misfit, other cells' first check-ups", "no", "misfit, other cells' first check-ups", None),
    ("Cramér–Rao-best design", "no", "CRB-best", None),
    ("Median grid design (benchmark)", "–", "median design", None),
    ("Median 30 % design (benchmark)", "–", "median 30% design", None),
    ("Random 35–50 % pairs (reference)", "no", "random 35-50 % pairs",
     float(re.search(r"random 35-50 % pairs\s+RMSE ([\d.]+)", conf)[1])),
    ("Best fixed design of the development cells", "yes (dev.)", None, float(re.search(r"F30 .*?RMSE ([\d.]+) pp", fix)[1])),
    ("Oracle design (best on labels; 2.4 h on confirmation)", "yes", "oracle (best on labels)",
     float(re.search(r"oracle fixed design on the confirmation cells \S+ ([\d.]+) pp", full)[1])),
]
out = []
for name, labels, key, c in rows:
    if key == "random 35-50 % pairs":
        m = re.search(r"random 35-50 % pairs\s+([\d.]+) pp, ([\d.]+) h", dev)
        d, h, ci = float(m[1]), float(m[2]), "–"
    elif key:
        d, h, ci = dev_row(key)
    else:  # chosen on the development cells, so its development RMSE is in-sample
        d = float(re.search(r"dev-cell RMSE ([\d.]+)", fix)[1])
        h, ci = 3.6, "in-sample"
    out.append({"procedure": name, "labels needed": labels, "check-up time (h)": h,
                "RMSE dev., 16 cells (pp)": d, "dev. minus random, 95 % CI (pp)": ci,
                "RMSE confirmation, 24 cells (pp)": c})
t = pd.DataFrame(out)
t.to_csv(FIGS / "table3_procedures.csv", index=False)
fmt = t.copy()
for col in ("RMSE dev., 16 cells (pp)", "RMSE confirmation, 24 cells (pp)"):
    fmt[col] = fmt[col].map(lambda v: "–" if pd.isna(v) else f"{v:.2f}")
fmt["check-up time (h)"] = fmt["check-up time (h)"].map(lambda v: f"{v:.1f}")
note = re.search(r"difference ([+-][\d.]+) pp; one-sided 95 % upper bound ([+-][\d.]+) pp; margin ([\d.]+)", conf)
note2 = re.search(r"rule - fixed ([+-][\d.]+) pp, 95 % CI \[([+-][\d.]+), ([+-][\d.]+)\], one-sided upper ([+-][\d.]+) vs margin ([\d.]+)", fix)
md = "\n".join(["| " + " | ".join(fmt.columns) + " |", "|" + "---|" * len(fmt.columns)]
                + ["| " + " | ".join(map(str, r)) + " |" for r in fmt.itertuples(index=False)]) + (
    f"\n\nConfirmation, pre-specified: rule minus random {note[1]} pp, one-sided 95 % upper bound {note[2]} pp "
    f"(margin {note[3]} pp): non-inferior. Rule minus best fixed design {note2[1]} pp, 95 % CI [{note2[2]}, {note2[3]}], "
    f"one-sided upper {note2[4]} vs margin {note2[5]} pp: non-inferior, superiority not shown. Cluster bootstrap "
    f"(experiments, then cells). Oracle rows use the labels and are not usable procedures.\n")
(FIGS / "table3_procedures.md").write_text(md, encoding="utf-8")
print(md)
