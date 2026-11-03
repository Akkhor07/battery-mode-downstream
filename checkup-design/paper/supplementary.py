"""Supplementary material (Tables S1-S6) as LaTeX, built from the generated table files in figs/ (run tables.py first).

    "F:/Research New/battery-pinn-degradation/.venv/Scripts/python.exe" supplementary.py
"""
from __future__ import annotations

import pandas as pd

from common import FIGS, HERE

OUT = HERE / "overleaf" / "supplementary.tex"
TABLES = [
    ("tableS1_dubarry", "Table S1. Quasi-OCV fits of synthetic C/25 curves with known modes (Dubarry), per degradation set "
     "(backs Fig.~4a)."),
    ("tableS2_sensitivity", "Table S2. Sensitivity of the conclusions to reference-mode errors, per scenario (backs Fig.~4b, c)."),
    ("tableS3_resistance", "Table S3. Resistance-growth prediction for every reference definition and split (backs Fig.~5)."),
    ("tableS4_reference_fits", "Table S4. Reference fits and the quality rule."),
    ("tableS5_lui_faithful", "Table S5. Faithful Lui et al.\\ forecast with bounded coefficients."),
    ("tableS6_ratio", "Table S6. The LAM$_\\mathrm{NE}$:LLI ratio as an early warning of a capacity knee."),
    ("tableS7_margins", "Table S7. Verdicts at equivalence margins of 5, 10 and 20\\% (rows of Table 3)."),
    ("tableS8_estimator", "Table S8. Implementation of the partial-window estimator."),
    ("tableS9_review_checks", "Table S9. Secondary checks added in response to review."),
]


def tex(v) -> str:
    v = "--" if (isinstance(v, float) and pd.isna(v)) else str(v)
    for a, b in (("\\", "\\textbackslash{}"), ("%", "\\%"), ("&", "\\&"), ("_", "\\_"), ("#", "\\#"),
                 ("<=", "$\\leq$"), (">=", "$\\geq$"), ("±", "$\\pm$"), ("→", "$\\to$"), ("–", "--"), ("−", "$-$"),
                 ("+/-", "$\\pm$")):
        v = v.replace(a, b)
    return v


def note(name: str) -> str:
    text = (FIGS / f"{name}.md").read_text(encoding="utf-8").split("\n\n", 1)
    return tex(text[1].strip()) if len(text) > 1 else ""


parts = [r"""\documentclass[a4paper,10pt]{article}
\usepackage[margin=18mm,landscape]{geometry}
\usepackage{booktabs,array,longtable}
\renewcommand{\arraystretch}{1.15}
\begin{document}
\begin{center}\Large Supplementary material\\[4pt]\large Do degradation-mode estimates improve battery life forecasts and
check-up design? A test against capacity alone on three real-cell datasets\\[4pt]\normalsize Md. Akkhor Hasan\end{center}
All tables are generated from the logged analysis outputs by the scripts in the code archive.
"""]
for name, caption in TABLES:
    df = pd.read_csv(FIGS / f"{name}.csv", dtype=str, keep_default_na=False)
    width = 0.95 / len(df.columns)
    cols = "".join(f"p{{{width:.3f}\\linewidth}}" for _ in df.columns)
    parts.append(f"\\section*{{{caption}}}\n\\footnotesize\n\\begin{{longtable}}{{@{{}}{cols}@{{}}}}\n\\toprule\n"
                 + " & ".join(tex(c) for c in df.columns) + r" \\" + "\n\\midrule\n\\endhead\n"
                 + "\n".join(" & ".join(tex(v) for v in row) + r" \\" for row in df.itertuples(index=False))
                 + "\n\\bottomrule\n\\end{longtable}\n" + note(name) + "\n\\normalsize\n\\clearpage\n")
parts.append("\\end{document}\n")
OUT.write_text("\n".join(parts), encoding="utf-8")
print("wrote", OUT)
