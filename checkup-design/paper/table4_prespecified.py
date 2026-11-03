"""Table 4: every test the paper reports, when its criterion was fixed, and whether it was met. Written by hand from
PLAN.md (steps 3-5) and the docstrings of the prognostics scripts (step 8, 9, 10, 11, 12), which record the criteria
as written before running; results are the logged values (Tables 2, 3, S1-S4). Failures are listed with the passes.
"""
from __future__ import annotations

import pandas as pd

from common import write

PRE, SEC, POST = "pre-specified (written before running)", "specified before computing; secondary", "post hoc / exploratory"
rows = [
    # (claim, data, criterion, status, result, verdict)
    ("Window rule is non-inferior to random 35-50 % pairs", "24 held-out Kirkaldy cells (expt 1-3)",
     "one-sided 95 % bound of the RMSE difference <= 10 % of the random-pair RMSE; two-stage bootstrap", PRE,
     "-1.33 pp; bound -0.82 pp vs margin 0.68 pp; 3.6 h vs 5.1 h", "met"),
    ("Window rule is non-inferior to the best fixed design of the development cells", "same 24 cells",
     "as above, comparator chosen on the 16 development cells", SEC,
     "-1.49 pp; 95 % CI [-3.56, +0.58]; bound +0.42 vs margin 0.69 pp", "met (superiority not shown)"),
    ("Reference modes add no forecasting value (replication)", "Michigan NMC, 19 cells",
     "observe check-ups 0-5; one-sided 95 % bound of the gain < 10 % of the capacity error, cluster bootstrap", PRE,
     "mode map +4 % (sqrt), +8 % (power), best plausible gain 2.0 / 6.2 %; after Lui et al. +19 % / +13 %, best "
     "plausible gain 4.5 % (sqrt) / 15.1 % (power)", "met in 3 of 4 rows (not for after Lui et al., power law)"),
    ("Reference modes add no forecasting value", "Kirkaldy NMC 40 cells; Wheeler LFP 20 cells",
     "same 10 % equivalence margin, applied to every setup", POST + " (margin set after the first comparisons)",
     "10 % gain excluded in 6 of 8 reference-mode setups; not excluded for cross-cell leave-one-cell-out (11.8 %) "
     "and LFP power law (unstable)", "met in 6 of 8"),
    ("Third lab on the Kirkaldy cell", "Stanford M50T, 10 cells", "quality rule and forecasting test as for Michigan", PRE,
     "1 of 10 cells passes the quality rule; test could not be run", "not testable"),
    ("Quality rule for reference fits", "all own refits", "fit <= 20 mV, R > 0, capacities off bounds, every check-up of a cell",
     "pre-specified (before any downstream result)", "cells kept: Kirkaldy 15 (charge) / 35 (discharge); Michigan 21 (discharge); "
     "Stanford 1", "applied unchanged"),
    ("Quasi-OCV fits recover known modes", "Dubarry synthetic curves, 20 sets",
     "abs. error <= 2 pp for truth <= 10 %, crosstalk <= 2 pp, kinetics-only changes give modes <= 2 pp", PRE,
     "19 of 20 sets fail; LLI mostly within 2 pp, LAM_NE not recoverable; combined path within 1.6 pp for LLI, LAM_PE",
     "not met"),
    ("Forecasting conclusion is robust to reference errors", "Michigan and Kirkaldy, 6 error scenarios x 200 draws",
     "mode map beats capacity by >= 10 % in at most 10 % of draws, every scenario", PRE,
     "worst scenario 3.5 % (Michigan, scale errors); all others 0 %", "met"),
    ("Window rule vs random pairs is robust to reference errors", "24 cells, 6 scenarios x 200 draws",
     "non-inferiority in >= 90 % of draws, every scenario", PRE, "100 % in every scenario", "met"),
    ("Window rule vs fixed design is robust to reference errors", "same", "non-inferiority in >= 90 % of draws, every scenario",
     PRE, "N1 100, N2 99.5, SC 96, KD 90.5, ALL 79.5, LK 0 %", "not met"),
    ("Faithful Lui et al. forecast (bounded coefficients, unanchored)", "Kirkaldy 40, Wheeler 20, Michigan 19 cells",
     "cluster best plausible gain < 10 % for bounded power law, unanchored", SEC + " (added after reading the full text)",
     "Kirkaldy +56 %, gain 0 %; Wheeler +10 %, gain 24 %; Michigan +1 %, gain 28 % (anchored: -13 %, gain 37 %)",
     "met on Kirkaldy only; a 10 % gain is not excluded on Wheeler and Michigan"),
    ("LAM_NE:LLI ratio warns of a knee (Baure & Dubarry)", "same cells and observation points",
     "ratio AUC lower bound > 0.5 and above capacity acceleration", SEC + " (added after reading the full text)",
     "Kirkaldy AUC 0.46 [0.24, 0.69] vs 0.72, 2 of 7 knees caught; Michigan 2 knee cells; Wheeler no knees",
     "not met (Kirkaldy); not testable (Michigan, Wheeler)"),
    ("Misfit ranks designs, CRB does not", "16 development cells (expt 4, 5); simulation",
     "Spearman correlation with realised error, cell bootstrap", POST,
     "CRB -0.03 [-0.12, +0.02]; misfit +0.48 [+0.42, +0.53]; 30 % restriction chosen after seeing these data",
     "motivated the pre-specified test above"),
    ("Modes improve resistance-growth prediction", "Kirkaldy NMC, 39 cells", "cluster bootstrap, Holm over two feature sets",
     POST, "published modes: Holm p 0.039 / 0.011; own refits: significant in 1 of 4 tests; estimated modes: never",
     "not claimed (suggestive only)"),
    ("Accuracy needed for a forecasting gain", "60 synthetic NMC cells (positive control)", "descriptive", POST,
     "forecast error -29 % with exact modes, +4 % at 1 pp noise, +35 % at 2 pp", "descriptive"),
]
t = pd.DataFrame(rows, columns=["claim", "data", "criterion", "when fixed", "result", "verdict"])
write(t, "table4_prespecified",
      "Margins are 10 % of the comparator's error throughout. 'Pre-specified' criteria are recorded with their date in "
      "PLAN.md (window design, sensitivity) or in the analysis script's header (Michigan, Stanford, Dubarry, quality "
      "rule), written before the analysis was run. Michigan rows corrected on 2026-10-28: the ageing axis of five "
      "45 degC cells did not start at their first check-up (fitted curves pass through the origin).")
