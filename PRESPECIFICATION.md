# Where each pre-specified criterion is written

The criteria were written into the analysis plans or the script headers, with their date, before the analysis
was run. This repository was created after the analyses (its history starts at publication), so the ordering is
documented by the text of these files, not by version history.

| Analysis | Status in the manuscript | Where the criterion is written |
|---|---|---|
| Window rule vs random 35-50 % pairs, 24 held-out cells | pre-specified | `checkup-design/PLAN.md`, "Step 3, confirmation ... (pre-specified 27 Oct 2026, written BEFORE running)" |
| Window rule vs best fixed design | secondary | `checkup-design/PLAN.md`, "Step 4 ... (specified 2026-10-27, BEFORE computing; added after step 3, so secondary)" |
| Reference-error sensitivity (window design and forecasting) | pre-specified | `checkup-design/PLAN.md`, "Step 5 ... (specified 2026-10-27, BEFORE computing)"; `prognostics/src/step12_refsens.py` |
| Michigan replication | pre-specified | `prognostics/src/step9_michigan.py` header, "Pre-specified test (written before any forecast was run)" |
| Third laboratory (Pozzato / Stanford) | pre-specified | `prognostics/src/step10_stanford.py` header |
| Quality rule for reference fits | pre-specified | `prognostics/src/step8_refit.py` and `step9_michigan.py` headers |
| Ground truth on synthetic curves | pre-specified | `prognostics/src/step11_dubarry.py` header, "Pass criteria (written before running)" |
| Faithful Lui et al. forecast, LAM_NE:LLI ratio | secondary | `prognostics/src/step13_lui_ratio.py` header |
| Checks added after the simulated review | secondary | `prognostics/src/step14_reviewer2.py`, `checkup-design/gap2/step6_bayes_crb.py` headers |

Exploratory analyses (margin for the Kirkaldy and Wheeler forecasts, CRB and misfit correlations, resistance,
positive control) are marked as such in Table 2 of the manuscript.

The Michigan analysis was corrected on 2026-10-28 (ageing axis of five cells); the outputs before the correction
are in `prognostics/data/pre_fix_2026-10-28/`.
