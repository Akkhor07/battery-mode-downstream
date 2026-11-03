# Gap 2: Fisher-optimal partial check-up windows

Moved here after Gap 1 failed its go/no-go (see battery-pinn-prognostics/PLAN.md, section 5).

## Main risk

Fisher-optimal design minimises the Cramer-Rao bound, i.e. the error from voltage noise under a correct model.
On real NMC cells the bound is 0.1-0.3 pp but the realised error is 3-5 pp, dominated by model mismatch. A design
that is optimal for noise may not be optimal on real cells. Step 1 tests this before anything is built.

## Step 1 (go/no-go): `gap2/step1_design_grid.py`

50 grid designs (window a on C/10, window b on C/2; 20 % or 30 %; start at 0/25/50/75/100 % of the feasible
range) plus 10 random 35-50 % pairs per check-up, on all 104 NMC check-ups, frozen PINN+LM, 3 seeds.
* Q1: does the CRB ranking of designs predict the ranking of realised RMSE (Spearman over designs, cell bootstrap)?
* Q2: does the CRB-best 20 % design reach the error of random 35-50 % pairs, in less check-up time?
Go if Q1 is clearly positive for LLI and LAM_PE; otherwise the paper must be about model-error-aware design.

## Later steps (after a go)

2. Sequential design as in the blueprint: first window -> encoder + LM -> local Fisher information -> choose the
   second window (position and C-rate) minimising the bound of the identifiable modes.
3. LFP (encoder only, LAM_NE and LLI), simulated test set (where the CRB should hold), timing of the check-up.
4. Figures and manuscript.

## Step 1 result (26 Oct 2026)

104 NMC check-ups, 50 grid designs + random 35-50 % pairs, PINN+LM, 3-seed ensemble (`results/`).
* The CRB does not predict real error: Spearman(CRB, realised RMSE) = -0.10 [-0.14, -0.04] for LLI and
  0.01 [-0.10, 0.13] for LAM_PE. CRB-best designs (windows at the end of discharge) are among the worst
  (LLI 10-14 pp).
* Design still matters a lot on real cells: LLI RMSE 2.8-14.6 pp, LAM_PE 3.3-10.0 pp across designs; the best
  grid design beats random 35-50 % pairs (5.3 pp) in less time (3.6 h vs 5.1 h).
* Design quality transfers between the two ageing experiments (rank correlation of design errors 0.86).
  Chosen on Expt 5 -> 3.9 pp on Expt 4 (best 3.8, median 7.5, random large 5.7). Chosen on Expt 4 -> 6.3 pp on
  Expt 5 (best 3.6, median 6.9, random large 5.2): the transfer is not reliable in both directions.
* A C/2 window at the end of discharge is always bad (10-11 pp): position-dependent model error, likely low-SOC
  diffusion / silicon effects that the model does not capture.

Verdict: noise-optimal (Fisher) design is a no-go on real cells. Candidate reframing: model-error-aware window
design. Next test: whether a label-free criterion (the real-window voltage misfit of each design, the JES
"whether to refine" signal) predicts realised error; and a simulated control showing the CRB does predict error
when the model is right.

## Step 2, simulated control (27 Oct 2026)

Simulated test set of the real-NMC Stage-1 run (150 states, true modes known), seed 0, same 50 designs.
* The CRB does not predict the error even here: Spearman(CRB, RMSE of LLI/LAM_PE) = 0.11. The CRB-best design
  gives 12.8 pp against 6.0 pp for the median design; RMSE ranges 3.1-16.7 pp across designs, while the CRB is
  0.1-1 pp. The estimator (encoder + 8 LM steps with prior) is far from efficient on fixed windows, so its own
  error, not voltage noise, sets the ranking. Correction to step 1: the CRB failure on real cells is therefore
  not (only) model mismatch.
* The label-free misfit does track the error in simulation: Spearman(misfit, RMSE) = 0.74.
* Real-cell misfit run (`--part real`) was stopped by the system for low memory; to be rerun.

## Step 2, real NMC (27 Oct 2026)
Full run done (`--part real --batch 50`, 3 seeds, both experiments; `results/nmc_design_misfit.parquet`, `results/step2_analyse.log`).
* Design-level Spearman with RMSE of LLI/LAM_PE: CRB -0.03 [-0.12, +0.02]; misfit +0.48 [+0.42, +0.53] (cell bootstrap). Within one window size the misfit gives +0.62 (20 %) and +0.60 (30 %): pooled, it is pulled down because short windows fit better but estimate worse.
* Procedure (choose the design per cell from the first check-up, score the later ones), RMSE in pp:
  misfit/own first 7.29 (2.5 h); misfit/other cells 8.12; CRB-best 10.45; median design 7.00; oracle 3.75 (3.6 h); random 35-50 % pairs 5.51 (5.1 h).
  Misfit picks are significantly worse than random large pairs and no better than the median design.
* Restricted to 30 % windows (3.6 h): misfit pick 4.95 pp vs median 30 % design 5.96 and random large 5.51 (5.1 h). Restricted to 20 %: 8.48 vs 8.12, no help.
* Verdict: the misfit ranks designs on average but a single beginning-of-life check-up is too noisy to pick a good design for an individual cell. The only positive is "30 % windows chosen by misfit match random 35-50 % pairs in 30 % less time", from 16 cells without a CI; too thin for a paper on its own.

## Step 3, confirmation on Expt 1, 2,2, 3 (pre-specified 27 Oct 2026, written BEFORE running)
Data: the 24 Kirkaldy cells not used in steps 1-2 (Expt 1: 9, Expt 2,2: 6, Expt 3: 9), check-ups with both C/10
and C/2 discharges (even RPTs). Same estimator (pinn_pair_wide seeds 0-2, 8 LM steps, prior s.d. 3), same 50 grid
designs and 10 random 35-50 % pairs per check-up, same window seed (40000).
Rule (fixed): for each cell, among the 25 grid designs of size 30 %, take the one with the lowest mean voltage
misfit (refined, mean of both windows) on the cell's first check-up; use it for all later check-ups.
Comparator: random 35-50 % pairs on the same later check-ups.
Primary metric: RMSE [pp] of relative LLI and LAM_PE over later check-ups, seed ensemble, as in step 2.
Success criterion: non-inferiority. One-sided 95 % upper bound of RMSE(rule) - RMSE(random) <= 10 % of
RMSE(random), two-stage bootstrap over experiments then cells. Reported regardless of outcome.
Secondary (descriptive): misfit-vs-error Spearman within size; median 30 % design; oracle; test hours.

### Step 3 result (2026-10-27): CONFIRMED
Pre-specified test on 24 new cells (Kirkaldy Expt 1, 2,2, 3; 132 later check-ups), seeds 0-2, `results/step3_confirm_analyse.log`:
- Rule (30 % design, lowest first-check-up misfit): RMSE 5.44 pp, 3.6 h. Random 35-50 % pairs: 6.78 pp, 5.1 h.
- Difference -1.33 pp; one-sided 95 % upper bound -0.82 pp (two-stage bootstrap) vs margin +0.68 pp -> NON-INFERIOR (confirmed).
- Rule better in every experiment: expt1 5.97 vs 7.54, expt2 2.84 vs 4.14, expt3 6.16 vs 7.36.
- Upper bound < 0 also implies superiority, but superiority was NOT pre-specified: report it as a secondary observation only.
- Secondary: Spearman(misfit, RMSE) +0.49 (20 %), +0.62 (30 %); oracle 4.21 / 4.61 pp, so per-cell choice leaves 0.8-1.2 pp on the table.

## Step 4, best-fixed-design comparator (specified 2026-10-27, BEFORE computing; added after step 3, so secondary)
Question (reviewer weakness "weak comparator"): is the misfit rule at least as good as the standard practice of
calibrating ONE fixed design on labelled cells and using it everywhere?
Comparators, both chosen only on the 16 development cells (Expt 4, 5; `nmc_design_misfit.parquet`) by the RMSE of
LLI/LAM_PE against the reference modes over later check-ups, then applied unchanged to the 24 confirmation cells:
  F30  = best 30 % grid design (same check-up time as the rule);  Fany = best grid design of any size.
Metric, data and bootstrap exactly as step 3 (two-stage, experiments then cells, 4000 draws).
Report: difference rule - comparator with one-sided 95 % upper bound vs 10 % margin (non-inferiority) and the
two-sided 95 % interval. Descriptive only: oracle fixed design chosen on the confirmation cells themselves
(not achievable; shows the headroom). Reported regardless of outcome.

### Step 4 result (2026-10-27), `results/step4_fixed.log`
- Best fixed design on the 16 dev cells is s0.3_a0.25_b0.75 for both F30 and Fany (dev RMSE 3.70 pp). On the 24
  confirmation cells it gives 6.93 pp (rank 19/25 among 30 % designs there): a calibrated fixed design does not transfer.
- Rule 5.44 vs fixed 6.93 pp, same 3.6 h: difference -1.49 pp, 95 % CI [-3.56, +0.58]; one-sided upper +0.42 vs
  margin 0.69 -> NON-INFERIOR. Not shown superior (CI includes 0).
- By experiment: expt1 rule 5.97 / fixed 5.68 (fixed slightly better), expt2 2.84 / 3.14, expt3 6.16 / 9.44.
- Wording: "the label-free rule is non-inferior to a design calibrated on labelled cells, and avoids the large
  failure of the calibrated design on one experiment (expt 3)". Do not claim superiority.

## Step 5, reference-error sensitivity (specified 2026-10-27, BEFORE computing)
Question: Dubarry (step11) shows quasi-OCV references can be wrong by ~2 pp, more under kinetic change. Could such
errors change our conclusions? We cannot recover the truth, so we perturb the references (ref -> ref + delta) with
the error patterns Dubarry showed and redo the analyses; 200 draws per scenario (relative modes, fractions):
  N1/N2  independent noise per check-up on LLI and LAM_PE, sd 1 / 2 pp
  SC     per-cell scale error: LLI*(1+a), LAM_PE*(1+b), a, b ~ U(-0.2, 0.2)
  LK     LLI leaking into LAM_PE: LAM_PE + 0.6*LLI (NMC532 pattern)
  KD     kinetic drift: per cell d ~ U(0, 5 pp), add d * (ageing progress 0..1) to LLI and LAM_PE
  ALL    SC + KD + N1 together
(a) Window design (24 confirmation cells): rule vs random 35-50 % and vs dev-calibrated fixed design (F30
    re-chosen on perturbed dev references). Report the fraction of draws in which non-inferiority still holds
    (500-draw two-stage bootstrap per draw) and the median difference.
(b) Forecasting (Michigan discharge refs, primary k=5, sqrt; Kirkaldy published refs, same design as step 7):
    mode map vs capacity; report the median ratio and the fraction of draws in which the mode map beats capacity
    by >= 10 %. Capacity is measured directly and is not perturbed.
Conclusion is called robust if non-inferiority holds in >= 90 % of draws in every scenario, and the mode map
never beats capacity by >= 10 % in more than 10 % of draws.

### Step 5 result, window design (2026-10-27), `results/step5_sensitivity.log`
- vs random 35-50 % pairs: ROBUST. Non-inferior in 100 % of draws in every scenario; median difference -1.13 to -1.33 pp.
- vs dev-calibrated fixed design: NOT robust by the pre-specified criterion (>= 90 % in every scenario).
  N1 100 %, N2 99.5 %, SC 96 %, KD 90.5 %, ALL 79.5 %, LK 0 %. The rule's point estimate stays better in every
  scenario (median -0.74 to -1.49 pp), but the bootstrap bound crosses the margin. Under LK the dev-chosen fixed
  design changes in every draw (leakage changes which design looks best on labels), which is what removes the margin.
  Note: step 11 showed LK (LLI leaking into LAM_PE) for the NMC532 pipeline, not for the Si-NMC811 pipeline these
  Kirkaldy cells use, so LK is the least relevant scenario here; it was pre-specified, so it is reported.
- Wording: "non-inferiority to random pairs is robust to plausible reference errors; non-inferiority to a calibrated
  fixed design holds for random and scale errors but not under combined or leakage-type errors".
