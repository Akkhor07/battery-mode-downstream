# Gap 1: forecasting degradation from partial-window mode estimates

Follow-up to the JES submission (code: github.com/Akkhor07/battery-pinn-degradation, doi:10.5281/zenodo.23424353).
Starting point: `D:\PINN_Battery_Research_Blueprint_Gap1_Execution_Plan.docx`. This file records what the data
and the literature say about that blueprint, and the plan that follows from it.

## 1. What the existing data supports (checked 26 Oct 2026)

Stage-1 outputs are reused without retraining (`src/stage1_export.py` -> `data/checkups_*.parquet`). The export
reproduces the JES numbers (NMC LLI 4.0 pp and LAM_PE 3.3 pp pooled over both experiments; LFP LLI 12.5 pp,
LAM_NE 6.7 pp), so the reuse is correct.

| | NMC (Kirkaldy, 16 cells) | LFP (Wheeler, 20 cells) |
|---|---|---|
| check-ups with Stage-1 estimates | 8 per cell (Expt 5), 5 (Expt 4); C/2 discharge only at every second RPT | 27-67 per cell, every 200 cycles |
| check-ups with capacity only | 16 (Expt 5), 10 (Expt 4) | same as above |
| end-of-test SOH | 0.70-0.89; several cells never reach 80% (censored RUL) | 0.32-0.64; all cross 80% and 70% |
| capacity knee (late fade rate / early) | only Expt 4 cells B and C (10 degC): 4.1x and 4.8x; all others 0.3-0.8x | none: 0.4-0.8x in all 20 cells |
| usable modes (JES CRB + misfit rule) | LLI, LAM_PE from PINN+LM | LAM_NE from the encoder; LLI biased by about -10 pp |

Consequences:
* Knee forecasting cannot be validated statistically on these real cells (2 knees in 36 cells). Knees can be a
  synthetic result plus two real case studies, not the headline.
* LFP is the real test bed for trajectory and RUL forecasting (20 cells, many check-ups, full life).
* NMC forecasting from the first 30-50% of check-ups means 2-4 points per cell: a case study, not a benchmark.

## 2. Corrections to the blueprint

1. Synthetic benchmark is circular. `generate_synthetic_knees.py` is not PyBaMM (the table says it is); it
   integrates exactly the SEI power law and Voight FFM laws that the loss then uses as priors, so the ODE priors
   are guaranteed to help. Replace with PyBaMM degradation submodels that produce knees mechanistically, e.g. the
   OKane2022 parameter set (SEI + lithium plating + stress-driven LAM, parameterised for the LG M50, the Kirkaldy
   cell). The priors are then tested against a model they were not derived from.
2. Diffusivities are not validated. The blueprint lists D_n and D_p as modes "already recovered accurately"; in
   the JES paper they are nuisance parameters and were never checked against references. Use them only as
   optional inputs, ablated.
3. Uncertainty mismatch on LFP. The stored Laplace s.d. belongs to the refined PINN, but the LFP decision rule uses
   the encoder output. Either recompute encoder uncertainty or use the ensemble (seed) spread on LFP.
4. Capacity layer. The blueprint hard-codes electrode capacities (5.45/5.30/5.05 Ah) and fixed stoichiometry
   limits. The Stage-1 decoder already has exact electrode balancing with the real OCPs and cut-off voltages;
   reuse it (it is differentiable).
5. Fair capacity baseline. For NMC the capacity-only model could use all 16 RPTs while the mode model has 8;
   report both the matched-check-up comparison and the all-RPT one.
6. Small numerical slips: misfit "37-62 mV" (paper: 37-57 mV); "q0 correlation 0.998" is not in the paper.

## 3. Prior work that limits the novelty claim

* Yu et al. 2021 (J. Power Sources): extrapolate LLI/LAM_PE/LAM_NE from a half-cell model to predict RUL; more
  accurate than capacity-only extrapolation on 8 cells. "Modes improve RUL" is therefore not new by itself.
* Jia et al. 2024 (IEEE TTE): electrode-level information in physics-guided ML for knee and trajectory prediction.
* Zhang et al. 2025 (arXiv): physics-informed degradation-mode diagnosis with knee-onset detection and prediction.
* Li et al. 2025 (Nat. Commun.): degradation-mode analysis needed to parameterise lifetime models.

Defensible gap: forecasting from modes that are estimated label-free from partial windows, with their
identifiability limits and calibrated uncertainty propagated into the forecast, and a direct test of when such
noisy mode estimates help over capacity alone. Note: all prior mode-based prognostics use full low-rate curves.

## 4. Plan

Step 0 (done): separate project, Stage-1 check-up export.

Step 1 (go/no-go, minutes of CPU): on the 20 LFP cells, compare simple per-cell forecasts from the first m
check-ups (m = 5, 10, 15): capacity extrapolation vs extrapolation of the estimated modes mapped to capacity, vs
the same with reference modes (upper bound). If noisy estimated modes do not beat capacity alone even with simple
models, a GRU will not rescue the paper and the framing must change before any training.

Step 2 (hours of CPU, ask first): PyBaMM OKane2022 full-life simulations with varied degradation parameters,
two partial windows extracted at each check-up and passed through the frozen Stage-1 encoder, so the synthetic
benchmark carries realistic Stage-1 errors rather than Gaussian noise.

Step 3: Stage-2 forecaster (GRU with monotone rate head and the Stage-1 balancing layer), baselines and ablations
as in blueprint Section 4.1, trained on Step-2 data, tested leave-protocol-group-out on LFP and as a case study on
NMC (including the two knee cells).

Step 4: figures, manuscript.

## 5. Step 1 result (26 Oct 2026): no-go on the LFP cells

`src/step1_gonogo.py`, full table in `data/step1_report.txt`. Forecasts from the check-ups up to 400/600/800
cycles (before any cell reaches 80 % SOH), 20 cells, paired cluster bootstrap over cells.

* Per-cell extrapolation, MAE of the cycle at 80 % SOH: capacity 345/215/121 cycles; estimated PINN modes
  6215/6947/6134 (much worse); reference modes 333/212/122 (no difference, intervals include 0).
* Leave-one-cell-out ridge: adding estimated or reference modes to capacity features never helps
  significantly (all intervals include 0 or favour capacity).
* Reason: on these cells 1 - SOH = 1.10 LLI_ref (R^2 = 0.999). Capacity fade is lithium loss, so even perfect
  modes add nothing a capacity curve does not already show, and the noisy estimates hurt when extrapolated.
* A hybrid (capacity basis + mode curves) is identical to capacity by the normal equations and was dropped.

Decision: Gap 1 paused; work moved to Gap 2 (`F:\Research New\battery-pinn-checkup-design`). Mode-based
forecasting can only help where the capacity-limiting mechanism changes (knees); revisit with data that has them.
