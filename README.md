# battery-mode-downstream

Analysis code, analysis plans and result logs for the manuscript

> Do degradation-mode estimates improve battery life forecasts and check-up design? A test against capacity alone on three real-cell datasets

The repository reproduces every number, figure and table of the manuscript from public data.

## Layout

| Folder | Content |
|---|---|
| `prognostics/src/` | Forecasting, resistance and reference-mode analyses (`step1` ... `step14`). Each script header states what it tests and, where applicable, the criterion fixed before it was run. |
| `prognostics/data/` | Result logs (`step*.txt`, `*.log`, `*.csv`) and derived tables: check-up tables with reference and estimated modes (`checkups_*.parquet`, `capacity_*.parquet`), independent refits (`refit/`), Michigan quasi-OCV fits (`michigan/modes.parquet`), fits of the synthetic Dubarry curves (`dubarry/fits.parquet`), estimator predictions (`stage1_preds/`). `pre_fix_2026-10-28/` holds the Michigan outputs before the correction described in the manuscript. |
| `prognostics/PLAN.md` | Analysis plan and dated decisions for the forecasting study. |
| `checkup-design/gap2/` | Window-design study: grid designs and Cramer-Rao bounds (`step1`), label-free misfit (`step2`), sensitivity (`step5`), Bayesian bound (`step6`). |
| `checkup-design/results/` | Window-design results and logs. |
| `checkup-design/PLAN.md` | Analysis plan with the dated pre-specified criteria (steps 3-5). |
| `checkup-design/paper/` | One script per figure and table; outputs in `paper/figs/`. They only read logged results. |
| `PRESPECIFICATION.md` | Where each pre-specified criterion is written down. |

## Data

Raw data are not redistributed; download them from the original sources:

- Kirkaldy et al. (2024), LG M50T, Zenodo, https://doi.org/10.5281/zenodo.10637534
- Wheeler et al. (2025), A123 graphite/LFP, Recherche Data Gouv, https://doi.org/10.57745/OLBXKT
- Mohtat et al. (2021), University of Michigan pouch cells, Deep Blue Data, https://doi.org/10.7302/7tw1-kc35
- Pozzato et al. (2022), LG M50T driving profiles, OSF (project qsabn)
- Dubarry (2022), synthetic curves with known modes, Mendeley Data 4k8f4t352y

## Estimator

The partial-window degradation-mode estimator and the simulation code are in
https://github.com/Akkhor07/battery-pinn-degradation and are used frozen. Its predictions on the real cells are
included here (`prognostics/data/stage1_preds/`, `checkups_*.parquet`, `checkup-design/results/`), so the downstream
analyses can be rerun without it.

## Running

Python 3.13 with numpy, pandas, scipy, matplotlib, pyarrow and PyBaMM. Run each script from its project folder,
for example

    cd prognostics && python -m src.step14_reviewer2
    cd checkup-design && python -m gap2.step6_bayes_crb
    cd checkup-design/paper && python fig1_forecast.py

Some scripts contain absolute paths to the author's data folders (`F:/Research New/...`); change them to your
download location.

## License

MIT (see `LICENSE`). The data belong to their original authors and keep their own licences.
