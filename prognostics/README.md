# battery-pinn-prognostics

Gap 1 follow-up to [battery-pinn-degradation](https://github.com/Akkhor07/battery-pinn-degradation) (JES
submission): forecasting degradation-mode, capacity and RUL trajectories from the label-free partial-window mode
estimates of the frozen Stage-1 estimator. See `PLAN.md` for the data audit, the corrections to the original
blueprint and the step plan.

The Stage-1 repository is used read-only (its trained runs, uncertainty files and raw data); set `STAGE1_ROOT` if
it is not at `F:\Research New\battery-pinn-degradation`. Use its environment:

```bash
"F:/Research New/battery-pinn-degradation/.venv/Scripts/python.exe" -m src.stage1_export
```

## Layout

```
src/stage1_export.py    Stage-1 check-up tables (no retraining)
data/                   generated tables (not versioned)
    checkups_nmc.parquet   16 NMC cells: modes from 4 estimators x 3 seeds, seed s.d., Laplace s.d. (raw and
                           calibrated on the other group), references, cycles, throughput, capacity, SOH
    checkups_lfp.parquet   20 LFP cells: same, plus model-free DVA LAM_NE
    capacity_nmc.parquet   capacity and published modes at all Kirkaldy RPTs (16 / 10 per cell)
```

Columns: `<mode>_<estimator>` with estimator in `pinn_enc`, `pinn_lm`, `cnn`, `cnn_dr` (relative to the cell's
first check-up, seed mean), `<mode>_<estimator>_seedsd`, `<mode>_ref`, `<mode>_sdraw`, `<mode>_sdcal`,
`logDn_pinn_*`, `logDp_pinn_*`, `cycles`, `Q_Ah`, `SOH`, `uid` (unique cell id).
