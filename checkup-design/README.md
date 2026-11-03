# battery-pinn-checkup-design

Gap 2 follow-up to [battery-pinn-degradation](https://github.com/Akkhor07/battery-pinn-degradation) (JES
submission): choosing the partial check-up windows (position, size, C-rate) with the Fisher information of the
Stage-1 model, so that a short abbreviated check-up identifies the degradation modes as well as a long one.

The Stage-1 repository is used read-only (trained runs, CRB engine, raw data); set `STAGE1_ROOT` if it is not at
`F:\Research New\battery-pinn-degradation`, and use its environment:

```bash
py="F:/Research New/battery-pinn-degradation/.venv/Scripts/python.exe"
$py -m gap2.step1_design_grid --part crb       # CRB of 50 window-pair designs (PyBaMM, a few minutes)
$py -m gap2.step1_design_grid --part eval      # frozen PINN+LM on every design and NMC check-up (~20 min)
$py -m gap2.step1_design_grid --part analyse
```

The package is called `gap2` (not `src`) so it does not clash with the Stage-1 `src` package on the path.
See `PLAN.md` for the go/no-go logic and the step plan.
