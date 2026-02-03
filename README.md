# Reviewer Submission Code

This package reproduces **all Methods computations** for the delayed adaptive feedback model and outputs.
## What it computes
- Success rate **S(τ)** for each (ε, σ, dφ) condition across τ
- Critical delay **τc** defined by **S(τc)=0.5** (linear interpolation)
- Recovery-time statistics (**mean/std/median**) for successful trials
- Optional per-trial audit table (includes pre-perturbation features)

## Files produced (in --outdir)
- `summary_long.csv` : long-form table with one row per τ and condition
- `tau_c_table.csv`  : one row per condition with τc
- `trials.csv`       : (optional) per-trial outputs for auditing
- `run_metadata.json`: full parameter + sweep metadata

## How to run (recommended manuscript settings)
```bash
python3 reviewer_numeric_sweep.py --outdir results_dt001_sin --dt 0.01 --feedback sin --n_trials 100 --seed 0
```

## Robustness checks
Time-step robustness:
```bash
python3 reviewer_numeric_sweep.py --outdir results_dt0005_sin --dt 0.005 --feedback sin --n_trials 100 --seed 0
python3 reviewer_numeric_sweep.py --outdir results_dt002_sin  --dt 0.02  --feedback sin --n_trials 100 --seed 0
```

Feedback nonlinearity robustness:
```bash
python3 reviewer_numeric_sweep.py --outdir results_dt001_tanh --dt 0.01 --feedback tanh --n_trials 100 --seed 0
```

## Quick sanity check (reduced sweep)
```bash
python3 reviewer_numeric_sweep.py --fast --outdir quick_check
```

## Notes
- Euler–Maruyama integration, Gaussian white noise (σ * sqrt(dt) * N(0,1))
- Delay implemented via circular buffer of the phase error e(t)
- Perturbation implemented as a reference-phase offset dφ applied after warm-up
