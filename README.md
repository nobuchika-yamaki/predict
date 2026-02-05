Reviewer Submission Code
This package reproduces all Methods computations for the delayed adaptive feedback model and outputs.

What it computes
Primary Analysis (reviewer_numeric_sweep.py):

Success rate S(τ) for each (ε, σ, dφ) condition across τ.

Critical delay τc defined by S(τc)=0.5 (linear interpolation).

Recovery-time statistics (mean/std/median) for successful trials.

Extended 2D Analysis (reproduction_extended_analysis.py):

Validates the model in 2D Phase-Velocity dynamics (with inertia and damping γ).

Performs a Damping Rate (γ) sweep to prove the invariance of τ 
c
​	
 .

Computes 95% Confidence Intervals (CI) via 1,000 bootstrap resamples for all τ 
c
​	
 estimates.

Files produced (in --outdir)
summary_long.csv: long-form table with one row per τ and condition.

tau_c_table.csv: one row per condition with τc.

extended_analysis_results.csv: (from 2D script) γ, τ 
c
​	
 , and 95% CI bounds.

run_metadata.json: full parameter + sweep metadata.

How to run (recommended manuscript settings)
1. Main 1D Sweep (Figure 2 & 3)

Bash
python3 reviewer_numeric_sweep.py --outdir results_1d --dt 0.01 --feedback sin --n_trials 100 --seed 0
2. Extended 2D & Stability Analysis (Figure 4 & 5)

This script reproduces the core physical argument: that τ 
c
​	
  is independent of the damping rate γ.

Bash
python3 reproduction_extended_analysis.py --n_trials 100 --n_boot 1000
Robustness checks
Time-step robustness:

Bash
python3 reviewer_numeric_sweep.py --outdir results_dt0005_sin --dt 0.005 --feedback sin --n_trials 100 --seed 0
Feedback nonlinearity robustness:

Bash
python3 reviewer_numeric_sweep.py --outdir results_dt001_tanh --dt 0.01 --feedback tanh --n_trials 100 --seed 0
Quick sanity check (reduced sweep)
Bash
python3 reviewer_numeric_sweep.py --fast --outdir quick_check
Scientific Logic in Extended Analysis
The reproduction_extended_analysis.py script specifically addresses the "Information Staleness" hypothesis. By sweeping the damping rate γ across a ten-fold range (γ∈[0.3,3.0]) in a 2D system, it demonstrates that the recoverability boundary τ 
c
​	
  remains statistically invariant. This confirms that the failure is not a result of physical sluggishness but a fundamental breakdown of temporal information.

Notes
Dynamics: Euler–Maruyama integration.

2D Model: Includes velocity coupling (dθ/dt=v) to test structural stability.

Uncertainty: All τ 
c
​	
  values in the extended analysis include bootstrap-based error bars to ensure statistical rigor.
