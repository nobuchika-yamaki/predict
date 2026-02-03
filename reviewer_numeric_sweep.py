#!/usr/bin/env python3
"""
Reviewer-submission code (numbers only; no figures)

Implements the delayed adaptive feedback model described in the manuscript:
  dθ/dt = ω + K(t) F(e(t-τ)) + σ ξ(t)
  dK/dt = ε [A - cos(e(t))] - μ (K(t) - K0)
  e(t)  = wrap(θ_ref(t) - θ(t)), θ_ref(t)=ω_ref t

Euler–Maruyama integration with circular delay buffer for e(t-τ).
Outputs numeric summaries:
  - Success rate S(τ) for each parameter condition
  - Critical delay τc (S=0.5 via linear interpolation)
  - Recovery-time statistics (mean, std, median) for successful trials
Optionally outputs per-trial data for auditing.

No plotting is performed.
"""

from __future__ import annotations
import argparse
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


def wrap_angle(x: np.ndarray | float) -> np.ndarray | float:
    """Map angle(s) to (-pi, pi]."""
    # Using modulo that works for numpy arrays and floats
    return (x + math.pi) % (2 * math.pi) - math.pi


def feedback_fn(name: str) -> Callable[[np.ndarray], np.ndarray]:
    name = name.lower()
    if name == "sin":
        return np.sin
    if name == "tanh":
        return np.tanh
    raise ValueError(f"Unknown feedback nonlinearity: {name}. Use 'sin' or 'tanh'.")


@dataclass(frozen=True)
class Params:
    # Core
    omega: float = 2 * math.pi
    omega_ref: float = 2 * math.pi
    K0: float = 1.0
    A: float = 0.5
    mu: float = 0.1

    # Protocol
    T_warm: float = 20.0
    T_obs: float = 20.0
    e_th: float = 0.05 * math.pi
    T_hold: float = 1.0

    # Numerical
    dt: float = 0.01

    # Sweep lists (defaults match manuscript)
    taus: Tuple[float, ...] = tuple(np.round(np.arange(0.0, 2.5001, 0.05), 8))
    eps_list: Tuple[float, ...] = (0.02, 0.05, 0.15)
    sig_list: Tuple[float, ...] = (0.0, 0.05, 0.15)
    dphi_list: Tuple[float, ...] = (0.2 * math.pi, 0.4 * math.pi, 0.6 * math.pi)

    n_trials: int = 100
    feedback: str = "sin"

    # For optional pre-state logging
    pre_drift_window: float = 0.5  # seconds


def simulate_one(
    p: Params,
    eps: float,
    sigma: float,
    tau: float,
    dphi: float,
    rng: np.random.Generator,
    log_trial: bool = False,
) -> Dict[str, float]:
    """
    Run one realization: warm-up, then perturb, then observe.
    Returns success flag, recovery time Ts (nan if fail), and pre-state features.
    """
    dt = p.dt
    n_warm = int(round(p.T_warm / dt))
    n_obs = int(round(p.T_obs / dt))
    n_hold = max(1, int(round(p.T_hold / dt)))
    drift_n = max(2, int(round(p.pre_drift_window / dt)))

    F = feedback_fn(p.feedback)

    # Delay buffer length for e(t-τ)
    delay_steps = int(round(tau / dt))
    buf_len = max(1, delay_steps + 1)  # at least 1
    e_buf = np.zeros(buf_len, dtype=float)
    buf_idx = 0

    # Initialize phase uniformly; K starts at K0 (can be changed if desired)
    theta = rng.uniform(-math.pi, math.pi)
    K = p.K0

    # Initial error and fill buffer with constant history
    # e(t) = wrap(ω_ref t - θ(t)) ; at t=0, θ_ref=0
    e0 = wrap_angle(0.0 - theta)
    e_buf[:] = e0

    # Helper to get delayed error
    def get_e_delayed() -> float:
        if delay_steps == 0:
            return e_buf[buf_idx]
        # delayed index relative to current write index
        j = (buf_idx - delay_steps) % buf_len
        return e_buf[j]

    # Warm-up integration
    # We integrate theta and K; we store current e into buffer each step.
    for i in range(n_warm):
        t = (i + 1) * dt
        theta_ref = p.omega_ref * t
        e = wrap_angle(theta_ref - theta)

        # Euler–Maruyama: noise term sigma * sqrt(dt) * N(0,1)
        dW = rng.normal(0.0, 1.0) * math.sqrt(dt)
        theta += (p.omega * dt) + (K * F(get_e_delayed()) * dt) + (sigma * dW)
        K += (eps * (p.A - math.cos(e)) - p.mu * (K - p.K0)) * dt

        # update buffer with current error (after using for K update)
        buf_idx = (buf_idx + 1) % buf_len
        e_buf[buf_idx] = e

    # Pre-perturbation features at end of warm-up
    t_pre = p.T_warm
    theta_ref_pre = p.omega_ref * t_pre
    e_pre = wrap_angle(theta_ref_pre - theta)
    K_pre = K

    # Estimate local error drift de/dt from finite differences over short window
    # We re-compute by simulating drift window without perturbation but without altering main state:
    # For simplicity & reproducibility, approximate drift using last drift_n stored errors in buffer.
    # Note: buffer holds e values at discrete times; use last drift_n values.
    if drift_n <= buf_len:
        # take last drift_n samples ending at current buf_idx
        idxs = [(buf_idx - k) % buf_len for k in range(drift_n)][::-1]
        e_hist = e_buf[idxs]
    else:
        # if drift window larger than buffer, fall back to repeating buffer
        reps = int(math.ceil(drift_n / buf_len))
        e_hist = np.tile(e_buf, reps)[:drift_n]
    de_dt = float(np.mean(np.diff(e_hist)) / dt)

    # Apply perturbation: shift reference phase -> effectively shift error by dphi at onset.
    # Implement by adding dphi to error used for success check; dynamics continue with theta_ref(t) unchanged,
    # but since the reference is shifted, the effective error becomes e + dphi.
    # Equivalent: at perturbation onset, define a phase offset added to theta_ref thereafter.
    ref_offset = dphi

    # Observation integration and recovery detection
    below_count = 0
    Ts = math.nan
    success = 0

    for j in range(n_obs):
        t = p.T_warm + (j + 1) * dt
        theta_ref = p.omega_ref * t + ref_offset
        e = wrap_angle(theta_ref - theta)

        dW = rng.normal(0.0, 1.0) * math.sqrt(dt)
        theta += (p.omega * dt) + (K * F(get_e_delayed()) * dt) + (sigma * dW)
        K += (eps * (p.A - math.cos(e)) - p.mu * (K - p.K0)) * dt

        buf_idx = (buf_idx + 1) % buf_len
        e_buf[buf_idx] = e

        if abs(e) < p.e_th:
            below_count += 1
            if below_count >= n_hold:
                # first time sustained criterion met
                Ts = (j + 1 - n_hold + 1) * dt
                success = 1
                break
        else:
            below_count = 0

    out = {
        "success": float(success),
        "Ts": float(Ts),
        "e_pre": float(e_pre),
        "K_pre": float(K_pre),
        "de_dt_pre": float(de_dt),
    }
    return out


def estimate_tau_c(taus: np.ndarray, S: np.ndarray) -> float:
    """
    Critical delay τc where S(τ)=0.5 via linear interpolation between adjacent τ.
    Returns nan if never crosses 0.5.
    """
    target = 0.5
    # Find first index where S falls below target (assuming typically decreasing with τ)
    for i in range(1, len(taus)):
        if (S[i-1] >= target and S[i] <= target) or (S[i-1] <= target and S[i] >= target):
            # interpolate
            t0, t1 = taus[i-1], taus[i]
            s0, s1 = S[i-1], S[i]
            if s1 == s0:
                return float((t0 + t1) / 2)
            frac = (target - s0) / (s1 - s0)
            return float(t0 + frac * (t1 - t0))
    return float("nan")


def run_sweep(
    p: Params,
    seed: int = 0,
    keep_trials: bool = False,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    rng_master = np.random.default_rng(seed)

    taus = np.array(p.taus, dtype=float)
    rows_summary: List[Dict[str, float]] = []
    rows_trials: List[Dict[str, float]] = []

    for eps in p.eps_list:
        for sigma in p.sig_list:
            for dphi in p.dphi_list:
                S_tau = np.zeros_like(taus)
                Ts_mean = np.full_like(taus, np.nan)
                Ts_std = np.full_like(taus, np.nan)
                Ts_median = np.full_like(taus, np.nan)

                for ti, tau in enumerate(taus):
                    # independent trials
                    succ = np.zeros(p.n_trials, dtype=float)
                    Ts = np.full(p.n_trials, np.nan, dtype=float)

                    for k in range(p.n_trials):
                        # Spawn per-trial RNG deterministically from master for reproducibility
                        trial_seed = int(rng_master.integers(0, 2**32 - 1))
                        rng = np.random.default_rng(trial_seed)
                        res = simulate_one(p, eps, sigma, tau, dphi, rng)

                        succ[k] = res["success"]
                        Ts[k] = res["Ts"]

                        if keep_trials:
                            rows_trials.append({
                                "eps": eps,
                                "sigma": sigma,
                                "dphi": dphi,
                                "tau": tau,
                                "trial": k,
                                "success": res["success"],
                                "Ts": res["Ts"],
                                "e_pre": res["e_pre"],
                                "K_pre": res["K_pre"],
                                "de_dt_pre": res["de_dt_pre"],
                                "seed": trial_seed,
                                "dt": p.dt,
                                "feedback": p.feedback,
                            })

                    S_tau[ti] = float(np.mean(succ))
                    if np.any(~np.isnan(Ts)):
                        Ts_valid = Ts[~np.isnan(Ts)]
                        Ts_mean[ti] = float(np.mean(Ts_valid))
                        Ts_std[ti] = float(np.std(Ts_valid, ddof=1)) if Ts_valid.size > 1 else 0.0
                        Ts_median[ti] = float(np.median(Ts_valid))

                tau_c = estimate_tau_c(taus, S_tau)

                # Save per-tau curve in summary (long form)
                for ti, tau in enumerate(taus):
                    rows_summary.append({
                        "eps": eps,
                        "sigma": sigma,
                        "dphi": dphi,
                        "tau": float(tau),
                        "S": float(S_tau[ti]),
                        "Ts_mean": float(Ts_mean[ti]) if not math.isnan(Ts_mean[ti]) else np.nan,
                        "Ts_std": float(Ts_std[ti]) if not math.isnan(Ts_std[ti]) else np.nan,
                        "Ts_median": float(Ts_median[ti]) if not math.isnan(Ts_median[ti]) else np.nan,
                        "tau_c_for_condition": float(tau_c),
                        "dt": p.dt,
                        "feedback": p.feedback,
                        "n_trials": p.n_trials,
                    })

    df_summary = pd.DataFrame(rows_summary)
    df_trials = pd.DataFrame(rows_trials) if keep_trials else None
    return df_summary, df_trials


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run delayed adaptive feedback sweeps and output numeric results (no figures)."
    )
    ap.add_argument("--outdir", default="results_numeric", help="Output directory.")
    ap.add_argument("--seed", type=int, default=0, help="Master RNG seed.")
    ap.add_argument("--dt", type=float, default=0.01, help="Time step (e.g., 0.005, 0.01, 0.02).")
    ap.add_argument("--feedback", choices=["sin", "tanh"], default="sin", help="Feedback nonlinearity.")
    ap.add_argument("--n_trials", type=int, default=100, help="Trials per condition.")
    ap.add_argument("--keep_trials", action="store_true", help="Save per-trial outputs for auditing.")
    ap.add_argument("--fast", action="store_true", help="Reduced sweep for quick sanity check.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    base = Params(dt=args.dt, feedback=args.feedback, n_trials=args.n_trials)

    if args.fast:
        # Reduced sweep for quick runtime check
        base = Params(
            dt=args.dt,
            feedback=args.feedback,
            n_trials=min(args.n_trials, 20),
            taus=tuple(np.round(np.arange(0.0, 2.0001, 0.10), 8)),
            eps_list=(0.05,),
            sig_list=(0.0, 0.15),
            dphi_list=(0.4 * math.pi,),
        )

    df_summary, df_trials = run_sweep(base, seed=args.seed, keep_trials=args.keep_trials)

    # Output long-form summary
    summary_path = os.path.join(args.outdir, "summary_long.csv")
    df_summary.to_csv(summary_path, index=False)

    # Condition-level τc table
    group_cols = ["eps", "sigma", "dphi", "dt", "feedback", "n_trials"]
    df_tau = (
        df_summary[group_cols + ["tau_c_for_condition"]]
        .drop_duplicates()
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    tau_path = os.path.join(args.outdir, "tau_c_table.csv")
    df_tau.to_csv(tau_path, index=False)

    # Optional per-trial output
    if df_trials is not None:
        trial_path = os.path.join(args.outdir, "trials.csv")
        df_trials.to_csv(trial_path, index=False)

    # Save run metadata
    meta = {
        "dt": base.dt,
        "feedback": base.feedback,
        "n_trials": base.n_trials,
        "taus": list(base.taus),
        "eps_list": list(base.eps_list),
        "sig_list": list(base.sig_list),
        "dphi_list": list(base.dphi_list),
        "T_warm": base.T_warm,
        "T_obs": base.T_obs,
        "e_th": base.e_th,
        "T_hold": base.T_hold,
        "omega": base.omega,
        "omega_ref": base.omega_ref,
        "K0": base.K0,
        "A": base.A,
        "mu": base.mu,
        "seed": args.seed,
        "keep_trials": args.keep_trials,
        "fast": args.fast,
    }
    with open(os.path.join(args.outdir, "run_metadata.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(meta, f, indent=2)

    # Print a compact terminal summary
    # (This is still "numbers", not figures.)
    print("Wrote:")
    print(" -", summary_path)
    print(" -", tau_path)
    if df_trials is not None:
        print(" -", os.path.join(args.outdir, "trials.csv"))
    print(" -", os.path.join(args.outdir, "run_metadata.json"))
    print()
    # Show a small preview of τc table
    print("Preview: tau_c_table (first 10 rows)")
    print(df_tau.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
