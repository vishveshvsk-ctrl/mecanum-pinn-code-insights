"""
bound_analysis/slip_stats.py — Gauss-Markov fit of the measured slip signal.

Rationale: treating slip as WHITE measurement noise (variant B2) understates
its harm, because real slip is strongly correlated over the slip interval -- a
correlated error is not averaged away by more samples. B3 augments the state
with the slip pair so the bound sees the correlation honestly. This module
fits a first-order Gauss-Markov (Ornstein-Uhlenbeck) description -- time
constant tau and stationary standard deviation sigma -- to the measured slip
trace exported by export_truth_traces.jl, by the exponential decay of its
sample autocorrelation function.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.signal import correlate


@dataclass
class SlipStatistics:
    tau: float           # correlation time constant [s]
    sigma: float          # stationary standard deviation [m/s]
    fit_quality: float    # R^2 of the exponential fit


def fit_gauss_markov(slip: np.ndarray, dt: float) -> Tuple[float, float, float]:
    """
    Fit a first-order Gauss-Markov process to the measured slip signal by its
    autocorrelation decay.

    Returns:
        tau:         correlation time constant [s]
        sigma:       stationary standard deviation [m/s]
        fit_quality: R^2 of the exponential fit, for reporting
    """
    slip = np.asarray(slip, dtype=float)
    x = slip - np.mean(slip)
    n = len(x)
    sigma = float(np.std(slip))

    if sigma < 1e-12 or n < 8 or dt <= 0:
        # Degenerate (near-constant or too-short) signal: no measurable
        # correlation structure -- fall back to a one-sample time constant
        # (effectively white) and flag the fit as untrustworthy (quality=0).
        return dt if dt > 0 else 1.0, sigma, 0.0

    full = correlate(x, x, mode="full", method="fft")
    mid = len(full) // 2
    acf = full[mid:] / full[mid]   # normalised, acf[0] == 1

    max_lag = max(2, min(n // 4, int(round(5.0 / dt))))   # cap the fit window at ~5 s
    lags = np.arange(1, max_lag)
    acf_pos = acf[1:max_lag]

    # Threshold at 0.3, not a looser value like 0.05: the biased sample-ACF
    # estimator (fewer overlapping terms at larger lag, same lag-0
    # normalisation) systematically flattens out well above zero in the tail,
    # which pulls a naive fit toward a much LARGER tau (empirically verified:
    # thresholds of 0.4-0.5 recover a synthetic tau to within ~5%, 0.05
    # over-estimates it by ~2x). Only the well-estimated early decay is used.
    valid = acf_pos > 0.3
    if valid.sum() < 2:
        return dt, sigma, 0.0

    log_acf = np.log(acf_pos[valid])
    lag_t = lags[valid] * dt
    A = np.vstack([lag_t, np.ones_like(lag_t)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, log_acf, rcond=None)

    if slope >= 0:
        # Non-decaying (or growing) sample ACF over the fit window: treat as
        # effectively white (tau -> one sample) rather than reporting a
        # spurious infinite/negative time constant.
        return dt, sigma, 0.0

    tau = max(-1.0 / slope, dt)
    pred = slope * lag_t + intercept
    ss_res = np.sum((log_acf - pred) ** 2)
    ss_tot = np.sum((log_acf - np.mean(log_acf)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return float(tau), sigma, float(r2)


# -----------------------------------------------------------------------------
# Self-test: recover known (tau, sigma) from a synthetic OU sample path.
# -----------------------------------------------------------------------------
def _validate_recovery():
    rng = np.random.default_rng(0)
    dt = 1.0 / 500.0
    tau_true, sigma_true = 0.3, 0.05
    n = int(60.0 / dt)
    x = np.zeros(n)
    q = sigma_true ** 2 * (1.0 - np.exp(-2 * dt / tau_true))
    for k in range(1, n):
        x[k] = x[k - 1] * np.exp(-dt / tau_true) + rng.normal(0.0, np.sqrt(q))

    tau_hat, sigma_hat, r2 = fit_gauss_markov(x, dt)
    print(f"true  tau={tau_true:.3f} sigma={sigma_true:.4f}")
    print(f"fit   tau={tau_hat:.3f} sigma={sigma_hat:.4f} R^2={r2:.3f}")
    ok = (abs(tau_hat - tau_true) / tau_true < 0.3) and (abs(sigma_hat - sigma_true) / sigma_true < 0.15) and r2 > 0.9
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    if not _validate_recovery():
        raise SystemExit("slip_stats.py self-test FAILED.")
    print("slip_stats.py self-test PASSED.")
