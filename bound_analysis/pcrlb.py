"""
bound_analysis/pcrlb.py — Tichavsky posterior CRLB information recursion.

The bound is a Riccati recursion, not an optimization: because the model is
additive-Gaussian and the Jacobians are evaluated along the KNOWN true
trajectory, the recursion reduces to a covariance propagation that runs in
seconds per trajectory. There is nothing to tune and nothing to converge.

Run this file directly (`python pcrlb.py`) to execute the two validation
checks from instructions/pcrlb-no-odometry-ellipse-bound.md §10:
  1. Straight-line constant-velocity, no measurements: sigma_pos(t) grows as
     t^(3/2) (double integrator driven by accelerometer noise).
  2. R -> 0 on the pose fix: sigma_pos / sigma_psi collapse to ~0 at every fix
     instant.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.linalg import solve_triangular, LinAlgError

from model import ImuKinematicModel, state_jacobian, process_noise, measurement_model


def _chol_inv(A: np.ndarray) -> np.ndarray:
    """Symmetric positive-definite inverse via Cholesky (brief §9: no explicit
    `inv`). Raises loudly on failure -- per the brief, that is a genuine signal
    the prior/Q/R is inconsistent, not something to silently regularize past."""
    try:
        L = np.linalg.cholesky(A)
    except LinAlgError as e:
        raise LinAlgError(
            "pcrlb: Cholesky failed -- matrix is not positive definite. "
            "This signals an inconsistent prior J0 / Q / R, not a numerical "
            "nuisance to paper over (see brief §9)."
        ) from e
    n = A.shape[0]
    Linv = solve_triangular(L, np.eye(n), lower=True)
    return Linv.T @ Linv


def pcrlb_recursion(model: ImuKinematicModel, truth: Dict[str, np.ndarray],
                     dt: float, J0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tichavsky posterior CRLB information recursion along a known true
    trajectory:

        J_{k+1} = inv( Q_k + F_k @ inv(J_k) @ F_k.T ) + H_{k+1}.T @ inv(R) @ H_{k+1}

    with F_k, H_k evaluated at the TRUE state. The bound on any unbiased
    estimator's error covariance is P >= inv(J_k).

    Args:
        model: variant-configured model
        truth: dict of [T] arrays -- t, psi, psidot, Vx, Vy, X, Y, slip
        dt:    trace timestep (uniform)
        J0:    initial information [n, n] (inverse of the initial-uncertainty prior)
    Returns:
        J:       [T, n, n]
        P_bound: [T, n, n]   (= inv(J), symmetrised)
    """
    n = model.n
    T = len(truth["t"])
    J = np.zeros((T, n, n))
    P_bound = np.zeros((T, n, n))

    J0 = 0.5 * (J0 + J0.T)
    J[0] = J0
    P_bound[0] = _chol_inv(J0)

    psi_arr, psidot_arr = truth["psi"], truth["psidot"]
    Vx_arr, Vy_arr, t_arr = truth["Vx"], truth["Vy"], truth["t"]
    slip_arr = truth.get("slip", np.zeros(T))

    for k in range(T - 1):
        F = state_jacobian(model, psi_arr[k], psidot_arr[k], Vx_arr[k], Vy_arr[k], dt)
        Q = process_noise(model, psi_arr[k], dt, t=t_arr[k], psidot=psidot_arr[k])

        Pk = _chol_inv(J[k])
        Pp = Q + F @ Pk @ F.T
        Pp = 0.5 * (Pp + Pp.T)
        Jp = _chol_inv(Pp)

        H, R = measurement_model(model, k + 1, slip_arr[k + 1], dt=dt)
        if H.shape[0] > 0:
            Rinv = _chol_inv(R)
            Jp = Jp + H.T @ Rinv @ H

        Jp = 0.5 * (Jp + Jp.T)
        J[k + 1] = Jp
        P_bound[k + 1] = _chol_inv(Jp)

    return J, P_bound


def bound_sigmas(P_bound: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Reduce the bound covariance to reportable scalar standard deviations.

    Returns dict of [T] arrays:
        sigma_pos  -- sqrt(P[3,3] + P[4,4])   horizontal position
        sigma_vel  -- sqrt(P[0,0] + P[1,1])   body velocity
        sigma_psi  -- sqrt(P[2,2])            heading
        sigma_bg   -- sqrt(P[7,7])            gyro bias
    """
    return {
        "sigma_pos": np.sqrt(P_bound[:, 3, 3] + P_bound[:, 4, 4]),
        "sigma_vel": np.sqrt(P_bound[:, 0, 0] + P_bound[:, 1, 1]),
        "sigma_psi": np.sqrt(P_bound[:, 2, 2]),
        "sigma_bg":  np.sqrt(P_bound[:, 7, 7]),
    }


# -----------------------------------------------------------------------------
# Self-test / validation (brief §10)
# -----------------------------------------------------------------------------
def _straight_line_truth(T_total: float, dt: float, Vx0: float = 1.0) -> Dict[str, np.ndarray]:
    n = round(T_total / dt) + 1
    t = np.linspace(0.0, T_total, n)
    return dict(
        t=t, psi=np.zeros(n), psidot=np.zeros(n),
        Vx=np.full(n, Vx0), Vy=np.zeros(n),
        X=Vx0 * t, Y=np.zeros(n), slip=np.zeros(n),
    )


def _validate_no_measurement_growth():
    """§10 check 1: no measurements -> sigma_pos(t) grows ~ t^(3/2)."""
    dt = 1.0 / 500.0
    T_total = 40.0
    truth = _straight_line_truth(T_total, dt)
    model = ImuKinematicModel(variant="B0", sigma_acc=0.05, sigma_gyro=0.005,
                               fix_rate_hz=0.0)   # fix_rate_hz<=0 -> no measurements at all
    # TIGHT initial prior (J0 = inv(P0) LARGE => P0 small): position and
    # velocity both well-known at t=0, so the growth we measure is purely the
    # process-noise-driven double-integrator effect, not an artifact of the
    # initial-velocity uncertainty also propagating into position through F.
    J0 = np.eye(model.n) * 1e6
    J, P = pcrlb_recursion(model, truth, dt, J0)
    sig = bound_sigmas(P)["sigma_pos"]

    t = truth["t"]
    # Fit log(sigma) = a + b*log(t) over the back half (skip the t~0 transient
    # dominated by the prior) and check the exponent is close to 1.5.
    mask = t > T_total * 0.3
    b, a = np.polyfit(np.log(t[mask]), np.log(sig[mask]), 1)
    print(f"[check 1] no-measurement sigma_pos(t) power-law exponent = {b:.3f}  (expect ~1.5)")
    ok = 1.2 <= b <= 1.8
    print(f"[check 1] {'PASS' if ok else 'FAIL'}")
    return ok


def _validate_perfect_fix_collapse():
    """§10 check 2: R -> 0 on the pose fix -> sigma_pos/sigma_psi collapse to
    ~0 at every fix instant."""
    dt = 1.0 / 500.0
    T_total = 5.0
    truth = _straight_line_truth(T_total, dt)
    model = ImuKinematicModel(variant="B0", sigma_acc=0.05, sigma_gyro=0.005,
                               sigma_pos=1e-6, sigma_psi=1e-6, fix_rate_hz=100.0)
    J0 = np.eye(model.n) * 1e-6
    J, P = pcrlb_recursion(model, truth, dt, J0)
    sig = bound_sigmas(P)
    steps_per_fix = round(1.0 / (model.fix_rate_hz * dt))
    fix_idxs = np.arange(steps_per_fix, len(truth["t"]), steps_per_fix)
    pos_at_fix = sig["sigma_pos"][fix_idxs]
    psi_at_fix = sig["sigma_psi"][fix_idxs]
    print(f"[check 2] sigma_pos at fix instants: max={pos_at_fix.max():.2e} (expect ~1e-6 scale)")
    print(f"[check 2] sigma_psi at fix instants: max={psi_at_fix.max():.2e} (expect ~1e-6 scale)")
    ok = (pos_at_fix.max() < 1e-4) and (psi_at_fix.max() < 1e-4)
    print(f"[check 2] {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ok1 = _validate_no_measurement_growth()
    ok2 = _validate_perfect_fix_collapse()
    if ok1 and ok2:
        print("All pcrlb.py validation checks PASSED.")
    else:
        raise SystemExit("pcrlb.py validation FAILED — see output above.")
