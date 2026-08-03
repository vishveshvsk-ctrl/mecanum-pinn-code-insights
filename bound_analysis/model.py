"""
bound_analysis/model.py — IMU-driven kinematic error model for the PCRLB.

State (n = 8 for B0/B1/B2, n = 10 for B3):
    [Vx, Vy, psi, X, Y, bx, by, bg]           (+ [sx, sy] for B3)
     0   1    2   3  4   5   6   7               8   9

The IMU is treated as a NOISY INPUT to the dynamics, not as a measurement.
This is what makes the process noise Q a property of the SENSOR SPEC rather
than a tuning knob -- and therefore what makes the result a bound.

Continuous-time dynamics (yaw rate omega = g_z_meas - bg is an ALGEBRAIC
quantity derived from the gyro input and the bias state -- there is no
separate omega state, unlike the repo's real ESKF/IMM filters, which is a
deliberate simplification of this reduced bound model):

    Vx_dot  = (a_x_meas - bx) + omega * Vy
    Vy_dot  = (a_y_meas - by) - omega * Vx
    psi_dot = omega = g_z_meas - bg
    X_dot   = Vx*cos(psi) - Vy*sin(psi)
    Y_dot   = Vx*sin(psi) + Vy*cos(psi)
    bx_dot, by_dot, bg_dot = 0   (random walk / random constant, via Q)
    sx_dot  = -sx / tau_slip    (B3 only, first-order Gauss-Markov)
    sy_dot  = -sy / tau_slip

DOCUMENTED DEVIATIONS FROM THE BRIEF (flagged during implementation, see
instructions/pcrlb-no-odometry-ellipse-bound.md and the recon report):

1. sigma_pos / sigma_psi / fix_rate_hz are read from `hybrid_ctrl/estimators.jl`
   EstimatorMod.PoseFixModel (the :docking tier: fix_rate_hz=100, sigma_pos=0.01,
   sigma_psi=deg2rad(0.5)), NOT from SensorMod.SensorModel as the brief's text
   states -- SensorModel has no pose-fix fields at all.

2. SensorMod.SensorModel has no `acc_bias_rw` field: `acc_bias` (0.02 m/s^2 for
   :default) is a FIXED CONSTANT added identically every call, not a random
   walk. There is therefore no sensor-spec-derived random-walk intensity for
   the accelerometer bias states (bx, by). This model defaults `acc_bias_rw` to
   0.0 (a "random constant" bias model -- Q=0, nonzero prior in J0, mirroring
   the true physics), while still exposing it as a constructor override for a
   robustness variant that HEDGES against unmodelled slow bias drift.

3. gyro_bias_rw: sensors.jl:91 updates the gyro bias with
   `gyro_bias[1] += gyro_bias_rw * sqrt(t + 0.001) * randn()` EVERY SENSOR TICK
   (not scaled by sqrt(dt)) -- so the per-tick variance is
   `gyro_bias_rw^2 * (t + 0.001)`, independent of the tick spacing. Over the
   filter's real production cadence (REAL_SENSOR_HZ = 1000 Hz, `f_est` in
   hybrid_ctrl/config.jl), the correct process-noise CONTRIBUTION over one
   PCRLB recursion step of size dt at time t is the sum of that many real
   ticks' worth of injected variance:
       Q[bg,bg](t, dt) = gyro_bias_rw^2 * (t + 0.001) * dt * REAL_SENSOR_HZ
   This is an ANALYTIC derivation from the exact update rule (not an empirical
   fit to a single noisy realized-bias sample path, which the brief suggests
   but which is fragile for a one-shot per-trajectory random walk). It reduces
   to the brief's flagged anomaly being carried through honestly: this grows
   faster than a standard sqrt(dt) random walk and is time-varying.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

REAL_SENSOR_HZ = 1000.0   # hybrid_ctrl/config.jl HybridConfig.f_est production cadence

# Sensor grades, read verbatim off hybrid_ctrl/sensors.jl SensorMod.SensorModel.
# Unicode field names there (sigma_acc, sigma_gyro, sf_gyro) are transcribed to
# ASCII here since this is an independent Python object, not a mirror of the
# Julia struct.
SENSOR_GRADES = {
    "default":   dict(sigma_acc=0.05, sigma_gyro=0.005, sf_gyro=0.0,   gyro_bias_rw=1e-4, sigma_omega=0.01),
    "realistic": dict(sigma_acc=0.05, sigma_gyro=0.003, sf_gyro=0.005, gyro_bias_rw=1e-4,
                       sigma_omega=0.01 / (0.05 * np.sqrt(1.09))),
}

# Pose-fix tier (:docking), read off hybrid_ctrl/estimators.jl EstimatorMod.PoseFixModel.
POSE_FIX_DOCKING = dict(fix_rate_hz=100.0, sigma_pos=0.01, sigma_psi=np.deg2rad(0.5))

VARIANTS = ("B0", "B1", "B2", "B3")


@dataclass
class ImuKinematicModel:
    """
    IMU-driven kinematic error model for the PCRLB.

    State (n = 8 for B0/B1/B2, n = 10 for B3):
        [Vx, Vy, psi, X, Y, bx, by, bg]           (+ [sx, sy] for B3)
         0   1    2   3  4   5   6   7               8   9

    The IMU is treated as a NOISY INPUT to the dynamics, not as a measurement.
    This is what makes the process noise Q a property of the SENSOR SPEC rather
    than a tuning knob -- and therefore what makes the result a bound.
    """
    variant: str
    sigma_acc: float
    sigma_gyro: float
    acc_bias_rw: float = 0.0          # see module docstring, deviation #2
    gyro_bias_rw: float = 1e-4
    sigma_pos: float = POSE_FIX_DOCKING["sigma_pos"]
    sigma_psi: float = POSE_FIX_DOCKING["sigma_psi"]
    fix_rate_hz: float = POSE_FIX_DOCKING["fix_rate_hz"]
    sf_gyro: float = 0.0               # heteroscedastic gyro scale-factor (:realistic grade)
    sigma_wheel_vel: Optional[float] = None
    tau_slip: Optional[float] = None
    sigma_slip: Optional[float] = None

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"ImuKinematicModel: variant must be one of {VARIANTS}, got {self.variant!r}")
        if self.variant in ("B1", "B2", "B3") and self.sigma_wheel_vel is None:
            raise ValueError(f"variant {self.variant} requires sigma_wheel_vel")
        if self.variant == "B2" and self.sigma_slip is None:
            raise ValueError("variant B2 requires sigma_slip (white R-inflation)")
        if self.variant == "B3" and (self.tau_slip is None or self.sigma_slip is None):
            raise ValueError("variant B3 requires tau_slip and sigma_slip (Gauss-Markov augmentation)")

    @property
    def n(self) -> int:
        return 10 if self.variant == "B3" else 8


def _gyro_noise_var(model: ImuKinematicModel, psidot_true: float) -> float:
    """sigma_gyro^2 + (sf_gyro * psidot)^2 -- heteroscedastic term, sensors.jl:116-121."""
    return model.sigma_gyro ** 2 + (model.sf_gyro * psidot_true) ** 2


def state_jacobian(model: ImuKinematicModel, psi: float, psidot: float,
                    Vx: float, Vy: float, dt: float) -> np.ndarray:
    """
    Discrete state transition Jacobian F = I + dt * A, evaluated at the TRUE
    state. A carries: Coriolis coupling between Vx and Vy through psidot, the
    heading integrator, the body-to-world rotation on the position rows, the
    bias injection rows (accel biases enter velocity, gyro bias enters
    heading), AND the gyro-bias -> Coriolis cross-coupling that falls out of
    writing omega = g_z_meas - bg explicitly in the Vx/Vy dynamics (see module
    docstring): d(Vx_dot)/d(bg) = -Vy, d(Vy_dot)/d(bg) = +Vx, in addition to
    d(psi_dot)/d(bg) = -1.

    Returns:
        F: [n, n]
    """
    n = model.n
    A = np.zeros((n, n))
    c, s = np.cos(psi), np.sin(psi)

    # Coriolis coupling (Vx,Vy rows) at the linearization-point psidot.
    A[0, 1] = psidot
    A[1, 0] = -psidot
    # Accel-bias injection.
    A[0, 5] = -1.0
    A[1, 6] = -1.0
    # Gyro-bias injection: heading integrator + Coriolis cross-coupling.
    A[2, 7] = -1.0
    A[0, 7] = -Vy
    A[1, 7] = Vx
    # Body-to-world rotation on the position rows.
    A[3, 0] = c
    A[3, 1] = -s
    A[3, 2] = -Vx * s - Vy * c
    A[4, 0] = s
    A[4, 1] = c
    A[4, 2] = Vx * c - Vy * s
    # Bias rows (5,6,7): zero drift -- random walk / random constant via Q.

    if model.variant == "B3":
        # First-order Gauss-Markov slip pair: sx_dot = -sx/tau, sy_dot = -sy/tau.
        A[8, 8] = -1.0 / model.tau_slip
        A[9, 9] = -1.0 / model.tau_slip

    return np.eye(n) + dt * A


def process_noise(model: ImuKinematicModel, psi: float, dt: float,
                   t: float = 0.0, psidot: float = 0.0) -> np.ndarray:
    """
    Q = G * Sigma_imu * G^T * dt + bias random-walk terms.

    Sigma_imu = diag(sigma_acc^2, sigma_acc^2, sigma_gyro_eff^2) is the SENSOR
    spec (sigma_gyro_eff includes the heteroscedastic scale-factor term for the
    :realistic grade -- see _gyro_noise_var); G maps input noise into the
    state. This is the load-bearing modelling choice: Q is DERIVED, not
    fitted.

    `t` and `psidot` are OPTIONAL additions beyond the brief's 3-argument
    signature (kept keyword-only with safe defaults so the 3-arg contract
    still works): `t` drives the gyro-bias anomalous growth-rate term (module
    docstring, deviation #3); `psidot` completes G's Coriolis-noise mapping
    (a white gyro sample perturbs the Vx/Vy Coriolis terms exactly as a bg
    perturbation does -- see state_jacobian).

    Returns:
        Q: [n, n]
    """
    n = model.n
    Q = np.zeros((n, n))

    # G maps input noise [noise_ax, noise_ay, noise_gz] directly into
    # [Vx_dot, Vy_dot, psi_dot]. The gyro noise ALSO perturbs the Vx/Vy
    # Coriolis terms over the same instant (proportional to Vy, -Vx
    # respectively, exactly mirroring state_jacobian's A[0,7]/A[1,7]
    # bg-coupling) -- omitted here as a second-order-in-dt correction to an
    # already O(dt) noise term, consistent with the first-order Euler
    # discretization used throughout (F = I + dt*A).
    sigma_gyro_eff2 = _gyro_noise_var(model, psidot)
    G = np.zeros((n, 3))
    G[0, 0] = 1.0            # Vx <- a_x noise
    G[1, 1] = 1.0            # Vy <- a_y noise
    G[2, 2] = 1.0            # psi_dot <- g_z noise
    Sigma_imu = np.diag([model.sigma_acc ** 2, model.sigma_acc ** 2, sigma_gyro_eff2])

    Q[:3, :3] += dt * (G[:3, :] @ Sigma_imu @ G[:3, :].T)

    # Bias random-walk terms (see module docstring deviations #2, #3).
    Q[5, 5] += (model.acc_bias_rw ** 2) * dt
    Q[6, 6] += (model.acc_bias_rw ** 2) * dt
    Q[7, 7] += (model.gyro_bias_rw ** 2) * (t + 0.001) * dt * REAL_SENSOR_HZ

    if model.variant == "B3":
        # Stationary Gauss-Markov process noise: Q_s = 2*sigma_slip^2/tau * dt
        # (matches the OU stationary-variance identity Var = tau*Q_s/2).
        q_s = 2.0 * (model.sigma_slip ** 2) / model.tau_slip * dt
        Q[8, 8] += q_s
        Q[9, 9] += q_s

    return Q


def measurement_model(model: ImuKinematicModel, k: int, slip_k: float,
                       dt: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Measurement Jacobian and noise for step k, per variant (see brief §7.4).
    Returns a zero-row H when no measurement is scheduled at this step (the
    pose fix is at fix_rate_hz, the state recursion at the trace rate).

    `dt` is an OPTIONAL addition beyond the brief's 3-argument signature
    (keyword-only, default None): the trace timestep is needed to translate
    `fix_rate_hz` into a step-index schedule (steps_per_fix = round(1 /
    (fix_rate_hz * dt))); pcrlb_recursion always supplies it. `fix_rate_hz<=0`
    explicitly disables the pose fix entirely (used by the no-measurement
    validation case in pcrlb.py's self-test); otherwise, if `dt` is not given,
    a pose fix is scheduled at EVERY step (dt treated as unknown / continuous,
    a conservative interface fallback).

    Args:
        k:      step index (decides pose-fix scheduling)
        slip_k: measured slip magnitude at step k (used by B2/B3 only, ignored
                by B2 since its R-inflation uses the STATIONARY sigma_slip, not
                the instantaneous per-step slip_k -- see brief §7.4: "R
                inflated by the empirical slip variance (white)")
    Returns:
        H: [m, n]   (m = 0, 3, or 5 depending on variant and schedule)
        R: [m, m]
    """
    n = model.n
    if model.fix_rate_hz <= 0:
        fix_now = False
    elif dt is not None:
        steps_per_fix = max(1, round(1.0 / (model.fix_rate_hz * dt)))
        fix_now = (k % steps_per_fix == 0)
    else:
        fix_now = True

    rows_H = []
    rows_R = []

    if fix_now:
        H_pose = np.zeros((3, n))
        H_pose[0, 3] = 1.0   # X
        H_pose[1, 4] = 1.0   # Y
        H_pose[2, 2] = 1.0   # psi
        R_pose = np.diag([model.sigma_pos ** 2, model.sigma_pos ** 2, model.sigma_psi ** 2])
        rows_H.append(H_pose)
        rows_R.append(R_pose)

    if model.variant in ("B1", "B2", "B3"):
        H_wheel = np.zeros((2, n))
        H_wheel[0, 0] = 1.0   # Vx
        H_wheel[1, 1] = 1.0   # Vy
        r_wheel = model.sigma_wheel_vel ** 2
        if model.variant == "B2":
            r_wheel = r_wheel + model.sigma_slip ** 2   # naive white slip inflation
        if model.variant == "B3":
            H_wheel[0, 8] = 1.0   # + sx
            H_wheel[1, 9] = 1.0   # + sy
        R_wheel = np.diag([r_wheel, r_wheel])
        rows_H.append(H_wheel)
        rows_R.append(R_wheel)

    if not rows_H:
        return np.zeros((0, n)), np.zeros((0, 0))

    H = np.vstack(rows_H)
    R = np.zeros((H.shape[0], H.shape[0]))
    row = 0
    for r in rows_R:
        m = r.shape[0]
        R[row:row + m, row:row + m] = r
        row += m
    return H, R
