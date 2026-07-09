"""Deployment-contract velocity front-end core: strapdown mechanization + the
fixed first-order complementary filter. Pure numpy, strictly causal recursions,
no learned components. This is the SAME code contract `sensor_frontend_v2.py`
(instructions/observer-gamma-only-5phase-retrain.md) later imports/mirrors —
keep it import-clean and free of any audit-only concerns.

Mechanization convention (code-verified against the sim EOM in
`tools_accel/accel_dynamics.py` / `run_one.jl`): the accelerometer observable
is the body-frame specific force `a_x = V̇x - psi_dot*Vy`,
`a_y = V̇y + psi_dot*Vx` (see `imu_observable.py`), so recovering velocity
from (a, gyro) alone means re-integrating the SAME transport-coupled ODE:
    dVx/dt = a_x + psi_dot*Vy
    dVy/dt = a_y - psi_dot*Vx
"euler" discretizes this directly; "rot_ab2" instead rotates the previous
velocity estimate by the exact planar rotation for -psi_dot*dt (norm-preserving,
handles the psi_dot*V coupling exactly rather than by linearization) and adds a
2nd-order (Adams-Bashforth) causal accel term.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

INTEGRATORS = ("euler", "rot_ab2")
MODES = ("full", "odom_only", "mech_only")

StepFn = Callable[[np.ndarray, int], np.ndarray]


def _euler_step_factory(a_meas: np.ndarray, gyro: np.ndarray, dt: float) -> StepFn:
    def step(v_prev: np.ndarray, k: int) -> np.ndarray:
        ax, ay = a_meas[k, 0], a_meas[k, 1]
        g = gyro[k]
        vx, vy = v_prev[0], v_prev[1]
        return np.array([vx + dt * (ax + g * vy), vy + dt * (ay - g * vx)])
    return step


def _rot_ab2_step_factory(a_meas: np.ndarray, gyro: np.ndarray, dt: float) -> StepFn:
    def step(v_prev: np.ndarray, k: int) -> np.ndarray:
        theta = -gyro[k] * dt
        c, s = math.cos(theta), math.sin(theta)
        vx, vy = v_prev[0], v_prev[1]
        v_rot_x = c * vx - s * vy
        v_rot_y = s * vx + c * vy
        if k == 0:
            ax, ay = a_meas[0, 0], a_meas[0, 1]
        else:
            ax = 1.5 * a_meas[k, 0] - 0.5 * a_meas[k - 1, 0]
            ay = 1.5 * a_meas[k, 1] - 0.5 * a_meas[k - 1, 1]
        return np.array([v_rot_x + dt * ax, v_rot_y + dt * ay])
    return step


def make_mech_step(a_meas: np.ndarray, gyro: np.ndarray, dt: float, integrator: str) -> StepFn:
    """Single source of truth for the one-step mechanization update, shared by
    `mechanize` (open-loop) and `complementary` (closed-loop)."""
    if integrator == "euler":
        return _euler_step_factory(a_meas, gyro, dt)
    if integrator == "rot_ab2":
        return _rot_ab2_step_factory(a_meas, gyro, dt)
    raise ValueError(f"unknown integrator {integrator!r}; expected one of {INTEGRATORS}")


def mechanize(a_meas: np.ndarray, gyro: np.ndarray, dt: float,
              integrator: str, v0: np.ndarray) -> np.ndarray:
    """
    Causal body-frame strapdown velocity integration (open-loop: no anchor).
    Args:  a_meas [T,2]; gyro [T]; integrator in {"euler", "rot_ab2"}; v0 [2]
    Returns: V_mech [T,2]
    """
    step = make_mech_step(a_meas, gyro, dt, integrator)
    T = a_meas.shape[0]
    V = np.empty((T, 2), dtype=np.float64)
    V[0] = v0
    for k in range(1, T):
        V[k] = step(V[k - 1], k)
    return V


def _complementary_alpha(crossover_hz: float, dt: float) -> float:
    """First-order complementary weight: alpha -> 1 (mechanized/HPF dominates)
    as crossover_hz -> 0; alpha -> 0 (odometry/LPF dominates) as crossover_hz
    grows. tau = 1/(2*pi*f_c); alpha = tau/(tau+dt)."""
    tau = 1.0 / (2.0 * math.pi * crossover_hz)
    return tau / (tau + dt)


def complementary(V_mech_step: StepFn, V_odom: np.ndarray,
                  crossover_hz: float, dt: float, mode: str,
                  v0: Optional[np.ndarray] = None) -> np.ndarray:
    """
    First-order complementary blend; mode in {"full", "odom_only", "mech_only"}.

    "full" is CLOSED-LOOP: at each step the mechanization advances from the
    previously BLENDED estimate (not an independently precomputed open-loop
    V_mech array), then blends with V_odom — the standard complementary-filter
    architecture, so accel bias is high-passed out rather than free-integrated.
    "mech_only" is open-loop (the unbounded-drift reference); "odom_only"
    returns the anchor alone.

    Returns: V_hat [T,2]
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    T = V_odom.shape[0]
    if mode == "odom_only":
        return V_odom.copy()

    V = np.empty((T, 2), dtype=np.float64)
    V[0] = V_odom[0] if v0 is None else v0

    if mode == "mech_only":
        for k in range(1, T):
            V[k] = V_mech_step(V[k - 1], k)
        return V

    # mode == "full"
    alpha = _complementary_alpha(crossover_hz, dt)
    beta = 1.0 - alpha
    for k in range(1, T):
        v_mech_k = V_mech_step(V[k - 1], k)
        V[k, 0] = alpha * v_mech_k[0] + beta * V_odom[k, 0]
        V[k, 1] = alpha * v_mech_k[1] + beta * V_odom[k, 1]
    return V
