#!/usr/bin/env python
# =============================================================================
# sensor_frontend_v2.py — deployment-contract sensor front-end for Observer v2.
#
# Produces the 5-channel sensor-real global input Gw = [V̂x, V̂y, ψ̇_gyro, a_x, a_y]
# at 500 Hz from deployment-available signals only:
#   * IMU: body-frame specific force (a_x, a_y) + gyro yaw rate ψ̇
#   * Wheel encoders: w_i, θ_i
#   * Fixed causal complementary filter: wheel-odometry anchor (LPF) + strapdown
#     mechanization (HPF) → V̂x, V̂y
#
# This is a self-contained port of the verified `code_insights/tools_accel/`
# pipeline, aligned with `instructions/observer-gamma-only-5phase-retrain.md`
# (rev 4).  It reads the pre-built acceleration sidecars in
# `data/.../accel/<stem>_accel.arrow` (dVx, dVy, dpsidot, dw1..4) produced by
# `tools_accel/make_accel_sidecars.py`.
# =============================================================================
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pyarrow.feather as feather

from . import config as C

G = 9.81
ANTI_ALIAS_CUTOFF_HZ = 200.0
SIM_HZ = 2000.0
TRAIN_HZ = 500.0


@dataclass
class SensorNoiseSpec:
    """Stage-2 sensor-corruption spec; stage 1 (campaign 1) runs noise=None."""
    seed: int = 0
    accel_noise_std: float = 0.05       # m/s^2, white per-sample
    accel_bias_std: float = 0.02        # m/s^2, constant per-trajectory bias
    mount_tilt_deg: float = 1.0         # deg, constant mounting-tilt magnitude
    gyro_noise_std: float = 0.005       # rad/s, white per-sample
    gyro_bias_std: float = 1e-4         # rad/s, constant per-trajectory bias


# ---------------------------------------------------------------------------
# Sidecar loading
# ---------------------------------------------------------------------------
REQUIRED_SIDECAR_COLS = ["dVx", "dVy", "dpsidot", "dw1", "dw2", "dw3", "dw4"]


def _sidecar_path(orig: Path) -> Path:
    return orig.parent / "accel" / f"{orig.stem}_accel.arrow"


def load_with_accel(orig: Path) -> Dict[str, np.ndarray]:
    """Load original Arrow + its acceleration sidecar, join columns, return
    a dict of numpy arrays.  Raises if the sidecar is missing or misaligned."""
    sidecar = _sidecar_path(orig)
    if not sidecar.exists():
        raise FileNotFoundError(f"no accel sidecar for {orig.name}: expected {sidecar}")

    orig_table = feather.read_table(orig)
    sidecar_table = feather.read_table(sidecar)
    if sidecar_table.num_rows != orig_table.num_rows:
        raise ValueError(f"{sidecar.name} row count {sidecar_table.num_rows} != "
                         f"original {orig_table.num_rows}")

    out: Dict[str, np.ndarray] = {}
    for name in orig_table.column_names:
        out[name] = orig_table.column(name).to_numpy()
    for name in sidecar_table.column_names:
        out[name] = sidecar_table.column(name).to_numpy()
    return out


# ---------------------------------------------------------------------------
# IMU observable synthesis
# ---------------------------------------------------------------------------
def specific_force(Vx: np.ndarray, Vy: np.ndarray, psi_dot: np.ndarray,
                   dVx: np.ndarray, dVy: np.ndarray) -> np.ndarray:
    """a_meas [T,2] = (a_x, a_y), the body-frame accelerometer specific force.

    The body EOM derivatives dVx/dVy already carry the ψ̇×V transport term;
    subtracting it recovers the specific-force observable (see
    tools_accel/imu_observable.py for the code-verified convention)."""
    a_x = dVx - psi_dot * Vy
    a_y = dVy + psi_dot * Vx
    return np.stack([a_x, a_y], axis=-1)


def _causal_lpf(x: np.ndarray, native_hz: float, cutoff_hz: float) -> np.ndarray:
    """One-pole causal IIR low-pass anti-alias filter (no lookahead)."""
    dt = 1.0 / native_hz
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    alpha = dt / (rc + dt)
    y = np.empty_like(x, dtype=np.float64)
    y[0] = x[0]
    for k in range(1, x.shape[0]):
        y[k] = y[k - 1] + alpha * (x[k] - y[k - 1])
    return y


def _decimate(x: np.ndarray, native_hz: float, target_hz: float) -> np.ndarray:
    if target_hz >= native_hz:
        return x
    step = int(round(native_hz / target_hz))
    x = _causal_lpf(x, native_hz, ANTI_ALIAS_CUTOFF_HZ)
    return x[::step]


def apply_noise(a_meas: np.ndarray, gyro: np.ndarray,
                spec: SensorNoiseSpec) -> Tuple[np.ndarray, np.ndarray]:
    """Stage-2 corruption: white noise + bias + mounting-tilt gravity leakage."""
    rng = np.random.default_rng(spec.seed)
    accel_bias = rng.normal(0.0, spec.accel_bias_std, size=2)
    tilt_mag = G * math.sin(math.radians(spec.mount_tilt_deg))
    tilt_angle = rng.uniform(0.0, 2.0 * math.pi)
    gravity_leak = tilt_mag * np.array([math.cos(tilt_angle), math.sin(tilt_angle)])
    accel_white = rng.normal(0.0, spec.accel_noise_std, size=a_meas.shape)
    a_noisy = a_meas + accel_bias + gravity_leak + accel_white

    gyro_bias = rng.normal(0.0, spec.gyro_bias_std)
    gyro_white = rng.normal(0.0, spec.gyro_noise_std, size=gyro.shape)
    gyro_noisy = gyro + gyro_bias + gyro_white
    return a_noisy, gyro_noisy


def build_imu_observable(Vx: np.ndarray, Vy: np.ndarray, psi_dot: np.ndarray,
                         dVx: np.ndarray, dVy: np.ndarray,
                         native_hz: float = SIM_HZ, target_hz: float = TRAIN_HZ,
                         spec: Optional[SensorNoiseSpec] = None,
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Build (a_meas [T,2], gyro [T]) at target_hz."""
    a_meas = specific_force(Vx, Vy, psi_dot, dVx, dVy)
    gyro = np.asarray(psi_dot, dtype=np.float64).copy()

    a_meas = _decimate(a_meas, native_hz, target_hz)
    gyro = _decimate(gyro, native_hz, target_hz)

    if spec is not None:
        a_meas, gyro = apply_noise(a_meas, gyro, spec)
    return a_meas.astype(np.float32), gyro.astype(np.float32)


# ---------------------------------------------------------------------------
# Wheel odometry
# ---------------------------------------------------------------------------
_LEVER = (C.L + C.H) / C.R
_H_OMEGA = np.array([
    [1.0 / C.R, -1.0 / C.R, -_LEVER],
    [1.0 / C.R,  1.0 / C.R,  _LEVER],
    [1.0 / C.R,  1.0 / C.R, -_LEVER],
    [1.0 / C.R, -1.0 / C.R,  _LEVER],
], dtype=np.float64)
_H_OMEGA_PINV = np.linalg.pinv(_H_OMEGA)


def wheel_odometry(w: np.ndarray) -> np.ndarray:
    """Kinematic least-squares body velocity [Vx, Vy] from wheel speeds w[T,4].

    The third component (psi_dot) is intentionally dropped — the gyro channel
    supplies ψ̇ directly, keeping V_odom a pure wheel-only anchor."""
    w = np.asarray(w, dtype=np.float64)
    V3 = w @ _H_OMEGA_PINV.T            # (T,4) @ (4,3) -> (T,3)
    return V3[:, :2].copy()


# ---------------------------------------------------------------------------
# Strapdown mechanization + complementary filter
# ---------------------------------------------------------------------------
StepFn = callable  # type: ignore[assignment]


def _euler_step_factory(a_meas: np.ndarray, gyro: np.ndarray, dt: float):
    def step(v_prev: np.ndarray, k: int) -> np.ndarray:
        ax, ay = a_meas[k, 0], a_meas[k, 1]
        g = gyro[k]
        vx, vy = v_prev[0], v_prev[1]
        return np.array([vx + dt * (ax + g * vy), vy + dt * (ay - g * vx)])
    return step


def _rot_ab2_step_factory(a_meas: np.ndarray, gyro: np.ndarray, dt: float):
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


def make_mech_step(a_meas: np.ndarray, gyro: np.ndarray, dt: float,
                   integrator: str = "rot_ab2"):
    if integrator == "euler":
        return _euler_step_factory(a_meas, gyro, dt)
    if integrator == "rot_ab2":
        return _rot_ab2_step_factory(a_meas, gyro, dt)
    raise ValueError(f"unknown integrator {integrator!r}")


def _complementary_alpha(crossover_hz: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * crossover_hz)
    return tau / (tau + dt)


def complementary_filter(V_mech_step, V_odom: np.ndarray,
                         crossover_hz: float, dt: float,
                         v0: Optional[np.ndarray] = None) -> np.ndarray:
    """Closed-loop first-order complementary filter.

    At each step the mechanization advances from the previously blended
    estimate, then blends with the odometry anchor.  This high-passes accel
    bias rather than free-integrating it."""
    T = V_odom.shape[0]
    V = np.empty((T, 2), dtype=np.float64)
    V[0] = V_odom[0] if v0 is None else v0
    alpha = _complementary_alpha(crossover_hz, dt)
    beta = 1.0 - alpha
    for k in range(1, T):
        v_mech_k = V_mech_step(V[k - 1], k)
        V[k, 0] = alpha * v_mech_k[0] + beta * V_odom[k, 0]
        V[k, 1] = alpha * v_mech_k[1] + beta * V_odom[k, 1]
    return V


# ---------------------------------------------------------------------------
# High-level front-end assembly
# ---------------------------------------------------------------------------
def build_sensor_frontend(cols: Dict[str, np.ndarray],
                          crossover_hz: float = 1.0,
                          integrator: str = "rot_ab2",
                          target_hz: float = TRAIN_HZ,
                          spec: Optional[SensorNoiseSpec] = None,
                          ) -> Dict[str, np.ndarray]:
    """From joined original+sidecar columns at 2000 Hz, build the v2 front-end.

    Returns dict with:
        Gw      [T500, 5] = [V̂x, V̂y, ψ̇_gyro, a_x, a_y]
        V_true  [T500, 2] = [Vx, Vy]      (loss-side audit reference)
        V_hat   [T500, 2] = [V̂x, V̂y]      (the front-end output)
        Vpx0_hat [T500, 4]                 (γ=0 contact velocities from V̂)
        Vpy0_hat [T500, 4]
        a_meas  [T500, 2]                  (specific force, for audit)
        psi_dot [T500]                     (gyro, decimated)
        time    [T500]                     (decimated time)
    """
    native_hz = SIM_HZ
    dt = 1.0 / target_hz

    # Stack wheel speeds.
    w = np.stack([cols[f"w{i}"] for i in range(1, C.N_WHEELS + 1)], axis=-1)
    theta = np.stack([cols[f"theta{i}"] for i in range(1, C.N_WHEELS + 1)], axis=-1)

    # IMU observable at target_hz.
    a_meas, gyro = build_imu_observable(
        cols["Vx"], cols["Vy"], cols["psi_dot"],
        cols["dVx"], cols["dVy"],
        native_hz=native_hz, target_hz=target_hz, spec=spec)

    # Wheel odometry + decimation to target_hz.
    V_odom_native = wheel_odometry(w)
    V_odom = np.stack([
        _decimate(V_odom_native[:, 0], native_hz, target_hz),
        _decimate(V_odom_native[:, 1], native_hz, target_hz),
    ], axis=-1)

    # Truth velocity for audit / ΔV target.
    V_true = np.stack([
        _decimate(cols["Vx"], native_hz, target_hz),
        _decimate(cols["Vy"], native_hz, target_hz),
    ], axis=-1)

    # Complementary filter (closed-loop, full mode).
    step = make_mech_step(a_meas, gyro, dt, integrator)
    V_hat = complementary_filter(step, V_odom, crossover_hz, dt, v0=V_odom[0])

    # Decimate theta and psi_dot for gamma=0 contact kinematics.
    psi_dot_dec = _decimate(cols["psi_dot"], native_hz, target_hz)
    theta_dec = _decimate(theta, native_hz, target_hz)
    time_dec = _decimate(cols["time"], native_hz, target_hz)

    # Folded wheel angle (same smooth approximator the simulator used).
    from .features import sawtooth_tanh
    sin_tt = np.empty((theta_dec.shape[0], C.N_WHEELS), dtype=np.float64)
    cos_tt = np.empty_like(sin_tt)
    for k in range(C.N_WHEELS):
        tt = sawtooth_tanh(theta_dec[:, k])
        sin_tt[:, k] = np.sin(tt)
        cos_tt[:, k] = np.cos(tt)

    # γ=0 contact velocities from the front-end estimate V̂ (NOT V_true).
    DY = C.RD * C.TAN_DELTA[None, :] * (sin_tt / cos_tt)
    Vpx0_hat = V_hat[:, 0:1] - psi_dot_dec[:, None] * (C.PY[None, :] + DY) - \
        (_decimate(w, native_hz, target_hz) * C.R)
    Vpy0_hat = V_hat[:, 1:2] + psi_dot_dec[:, None] * C.PX[None, :]

    Gw = np.concatenate([
        V_hat.astype(np.float32),
        gyro[:, None],
        a_meas.astype(np.float32),
    ], axis=-1)

    return {
        "Gw": Gw,
        "V_true": V_true.astype(np.float32),
        "V_hat": V_hat.astype(np.float32),
        "Vpx0_hat": Vpx0_hat.astype(np.float32),
        "Vpy0_hat": Vpy0_hat.astype(np.float32),
        "a_meas": a_meas,
        "psi_dot": psi_dot_dec.astype(np.float32),
        "time": time_dec.astype(np.float32),
        "w": _decimate(w, native_hz, target_hz).astype(np.float32),
        "theta": theta_dec.astype(np.float32),
        "sin_tt": sin_tt.astype(np.float32),
        "cos_tt": cos_tt.astype(np.float32),
    }
