"""Accelerometer + gyro observable synthesis from the exact-dynamics sidecar,
plus optional stage-2 sensor corruption and anti-alias-LPF + decimation.

Accelerometer convention (code-verified, matches `accel_dynamics.py` / the sim
EOM in `run_one.jl`): the body-frame velocity derivatives `dVx, dVy` (from the
`accel/<stem>_accel.arrow` sidecar) already carry the psi_dot*V transport
("Coriolis") coupling baked in by the Newton-Euler body EOM (see
`run_one.jl::dynamics_full_mf_asmc!` RHS0/RHS1). An accelerometer measures
specific force ONLY — it does not sense the fictitious transport term that
arises purely from expressing velocity components in a rotating (body) frame.
Subtracting that term recovers the specific-force observable:
    a_x = dVx - psi_dot*Vy
    a_y = dVy + psi_dot*Vx
Gyro is the direct yaw rate: gyro = psi_dot.

SensorNoiseSpec fields mirror the categories named (not yet numerically pinned)
in `instructions/observer-gamma-only-5phase-retrain.md` (rev-4 A2 brief,
config_v2.py's planned sensor knobs: "accel noise/bias/tilt, gyro noise/bias").
No real IMU datasheet exists in this repo; per this brief's own §11 ("stage-2
uses placeholder spec values; calibration is follow-up"), the numeric defaults
below are placeholders borrowed from the closest same-repo precedent with the
same physical units/shape — `code_insights/hybrid_ctrl/sensors.jl`'s
`SensorModel` (a different subsystem's IMU model, not re-derived from scratch).
Mounting-tilt gravity leakage has no repo precedent at all; modeled here as a
constant per-trajectory horizontal-plane vector of magnitude g*sin(tilt), random
direction, seeded — the standard first-order model of a fixed mounting error.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

G = 9.81
# Anti-alias corner: comfortably under the 500 Hz band's Nyquist (250 Hz) so
# LuGre stick-slip chatter (the companion brief cites content above 250 Hz)
# cannot fold into the decimated band; real IMUs low-pass internally, so this
# stays on for both noise stages.
ANTI_ALIAS_CUTOFF_HZ = 200.0


@dataclass
class SensorNoiseSpec:
    seed: int = 0
    accel_noise_std: float = 0.05       # m/s^2, white, per-sample
    accel_bias_std: float = 0.02        # m/s^2, constant per-trajectory bias (per axis)
    mount_tilt_deg: float = 1.0         # deg, constant mounting-tilt magnitude
    gyro_noise_std: float = 0.005       # rad/s, white, per-sample
    gyro_bias_std: float = 1e-4         # rad/s, constant per-trajectory bias


def specific_force(Vx: np.ndarray, Vy: np.ndarray, psi_dot: np.ndarray,
                    dVx: np.ndarray, dVy: np.ndarray) -> np.ndarray:
    """a_meas [T,2] = (a_x, a_y), the code-verified accelerometer observable."""
    a_x = dVx - psi_dot * Vy
    a_y = dVy + psi_dot * Vx
    return np.stack([a_x, a_y], axis=-1)


def _causal_lpf(x: np.ndarray, native_hz: float, cutoff_hz: float) -> np.ndarray:
    """One-pole causal IIR low-pass (the discrete-time image of a simple analog
    anti-alias filter; deployment-realistic, no lookahead)."""
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


def apply_noise(a_meas: np.ndarray, gyro: np.ndarray, spec: SensorNoiseSpec) -> Tuple[np.ndarray, np.ndarray]:
    """Stage-2 corruption: accel white noise + constant bias + mounting-tilt
    gravity leakage; gyro white noise + constant bias. Seeded (spec.seed) for
    reproducibility."""
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
                          native_hz: float = 2000.0, target_hz: float = 500.0,
                          spec: Optional[SensorNoiseSpec] = None,
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """From joined original+sidecar arrays at native rate, build (a_meas, gyro)
    at `target_hz`: anti-alias LPF + decimate, then optional stage-2 corruption.

    Returns: a_meas [T,2], gyro [T]  (at target_hz; T = native_T if target_hz
        >= native_hz, else native_T // (native_hz/target_hz))
    """
    a_meas = specific_force(Vx, Vy, psi_dot, dVx, dVy)
    gyro = np.asarray(psi_dot, dtype=np.float64).copy()

    a_meas = _decimate(a_meas, native_hz, target_hz)
    gyro = _decimate(gyro, native_hz, target_hz)

    if spec is not None:
        a_meas, gyro = apply_noise(a_meas, gyro, spec)
    return a_meas, gyro
