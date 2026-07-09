"""Wheel-odometry body-velocity estimate: kinematic least-squares map from
per-wheel angular velocity to body-frame (Vx, Vy). Slip-corrupted by nature —
that is the point of the drift audit (it's the LPF anchor being stress-tested).

Geometry constants (H, L, R) come from `observer_v1_py/mecanum_observer/config.py`
(itself a verified mirror of `base.toml [platform.geometry]` — see
CLAUDE.md's authority rule). The wheel Jacobian (O-configuration, body twist ->
wheel speed) mirrors the existing `_wheel_jacobian` construction in
`code_insights/hybrid_ctrl/estimators.jl` for the same geometry, ported to numpy;
`theta` (wheel roll angle) is accepted for interface completeness but not used —
this project already treats `w` (angular velocity) as a directly measurable
column (see `mecanum_pinn.data._ARROW_STATE_COLS`), so there's no need to
differentiate `theta` to recover it.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "observer_v1_py"))
from mecanum_observer.config import H, L, R  # noqa: E402

# Wheel Jacobian H_omega: body twist [Vx, Vy, psi_dot] -> wheel angular velocity
# [T,4] = H_omega @ [Vx,Vy,psi_dot], O-configuration, wheel order 1..4:
#   w1 = (Vx - Vy - (L+H)*psi_dot)/R
#   w2 = (Vx + Vy + (L+H)*psi_dot)/R
#   w3 = (Vx + Vy - (L+H)*psi_dot)/R
#   w4 = (Vx - Vy + (L+H)*psi_dot)/R
_LEVER = (L + H) / R
_H_OMEGA = np.array([
    [1.0 / R, -1.0 / R, -_LEVER],
    [1.0 / R,  1.0 / R,  _LEVER],
    [1.0 / R,  1.0 / R, -_LEVER],
    [1.0 / R, -1.0 / R,  _LEVER],
])
_H_OMEGA_PINV = np.linalg.pinv(_H_OMEGA)  # (3,4); constant Jacobian -> a fixed linear map


def wheel_odometry(w: np.ndarray, theta: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Kinematic least-squares body velocity from wheel angular velocities.

    Args: w [T,4] wheel angular velocity (rad/s); theta [T,4] accepted, unused.
    Returns: V_odom [T,2] = (Vx, Vy). (The Jacobian's 3rd row, psi_dot, is
        dropped here — the drift audit's gyro channel comes from `imu_observable`,
        keeping V_odom a PURE wheel-only estimate for clean attribution.)
    """
    w = np.asarray(w, dtype=np.float64)
    V3 = w @ _H_OMEGA_PINV.T  # (T,4) @ (4,3) -> (T,3) = [Vx, Vy, psi_dot]
    return V3[:, :2].copy()
