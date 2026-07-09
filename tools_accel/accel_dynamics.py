"""Exact-dynamics body + wheel accelerations from stored Arrow columns (pure numpy).

Numpy port of the VERIFIED torch implementation in
`Mecanum_PINN_Mamba_ForceRecon_v1/mecanum_pinn/physics.py::ne_rhs` (residual 0.000
vs a real Arrow file for the roller->body transform; the body EOM matches
`run_one.jl::dynamics_full_mf_asmc!` lines 716-722 with the per-wheel spin moment
Mz_i DROPPED by design — tiny body-frame lever, see physics.py's comment block).
Port, don't re-derive: every constant below is copied from `RobotParams` defaults
in physics.py, which are themselves mirrors of `base.toml` (`[platform.*]`) and
verified identical across all six trajectory_files_run_*/base.toml configs in
this repo (only mu/chi differ across configs; the geometry/mass/COM constants do
not).

EOM convention tag: "ne_rhs_v1_mz_dropped" — records that Mz_i is excluded from
the yaw-moment balance, matching physics.py exactly (not the full run_one.jl RHS2,
which does include it). Any sidecar consumer comparing against a from-scratch
ODE re-derivation that keeps Mz_i should expect a small systematic offset in
dpsidot, bounded by the per-wheel spin moment's contribution to the yaw axis.

State inputs are all EXISTING Arrow columns (Vx, Vy, psi_dot, w1..4, Msat_1..4,
Fpar_1..4, Fperp_1..4) — no ODE re-run, no invented physics.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

EOM_CONVENTION = "ne_rhs_v1_mz_dropped"
GENERATOR_VERSION = "accel_dynamics_v1"

# ============================================================
# Physical constants (base.toml [platform.*], identical across all six
# trajectory_files_run_{0p3,0p5,0p8}_{main,quad}/base.toml configs)
# ============================================================
H = 0.235          # half-length (m)
L = 0.150          # half-width (m)
R1 = 0.05          # wheel outer radius (m)
RA = 0.0355        # roller axle distance (m) -- unused here (no roller-spin term)

M = 30.0           # platform mass (kg)
M_WHEEL = 1.4      # per-wheel mass (kg)
JW = 5.87e-3       # wheel inertia (kg m^2)
IS = 4.42          # platform moment of inertia about Oz (kg m^2)
MS = M + 4.0 * M_WHEEL   # sprung+wheels mass, ms

AX = 1.6e-2        # COM offset X (m)
AY = -2.6e-2       # COM offset Y (m)

# Drivetrain viscous coefficient, selected by the filename's friction_case (fc).
P1_BY_CASE = {1: 0.11, 2: 0.011}

# O-configuration roller angles + wheel centers (order: wheel 1..4).
DELTA = np.array([-np.pi / 4, np.pi / 4, np.pi / 4, -np.pi / 4], dtype=np.float64)
WC_X = np.array([H, H, -H, -H], dtype=np.float64)
WC_Y = np.array([L, -L, L, -L], dtype=np.float64)
COS_DELTA = np.cos(DELTA)
SIN_DELTA = np.sin(DELTA)

_M_BODY = np.array([
    [MS, 0.0, -M * AY],
    [0.0, MS, M * AX],
    [-M * AY, M * AX, IS],
], dtype=np.float64)
M_BODY_INV = np.linalg.inv(_M_BODY)


def roller_to_body(Fpar: np.ndarray, Fperp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Roller-frame (F_par, F_perp) -> body-frame (Fx, Fy) per wheel.

    F_par/F_perp: (..., 4). Returns Fx, Fy: (..., 4).
    """
    Fx = Fpar * COS_DELTA - Fperp * SIN_DELTA
    Fy = Fpar * SIN_DELTA + Fperp * COS_DELTA
    return Fx, Fy


def body_wheel_accels(Vx: np.ndarray, Vy: np.ndarray, psi_dot: np.ndarray,
                       w: np.ndarray, Msat: np.ndarray,
                       Fpar: np.ndarray, Fperp: np.ndarray,
                       friction_case: int = 1,
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact-dynamics accelerations from stored columns (no ODE re-run).

    Args:
        Vx, Vy, psi_dot: (T,)
        w, Msat, Fpar, Fperp: (T, 4)
        friction_case: 1 or 2, parsed from the filename's `_case<fc>_` group;
            selects the drivetrain viscous coefficient p1.

    Returns:
        dVx, dVy, dpsidot: (T,) float32
        dw: (T, 4) float32
    """
    Vx = np.asarray(Vx, dtype=np.float64)
    Vy = np.asarray(Vy, dtype=np.float64)
    psi_dot = np.asarray(psi_dot, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    Msat = np.asarray(Msat, dtype=np.float64)
    Fpar = np.asarray(Fpar, dtype=np.float64)
    Fperp = np.asarray(Fperp, dtype=np.float64)

    try:
        p1 = P1_BY_CASE[int(friction_case)]
    except KeyError:
        raise ValueError(f"unknown friction_case {friction_case!r}; expected one of {sorted(P1_BY_CASE)}")

    Fx, Fy = roller_to_body(Fpar, Fperp)          # (T, 4)

    RHS0 = Fx.sum(-1) + MS * psi_dot * Vy + M * AX * psi_dot ** 2
    RHS1 = Fy.sum(-1) - MS * psi_dot * Vx + M * AY * psi_dot ** 2
    RHS2 = (WC_X * Fy - WC_Y * Fx).sum(-1) - M * psi_dot * (AX * Vx + AY * Vy)
    RHS = np.stack([RHS0, RHS1, RHS2], axis=-1)   # (T, 3)
    dv = RHS @ M_BODY_INV.T                       # (T, 3) == M_inv @ RHS per-row (M_inv symmetric)

    dw = (Msat - Fx * R1 - p1 * w) / JW           # (T, 4)

    dVx = dv[..., 0].astype(np.float32)
    dVy = dv[..., 1].astype(np.float32)
    dpsidot = dv[..., 2].astype(np.float32)
    dw = dw.astype(np.float32)
    return dVx, dVy, dpsidot, dw
