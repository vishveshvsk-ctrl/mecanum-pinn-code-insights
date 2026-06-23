"""Robot physical parameters, geometry, and small kinematic helpers.

This module is dependency-free apart from numpy + torch. Anything that
needs (mu, chi) or motion-case info lives elsewhere; this module only
contains intrinsic-platform quantities (mass distribution, wheel layout,
roller geometry).

`RobotParams.finalize(p1_wheels=...)` computes the derived torch tensors
on the currently-bound device. Call it once at the start of a run after
the device is selected.

Force-magnitude normalization constants (`F_MAX`, `MZ_MAX`) live at module
level here so both data.py (for trajectory normalization at __getitem__
time) and models.py (for matching units in `_reconstruct_forces`) share
one source of truth. The corresponding torch tensors are also stored on
RobotParams.finalize() so the model can register them as device-resident
buffers; the module-level Python floats are the canonical values.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


# ============================================================
# Force-magnitude normalization constants
# ============================================================
# Default robot mass parameters (must match RobotParams dataclass defaults
# below). If you change the robot mass for a custom configuration, also
# re-normalize your trajectory data — these constants determine the scale
# `data.py` uses when dividing physical forces.
_M_DEFAULT       = 30.0    # platform mass [kg]
_M_WHEEL_DEFAULT = 1.4     # per-wheel mass [kg]
_G               = 9.81    # gravitational accel [m/s^2]

# F_MAX = m_total · g = total static weight of the robot.
# MZ_MAX = F_MAX / 400 — chosen so the moment scale doesn't get washed out
# by the force scale in MSE losses. Matches the v12-era convention.
F_MAX:  float = (_M_DEFAULT + 4.0 * _M_WHEEL_DEFAULT) * _G / 4.0    # 349.236 N
MZ_MAX: float = F_MAX / 8000.0                                  # 0.873 N·m


@dataclass
class RobotParams:
    """Frozen physical container. p1 is a buffer (not trainable)."""

    # Geometry (m)
    h: float = 0.235
    l: float = 0.150
    R1: float = 0.05
    Ra: float = 0.0355

    # Masses and inertias
    m: float = 30.0
    m_wheel: float = 1.4
    Jw_1: float = 5.87e-3
    ms: Optional[float] = None       # filled in finalize()
    Is: float = 4.42

    # COM offset
    aX: float = 1.6e-2
    aY: float = -2.6e-2

    # Frozen viscous bearing friction (read by physics residual; not trainable)
    p1: Optional[torch.Tensor] = None

    # Filled in finalize()
    N_total: Optional[float] = None
    N_per_roller: Optional[torch.Tensor] = None
    F_max:  Optional[torch.Tensor] = None    # force-axis normalization (scalar)
    Mz_max: Optional[torch.Tensor] = None    # moment-axis normalization (scalar)
    wc_x:  Optional[torch.Tensor] = None
    wc_y:  Optional[torch.Tensor] = None
    delta: Optional[torch.Tensor] = None
    M_inv: Optional[torch.Tensor] = None
    M:     Optional[torch.Tensor] = None

    # Numerical
    eps_v: float = 1e-6

    # Display bounds for the downstream (mu, chi) estimation utility
    mu_min: float = 0.05
    mu_max: float = 1.50
    chi_min: float = 0.0
    chi_max: float = 0.012

    grad_clip: float = 1.0
    Max_torque: float = 10.0

    def finalize(self, p1_wheels: float, device: Optional[torch.device] = None
                 ) -> 'RobotParams':
        """Populate derived tensors on `device` (defaults to current torch device).

        Idempotent: calling twice with the same device just rebuilds the
        tensors. If you change device mid-run, call finalize again.
        """
        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )

        self.p1 = torch.tensor(p1_wheels, dtype=torch.float32, device=device)
        self.ms = self.m + 4.0 * self.m_wheel
        self.N_total = self.m * 9.81

        # Force-magnitude normalization scalars on device. These mirror the
        # module-level F_MAX/MZ_MAX constants but use the actual `ms`, so a
        # custom RobotParams (different m or m_wheel) gets a consistent scale.
        # WARNING: if you customize m / m_wheel, the values here will diverge
        # from the module-level F_MAX/MZ_MAX that data.py uses to normalize
        # trajectories at __getitem__. In that case you must re-normalize
        # your dataset (or override F_MAX/MZ_MAX in physics.py) before
        # training, or the model output and target won't match.
        self.F_max  = torch.tensor(self.ms * 9.81, dtype=torch.float32, device=device)
        self.Mz_max = torch.tensor(self.ms * 9.81 / 400.0, dtype=torch.float32, device=device)

        N1 = self.N_total/4 * (1 + self.aX/self.h + self.aY/self.l) + self.m_wheel * 9.81
        N2 = self.N_total/4 * (1 + self.aX/self.h - self.aY/self.l) + self.m_wheel * 9.81
        N3 = self.N_total/4 * (1 - self.aX/self.h + self.aY/self.l) + self.m_wheel * 9.81
        N4 = self.N_total/4 * (1 - self.aX/self.h - self.aY/self.l) + self.m_wheel * 9.81
        self.N_per_roller = torch.tensor(
            [N1, N2, N3, N4], dtype=torch.float32, device=device,
        )

        self.wc_x  = torch.tensor([+self.h, +self.h, -self.h, -self.h],
                                  dtype=torch.float32, device=device)
        self.wc_y  = torch.tensor([+self.l, -self.l, +self.l, -self.l],
                                  dtype=torch.float32, device=device)
        self.delta = torch.tensor([-np.pi/4, np.pi/4, np.pi/4, -np.pi/4],
                                  dtype=torch.float32, device=device)

        M_np = np.array([
            [self.ms, 0,        -self.m * self.aY],
            [0,       self.ms,   self.m * self.aX],
            [-self.m * self.aY,  self.m * self.aX, self.Is],
        ], dtype=np.float32)
        self.M     = torch.tensor(M_np,                dtype=torch.float32, device=device)
        self.M_inv = torch.tensor(np.linalg.inv(M_np), dtype=torch.float32, device=device)
        return self


@dataclass
class Geometry:
    """Subset of RobotParams that the inverse model needs as a lightweight
    handle. Cloned tensors so the inverse module owns its own buffers."""
    wc_x:      torch.Tensor    # (n_wheels,)
    wc_y:      torch.Tensor    # (n_wheels,)
    tan_delta: torch.Tensor    # (n_wheels,)
    R:         float
    Ra:        float


def make_geometry(rp: RobotParams) -> Geometry:
    """Build a Geometry handle from a finalized RobotParams. Call after
    rp.finalize() so the tensors exist."""
    return Geometry(
        wc_x=rp.wc_x.clone(),
        wc_y=rp.wc_y.clone(),
        tan_delta=torch.tan(rp.delta).clone(),
        R=rp.R1,
        Ra=rp.Ra,
    )


def sawtooth_approx(theta: torch.Tensor, k: float = 40.0) -> torch.Tensor:
    """Tanh-peak smoothed sawtooth. Period pi/6, range +-pi/12.

    Used inside the inverse model's slip approximation. Higher k gives a
    sharper sawtooth at the cost of derivative smoothness — k=40 is the
    well-tested default from v11.
    """
    s = torch.sin(12.0 * theta)
    c = torch.cos(12.0 * theta)
    return torch.atan2(k * s, k * c + 1.0) / 12.0
