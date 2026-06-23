"""Robot physical parameters, geometry, force-frame transforms, and (next) the
analytical Newton-Euler integrator for the Mamba force-reconstruction PINN.

Adapted from train_GPU_PINN_v14_py/mecanum_pinn/physics.py. Differences:
  - Forces are roller-frame F_par / F_perp (8 channels), Mz DROPPED.
  - F_MAX normalizes both F_par and F_perp; MZ_MAX removed.
  - Adds the VERIFIED roller->body rotation and body generalized-force assembly.

VERIFIED (residual 0.000 vs a real Arrow file, all 4 wheels):
    Fx_i = Fpar_i*cos(delta_i) - Fperp_i*sin(delta_i)
    Fy_i = Fpar_i*sin(delta_i) + Fperp_i*cos(delta_i)
with delta = [-pi/4, +pi/4, +pi/4, -pi/4]  (O-configuration).

NOTE: the full analytical integrator (`forward_integrate`) — body Coriolis +
wheel rotational dynamics — is added in the next step, with the plant ODE terms
extracted from run_one.jl. The transforms below are complete and data-verified.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


# ============================================================
# Force-magnitude normalization (roller-frame F_par / F_perp)
# ============================================================
_M_DEFAULT       = 30.0    # platform mass [kg]
_M_WHEEL_DEFAULT = 1.4     # per-wheel mass [kg]
_G               = 9.81

# Per-wheel static weight ~ the friction-circle scale of a single contact.
# Used by data.py to normalize F_par/F_perp at __getitem__ time and by the
# model to match units. Single source of truth (no Mz scale anymore).
F_MAX: float = (_M_DEFAULT + 4.0 * _M_WHEEL_DEFAULT) * _G / 4.0    # 87.309 N

# Force channel order used everywhere downstream: [Fpar_1..4, Fperp_1..4]
N_FORCE = 8


@dataclass
class RobotParams:
    """Frozen physical container (cloned from v14; Mz scale removed)."""
    # Geometry (m)
    h: float = 0.235
    l: float = 0.150
    R1: float = 0.05
    Ra: float = 0.0355
    # Masses / inertias
    m: float = 30.0
    m_wheel: float = 1.4
    Jw_1: float = 5.87e-3
    ms: Optional[float] = None
    Is: float = 4.42
    # COM offset
    aX: float = 1.6e-2
    aY: float = -2.6e-2
    # Frozen drivetrain viscous (read by wheel dynamics; not trainable)
    p1: Optional[torch.Tensor] = None
    # Derived (finalize)
    N_total: Optional[float] = None
    N_per_roller: Optional[torch.Tensor] = None
    F_max:  Optional[torch.Tensor] = None
    wc_x:  Optional[torch.Tensor] = None
    wc_y:  Optional[torch.Tensor] = None
    delta: Optional[torch.Tensor] = None
    cos_delta: Optional[torch.Tensor] = None
    sin_delta: Optional[torch.Tensor] = None
    M:     Optional[torch.Tensor] = None
    M_inv: Optional[torch.Tensor] = None
    # Numerical / bounds
    eps_v: float = 1e-6
    mu_min: float = 0.05
    mu_max: float = 1.50
    chi_min: float = 0.0
    chi_max: float = 0.012
    grad_clip: float = 1.0
    Max_torque: float = 10.0

    def finalize(self, p1_wheels: float,
                 device: Optional[torch.device] = None) -> 'RobotParams':
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.p1 = torch.tensor(p1_wheels, dtype=torch.float32, device=device)
        self.ms = self.m + 4.0 * self.m_wheel
        self.N_total = self.m * _G
        self.F_max = torch.tensor(F_MAX, dtype=torch.float32, device=device)

        N1 = self.N_total/4*(1 + self.aX/self.h + self.aY/self.l) + self.m_wheel*_G
        N2 = self.N_total/4*(1 + self.aX/self.h - self.aY/self.l) + self.m_wheel*_G
        N3 = self.N_total/4*(1 - self.aX/self.h + self.aY/self.l) + self.m_wheel*_G
        N4 = self.N_total/4*(1 - self.aX/self.h - self.aY/self.l) + self.m_wheel*_G
        self.N_per_roller = torch.tensor([N1, N2, N3, N4],
                                         dtype=torch.float32, device=device)

        self.wc_x  = torch.tensor([+self.h, +self.h, -self.h, -self.h],
                                  dtype=torch.float32, device=device)
        self.wc_y  = torch.tensor([+self.l, -self.l, +self.l, -self.l],
                                  dtype=torch.float32, device=device)
        self.delta = torch.tensor([-np.pi/4, np.pi/4, np.pi/4, -np.pi/4],
                                  dtype=torch.float32, device=device)
        self.cos_delta = torch.cos(self.delta)
        self.sin_delta = torch.sin(self.delta)

        M_np = np.array([
            [self.ms, 0,                 -self.m * self.aY],
            [0,       self.ms,            self.m * self.aX],
            [-self.m * self.aY, self.m * self.aX, self.Is],
        ], dtype=np.float32)
        self.M     = torch.tensor(M_np,                dtype=torch.float32, device=device)
        self.M_inv = torch.tensor(np.linalg.inv(M_np), dtype=torch.float32, device=device)
        return self


# ============================================================
# Force-frame transforms  (VERIFIED, residual 0.000 vs data)
# ============================================================
def roller_to_body(F_par: torch.Tensor, F_perp: torch.Tensor,
                   cos_delta: torch.Tensor, sin_delta: torch.Tensor):
    """Roller-frame (F_par, F_perp) -> body-frame (Fx, Fy) per wheel.

    Standard 2D rotation by the roller angle delta_i. Shapes broadcast on the
    last (wheel) axis: F_* are (..., 4), cos/sin_delta are (4,).
    """
    Fx = F_par * cos_delta - F_perp * sin_delta
    Fy = F_par * sin_delta + F_perp * cos_delta
    return Fx, Fy


def body_generalized_force(Fx: torch.Tensor, Fy: torch.Tensor,
                           wc_x: torch.Tensor, wc_y: torch.Tensor) -> torch.Tensor:
    """Per-wheel body forces -> platform generalized force [Qx, Qy, Mz_body].

    Qx = sum Fx_i ; Qy = sum Fy_i ; Mz = sum (wc_x_i*Fy_i - wc_y_i*Fx_i).
    Returns (..., 3).
    """
    Qx = Fx.sum(dim=-1)
    Qy = Fy.sum(dim=-1)
    Mz = (wc_x * Fy - wc_y * Fx).sum(dim=-1)
    return torch.stack([Qx, Qy, Mz], dim=-1)


@dataclass
class Geometry:
    wc_x: torch.Tensor
    wc_y: torch.Tensor
    cos_delta: torch.Tensor
    sin_delta: torch.Tensor
    tan_delta: torch.Tensor
    R: float
    Ra: float


def make_geometry(rp: RobotParams) -> Geometry:
    return Geometry(
        wc_x=rp.wc_x.clone(), wc_y=rp.wc_y.clone(),
        cos_delta=rp.cos_delta.clone(), sin_delta=rp.sin_delta.clone(),
        tan_delta=torch.tan(rp.delta).clone(), R=rp.R1, Ra=rp.Ra,
    )


def sawtooth_approx(theta: torch.Tensor, k: float = 40.0) -> torch.Tensor:
    """Tanh-smoothed sawtooth, period pi/6, range +-pi/12 (12 rollers)."""
    s = torch.sin(12.0 * theta)
    c = torch.cos(12.0 * theta)
    return torch.atan2(k * s, k * c + 1.0) / 12.0


# ============================================================
# Analytical Newton-Euler integrator  (Heun / explicit trapezoidal, O(dt^3))
# ============================================================
# Plant RHS verified against run_one.jl `dynamics_full_mf_asmc!` (lines 716-722):
#   body : dv = M_inv @ [RHS0, RHS1, RHS2]
#       RHS0 = sum Fx_i + ms*psi_dot*Vy + m*aX*psi_dot^2
#       RHS1 = sum Fy_i - ms*psi_dot*Vx + m*aY*psi_dot^2
#       RHS2 = sum(px*Fy_i - py*Fx_i [+ Mz_i]) - m*psi_dot*(aX*Vx + aY*Vy)
#   wheel: dw_i = (Msat_i - Fx_i*R - p1*w_i) / Jw     (reaction = Fx_i * R)
#   theta: dtheta_i = w_i
# The per-wheel spin moment Mz_i is DROPPED by design (tiny body-frame lever;
# see chi-identifiability). No external disturbance enters the body EOM (the DOB
# only feeds the controller), so forces are the sole interface. All tensors are
# PHYSICAL units; the caller (losses) de-normalizes before and re-normalizes
# after. State layout: [Vx, Vy, psi_dot, w1..4, theta1..4]  (11).


def ne_rhs(state: torch.Tensor, F: torch.Tensor, Msat: torch.Tensor,
           rp: RobotParams) -> torch.Tensor:
    """Continuous-time RHS d/dt[Vx,Vy,psi_dot, w1..4, theta1..4] (physical units).

    state (...,11); F (...,8)=[Fpar_1..4, Fperp_1..4]; Msat (...,4). -> (...,11).
    Requires a finalized RobotParams (rp.finalize(p1) called).
    """
    Vx = state[..., 0]; Vy = state[..., 1]; pd = state[..., 2]
    w  = state[..., 3:7]
    Fpar = F[..., 0:4]; Fperp = F[..., 4:8]
    Fx, Fy = roller_to_body(Fpar, Fperp, rp.cos_delta, rp.sin_delta)   # (...,4)

    px, py = rp.wc_x, rp.wc_y
    ms, m, aX, aY = rp.ms, rp.m, rp.aX, rp.aY
    RHS0 = Fx.sum(-1) + ms * pd * Vy + m * aX * pd * pd
    RHS1 = Fy.sum(-1) - ms * pd * Vx + m * aY * pd * pd
    RHS2 = (px * Fy - py * Fx).sum(-1) - m * pd * (aX * Vx + aY * Vy)
    RHS  = torch.stack([RHS0, RHS1, RHS2], dim=-1)            # (...,3)
    dv   = RHS @ rp.M_inv.t()                                 # = M_inv @ RHS

    dw  = (Msat - Fx * rp.R1 - rp.p1 * w) / rp.Jw_1           # (...,4)
    dth = w                                                   # theta_dot = w
    return torch.cat([dv, dw, dth], dim=-1)


def _fold_theta(state: torch.Tensor) -> torch.Tensor:
    th = state[..., 7:11]
    th = torch.atan2(torch.sin(12.0 * th), torch.cos(12.0 * th)) / 12.0
    return torch.cat([state[..., :7], th], dim=-1)


def forward_integrate(state: torch.Tensor, F: torch.Tensor, Msat: torch.Tensor,
                      rp: RobotParams, dt: float,
                      F_next: Optional[torch.Tensor] = None,
                      Msat_next: Optional[torch.Tensor] = None) -> torch.Tensor:
    """One Heun (explicit trapezoidal) step; local error O(dt^3).

    Force/torque are sampled at BOTH endpoints (F, F_next) — this is what lifts
    the scheme from O(dt^2) (zero-order-hold force impulse) to O(dt^3). In an
    autoregressive rollout, F_next is the network's force at the next step, so no
    extra network evaluation is needed. If F_next/Msat_next are None the step
    falls back to held force/torque (O(dt^2) for that one step).

    All tensors PHYSICAL units; theta re-folded to +-pi/12.
    """
    if F_next is None:
        F_next = F
    if Msat_next is None:
        Msat_next = Msat
    f1   = ne_rhs(state, F, Msat, rp)
    star = state + dt * f1                        # Euler predictor
    f2   = ne_rhs(star, F_next, Msat_next, rp)    # corrector @ next-step force/torque
    nxt  = state + 0.5 * dt * (f1 + f2)
    return _fold_theta(nxt)
