#!/usr/bin/env python
# =============================================================================
# physics_v2hy3.py — steady-state (bristle-free) Adamov friction + regime-split
# physics helpers for Observer v2-Hy3.
#
# These functions are backend-agnostic (xp = numpy or torch) wherever possible,
# but the Hy3 loss path uses them as torch operations.  The steady-state force
# law is the 1:1 transcription of `lugre_ss_friction` in `run_one.jl:343-355`.
# =============================================================================
from __future__ import annotations

import math
from typing import Tuple

import torch

from . import config as C
from .physics import contact_from_gamma

_8_3PI = 8.0 / (3.0 * math.pi)
_16_3PI = 16.0 / (3.0 * math.pi)


def _as(ref, arr):
    """Broadcast a length-4 numpy constant to ref's backend/device as [1,...,4]."""
    if isinstance(ref, torch.Tensor):
        return torch.as_tensor(arr, dtype=ref.dtype, device=ref.device)
    return arr


def mindlin_fslip(xp, V: torch.Tensor, c: torch.Tensor, iters: int) -> torch.Tensor:
    """Picard fixed-point for the Mindlin slip fraction.

    Mirrors run_one.jl:336-342:
        x = V/(c+V); fs = 1
        repeat iters:
            b = max(1-x, 1e-9)
            fs = 1 - b^(2/3)
            x = V/(fs*c + V)
        b = max(1-x, 1e-9)
        return 1 - b^(2/3)
    """
    x = V / (c + V)
    fs = xp.ones_like(V)
    eps = xp.full_like(V, 1e-9)
    for _ in range(iters):
        b = xp.maximum(1.0 - x, eps)
        fs = 1.0 - b ** (2.0 / 3.0)
        x = V / (fs * c + V)
    b = xp.maximum(1.0 - x, eps)
    return 1.0 - b ** (2.0 / 3.0)


def lugre_ss_forces(xp, mu, N, chi, w_z, Vpx, Vpy, mindlin_iters: int = 2
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Steady-state (gross-slip) Adamov friction forces, bristle-eliminated.

    1:1 transcription of run_one.jl `lugre_ss_friction` (lines 343-355),
    with use_mindlin=True.  No zx/zy/zs arguments.

    Args:
        xp:  math module (numpy) or torch
        mu:  [...] or [...,1]
        N:   scalar or [...,4]
        chi: [...] or [...,1]
        w_z: [...,4]
        Vpx, Vpy: [...,4]
        mindlin_iters: Picard iterations for slip fraction
    Returns:
        Fx, Fy, Mz  (each [...,4])
    """
    er = C.LG_EPS_REG
    Vp = xp.sqrt(Vpx * Vpx + Vpy * Vpy + er * er)
    awz = xp.sqrt(w_z * w_z + er * er)
    c_t = _8_3PI * awz * chi

    fsl = mindlin_fslip(xp, Vp, c_t, mindlin_iters)
    s_t = fsl * c_t + Vp
    s_s = _16_3PI * awz * chi + 5.0 * Vp

    # stribeck_g from physics.py == run_one.jl
    sti = C.LG_STICTION_RATIO - 1.0
    g_t = mu * (1.0 + sti * xp.exp(-((s_t / C.LG_V_STR) ** 2)))
    g_s = mu * (1.0 + sti * xp.exp(-((s_s / C.LG_W_STR) ** 2)))

    Fx = -N * g_t * Vpx / s_t
    Fy = -N * g_t * Vpy / s_t
    Mz = -N * chi * chi * g_s * w_z / s_s
    return Fx, Fy, Mz


def gamma_noslip(Vpy0: torch.Tensor, cti: torch.Tensor) -> torch.Tensor:
    """Closed-form no-slip roller spin — nd711 §5.1 Model 1.

    Model 1 (nonholonomic, VPiX ≡ VPiY ≡ 0) splits the two contact conditions:
    VPiX = 0 fixes the WHEEL spin, VPiY = 0 fixes the ROLLER spin. We MEASURE the
    wheel speed, so only the lateral condition is ours to impose:

        Vpy = Vpy0 + γ·dVpy_dg = 0
        γ_noslip = −Vpy0 / (cos δ·(R·cos θt − Rd))
                 = (V_Y + Ω·ρ_Xi) / ((Rd − R)·cos δ_i)      at θt → 0  (the paper)

    Note this is NOT `gamma_kin` below, which least-squares BOTH components and so
    mixes in Vpx0 — i.e. the measured wheel speed and V_X, both slip-contaminated.
    γ depends on V_Y and Ω only; the wheel speed is not an input to it.

    Args:
        Vpy0: [...,4] lateral contact velocity at γ=0 (V_y + ψ̇·PX)
        cti:  [...,4] cos(θ̃), the folded wheel angle
    """
    cd = _as(Vpy0, C.COS_DELTA)
    den = cd * (C.R * cti - C.RD)
    return -Vpy0 / den


def gamma_kin(Vpx0: torch.Tensor, Vpy0: torch.Tensor,
              dVpx_dg: torch.Tensor, dVpy_dg: torch.Tensor,
              eps: float = C.LG_EPS_REG) -> torch.Tensor:
    """Stick-region kinematic γ from the pure-rolling least-squares condition.

    Over-determined 2 equations (Vpx=Vpy=0) in 1 unknown γ; least-squares:
        γ_kin = -(Vpx0*dVpx_dg + Vpy0*dVpy_dg) / (dVpx_dg^2 + dVpy_dg^2 + eps)
    """
    num = Vpx0 * dVpx_dg + Vpy0 * dVpy_dg
    den = dVpx_dg * dVpx_dg + dVpy_dg * dVpy_dg + eps
    return -num / den


def roller_residual_ss(gamma: torch.Tensor, mu: torch.Tensor, chi: torch.Tensor,
                       psi_dot: torch.Tensor,
                       Vpx0: torch.Tensor, Vpy0: torch.Tensor,
                       cti: torch.Tensor, sti: torch.Tensor,
                       mindlin_iters: int = 2) -> torch.Tensor:
    """Steady-state roller torque balance residual (per wheel).

    r = P2*gamma + Fx*dVpx/dg + Fy*dVpy/dg   -> 0 in quasi-static slip

    Forces come from lugre_ss_forces (bristle-free).  The Mz*dwz/dg term is
    dropped (same project decision as v1 `roller_residual`).
    """
    Vpx, Vpy, w_z, dVpx, dVpy, _ = contact_from_gamma(
        gamma, psi_dot, Vpx0, Vpy0, cti, sti)
    N = _as(gamma, C.N_PER_ROLLER)
    Fx, Fy, _ = lugre_ss_forces(torch, mu[..., None], N, chi[..., None],
                                w_z, Vpx, Vpy, mindlin_iters)
    return C.P2 * gamma + Fx * dVpx + Fy * dVpy
