#!/usr/bin/env python
# =============================================================================
# physics.py — torch port of the LuGre+Adamov friction law and the roller /
# wheel torque-balance residuals used by the physics loss (Regime D etc.).
#
# The force law is written backend-agnostically (`xp` = numpy OR torch, using
# only sqrt/exp/maximum/minimum which share an API) so the SAME code path is
# (a) verified against stored Arrow forces in numpy and (b) used in torch for
# the loss. This is the :lugre_adamov model with use_mindlin=True — a 1:1
# transcription of lugre_dyn_rates (run_one.jl) at friction_case=1.
#
# Forces are recomputed from the PREDICTED states (gamma via the contact
# kinematics, plus the bristles zx/zy/zs), with chi carried per-sample as a
# loss-side constant (never a network input).
# =============================================================================
from __future__ import annotations

import math

from . import config as C

_8_3PI = 8.0 / (3.0 * math.pi)
_16_3PI = 16.0 / (3.0 * math.pi)


def lugre_forces(xp, mu, N, chi, w_z, Vpx, Vpy, zx, zy, zs):
    """Per-contact LuGre+Adamov forces (Fx, Fy, Mz) from states.

    All args broadcast; `mu`, `N`, `chi` may be scalars or per-sample tensors.
    Mirrors run_one.jl:294-332 exactly (coupling=:adamov, use_mindlin=True)."""
    er = C.LG_EPS_REG
    Vp = xp.sqrt(Vpx * Vpx + Vpy * Vpy + er * er)
    awz = xp.sqrt(w_z * w_z + er * er)
    c_t = _8_3PI * awz * chi

    # Mindlin state-based slip-fraction ramp (b = remaining stiction fraction)
    znorm = xp.sqrt(zx * zx + zy * zy + 1e-18)
    dstar = (C.LG_STICTION_RATIO * mu) / C.LG_SIGMA0          # breakaway deflection
    sfrac = xp.minimum(xp.maximum(znorm / dstar, 0.0 * znorm), 1.0 + 0.0 * znorm)
    b = xp.maximum(1.0 - sfrac, 1e-9 + 0.0 * znorm)
    fsl = 1.0 - b ** (2.0 / 3.0)

    s_t = fsl * c_t + Vp
    s_s = _16_3PI * awz * chi + 5.0 * Vp

    g_t = mu * (1.0 + (C.LG_STICTION_RATIO - 1.0) * xp.exp(-((s_t / C.LG_V_STR) ** 2)))
    g_s = mu * (1.0 + (C.LG_STICTION_RATIO - 1.0) * xp.exp(-((s_s / C.LG_W_STR) ** 2)))

    dzx = Vpx - C.LG_SIGMA0 * s_t / g_t * zx
    dzy = Vpy - C.LG_SIGMA0 * s_t / g_t * zy
    dzs = w_z - C.LG_SIGMA0_S * s_s / g_s * zs

    Fx = -N * (C.LG_SIGMA0 * zx + C.LG_SIGMA1 * dzx + C.LG_SIGMA2 * Vpx)
    Fy = -N * (C.LG_SIGMA0 * zy + C.LG_SIGMA1 * dzy + C.LG_SIGMA2 * Vpy)
    Mz = -N * chi * chi * (C.LG_SIGMA0_S * zs + C.LG_SIGMA1_S * dzs + C.LG_SIGMA2_S * w_z)
    return Fx, Fy, Mz


# ---------------------------------------------------------------------------
# Contact kinematics from gamma (per wheel). cti/sti are cos/sin of theta_tilde
# at the prediction time; DY = Rd*tan(delta)*tan(theta_tilde). All inputs except
# gamma are measurable. Returns Vpx, Vpy, w_z and the gamma-sensitivities.
# ---------------------------------------------------------------------------
def contact_from_gamma(gamma, psi_dot, Vpx0, Vpy0, cti, sti):
    """gamma: [...,4]; psi_dot: [...]; Vpx0/Vpy0/cti/sti: [...,4].
    Returns (Vpx, Vpy, w_z, dVpx_dg, dVpy_dg, dwz_dg), each [...,4]."""
    sd = _as(gamma, C.SIN_DELTA); cd = _as(gamma, C.COS_DELTA); td = _as(gamma, C.TAN_DELTA)
    DY = C.RD * td * (sti / cti)                       # Rd*tan(delta)*tan(theta_t)
    dVpx_dg = sd * (C.RD * cti - C.R) + DY * cd * sti
    dVpy_dg = cd * (C.R * cti - C.RD)
    dwz_dg = sti * cd
    Vpx = Vpx0 + gamma * dVpx_dg
    Vpy = Vpy0 + gamma * dVpy_dg
    w_z = psi_dot[..., None] + gamma * dwz_dg
    return Vpx, Vpy, w_z, dVpx_dg, dVpy_dg, dwz_dg


def _as(ref, arr):
    """Broadcast a length-4 numpy constant to ref's backend/device as [1,...,4]."""
    try:
        import torch
        if isinstance(ref, torch.Tensor):
            return torch.as_tensor(arr, dtype=ref.dtype, device=ref.device)
    except Exception:
        pass
    return arr


def roller_residual(xp, gamma, zx, zy, zs, mu, chi,
                    psi_dot, Vpx0, Vpy0, cti, sti):
    """Quasi-static roller torque balance (inertial term dropped — ~1-3%, and
    its finite-diff estimate is noise-dominated). The Mz*dwz/dg term is also
    dropped: Mz carries chi^2 (~2.5e-5), so it is negligible here and Mz is
    low-SNR / not used per project decision.

      r = p2*gamma + Fx*dVpx/dg + Fy*dVpy/dg   (per wheel)  -> 0

    gamma/zx/zy/zs: [...,4]; mu/chi: [...]; returns r [...,4]."""
    Vpx, Vpy, w_z, dVpx, dVpy, _ = contact_from_gamma(gamma, psi_dot, Vpx0, Vpy0, cti, sti)
    N = _as(gamma, C.N_PER_ROLLER)
    Fx, Fy, _ = lugre_forces(xp, mu[..., None], N, chi[..., None],
                             w_z, Vpx, Vpy, zx, zy, zs)
    return C.P2 * gamma + Fx * dVpx + Fy * dVpy


def wheel_residual(xp, gamma, zx, zy, zs, mu, chi,
                   psi_dot, Vpx0, Vpy0, cti, sti, Msat, w, w_dot):
    """Wheel torque balance (RHS fully measurable):
      r = J_wheel*w_dot - (Msat - R*Fx - p1*w)   (per wheel)  -> 0
    Fx from the predicted-state LuGre forces. Msat/w/w_dot: [...,4]."""
    Vpx, Vpy, w_z, *_ = contact_from_gamma(gamma, psi_dot, Vpx0, Vpy0, cti, sti)
    N = _as(gamma, C.N_PER_ROLLER)
    Fx, _, _ = lugre_forces(xp, mu[..., None], N, chi[..., None], w_z, Vpx, Vpy, zx, zy, zs)
    return C.J_WHEEL * w_dot - (Msat - C.R * Fx - C.P1 * w)
