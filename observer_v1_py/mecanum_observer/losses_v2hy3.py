#!/usr/bin/env python
# =============================================================================
# losses_v2hy3.py — supervised γ + ΔV loss + regime-split physics for Hy3.
#
# Physics loss is GT-free at deployment:
#   Vpx0_u = Vpx0_hat + ΔV̂x ;  Vpy0_u = Vpy0_hat + ΔV̂y
#   (Vpx0_hat ALREADY carries V̂ from the front-end, so ONLY the ΔV̂ correction is
#    added here — adding V̂+ΔV̂ double-counts V̂. See sensor_frontend_v2.py:302.)
#   stick: γ̂_phys → γ_kin(Vpx0_u, Vpy0_u)
#   slip:  roller_residual_ss(γ̂_phys, Vpx0_u, Vpy0_u)
#   gate g = σ((|Vp| − gate_center)/gate_width)  (model-derived)
#   both branches scaled to normalized-γ units for dimensionless blending.
# =============================================================================
from __future__ import annotations

from typing import Dict, Tuple

import torch

from . import config as C
from . import physics_v2hy3 as P
from .config_v2 import GAMMA_P95_DEFAULT
from .config_v2hy3 import ObserverConfigV2Hy3


def supervised_loss(gamma_hat: torch.Tensor, dv_hat: torch.Tensor,
                    y_gamma: torch.Tensor, y_dv: torch.Tensor,
                    sup_weight: torch.Tensor, w_dv: float
                    ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Weighted supervised loss on both heads.

    L_sup = MSE(γ̂, y_gamma; sup_weight) + w_dv · MSE(ΔV̂, y_dv).
    """
    se_gamma = ((gamma_hat - y_gamma) ** 2).mean(dim=-1)      # [B]
    loss_gamma = (se_gamma * sup_weight).mean()

    se_dv = ((dv_hat - y_dv) ** 2).mean(dim=-1)               # [B]
    loss_dv = se_dv.mean()

    loss = loss_gamma + w_dv * loss_dv
    log = {
        "mse_gamma": float(se_gamma.mean().detach()),
        "mse_gamma_weighted": float(loss_gamma.detach()),
        "mse_dv": float(se_dv.mean().detach()),
        "mse_dv_weighted": float(loss_dv.detach() * w_dv),
        "sup_gamma": float(loss_gamma.detach()),
        "sup_dv": float(loss_dv.detach()),
    }
    return loss, log


def derived_contact_slip(gamma_hat: torch.Tensor, dv_hat: torch.Tensor,
                         V_hat: torch.Tensor, phys: Dict[str, torch.Tensor],
                         cfg: ObserverConfigV2Hy3,
                         gamma_std_t: torch.Tensor, gamma_mean_t: torch.Tensor
                         ) -> torch.Tensor:
    """Model-DERIVED per-wheel contact slip speed |Vp|  [B,4], physical units.

    Identical to the quantity the physics gate consumes:
        Vpx0_u = Vpx0_hat + ΔV̂x ; Vpy0_u = Vpy0_hat + ΔV̂y
        Vpx,Vpy = contact_from_gamma(γ̂_phys, ψ̇, Vpx0_u, Vpy0_u, cti, sti)
        |Vp| = sqrt(Vpx² + Vpy² + eps²)
    Supervising this toward the true vpm forces (γ̂, ΔV̂) to reconstruct the true
    contact slip — which is what makes the gate usable. No new network head.

    V_hat is accepted but UNUSED: Vpx0_hat is built from V̂ already, so the ΔV̂ head
    supplies only the correction. (Kept in the signature for call-site stability.)
    """
    gamma_phys = gamma_hat * gamma_std_t + gamma_mean_t
    dv_phys = dv_hat * torch.tensor(cfg.dv_scale, dtype=dv_hat.dtype,
                                    device=dv_hat.device)
    # Vpx0_hat ALREADY carries V̂ (front-end: V̂ − ψ̇·(PY+DY) − w·R) -> add ONLY the
    # ΔV̂ correction. Adding V̂+ΔV̂ double-counts V̂ (~0.5-1 m/s) and was the run-1 bug.
    Vpx0_u = phys["Vpx0_hat"] + dv_phys[:, 0:1]
    Vpy0_u = phys["Vpy0_hat"] + dv_phys[:, 1:2]
    Vpx, Vpy, _, _, _, _ = P.contact_from_gamma(
        gamma_phys, phys["psi_dot"], Vpx0_u, Vpy0_u, phys["cti"], phys["sti"])
    return torch.sqrt(Vpx * Vpx + Vpy * Vpy + C.LG_EPS_REG * C.LG_EPS_REG)


def slip_consistency_loss(v_slip: torch.Tensor, vpm_true: torch.Tensor,
                          vpm_scale: float) -> torch.Tensor:
    """Normalized MSE between derived |Vp| and the true per-wheel vpm."""
    return ((v_slip - vpm_true) ** 2).mean() / (vpm_scale * vpm_scale)


def physics_loss_hy3(gamma_hat: torch.Tensor, dv_hat: torch.Tensor,
                     V_hat: torch.Tensor, phys: Dict[str, torch.Tensor],
                     cfg: ObserverConfigV2Hy3,
                     gamma_std_t: torch.Tensor, gamma_mean_t: torch.Tensor
                     ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Regime-split, GT-free physics loss for Hy3.

    Args:
        gamma_hat: [B,4] predicted γ, normalized
        dv_hat:    [B,2] predicted ΔV, normalized by dv_scale
        V_hat:     [B,2] front-end velocity V̂ (physical units), from Gw
        phys:      dict with Vpx0_hat, Vpy0_hat, cti, sti, psi_dot, mu, chi, w
        gamma_std_t/gamma_mean_t: 1x1 tensors for in-graph γ de-normalization
    Returns:
        (scalar loss, log dict)
    """
    # De-normalize network outputs (in-graph).
    gamma_phys = gamma_hat * gamma_std_t + gamma_mean_t          # [B,4]
    dv_phys = dv_hat * torch.tensor(cfg.dv_scale, dtype=dv_hat.dtype,
                                    device=dv_hat.device)        # [B,2]

    # Vpx0_hat ALREADY carries V̂ (front-end: V̂ − ψ̇·(PY+DY) − w·R) -> add ONLY the
    # ΔV̂ correction (broadcast over 4 wheels). Adding V̂+ΔV̂ double-counts V̂
    # (~0.5-1 m/s) and was the run-1 bug. `dv_corr_*` are VELOCITIES (the ΔV head) —
    # not the sidecar dVx/dVy, which are accelerations.
    dv_corr_x = dv_phys[:, 0:1]                                   # [B,1]
    dv_corr_y = dv_phys[:, 1:2]                                   # [B,1]

    Vpx0_u = phys["Vpx0_hat"] + dv_corr_x                         # [B,4]
    Vpy0_u = phys["Vpy0_hat"] + dv_corr_y                         # [B,4]

    # Geometry sensitivities at the corrected base (γ-independent).
    _, _, _, dVpx_dg, dVpy_dg, _ = P.contact_from_gamma(
        torch.zeros_like(Vpx0_u), phys["psi_dot"], Vpx0_u, Vpy0_u,
        phys["cti"], phys["sti"])

    # Stick residual: γ̂_phys should match the pure-rolling kinematic γ.
    gamma_kin = P.gamma_kin(Vpx0_u, Vpy0_u, dVpx_dg, dVpy_dg)
    r_stick = (gamma_phys - gamma_kin) / GAMMA_P95_DEFAULT

    # Slip residual: steady-state roller torque balance.
    r_roll = P.roller_residual_ss(
        gamma_phys, phys["mu"], phys["chi"], phys["psi_dot"],
        Vpx0_u, Vpy0_u, phys["cti"], phys["sti"], cfg.mindlin_iters)
    r_slip = r_roll / (C.P2 * GAMMA_P95_DEFAULT)

    # Model-derived slip gate from γ̂_phys and V_used.
    Vpx, Vpy, _, _, _, _ = P.contact_from_gamma(
        gamma_phys, phys["psi_dot"], Vpx0_u, Vpy0_u, phys["cti"], phys["sti"])
    Vp_mag = torch.sqrt(Vpx * Vpx + Vpy * Vpy + C.LG_EPS_REG * C.LG_EPS_REG)
    g = torch.sigmoid((Vp_mag - cfg.gate_center) / cfg.gate_width)

    # Regime-blended loss, per wheel.
    loss_per_wheel = (1.0 - g) * (r_stick ** 2) + g * (r_slip ** 2)
    loss = loss_per_wheel.mean()

    # Per-wheel log keys + gate-distribution diagnostics (all fractions, so the
    # training-side sum/nb aggregation yields the correct per-epoch mean).
    # g→1 is slip, g→0 is stick; g_stick_frac answers "does the stick branch ever
    # fire?" — the open question from the smoke, where g_mean pinned at ~1.0.
    g_det = g.detach()
    log: Dict[str, float] = {
        "g_mean": float(g_det.mean()),
        "g_stick_frac": float((g_det < 0.5).float().mean()),   # fraction in stick regime
        "g_lt01_frac": float((g_det < 0.1).float().mean()),    # deep-stick fraction
    }
    for i in range(C.N_WHEELS):
        log[f"phys_stick_w{i+1}"] = float((r_stick[:, i] ** 2).mean().detach())
        log[f"phys_slip_w{i+1}"] = float((r_slip[:, i] ** 2).mean().detach())

    return loss, log
