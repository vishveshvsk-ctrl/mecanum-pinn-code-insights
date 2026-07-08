#!/usr/bin/env python
# =============================================================================
# losses_v2.py — γ-only supervised loss + physics-regularised refinement.
#
# Supervised branch: weighted MSE on the normalised γ prediction.
# Physics branch: de-normalise γ̂, assemble the 4-state tuple
#   (γ̂, zx_lab, zy_lab, zs=0)
# and feed it to the existing v1 residual/integrated physics channels plus the
# promoted roller torque-balance term.  zx/zy are label tensors (detached), so
# all gradients reach γ̂ alone.
# =============================================================================
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from . import config as C
from . import physics as P
from .config_v2 import ROLLER_SCALE


def supervised_gamma_loss(gamma_hat: torch.Tensor, y: torch.Tensor,
                          sup_weight: torch.Tensor
                          ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Weighted MSE on normalised γ.

    Args:
        gamma_hat, y: [B, 4] normalised γ predictions and targets.
        sup_weight: [B] per-window scalar weight.
    Returns:
        (scalar loss, log dict)
    """
    se = ((gamma_hat - y) ** 2).mean(dim=-1)          # [B]
    loss = (se * sup_weight).mean()
    log = {
        "mse_gamma": float(se.mean().detach()),
        "mse_gamma_weighted": float(loss.detach()),
    }
    return loss, log


def _build_pred_state(gamma_hat_phys: torch.Tensor, phys: Dict[str, torch.Tensor]
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble (gamma, zx, zy, zs) from predicted γ and label bristles."""
    zx = phys["zx_lab"]
    zy = phys["zy_lab"]
    zs = torch.zeros_like(gamma_hat_phys)
    # Sanity: labels must not carry gradients.
    assert not zx.requires_grad and not zy.requires_grad
    return gamma_hat_phys, zx, zy, zs


def physics_loss_v2(gamma_hat_phys: torch.Tensor, phys: Dict[str, torch.Tensor],
                    variant: str, w_roller: float,
                    roller_slip_weighting: bool,
                    Minv: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Physics loss for v2.

    Args:
        gamma_hat_phys: [B, 4] predicted roller spin, PHYSICAL units, in-graph.
        phys: dict with measurables + mu/chi + zx_lab/zy_lab (physical, detached)
              + slip_mag + variant-specific terms.
        variant: {"residual", "integrated"}.
        w_roller: weight on the promoted roller term.
        roller_slip_weighting: if True, weight roller residual by slip magnitude.
        Minv: (3,3) torch inverse body mass matrix (required for integrated).
    Returns:
        (scalar loss, log dict) with per-wheel `phys_roller_w{i}` keys.
    """
    gamma, zx, zy, zs = _build_pred_state(gamma_hat_phys, phys)
    loss = gamma_hat_phys.new_zeros(())
    log: Dict[str, float] = {}

    if variant == "residual":
        r_wheel = P.wheel_residual(torch, gamma, zx, zy, zs,
                                   phys["mu"], phys["chi"],
                                   phys["psi_dot"], phys["Vpx0"], phys["Vpy0"],
                                   phys["cti"], phys["sti"],
                                   phys["Msat"], phys["w"], phys["w_dot"]) / C.WHEEL_SCALE
        sw = (r_wheel ** 2).mean(0)
        loss = loss + sw.sum()
        for i in range(4):
            log[f"phys_wheel_w{i+1}"] = float(sw[i].detach())

        r0, r1, r2 = P.body_residual(torch, gamma, zx, zy, zs,
                                     phys["mu"], phys["chi"],
                                     phys["psi_dot"], phys["Vpx0"], phys["Vpy0"],
                                     phys["cti"], phys["sti"],
                                     phys["Vx"], phys["Vy"],
                                     phys["dVx"], phys["dVy"], phys["dpsi_dot"])
        lx, ly, lyaw = (r0 ** 2).mean(), (r1 ** 2).mean(), (r2 ** 2).mean()
        loss = loss + lx + ly + lyaw
        log.update(phys_body_x=float(lx.detach()),
                   phys_body_y=float(ly.detach()),
                   phys_body_yaw=float(lyaw.detach()))

    elif variant == "integrated":
        if Minv is None:
            raise ValueError("integrated variant requires Minv")
        Vx_n, Vy_n, pd_n, w_n = P.integrated_step(
            torch, gamma, zx, zy, zs, phys["mu"], phys["chi"], phys["psi_dot"],
            phys["Vpx0"], phys["Vpy0"], phys["cti"], phys["sti"],
            phys["Vx"], phys["Vy"], phys["w"], phys["Msat"], Minv, C.T_S)
        eVx = (Vx_n - phys["Vx_next"]) / C.PRED_P95["Vx"]
        eVy = (Vy_n - phys["Vy_next"]) / C.PRED_P95["Vy"]
        ePd = (pd_n - phys["psi_dot_next"]) / C.PRED_P95["psi_dot"]
        eW = (w_n - phys["w_next"]) / C.PRED_P95["w"]
        lVx, lVy, lPd = (eVx ** 2).mean(), (eVy ** 2).mean(), (ePd ** 2).mean()
        sw = (eW ** 2).mean(0)
        loss = loss + lVx + lVy + lPd + sw.sum()
        log.update(phys_int_Vx=float(lVx.detach()),
                   phys_int_Vy=float(lVy.detach()),
                   phys_int_psidot=float(lPd.detach()))
        for i in range(4):
            log[f"phys_int_w{i+1}"] = float(sw[i].detach())

    else:
        raise ValueError(f"Unknown physics_variant: {variant}")

    # Promoted roller term — trained in v2.
    r_roll = P.roller_residual(torch, gamma, zx, zy, zs,
                               phys["mu"], phys["chi"],
                               phys["psi_dot"], phys["Vpx0"], phys["Vpy0"],
                               phys["cti"], phys["sti"]) / ROLLER_SCALE
    sr = r_roll ** 2  # [B,4]
    if roller_slip_weighting:
        # Upweight high-slip rollers; guard the denominator.
        slip = phys["slip_mag"]
        weights = slip / (slip + 0.1)
        sr = sr * weights
    sr_per_wheel = sr.mean(0)
    loss = loss + w_roller * sr_per_wheel.sum()
    for i in range(4):
        log[f"phys_roller_w{i+1}"] = float(sr_per_wheel[i].detach())

    return loss, log
