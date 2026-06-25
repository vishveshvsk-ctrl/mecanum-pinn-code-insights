#!/usr/bin/env python
# =============================================================================
# losses.py — per-state normalised supervised loss + physics-loss variants.
#
# The physics-loss variants express the SAME one-step Newton-Euler plant
# constraint under two different scalings/normalisations:
#   - "residual"  : instantaneous force/accel residual (wheel torque + body EOM)
#   - "integrated": one Heun step from measured state using predicted forces,
#                   compared to measured next sample, normalised by p95.
# The (dropped) roller/gamma torque-balance term is computed every call for
# monitoring only and is never backpropagated.
# =============================================================================
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from . import config as C
from . import physics as P
from .config import N_STATES, TARGET_STATES


def observer_loss(pred: torch.Tensor, target: torch.Tensor
                  ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """pred/target: [B, 4 wheels, 4 states] in normalised units.

    Returns (scalar loss, per-state MSE dict for logging)."""
    se = (pred - target) ** 2                          # [B,4,4]
    per_state = se.mean(dim=(0, 1))                     # [4]
    loss = per_state.mean()
    log = {f"mse_{TARGET_STATES[s]}": float(per_state[s].detach())
           for s in range(N_STATES)}
    return loss, log


def physics_loss(pred_phys: torch.Tensor, phys: Dict[str, torch.Tensor],
                 variant: str = "integrated",
                 Minv: Optional[torch.Tensor] = None
                 ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Physics-loss dispatch: residual vs integrated. Roller is monitor-only.

    pred_phys: [B, 4 wheels, 4 states] = (gamma,zx,zy,zs_dummy) in physical units.
    phys: per-sample measurables + mu/chi + body-state terms from make_windows.
    variant: {"residual", "integrated"}.
    Minv: (3,3) torch inverse body mass matrix on device (required for integrated).
    Returns (scalar loss, per-component log dict).
    """
    gamma = pred_phys[:, :, 0]; zx = pred_phys[:, :, 1]; zy = pred_phys[:, :, 2]
    zs = torch.zeros_like(gamma)         # zs not predicted; Mz unused
    loss = pred_phys.new_zeros(())
    log: Dict[str, float] = {}

    if variant == "residual":
        # wheel (4 channels, /WHEEL_SCALE)
        r_wheel = P.wheel_residual(torch, gamma, zx, zy, zs,
                                   phys["mu"], phys["chi"],
                                   phys["psi_dot"], phys["Vpx0"], phys["Vpy0"],
                                   phys["cti"], phys["sti"],
                                   phys["Msat"], phys["w"], phys["w_dot"]) / C.WHEEL_SCALE
        sw = (r_wheel ** 2).mean(0)      # [4]
        loss = loss + sw.sum()
        for i in range(4):
            log[f"phys_wheel_w{i+1}"] = float(sw[i].detach())

        # body (3 channels, scaled inside body_residual)
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
        eW = (w_n - phys["w_next"]) / C.PRED_P95["w"]            # [B,4]
        lVx, lVy, lPd = (eVx ** 2).mean(), (eVy ** 2).mean(), (ePd ** 2).mean()
        sw = (eW ** 2).mean(0)                                    # [4]
        loss = loss + lVx + lVy + lPd + sw.sum()
        log.update(phys_int_Vx=float(lVx.detach()),
                   phys_int_Vy=float(lVy.detach()),
                   phys_int_psidot=float(lPd.detach()))
        for i in range(4):
            log[f"phys_int_w{i+1}"] = float(sw[i].detach())

    else:
        raise ValueError(f"Unknown physics_variant: {variant}")

    # ROLLER — MONITOR ONLY (never in loss), per wheel:
    r_roll = P.roller_residual(torch, gamma, zx, zy, zs,
                               phys["mu"], phys["chi"],
                               phys["psi_dot"], phys["Vpx0"], phys["Vpy0"],
                               phys["cti"], phys["sti"]) / C.ROLLER_SCALE
    sr = (r_roll ** 2).mean(0)
    for i in range(4):
        log[f"phys_roller_w{i+1}_MON"] = float(sr[i].detach())

    return loss, log
