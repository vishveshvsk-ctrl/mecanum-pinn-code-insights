#!/usr/bin/env python
# =============================================================================
# losses.py — per-state normalised supervised loss.
#
# Targets are z-scored per state in the data pipeline, so plain MSE in that
# space IS the per-state normalised MSE -> the 4 states contribute on equal
# footing and the loss is not dominated by the largest-magnitude state. We do
# NOT up-weight z: its expected irreducible floor (non-uniqueness, handoff §4.4)
# is a result to MEASURE, not a term to fight. Purely supervised for now (the
# self-supervised future-measurable term stays deferred, handoff §5).
# =============================================================================
from __future__ import annotations

from typing import Dict, Tuple

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


def physics_loss(pred_phys: torch.Tensor, phys: Dict[str, torch.Tensor]
                 ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Quasi-static roller + wheel torque-balance residuals from the PREDICTED
    (physical) states. pred_phys: [B,4 wheels,4 states] = (gamma,zx,zy,zs) in
    physical units. `phys` carries the per-sample measurables + mu/chi at the
    prediction time. Mz term is excluded (low-SNR, chi^2-tiny). Returns
    (scalar loss, per-term dict)."""
    gamma = pred_phys[:, :, 0]; zx = pred_phys[:, :, 1]; zy = pred_phys[:, :, 2]
    zs = torch.zeros_like(gamma)         # zs not predicted; Mz unused in residuals
    r_roll = P.roller_residual(torch, gamma, zx, zy, zs, phys["mu"], phys["chi"],
                               phys["psi_dot"], phys["Vpx0"], phys["Vpy0"],
                               phys["cti"], phys["sti"]) / C.ROLLER_TAU
    r_wheel = P.wheel_residual(torch, gamma, zx, zy, zs, phys["mu"], phys["chi"],
                               phys["psi_dot"], phys["Vpx0"], phys["Vpy0"],
                               phys["cti"], phys["sti"],
                               phys["Msat"], phys["w"], phys["w_dot"]) / C.MAX_TORQUE
    l_roll = (r_roll ** 2).mean()
    l_wheel = (r_wheel ** 2).mean()
    loss = l_roll + l_wheel
    return loss, {"phys_roller": float(l_roll.detach()),
                  "phys_wheel": float(l_wheel.detach())}
