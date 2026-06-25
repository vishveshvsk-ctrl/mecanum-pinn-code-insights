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


# ---------------------------------------------------------------------------
# Mass/inertia constants for the velocity-propagation metric. These are the
# same platform parameters used by the force-reconstruction integrator
# (Mecanum_PINN_Mamba_ForceRecon_v1/mecanum_pinn/physics.py).
# ---------------------------------------------------------------------------
_MS = C.M_PLATFORM + 4.0 * C.M_WHEEL


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


def velocity_propagation_loss(pred_phys: torch.Tensor, phys: Dict[str, torch.Tensor],
                              dt: float | None = None,
                              normalize: str = "none"
                              ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Analytically integrate platform velocity one step and compare to measurement.

    This is the force-recon-style Newton-Euler consistency check adapted to the
    observer: predicted hidden states -> LuGre forces -> body generalized force ->
    [Vx, Vy, psi_dot] propagation. It can be used as either an extra loss term or
    (more commonly) a diagnostic metric.

    Inputs
    ------
    pred_phys: [B, 4 wheels, 4] physical predictions (gamma, zx, zy, zs_dummy).
    phys: dict with keys returned by data.make_windows when physics_loss=True,
          including ph_Vx, ph_Vy, ph_psi_dot, ph_Vx_next, ph_Vy_next,
          ph_psid_next, ph_vp_valid, plus the usual contact kinematics.
    dt: time step in seconds. None -> 1 / TRAIN_HZ (2 ms @ 500 Hz).
    normalize: "none" | "p95" | "range".
        - "none": raw SI MSE (m/s)^2 and (rad/s)^2.
        - "p95": divide velocities by their p95 scales before squaring.
                 Uses frozen deployment scales (VX_P95, VY_P95, WZ_P95).
        - "range": divide by per-batch range (max - min) of the target.

    Returns (loss, log_dict).  The scalar loss averages Vx/Vy/psi_dot MSE.
    """
    if dt is None:
        dt = C.T_S

    gamma = pred_phys[:, :, 0]
    zx = pred_phys[:, :, 1]
    zy = pred_phys[:, :, 2]
    zs = torch.zeros_like(gamma)

    # Recompute contact kinematics and body-frame forces from predicted states.
    Vpx, Vpy, w_z, *_ = P.contact_from_gamma(
        gamma, phys["psi_dot"], phys["Vpx0"], phys["Vpy0"], phys["cti"], phys["sti"]
    )
    N = P._as(gamma, C.N_PER_ROLLER)
    # mu/chi are [B] in the phys dict; expand to [B,1] so lugre_forces broadcasts
    # over the wheel axis correctly.
    mu_b = phys["mu"][:, None] if phys["mu"].ndim == 1 else phys["mu"]
    chi_b = phys["chi"][:, None] if phys["chi"].ndim == 1 else phys["chi"]
    Fx, Fy, _ = P.lugre_forces(torch, mu_b, N, chi_b, w_z, Vpx, Vpy, zx, zy, zs)

    # Current measured platform velocity.
    state = torch.stack([phys["Vx"], phys["Vy"], phys["psi_dot"]], dim=-1)  # [B, 3]

    # Analytical one-step Euler propagation.
    state_pred = P.forward_velocity_step(
        state, Fx, Fy, dt,
        px=P._as(state, C.PX), py=P._as(state, C.PY),
        ms=_MS, m=C.M_PLATFORM, aX=C.AX, aY=C.AY, Is=C.Is,
        method="euler",
    )

    # Target: measured next-step velocity. Only valid where a t+1 sample exists.
    target = torch.stack([phys["Vx_next"], phys["Vy_next"], phys["psid_next"]], dim=-1)
    valid = phys["vp_valid"].to(dtype=torch.bool)       # [B]
    if valid.any():
        err = state_pred[valid] - target[valid]          # [B_valid, 3]
    else:
        err = torch.zeros_like(state_pred)

    if normalize == "p95":
        # Frozen deployment p95 scales. VX/VY p95 not currently in config,
        # so we use the empirical training-set p95 ~ 0.5 m/s unless overridden.
        # Keep this explicit so the user can swap in the deployment value.
        vx_p95 = getattr(C, "VX_P95", 0.5)
        vy_p95 = getattr(C, "VY_P95", 0.5)
        wz_p95 = C.WZ_P95
        scales = torch.tensor([vx_p95, vy_p95, wz_p95],
                              dtype=err.dtype, device=err.device)
        err = err / scales
    elif normalize == "range":
        rng = (target[valid].max(dim=0)[0] - target[valid].min(dim=0)[0]).clamp_min(1e-6)
        err = err / rng

    mse = (err ** 2).mean(dim=0)                         # [3]
    loss = mse.mean()

    log = {
        "vp_Vx": float(mse[0].detach()),
        "vp_Vy": float(mse[1].detach()),
        "vp_psi_dot": float(mse[2].detach()),
        "vp_n_valid": int(valid.sum().item()),
    }
    return loss, log
