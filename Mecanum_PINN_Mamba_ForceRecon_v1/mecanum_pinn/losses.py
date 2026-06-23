"""Losses: force grounding + discrete NE-integration residual + consistency +
parameter-ID, with slip/spin confidence gating.

Physics loss is the DISCRETE Newton-Euler residual (reverse-mode; no JVP): the
predicted force, integrated one step by physics.forward_integrate (Heun, O(dt^3),
force sampled at both endpoints), must reproduce the measured next state. The
same `ne_residual` is reused by the forward and inverse stages.

Batch (from data.MecanumTrajectoryDataset): (S, U, T, S_next, F_sim, mu, chi),
S/U/S_next/F_sim NORMALIZED (by data.state_max/control_max/force_max). The
integrator works in PHYSICAL units, so we de-normalize before and re-normalize
after using the torch scaling tensors populated by data.init_torch_globals().
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from . import data
from .physics import RobotParams, forward_integrate


# ============================================================
# De/re-normalization helpers (use the device-resident scaling tensors)
# ============================================================
def _state_max() -> torch.Tensor:
    return data.state_max_torch                     # (11,)


def _ctrl_max() -> torch.Tensor:
    return data.control_max_torch.squeeze(0)        # (4,)


def _force_max() -> torch.Tensor:
    return data.force_max_torch.squeeze(0)          # (8,)


# ============================================================
# Shared discrete Newton-Euler residual
# ============================================================
def ne_residual(S_norm: torch.Tensor, U_norm: torch.Tensor,
                F_phys: torch.Tensor, S_next_norm: torch.Tensor,
                rp: RobotParams, dt: float) -> torch.Tensor:
    """One-step Heun residual over a window. All sequence tensors aligned on dim 1.

    S_norm,(S_next_norm) (B,L,11) normalized; U_norm (B,L,4) normalized;
    F_phys (B,L,8) PHYSICAL. Predicts S_{t+1} from S_t for t=0..L-2 using
    F_t and F_{t+1} (full O(dt^3) Heun). Returns scalar MSE (normalized units).
    """
    sm = _state_max(); cm = _ctrl_max()
    S_phys = S_norm * sm
    U_phys = U_norm * cm

    state_t = S_phys[:, :-1]
    F_t,  F_next  = F_phys[:, :-1], F_phys[:, 1:]
    U_t,  U_next  = U_phys[:, :-1], U_phys[:, 1:]
    S_pred = forward_integrate(state_t, F_t, U_t, rp, dt,
                               F_next=F_next, Msat_next=U_next)   # (B,L-1,11) physical
    S_pred_norm = S_pred / sm
    return F.mse_loss(S_pred_norm, S_next_norm[:, :-1])


# ============================================================
# Forward-stage losses
# ============================================================
def forward_losses(F_phys: torch.Tensor, batch: Tuple, rp: RobotParams,
                   cfg: Dict) -> Dict[str, torch.Tensor]:
    """Component losses for the forward stage. Weighting is done by the trainer."""
    S, U, _T, S_next, F_sim, _mu, _chi = batch
    grnd = F.mse_loss(F_phys / _force_max(), F_sim)
    phys = ne_residual(S, U, F_phys, S_next, rp, cfg['dt'])
    return {'grnd': grnd, 'phys': phys}


# ============================================================
# Inverse-stage losses
# ============================================================
def inverse_losses(F_inv: torch.Tensor, F_fwd_phys: torch.Tensor,
                   batch: Tuple, rp: RobotParams, cfg: Dict) -> Dict[str, torch.Tensor]:
    """Component losses for the inverse stage (weighting done by the trainer).

    mu-ID is NOT here -- it is a TEST-TIME residual readout (see evaluation.evaluate_mu_id);
    chi-ID is deferred. Consistency is computed for monitoring only (w_cons=0).
    """
    S, U, _T, S_next, F_sim, _mu, _chi = batch
    off = max(2, int(cfg['inv_window'])) - 1
    fm = _force_max()
    grnd = F.mse_loss(F_inv / fm, F_sim[:, off:])
    cons = F.mse_loss(F_inv / fm, F_fwd_phys[:, off:] / fm)          # monitored, not trained
    phys = ne_residual(S[:, off:], U[:, off:], F_inv, S_next[:, off:], rp, cfg['dt'])
    return {'grnd': grnd, 'cons': cons, 'phys': phys}
