"""Physics residuals, time-weighted losses, autocast, and the compute_losses dispatcher.

v13 — forward physics residual is JVP-based, identical mechanism to v12.
Each output of forward_path depends on the T input (per-wheel GRU sees T as
a feature; dec3 takes T directly), so
    S_dot = jvp(forward_path, (T,), (ones_like(T),))
returns a meaningful directional derivative that the residual treats as
dS/dt. The model learns to make this sensitivity match the physics RHS.

v14 θ-rework:
  Forward S_pred is now 7-dim (V + ω) — θ is not predicted. The
  state-loss target is sliced to S_next[..., :7]. The forward physics
  residual drops its θ-identity term (R_theta) and is left with the
  3 platform-velocity and 4 wheel-velocity components; no C_dot / S_dot
  plumbing is needed any more.

Per-component diagnostics: every loss component that admits a structural
breakdown is exposed alongside its scalar form, under the same key with a
`_per_comp` suffix. These are tensors of fixed length that get accumulated
diagnostically by `_epoch_loop` and printed per epoch — they're for human
inspection only, never enter `aggregate_total_loss`.

  Forward stage                 Shape  Component order
  ----------------------------- -----  ------------------------------------
  state_per_comp                (7,)   V_X, V_Y, Ω, ω_1..4
  grnd_per_comp                 (12,)  Fx_1..4, Fy_1..4, Mz_1..4
  phys_per_comp                 (7,)   V_X_resid, V_Y_resid, Ω_resid,
                                       ω_1..4_resid

  Inverse stage                 Shape  Component order
  ----------------------------- -----  ------------------------------------
  grnd_per_comp                 (12,)  Fx_1..4, Fy_1..4, Mz_1..4
  cons_per_comp                 (12,)  Fx_1..4, Fy_1..4, Mz_1..4
  phys_per_comp                 (7,)   V_X_resid, V_Y_resid, Ω_resid,
                                       ω_1..4_resid

Note: the physics per-component values are *normalized* residuals (divided
by force_max and control_max). Don't compare their magnitudes directly
against state_per_comp or grnd_per_comp; what's diagnostic is the relative
balance *within* each breakdown.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch.func import jvp

from . import data as _data    # imported as module so we see post-init torch tensors
from .physics import RobotParams

# ============================================================
# Continuous-style forward residual (JVP-T derived rates)
# ============================================================
def physics_residual_continuous_forward(
    rp: RobotParams, state_curr_pred, control_t, F_pred, p1,
    v_dot, w_dot, force_max, control_max,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (per_sample_scalar, per_sample_per_comp).

    per_sample_scalar : (B*K,)         — original loss target
    per_sample_per_comp : (B*K, 7)     — V_X, V_Y, Ω, ω_1..4 sq residuals

    state_curr_pred is 7-dim here ([V, ω] only); θ-identity residual is
    no longer computed since the forward model doesn't predict θ.
    """
    Vx, Vy, psi_dot = state_curr_pred[:, 0], state_curr_pred[:, 1], state_curr_pred[:, 2]
    w_curr     = state_curr_pred[:, 3:7]

    px = rp.wc_x.unsqueeze(0)
    py = rp.wc_y.unsqueeze(0)
    Fx = F_pred[:, 0:4]
    Fy = F_pred[:, 4:8]
    Mz = F_pred[:, 8:12]

    RHS0 = Fx.sum(dim=1) + rp.ms * psi_dot * Vy + rp.m * rp.aX * (psi_dot**2)
    RHS1 = Fy.sum(dim=1) - rp.ms * psi_dot * Vx + rp.m * rp.aY * (psi_dot**2)
    RHS2 = ((px * Fy - py * Fx).sum(dim=1)
            + Mz.sum(dim=1)
            - rp.m * psi_dot * (rp.aX * Vx + rp.aY * Vy))
    RHS  = torch.stack([RHS0, RHS1, RHS2], dim=1)

    scaler = torch.stack([force_max, force_max, control_max])
    R_platform = (v_dot @ rp.M.T - RHS) / scaler                  # (B*K, 3)
    R_wheel    = (rp.Jw_1 * w_dot - (control_t - rp.R1 * Fx - p1.unsqueeze(0) * w_curr)) / control_max  # (B*K, 4)

    R_platform_sq = R_platform.pow(2)                             # (B*K, 3)
    R_wheel_sq    = R_wheel.pow(2)                                # (B*K, 4)

    res = (R_platform_sq.mean(dim=1)
           + R_wheel_sq.mean(dim=1))                              # (B*K,)

    per_comp_sq = torch.cat([R_platform_sq, R_wheel_sq], dim=1)   # (B*K, 7)
    return res, per_comp_sq


def physics_loss_time_weighted_continuous_forward(
    rp: RobotParams, S_pred, U, T, F_fwd, p1, v_dot, w_dot,
    force_max_torch, control_max_torch, state_max_torch,
    tau: float = 0.5, k_steps: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (phys_loss_scalar, phys_per_comp_7).

    S_pred is (B, L, 7) here. v_dot is (B, L, 3) and w_dot is (B, L, 4),
    both directional derivatives of S_pred w.r.t. T from the JVP call.
    """
    B, total_len, _ = U.shape
    K = min(k_steps, total_len)

    S_k = S_pred[:, :K, :]            # (B, K, 7)
    U_k = U[:, :K, :]
    F_k = F_fwd[:, :K, :]
    v_dot_k, w_dot_k = v_dot[:, :K, :], w_dot[:, :K, :]

    T_k = T[:, :K]
    if T_k.dim() == 2:
        T_k = T_k.unsqueeze(-1)
    t0 = T_k[:, 0:1, :]
    w_k = torch.exp(-(T_k - t0) / tau)

    # De-normalize S_pred and derivatives to physical units. state_max_torch
    # is 11-dim ([V (3), ω (4), θ (4)]); only the first 7 entries are
    # relevant for S_pred.
    state_max_7 = state_max_torch[:7]                              # (7,)
    state_curr_flat = S_k.reshape(B * K, -1) * state_max_7
    control_flat    = U_k.reshape(B * K, -1) * control_max_torch
    F_fwd_flat      = F_k.reshape(B * K, -1) * force_max_torch
    v_dot_flat      = v_dot_k.reshape(B * K, -1) * state_max_torch[0:3]
    w_dot_flat      = w_dot_k.reshape(B * K, -1) * state_max_torch[3:7]
    w_k_flat        = w_k.reshape(B * K)

    res_flat, per_comp_sq = physics_residual_continuous_forward(
        rp=rp, state_curr_pred=state_curr_flat, control_t=control_flat,
        F_pred=F_fwd_flat, p1=p1,
        v_dot=v_dot_flat, w_dot=w_dot_flat,
        force_max=force_max_torch[0], control_max=control_max_torch[0],
    )
    # Scalar loss: time-weight, mean over batch, sum over time
    weighted = (w_k_flat * res_flat).view(B, K)
    phys_loss = weighted.mean(dim=0).sum()

    # Per-component: same weighting & aggregation, kept per-component
    weighted_pc = (w_k_flat.unsqueeze(1) * per_comp_sq).view(B, K, -1)
    phys_per_comp = weighted_pc.mean(dim=0).sum(dim=0)            # (7,)

    return phys_loss, phys_per_comp


# ============================================================
# Discrete inverse residual — K off-by-one fix for short sequences
# ============================================================
def physics_residual_discrete_inverse(
    rp: RobotParams, state_t, state_tp1, control_t, dt_k, F_inv, p1,
    force_max, control_max,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (per_sample_scalar, per_sample_per_comp_7).

    per_sample_per_comp : (B*K, 7) — V_X, V_Y, Ω, ω_1..4 sq residuals
    (No θ component in the discrete inverse residual.)
    """
    Vx, Vy, psi_dot = state_t[:, 0], state_t[:, 1], state_t[:, 2]
    w = state_t[:, 3:7]
    Vx_n, Vy_n, psi_dot_n = state_tp1[:, 0], state_tp1[:, 1], state_tp1[:, 2]
    w_n = state_tp1[:, 3:7]

    px = rp.wc_x.unsqueeze(0)
    py = rp.wc_y.unsqueeze(0)
    Fx = F_inv[:, 0:4]
    Fy = F_inv[:, 4:8]
    Mz = F_inv[:, 8:12]

    RHS0 = Fx.sum(dim=1) + rp.ms * psi_dot * Vy + rp.m * rp.aX * (psi_dot**2)
    RHS1 = Fy.sum(dim=1) - rp.ms * psi_dot * Vx + rp.m * rp.aY * (psi_dot**2)
    RHS2 = ((px * Fy - py * Fx).sum(dim=1)
            + Mz.sum(dim=1)
            - rp.m * psi_dot * (rp.aX * Vx + rp.aY * Vy))
    RHS  = torch.stack([RHS0, RHS1, RHS2], dim=1)

    v    = torch.stack([Vx,   Vy,   psi_dot  ], dim=1)
    v_n  = torch.stack([Vx_n, Vy_n, psi_dot_n], dim=1)
    dt_col = dt_k.unsqueeze(1)

    scaler = torch.stack([force_max, force_max, control_max])
    R_platform = ((v_n - v) @ rp.M.T - dt_col * RHS) / scaler                                          # (B*K, 3)
    R_wheel    = (rp.Jw_1 * (w_n - w) - dt_col * (control_t - rp.R1 * Fx - p1.unsqueeze(0) * w_n)) / control_max  # (B*K, 4)

    R_platform_sq = R_platform.pow(2)
    R_wheel_sq    = R_wheel.pow(2)
    res = R_platform_sq.mean(dim=1) + R_wheel_sq.mean(dim=1)      # (B*K,)
    per_comp_sq = torch.cat([R_platform_sq, R_wheel_sq], dim=1)   # (B*K, 7)
    return res, per_comp_sq


def physics_loss_time_weighted_discrete_inverse(
    rp: RobotParams, S_true, U, T, F_inv, p1,
    force_max_torch, control_max_torch, state_max_torch,
    tau: float = 0.5, k_steps: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverse residual: (S[t+1] − S[t])/dt vs RHS(state_t).

    Returns (phys_loss_scalar, phys_per_comp_7).

    v12.5: K = min(k_steps, L − 1) (was `min(k_steps, L)` in v12). With L
    states we have at most L−1 transitions; the v12 form over-runs the
    buffer at L = k_steps. Invisible when seq_len > k_steps.
    """
    B, total_len, _ = U.shape
    K = min(k_steps, max(1, total_len - 1))

    S_k = S_true[:, :K+1, :]
    U_k = U[:, :K, :]
    F_k = F_inv[:, :K, :]

    T_k = T[:, :K+1]
    if T_k.dim() == 2:
        T_k = T_k.unsqueeze(-1)
    t0    = T_k[:, 0:1, :]
    t_k   = T_k[:, :-1, :]
    t_kp1 = T_k[:, 1:,  :]
    dt_k = (t_kp1 - t_k).clamp(min=1e-6)
    w_k  = torch.exp(-(t_k - t0) / tau)

    state_t_flat   = S_k[:, :-1, :].reshape(B * K, -1) * state_max_torch
    state_tp1_flat = S_k[:, 1:,  :].reshape(B * K, -1) * state_max_torch
    control_flat   = U_k.reshape(B * K, -1) * control_max_torch
    F_inv_flat     = F_k.reshape(B * K, -1) * force_max_torch
    dt_k_flat      = dt_k.reshape(B * K)
    w_k_flat       = w_k.reshape(B * K)

    res_flat, per_comp_sq = physics_residual_discrete_inverse(
        rp=rp, state_t=state_t_flat, state_tp1=state_tp1_flat,
        control_t=control_flat, dt_k=dt_k_flat, F_inv=F_inv_flat, p1=p1,
        force_max=force_max_torch[0], control_max=control_max_torch[0],
    )
    weighted = (w_k_flat * res_flat).view(B, K)
    phys_loss = weighted.mean(dim=0).sum()

    weighted_pc = (w_k_flat.unsqueeze(1) * per_comp_sq).view(B, K, -1)
    phys_per_comp = weighted_pc.mean(dim=0).sum(dim=0)            # (7,)

    return phys_loss, phys_per_comp


# ============================================================
# Autocast + dispatcher
# ============================================================
def autocast_ctx(config: Dict[str, Any]):
    """Context manager for mixed-precision forward/loss.

    bfloat16 on CUDA: keeps the physics-residual exp(-(t-t0)/tau) and
    force/moment dynamic range numerically clean without a GradScaler.

    When config['amp_enabled'] is False (e.g. the 24 GB Quadro Turing
    tier, which has no native bf16 path) we return a true nullcontext --
    skipping autocast entirely instead of asking it to be a no-op -- so
    the graph that torch.compile traces is identical to a plain fp32 run.
    """
    enabled = bool(config.get('amp_enabled', True)) and config['device'].type == 'cuda'
    if not enabled:
        return nullcontext()
    return torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True)


def _forward_losses(model, batch, rp: RobotParams, config: Dict[str, Any],
                    skip_phys: bool = False):
    S, U, T, S_next, F_sim, mu_b, chi_b = [x.to(config['device']) for x in batch]

    # S_pred is 7-dim (V + ω) — θ is not predicted. Match the loss target
    # by slicing S_next to the same 7 components.
    S_next_pred = S_next[..., :7]

    if skip_phys:
        S_pred, F_fwd, H_t = model.forward_path(
            S, U, T, mu_b, chi_b, rp.N_per_roller,
        )
        phys_loss = torch.tensor(0.0, device=config['device'])
        phys_per_comp = torch.zeros(7, device=config['device'])
    else:
        T_model = T.clone().detach().requires_grad_(True)

        def model_wrapper(t_input):
            S_pred, F_fwd, H_t = model.forward_path(
                S, U, t_input, mu_b, chi_b, rp.N_per_roller,
            )
            return S_pred, (F_fwd, H_t)

        S_pred, S_dot, (F_fwd, H_t) = jvp(
            model_wrapper, (T_model,),
            (torch.ones_like(T_model),), has_aux=True,
        )

        # S_pred is 7-dim; S_dot follows the same layout.
        v_dot = S_dot[..., 0:3]
        w_dot = S_dot[..., 3:7]

        phys_loss, phys_per_comp = physics_loss_time_weighted_continuous_forward(
            rp=rp, S_pred=S_pred, U=U, T=T, F_fwd=F_fwd, p1=rp.p1,
            v_dot=v_dot, w_dot=w_dot,
            force_max_torch=_data.force_max_torch[0],
            control_max_torch=_data.control_max_torch[0],
            state_max_torch=_data.state_max_torch,
            tau=config['tau'], k_steps=config['k_steps'],
        )

    # Per-component breakdowns for diagnostics (not in aggregate loss).
    # state_per_comp is 7-dim (no θ) since S_pred is 7-dim.
    state_per_comp = ((S_pred - S_next_pred) ** 2).mean(dim=(0, 1)).detach()  # (7,)
    grnd_per_comp  = ((F_fwd  - F_sim     ) ** 2).mean(dim=(0, 1)).detach()   # (12,)

    return {
        'state':           F.mse_loss(S_pred, S_next_pred),
        'grnd':            F.mse_loss(F_fwd, F_sim),
        'phys':            phys_loss,
        'state_per_comp':  state_per_comp,
        'grnd_per_comp':   grnd_per_comp,
        'phys_per_comp':   phys_per_comp.detach(),
        'F_fwd':           F_fwd.detach(),
        'H_t':             H_t.detach(),
    }


def _inverse_losses(model, batch, rp: RobotParams, config: Dict[str, Any],
                    skip_phys: bool = False):
    S, U, T, S_next, F_sim, mu_b, chi_b = [x.to(config['device']) for x in batch]

    # Forward pass produces detached H_t and F_fwd for the inverse model's
    # consistency loss. S_pred is 7-dim now but not used here.
    with torch.no_grad():
        _, F_fwd, H_t = model.forward_path(
            S, U, T.clone(), mu_b, chi_b, rp.N_per_roller,
        )
    F_fwd_d = F_fwd.detach()
    H_d     = H_t.detach() if model.inverse_model.use_H else None

    # Inverse model still takes 11-dim S_curr and S_next.
    F_inv = model.inverse_path(
        S_curr=S, S_next=S_next, U=U,
        N_per_wheel=rp.N_per_roller, H_detached=H_d,
    )

    if skip_phys:
        phys_loss = torch.tensor(0.0, device=config['device'])
        phys_per_comp = torch.zeros(7, device=config['device'])
    else:
        phys_loss, phys_per_comp = physics_loss_time_weighted_discrete_inverse(
            rp=rp, S_true=S, U=U, T=T, F_inv=F_inv, p1=rp.p1,
            force_max_torch=_data.force_max_torch[0],
            control_max_torch=_data.control_max_torch[0],
            state_max_torch=_data.state_max_torch,
            tau=config['tau'], k_steps=config['k_steps'],
        )

    # Per-component breakdowns for diagnostics (not in aggregate loss)
    grnd_per_comp = ((F_inv - F_sim ) ** 2).mean(dim=(0, 1)).detach()    # (12,)
    cons_per_comp = ((F_inv - F_fwd_d) ** 2).mean(dim=(0, 1)).detach()   # (12,)

    return {
        'grnd':           F.mse_loss(F_inv, F_sim),
        'cons':           F.mse_loss(F_inv, F_fwd_d),
        'phys':           phys_loss,
        'grnd_per_comp':  grnd_per_comp,
        'cons_per_comp':  cons_per_comp,
        'phys_per_comp':  phys_per_comp.detach(),
        'F_inv':          F_inv.detach(),
    }


def compute_losses(model, batch, rp: RobotParams, config: Dict[str, Any],
                   stage: str, w_phys: float = None):
    """Dispatcher.

    Skips physics-residual computation when w_phys == 0 and
    config['always_compute_phys'] is False; in that case `phys` is set
    to 0.0 so plotting / aggregation code stays uniform across phases.

    The full forward + loss path runs under bf16 autocast when
    config['amp_enabled'] is True (or under nullcontext otherwise).
    backward() should be called outside this context (standard idiom).
    """
    always = config.get('always_compute_phys', False)
    skip_phys = (w_phys is not None and w_phys == 0.0 and not always)

    with autocast_ctx(config):
        if stage == 'forward':
            return _forward_losses(model, batch, rp, config, skip_phys=skip_phys)
        elif stage == 'inverse':
            return _inverse_losses(model, batch, rp, config, skip_phys=skip_phys)
    raise ValueError(stage)


def aggregate_total_loss(losses: Dict[str, torch.Tensor], stage: str,
                         w_grnd: float, w_phys: float, w_cons: float
                         ) -> torch.Tensor:
    """Combine component losses into a scalar 'total' for backward().

    Per-component diagnostic tensors (anything with a `_per_comp` suffix)
    are not part of the loss aggregation.
    """
    if stage == 'forward':
        total = losses['state']
        if w_grnd > 0: total = total + w_grnd * losses['grnd']
        if w_phys > 0: total = total + w_phys * losses['phys']
    else:
        total = torch.tensor(0.0, device=losses['grnd'].device,
                             dtype=losses['grnd'].dtype)
        if w_grnd > 0: total = total + w_grnd * losses['grnd']
        if w_phys > 0: total = total + w_phys * losses['phys']
        if w_cons > 0: total = total + w_cons * losses['cons']
    return total
